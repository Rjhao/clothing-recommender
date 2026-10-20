"""知识图谱推荐基线：MKR / KGIN / CKAN

与 ``baselines_real.py`` 同接口、同评估协议（论文 表4-4 实验参数配置）：

    embedding_dim=64, Adam, lr=0.001, weight_decay=0,
    epochs=30, 训练负采样 1:4, BPR Loss, λ_KG=0.05, seed=42

评估统一走 ``baselines_real.evaluate_batch``（sampled-100 leave-one-out，
与 BPR-MF / LightGCN / RippleNet / KGCL 完全一致的候选集构造）。

三者共用 ``build_hetero_graph`` 产出的同一张协同知识图（CKG）：
6 类节点（user/item/l3/l2/l1/brand），4 类关系，87,370 节点 / 424,866 边。

    rel 0  user-item        （训练正样本边）
    rel 1  item-category    （item→L3 或 item→L2）
    rel 2  subcategory      （L3→L2、L2→L1）
    rel 3  item-brand

用法：
    python -m recommender.baselines_real --data data/amazon_fashion_real.json \
        --baselines mkr kgin ckan --epochs 30
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.kgat.hetero_graph import N_RELATIONS, build_hetero_graph, build_kg_triplets
from recommender.kgat.kg_embedding import TransR

logging = None  # 由 main() 注入，避免与 baselines_real 的 logger 冲突


# ============================================================================
# 共享工具
# ============================================================================

def node_class(n, off):
    """节点 id → 类别代号 'u'/'i'/'l3'/'l2'/'l1'/'b'。"""
    if n < off['n_users']:
        return 'u'
    if n < off['item'] + off['n_items']:
        return 'i'
    if n < off['l2']:
        return 'l3'
    if n < off['l1']:
        return 'l2'
    if n < off['brand']:
        return 'l1'
    return 'b'


def build_item_attr_index(edge_index, edge_type, node_offsets, max_attrs=2):
    """抽取每个商品的属性实体及其关系类型，供 MKR / CKAN 使用。

    CKG 中 item 的属性边只有两种：rel 1（类目）与 rel 3（品牌），且
    ``build_hetero_graph`` 双向存边，这里按邻居 id 去重。

    Returns:
        attr_ent:  LongTensor [n_items, max_attrs]  属性实体全局 id（-1 填充）
        attr_rel:  LongTensor [n_items, max_attrs]  该属性对应的 TransR 关系号
        attr_mask: BoolTensor  [n_items, max_attrs]  真实属性位
    """
    n_items = node_offsets['n_items']
    lo, hi = node_offsets['item'], node_offsets['item'] + n_items

    per_item = [[] for _ in range(n_items)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, t, ty in zip(src, dst, edge_type.tolist()):
        if ty not in (1, 3):
            continue
        s_item, t_item = lo <= s < hi, lo <= t < hi
        if s_item and t_item:          # 理论上不会出现
            continue
        if s_item:
            per_item[s - lo].append((t, ty))
        elif t_item:
            per_item[t - lo].append((s, ty))

    ent = torch.full((n_items, max_attrs), -1, dtype=torch.long)
    rel = torch.zeros((n_items, max_attrs), dtype=torch.long)
    mask = torch.zeros((n_items, max_attrs), dtype=torch.bool)
    for i, attrs in enumerate(per_item):
        # 按关系号排序，保证同一商品的属性位置稳定（可复现）
        uniq = {}
        for node, ty in attrs:
            uniq[(node, ty)] = True
        for slot, ((node, ty)) in enumerate(sorted(uniq, key=lambda x: (x[1], x[0]))):
            if slot >= max_attrs:
                break
            ent[i, slot] = node
            rel[i, slot] = ty
            mask[i, slot] = True
    return ent, rel, mask


def build_directed_adj(edge_index, edge_type, node_offsets, n_memory=0, seed=42):
    """把无向 CKG 拆成 KGIN 所需的 8 个有向关系邻接表（sparse COO）。

    CKG 存的是无向边，但 KGIN 的关系语义是有向的（user→item 表示"交互"，
    item→user 表示其反向）。这里按 (边类型, 源节点类别) 映射到 8 个关系号。

    ``n_memory=0`` 表示不裁剪，保留完整邻接。**不做节点级邻居裁剪**：
    KGIN 原文的"关系感知剪枝"是对 user→item 之间的**路径**去重（避免重复
    实体），不是限制每个节点的邻居数。本图谱中高入度节点（如类目→商品，
    单个 L2 类目下有上千件服装）一旦按节点裁剪，商品侧的 KG 上下文会在
    两层传播内被完全抹平，实测会导致表示归零、模型退化。

    Returns:
        adjs: list[torch.Tensor]，长度 8，每个为 sparse_coo [n_total, n_total]
        rel_names: list[str]，用于日志
    """
    n_total = node_offsets['n_total']
    rng = np.random.default_rng(seed)

    rel_names = [
        'user->item', 'item->user',
        'item->category', 'category->item',
        'category->category', 'category->category(rev)',
        'item->brand', 'brand->item',
    ]
    buckets = [[] for _ in range(8)]

    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, t, ty in zip(src, dst, edge_type.tolist()):
        sc, tc = node_class(s, node_offsets), node_class(t, node_offsets)
        if ty == 0:
            r = 0 if sc == 'u' else 1
        elif ty == 1:
            r = 2 if sc == 'i' else 3
        elif ty == 2:
            r = 4 if sc in ('l3', 'l2') else 5
        elif ty == 3:
            r = 6 if sc == 'i' else 7
        else:
            continue
        buckets[r].append((s, t))

    adjs = []
    for r in range(8):
        pairs = buckets[r]
        if not pairs:
            adjs.append(None)
            continue
        by_src = {}
        for s, t in pairs:
            by_src.setdefault(s, []).append(t)
        rows, cols = [], []
        for s, ts in by_src.items():
            if n_memory and len(ts) > n_memory:
                sel = rng.choice(len(ts), size=n_memory, replace=False)
                ts = [ts[i] for i in sel]
            rows.extend([s] * len(ts))
            cols.extend(ts)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        # KGIN 式归一化：1/|N_r(i)|
        deg = np.bincount(rows, minlength=n_total).astype(np.float32)
        vals = 1.0 / np.maximum(deg[rows], 1.0)
        idx = torch.from_numpy(np.stack([rows, cols]))
        adjs.append(torch.sparse_coo_tensor(
            idx, torch.from_numpy(vals), (n_total, n_total)).coalesce())
    return adjs, rel_names


def _transpose_csr(csr, n):
    """CSR → 转置 CSR（用于自定义反向）。"""
    coo = csr.to_sparse_coo().coalesce()
    idx = coo.indices()
    return torch.sparse_coo_tensor(
        torch.stack([idx[1], idx[0]]), coo.values(), (n, n)).coalesce().to_sparse_csr()


class _SparseMM(torch.autograd.Function):
    """y = A @ x 的自定义反向。

    PyTorch 通用稀疏 autograd 在 3 层 × 8 关系的规模上把前向 35ms 放大到
    前向+反向 286ms。这里预先存好转置 CSR，反向直接一次 ``Aᵀ @ g``，
    邻接的 value 由度数归一化得到、不参与求导。
    """

    @staticmethod
    def forward(ctx, adj, adj_t, x):
        ctx.adj_t = adj_t
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, grad_out):
        return None, None, torch.sparse.mm(ctx.adj_t, grad_out.contiguous())


def build_user_history(train_pos, n_users, max_hist=32):
    """用户 → 训练交互商品列表（定长 padding），供 CKAN 的知识聚合使用。"""
    rows = [[] for _ in range(n_users)]
    for u, i in train_pos:
        if len(rows[u]) < max_hist:
            rows[u].append(i)
    hist = torch.zeros((n_users, max_hist), dtype=torch.long)
    mask = torch.zeros((n_users, max_hist), dtype=torch.bool)
    for u, items in enumerate(rows):
        for k, it in enumerate(items):
            hist[u, k] = it
            mask[u, k] = True
    return hist, mask


class TransE(nn.Module):
    """TransE：score(h,r,t) = -||h + r - t||²。CKAN 的知识嵌入分支使用。"""

    def __init__(self, n_entities, n_relations, dim=64):
        super().__init__()
        self.n_entities = n_entities
        self.entity_embedding = nn.Embedding(n_entities, dim)
        self.relation_embedding = nn.Embedding(n_relations, dim)
        nn.init.normal_(self.entity_embedding.weight, 0, 0.01)
        nn.init.normal_(self.relation_embedding.weight, 0, 0.01)

    def forward(self, h, r, t):
        he = self.entity_embedding(h)
        re = self.relation_embedding(r)
        te = self.entity_embedding(t)
        return -((he + re - te) ** 2).sum(-1)

    def margin_loss(self, h, r, t, neg_t, margin=1.0):
        pos = self.forward(h, r, t)
        neg = self.forward(h, r, neg_t)
        return F.relu(margin - pos + neg).mean()


# ============================================================================
# MKR  (Wang et al., SIGIR 2019)
# ============================================================================
class MKR(nn.Module):
    """Knowledge Graph Embedding for Recommendation.

    两阶段：BPR 训练 MKR 层的推荐损失，同时用 TransR 的间隔排序损失
    约束 KG 三元组，两者共享同一份实体/关系嵌入。

    MKR 层（对商品 v 的每个属性实体 t，关系 r）::

        h_v^r = h_v + W_r^T (W_r h_v - W_r t)
        α(v,i) = softmax_i( e_v^T tanh(W [h_v ; t_i]) )
        h~_v   = Σ_i α(v,i) · h_v^{r_i}
        ŷ_{u,v} = h_u · h~_v
    """

    def __init__(self, n_users, n_items, dim=64, graph=None, triplets=None,
                 max_attrs=2, n_relations=N_RELATIONS):
        super().__init__()
        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)      # h_v，MKR 层与 KG 头共用
        self.att = nn.Linear(2 * dim, dim, bias=False)
        self.transR = TransR(graph['n_total'], n_relations, dim, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

        ent, rel, mask = build_item_attr_index(
            graph['edge_index'], graph['edge_type'], graph['node_offsets'], max_attrs)
        self.register_buffer('attr_ent', ent)
        self.register_buffer('attr_rel', rel)
        self.register_buffer('attr_mask', mask)
        self.register_buffer('attr_ent_c', ent.clamp(min=0))
        self.register_buffer('triplets', triplets if triplets is not None
                             else torch.empty((0, 3), dtype=torch.long))

    def item_repr(self):
        """返回增强后的商品表示 h~ [n_items, d]。

        按关系分组做投影，避免为每个 (item, attr) 存一份 [d,d] 矩阵。
        """
        h = self.item_emb.weight                                   # [n_items, d]
        t = self.transR.entity_embedding(self.attr_ent_c)         # [n_items, A, d]
        A = self.attr_mask.size(1)

        hv_slot = torch.zeros(self.n_items, A, self.dim, device=h.device)
        logits = torch.zeros(self.n_items, A, device=h.device)
        for r in range(N_RELATIONS):
            sel = (self.attr_rel == r) & self.attr_mask          # [n_items, A]
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=False)                    # [K, 2] = (item, slot)
            items, slots = idx[:, 0], idx[:, 1]
            M = self.transR.M[r]                                  # [d, d]
            hv, tv = h[items], t[items, slots]
            # h_v^r = h_v + W_r^T (W_r h_v - W_r t)
            hv_slot = hv_slot.index_put((items, slots), hv + (hv @ M - tv @ M) @ M.T)
            e = self.att(torch.cat([hv, tv], dim=-1))             # [K, d]
            logits = logits.index_put((items, slots), (hv * e).sum(-1))

        logits = logits.masked_fill(~self.attr_mask, float('-inf'))
        alpha = torch.softmax(logits, dim=1)                      # [n_items, A]
        return (alpha.unsqueeze(-1) * hv_slot).sum(1)

    def _kg_loss(self, dev, n_sample=8192):
        trip = self.triplets
        if trip.numel() == 0:
            return torch.zeros((), device=dev)
        if n_sample and trip.size(0) > n_sample:
            sel = torch.randint(0, trip.size(0), (n_sample,))
            trip = trip[sel]
        h, r, t = trip[:, 0].to(dev), trip[:, 1].to(dev), trip[:, 2].to(dev)
        neg = torch.randint(0, self.transR.n_entities, t.shape, device=dev)
        pos = self.transR(h, r, t)
        negs = self.transR(h, r, neg)
        return F.relu(1.0 - pos + negs).mean()

    def fit(self, train_pos, train_neg, epochs=30, batch=256, lr=0.001,
            dev='cuda', lambda_kg=0.05):
        self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pos)
        u = torch.LongTensor([p[0] for p in train_pos])
        ip = torch.LongTensor([p[1] for p in train_pos])
        # 保留 (uid, [negs]) 元组：下面按 all_negs[j][1] 取整条负样本列表
        all_negs = list(train_neg)

        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = bpr_t = kg_t = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                ub, pb = u[idx].to(dev), ip[idx].to(dev)
                nb = torch.LongTensor([
                    int(np.random.choice(all_negs[j][1])) for j in idx.tolist()
                ]).to(dev)
                h_tilde = self.item_repr()                       # [n_items, d]
                pos = (self.user_emb(ub) * h_tilde[pb]).sum(-1)
                neg = (self.user_emb(ub) * h_tilde[nb]).sum(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                kg = self._kg_loss(dev)
                loss = bpr + lambda_kg * kg
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx); bpr_t += bpr.item() * len(idx)
                kg_t += kg.item() * len(idx)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  MKR ep {ep+1}/{epochs} loss={tot/n:.4f} bpr={bpr_t/n:.4f} kg={kg_t/n:.4f}", flush=True)

    @torch.no_grad()
    def score_batch(self, uids, cand):
        dev = self.user_emb.weight.device
        uids_t = torch.as_tensor(uids, device=dev)
        cand_t = torch.as_tensor(cand, device=dev)
        h_tilde = self.item_repr()
        return (self.user_emb(uids_t).unsqueeze(1) * h_tilde[cand_t]).sum(-1).cpu().numpy()


# ============================================================================
# KGIN  (Wang et al., TOIS 2021)
# ============================================================================
class KGIN(nn.Module):
    """Knowledge Graph Enhanced User-Item Interaction Graph for Recommendation.

    关系感知路径聚合：在 CKG 上把无向边拆成 8 个有向关系，对每个
    (节点, 关系) 做关系感知剪枝（保留至多 n_memory 个邻居），逐层按关系
    分别聚合、再用注意力融合，得到的路径表示与节点基础表示拼接后经 MLP
    输出预测分数::

        z_i^r = Σ_{u ∈ N_r(i)} (1/|N_r(i)|) ⊙ f_r ⊙ (x_u ⊙ f_r)
        x^l   = Σ_r softmax(β_r) ⊙ z^{l-1}_r
        ŷ_{u,v} = MLP([x_u ; x_v])
    """

    def __init__(self, n_users, n_items, dim=64, graph=None,
                 n_layers=3, n_memory=0, seed=42):
        super().__init__()
        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.node_emb = nn.Embedding(graph['n_total'], dim)
        self.rel_emb = nn.Embedding(8, dim)
        self.rel_att = nn.Linear(dim, 1, bias=False)      # 关系融合注意力打分
        self.predict = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, 1))
        nn.init.normal_(self.node_emb.weight, 0, 0.01)
        nn.init.normal_(self.rel_emb.weight, 0, 1.0)   # 归一化后使用，取 KGIN 默认尺度

        adjs, names = build_directed_adj(
            graph['edge_index'], graph['edge_type'], graph['node_offsets'],
            n_memory=n_memory, seed=seed)
        self.n_layers = n_layers
        self.rel_names = names
        self._n_total = graph['n_total']
        self._adj_coo = [a for a in adjs if a is not None]
        self._rel_slots = [i for i, a in enumerate(adjs) if a is not None]
        self._adjs = None  # 由 to() 构建

    def to(self, dev):
        super().to(dev)
        if self._adjs is None:
            pairs = []
            for a in self._adj_coo:
                csr = a.to(dev).to_sparse_csr()
                pairs.append((csr, _transpose_csr(csr, self._n_total)))
            self._adjs = pairs
        return self

    def propagate(self):
        """k 层关系感知聚合，返回全部节点表示 [n_total, d]。

        关系向量统一归一化到单位范数后再做 ``⊙ f_r ⊙`` 双重加权——若沿用
        仓库惯例的 N(0,0.01) 初始化，每层乘 f_r² 会让信号逐层衰减 1e-4，
        三层后在 float32 下直接下溢为 0（实测表现为 HR@10=1.0 的退化输出）。
        层间加残差，保证节点自身表示不被邻居平均淹没。
        """
        x = self.node_emb.weight
        for _ in range(self.n_layers):
            zs, rs = [], []
            for slot, r in enumerate(self._rel_slots):
                csr, csr_t = self._adjs[slot]
                z = _SparseMM.apply(csr, csr_t, x)           # [n_total, d]
                f = F.normalize(self.rel_emb.weight[r], dim=0)   # [d], ‖f‖=1
                z = z * f * f                                  # ⊙ f_r ⊙ (· ⊙ f_r)
                zs.append(z)
                rs.append(f)
            Z = torch.stack(zs)                              # [R, n_total, d]
            F_ = torch.stack(rs).unsqueeze(1)                # [R, 1, d]
            att = self.rel_att(Z * F_).squeeze(-1)           # [R, n_total]
            alpha = torch.softmax(att, dim=0).unsqueeze(-1)  # [R, n_total, 1]
            x = x + (alpha * Z).sum(0)                       # 残差
        return x

    def _represent(self, nodes):
        h = self.propagate()
        return h[nodes]

    def fit(self, train_pos, train_neg, epochs=30, batch=256, lr=0.001, dev='cuda'):
        self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pos)
        u = torch.LongTensor([p[0] for p in train_pos])
        ip = torch.LongTensor([p[1] for p in train_pos])
        # 保留 (uid, [negs]) 元组：下面按 all_negs[j][1] 取整条负样本列表
        all_negs = list(train_neg)
        item_off = self.n_users

        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                ub, pb = u[idx].to(dev), ip[idx].to(dev)
                nb = torch.LongTensor([
                    int(np.random.choice(all_negs[j][1])) for j in idx.tolist()
                ]).to(dev)
                h = self.propagate()                          # [n_total, d]
                # KGIN 打分：MLP([用户关系路径表示; 商品关系路径表示])
                pos = self.predict(torch.cat([h[ub], h[pb + item_off]], -1)).squeeze(-1)
                neg = self.predict(torch.cat([h[ub], h[nb + item_off]], -1)).squeeze(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); bpr.backward(); opt.step()
                tot += bpr.item() * len(idx)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  KGIN ep {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)

    @torch.no_grad()
    def score_batch(self, uids, cand):
        dev = self.node_emb.weight.device
        uids_t = torch.as_tensor(uids, device=dev)
        cand_t = torch.as_tensor(cand, device=dev)
        h = self.propagate()
        uh = h[uids_t]                              # [B, d]
        ih = h[cand_t + self.n_users]               # [B, N, d]
        ue = uh.unsqueeze(1).expand_as(ih)          # [B, N, d]
        out = torch.cat([ue, ih], dim=-1)
        return self.predict(out).squeeze(-1).cpu().numpy()


# ============================================================================
# CKAN  (Wang et al., WWW 2019)
# ============================================================================
class CKAN(nn.Module):
    """Collaborative Knowledge-Aware Networks for Recommendation.

    知识感知嵌入 + 通道级注意力：从用户交互过的商品侧聚合知识向量，
    每个用户对知识维度有一组通道权重 w_u，实体注意力 β_u 在其知识邻域上
    做 softmax::

        α_u(e)_k = w_u[k]                       （通道权重，逐维）
        β_u(e)   = softmax_{e ∈ N_u}(⟨e_u, e⟩)
        F_u      = Σ_e β_u(e) · (α_u(e) ⊙ k_e)
        ŷ_{u,v}  = σ( (e_u + F_u) · (e_v + F_v) )

    损失 = BPR + λ_KG · TransE 三元组间隔排序损失。
    """

    def __init__(self, n_users, n_items, dim=64, graph=None, triplets=None,
                 max_hist=32, n_relations=N_RELATIONS):
        super().__init__()
        self.n_users, self.n_items, self.dim = n_users, n_items, dim
        self.user_emb = nn.Embedding(n_users, dim)          # e_u
        self.item_emb = nn.Embedding(n_items, dim)          # e_v
        self.transe = TransE(graph['n_total'], n_relations, dim)
        self.channel = nn.Linear(dim, dim)                   # → 通道权重 w_u
        self.hist_emb = nn.Embedding(n_items, dim)           # 历史商品的协同表示
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.normal_(self.hist_emb.weight, 0, 0.01)
        self.register_buffer('triplets', triplets if triplets is not None
                             else torch.empty((0, 3), dtype=torch.long))
        self.max_hist = max_hist

    def set_history(self, train_pos):
        hist, mask = build_user_history(train_pos, self.n_users, self.max_hist)
        self.register_buffer('hist', hist)
        self.register_buffer('hist_mask', mask)

    def user_knowledge(self, uids):
        """用户侧知识聚合：F_u = Σ_e β_u(e) · (w_u ⊙ k_e)，e 取自 u 的交互商品。"""
        w = torch.sigmoid(self.channel(self.user_emb(uids)))          # [B, d] 通道权重
        h = self.hist[uids]                                          # [B, H]
        m = self.hist_mask[uids].float()                             # [B, H]
        k = self.transe.entity_embedding(h + self.n_users)           # [B, H, d]
        e = self.hist_emb(h)                                         # [B, H, d]
        logits = (self.user_emb(uids).unsqueeze(1) * e).sum(-1) / (self.dim ** 0.5)
        # 用 -1e9 而非 -inf：历史全空时 softmax 仍给出有限值，乘 m 后归零
        beta = torch.softmax(logits.masked_fill(m == 0, -1e9), dim=-1) * m
        return (beta.unsqueeze(-1) * (w.unsqueeze(1) * k)).sum(1)    # [B, d]

    def item_knowledge(self, iids):
        """商品侧知识：商品自身的 KG 知识向量（无需再聚合）。"""
        return self.transe.entity_embedding(iids + self.n_users)

    def _reps(self, uids, iids):
        fu = self.user_knowledge(uids)
        fv = self.item_knowledge(iids)
        return self.user_emb(uids) + fu, self.item_emb(iids) + fv

    def _kg_loss(self, dev, n_sample=8192):
        trip = self.triplets
        if trip.numel() == 0:
            return torch.zeros((), device=dev)
        if n_sample and trip.size(0) > n_sample:
            sel = torch.randint(0, trip.size(0), (n_sample,))
            trip = trip[sel]
        h, r, t = trip[:, 0].to(dev), trip[:, 1].to(dev), trip[:, 2].to(dev)
        neg = torch.randint(0, self.transe.n_entities, t.shape, device=dev)
        return self.transe.margin_loss(h, r, t, neg)

    def fit(self, train_pos, train_neg, epochs=30, batch=256, lr=0.001,
            dev='cuda', lambda_kg=0.05):
        self.set_history(train_pos)
        self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pos)
        u = torch.LongTensor([p[0] for p in train_pos])
        ip = torch.LongTensor([p[1] for p in train_pos])
        # 保留 (uid, [negs]) 元组：下面按 all_negs[j][1] 取整条负样本列表
        all_negs = list(train_neg)

        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = bpr_t = kg_t = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                ub, pb = u[idx].to(dev), ip[idx].to(dev)
                nb = torch.LongTensor([
                    int(np.random.choice(all_negs[j][1])) for j in idx.tolist()
                ]).to(dev)
                eu, ev = self._reps(ub, pb)
                _, eneg = self._reps(ub, nb)
                pos = (eu * ev).sum(-1)
                neg = (eu * eneg).sum(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                kg = self._kg_loss(dev)
                loss = bpr + lambda_kg * kg
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx); bpr_t += bpr.item() * len(idx)
                kg_t += kg.item() * len(idx)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  CKAN ep {ep+1}/{epochs} loss={tot/n:.4f} bpr={bpr_t/n:.4f} kg={kg_t/n:.4f}", flush=True)

    @torch.no_grad()
    def score_batch(self, uids, cand):
        dev = self.user_emb.weight.device
        uids_t = torch.as_tensor(uids, device=dev)
        cand_t = torch.as_tensor(cand, device=dev)
        eu, ev = self._reps(uids_t, cand_t)
        # eu [B, d], ev [B, N, d] -> [B, N]
        return (eu.unsqueeze(1) * ev).sum(-1).cpu().numpy()


# ============================================================================
# 独立运行入口（供冒烟测试 / 单模型调试）
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--models', nargs='+', default=['mkr', 'kgin', 'ckan'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--lambda-kg', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-users', type=int, default=0, help='>0 则只取前 N 个测试用户（冒烟）')
    ap.add_argument('--kgin-layers', type=int, default=3)
    ap.add_argument('--kgin-memory', type=int, default=0, help='每节点每关系邻居上限；0=不裁剪(默认)')
    ap.add_argument('--output', default='experiments/results/kg_baselines.json')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format='%(asctime)s - %(message)s')
    log = _lg.getLogger(__name__)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"device={dev}")

    from recommender.train import load_data, prepare_training_data
    from recommender.evaluate import build_loo_test
    from recommender.baselines_real import evaluate_batch

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pos = [(p[0], p[1]) for p, _ in data['train_data']]
    train_neg = [(p[0], ng) for p, ng in data['train_data']]
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                               n_negative=100, seed=args.seed)
    if args.max_users:
        test_data = test_data[:args.max_users]

    idx2user = {v: k for k, v in data['user2idx'].items()}
    idx2item = {v: k for k, v in data['item2idx'].items()}
    raw_pairs = [(idx2user[u], idx2item[i]) for u, i in train_pos]
    ei, et, n_total, off = build_hetero_graph(
        products, raw_pairs, data['user2idx'], data['item2idx'])
    graph = {'edge_index': ei, 'edge_type': et, 'n_total': n_total, 'node_offsets': off}
    triplets = build_kg_triplets(products, data['item2idx'], off)
    log.info(f"graph nodes={n_total} edges={ei.size(1)} triplets={tuple(triplets.shape)} "
             f"test_users={len(test_data)}")

    import json
    results = {}
    for name in args.models:
        log.info(f"==== {name} ====")
        t0 = time.time()
        if name == 'mkr':
            m = MKR(n_u, n_i, dim=64, graph=graph, triplets=triplets)
            m.fit(train_pos, train_neg, epochs=args.epochs, batch=args.batch,
                  lr=args.lr, dev=dev, lambda_kg=args.lambda_kg)
        elif name == 'kgin':
            m = KGIN(n_u, n_i, dim=64, graph=graph,
                     n_layers=args.kgin_layers, n_memory=args.kgin_memory, seed=args.seed)
            m.fit(train_pos, train_neg, epochs=args.epochs, batch=args.batch,
                  lr=args.lr, dev=dev)
        elif name == 'ckan':
            m = CKAN(n_u, n_i, dim=64, graph=graph, triplets=triplets)
            m.fit(train_pos, train_neg, epochs=args.epochs, batch=args.batch,
                  lr=args.lr, dev=dev, lambda_kg=args.lambda_kg)
        else:
            log.warning(f"unknown model {name}")
            continue
        log.info(f"  trained in {time.time()-t0:.0f}s, evaluating...")
        r = evaluate_batch(m.score_batch, test_data, [5, 10, 20])
        log.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f} "
                 f"P@10={r['precision@10']:.4f} HR@20={r['hr@20']:.4f} "
                 f"NDCG@20={r['ndcg@20']:.4f} P@20={r['precision@20']:.4f}")
        results[name] = r
        del m
        if dev == 'cuda':
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'data': args.data, 'args': vars(args), 'results': results},
                  f, indent=2, ensure_ascii=False)
    log.info(f"saved -> {out}")


if __name__ == '__main__':
    main()