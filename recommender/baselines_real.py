"""真实数据规模下的基线（sparse + 批量评估，64k 用户 / 21k 商品）。

原 baselines.py 在 64k² dense sim 矩阵下 OOM，本模块用 scipy.sparse +
top-K 邻居截断 + 分批评估。

基线（与原版口径一致）:
    popular   流行度
    usercf    User-based CF, sparse + top-K 邻居
    itemcf    Item-based CF, sparse item-item 共现
    bprmf     BPR-MF (GPU)
    lightgcn  LightGCN (GPU, sparse 邻接)
    ripplen   RippleNet (GPU, 多跳 KG)
    kgcl      KGCL (GPU, 图对比)

评估：sampled-100 leave-one-out（同 evaluate.py 协议），但按用户分批
计算候选分数，避免逐用户小 forward 的开销（与 evaluate_fast 一致）。

Usage:
    python -m recommender.baselines_real --data data/amazon_fashion_real.json \
        --baselines popular usercf itemcf bprmf lightgcn ripplen kgcl --epochs 30
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k, compute_precision_at_k, compute_recall_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------- 通用批量评估 ----------------
def evaluate_batch(score_fn, test_data, k_values, batch_users=512):
    """score_fn(user_idxs[B], item_idxs[B, N]) -> scores[B, N]."""
    res = {f'{m}@{k}': [] for m in ('hr', 'ndcg', 'precision', 'recall') for k in k_values}
    n_done = 0
    t0 = time.time()
    for start in range(0, len(test_data), batch_users):
        batch = test_data[start:start + batch_users]
        B = len(batch)
        # 收集候选
        N = 1 + max(len(negs) for _, _, negs in batch)
        uids = np.zeros(B, dtype=np.int64)
        cand = np.full((B, N), -1, dtype=np.int64)
        mask = np.zeros((B, N), dtype=bool)
        for bi, (uid_idx, pos_items, neg_items) in enumerate(batch):
            uids[bi] = uid_idx
            cand[bi, 0] = pos_items[0]
            mask[bi, 0] = True
            for j, ng in enumerate(neg_items):
                cand[bi, 1 + j] = ng
                mask[bi, 1 + j] = True
        scores = score_fn(uids, cand)  # [B, N]
        scores = np.where(mask, scores, -1e18)
        order = np.argsort(-scores, axis=1)
        cand_orig = cand.copy()
        for bi in range(B):
            pos = [int(cand_orig[bi, 0])]
            ranked = cand_orig[bi, order[bi]].tolist()
            for k in k_values:
                res[f'hr@{k}'].append(compute_hr_at_k(pos, ranked, k))
                res[f'ndcg@{k}'].append(compute_ndcg_at_k(pos, ranked, k))
                res[f'precision@{k}'].append(compute_precision_at_k(pos, ranked, k))
                res[f'recall@{k}'].append(compute_recall_at_k(pos, ranked, k))
        n_done += B
        if n_done % (batch_users * 20) == 0 or n_done >= len(test_data):
            logger.info(f"  eval {n_done}/{len(test_data)} {time.time()-t0:.0f}s")
    return {k: float(np.mean(v)) for k, v in res.items()}


# ---------------- Popular ----------------
class Popular:
    def __init__(self, n_items):
        self.n_items = n_items
        self.pop = np.zeros(n_items, dtype=np.float32)

    def fit(self, train_pairs):
        for u, i in train_pairs:
            self.pop[i] += 1

    def score_batch(self, uids, cand):
        return np.broadcast_to(self.pop[cand], (len(uids), cand.shape[1])).copy()


# ---------------- UserCF (sparse + top-K 邻居) ----------------
class UserCF:
    """User-based CF, sparse cos-sim, top-K 邻居截断。"""
    def __init__(self, n_users, n_items, top_k=200):
        self.n_users, self.n_items = n_users, n_items
        self.top_k = top_k
        self.ui = None   # csr [n_u, n_i]
        self.sim_topk = None  # csr [n_u, n_u], 每行 ≤ top_k 非零

    def fit(self, train_pairs):
        rows = np.array([p[0] for p in train_pairs], dtype=np.int32)
        cols = np.array([p[1] for p in train_pairs], dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        self.ui = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        # cosine: 行归一化
        norm = np.sqrt(np.array(self.ui.multiply(self.ui).sum(axis=1)).flatten())
        norm[norm == 0] = 1
        Dinv = sp.diags(1.0 / norm)
        ui_n = Dinv @ self.ui  # 行归一化
        sim = ui_n @ ui_n.T    # sparse cosine
        sim.setdiag(0)
        # top-K per row
        sim = sim.tolil()
        rows_keep, cols_keep, vals_keep = [], [], []
        for u in range(self.n_users):
            row = np.array(sim.data[u])
            idx = sim.rows[u]
            if len(row) <= self.top_k:
                rows_keep.extend([u] * len(row))
                cols_keep.extend(idx); vals_keep.extend(row.tolist())
            else:
                top = np.argpartition(-row, self.top_k)[:self.top_k]
                rows_keep.extend([u] * self.top_k)
                cols_keep.extend([idx[t] for t in top])
                vals_keep.extend([row[t] for t in top])
        self.sim_topk = sp.csr_matrix(
            (vals_keep, (rows_keep, cols_keep)), shape=(self.n_users, self.n_users))
        logger.info(f"UserCF fit: ui nnz={self.ui.nnz:,}, sim_topk nnz={self.sim_topk.nnz:,}")

    def score_batch(self, uids, cand):
        # 向量化：sim_topk[batch] @ ui -> [B, n_i] dense, gather cand.
        out = np.zeros((len(uids), cand.shape[1]), dtype=np.float32)
        sim_sub = self.sim_topk[uids]  # csr [B, n_u]
        full = sim_sub @ self.ui       # csr [B, n_i]
        full_dense = full.toarray()    # [B, n_i]
        # 逐 (b, k) gather，cand=-1 跳过
        B, N = out.shape
        b_idx = np.repeat(np.arange(B), N)
        k_idx = np.tile(np.arange(N), B)
        c_flat = cand.reshape(-1)
        valid = c_flat >= 0
        out[b_idx[valid], k_idx[valid]] = full_dense[b_idx[valid], c_flat[valid]]
        # 排除已交互
        for bi, u in enumerate(uids):
            interacted = self.ui.getrow(u).indices
            if len(interacted):
                pos_mask = np.isin(cand[bi], interacted) & (cand[bi] >= 0)
                out[bi, pos_mask] = -1e9
        return out


# ---------------- ItemCF (sparse item-item 共现) ----------------
class ItemCF:
    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.ui = None
        self.sim = None  # csr [n_i, n_i] cosine

    def fit(self, train_pairs):
        rows = np.array([p[0] for p in train_pairs], dtype=np.int32)
        cols = np.array([p[1] for p in train_pairs], dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        self.ui = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        # i-i 共现 = ui.T @ ui
        co = self.ui.T @ self.ui  # sparse
        # cosine
        norm = np.sqrt(np.array(co.multiply(co).sum(axis=1)).flatten())
        norm[norm == 0] = 1
        Dinv = sp.diags(1.0 / norm)
        self.sim = Dinv @ co  # 行归一化
        self.sim.setdiag(0)
        logger.info(f"ItemCF fit: sim nnz={self.sim.nnz:,}")

    def score_batch(self, uids, cand):
        # 向量化：scores[u, i] = sum_{j in interacted(u)} sim[i, j]
        ui_sub = self.ui[uids]  # csr [B, n_i]
        scores_full = (ui_sub @ self.sim.T).toarray()  # [B, n_i]
        out = np.zeros((len(uids), cand.shape[1]), dtype=np.float32)
        B, N = out.shape
        b_idx = np.repeat(np.arange(B), N)
        k_idx = np.tile(np.arange(N), B)
        c_flat = cand.reshape(-1)
        valid = c_flat >= 0
        out[b_idx[valid], k_idx[valid]] = scores_full[b_idx[valid], c_flat[valid]]
        # 排除已交互
        for bi, u in enumerate(uids):
            interacted = self.ui.getrow(u).indices
            if len(interacted):
                pos_mask = np.isin(cand[bi], interacted) & (cand[bi] >= 0)
                out[bi, pos_mask] = -1e9
        return out


# ---------------- BPR-MF (GPU) ----------------
class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        self.n_items = n_items

    def to(self, dev):
        self.user_emb = self.user_emb.to(dev); self.item_emb = self.item_emb.to(dev); return self

    def parameters(self):
        return list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, train_pos, train_neg, epochs=30, batch=256, lr=0.001, dev='cuda'):
        u = torch.LongTensor([p[0] for p in train_pos]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pos]).to(dev)
        # train_neg[i] = (uid, [neg_items]); 每 batch 随机抽一个
        all_negs = [n[1] for n in train_neg]
        # nn.Embedding 0.01 std init + L2 正则会压制梯度(BPR 文献标准做法是 wd=0)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pos)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ineg = torch.LongTensor([
                    int(all_negs[j][np.random.randint(0, len(all_negs[j]))])
                    for j in idx.tolist()
                ]).to(dev)
                pos = (self.user_emb.weight[u[idx]] * self.item_emb.weight[ip[idx]]).sum(-1)
                neg = (self.user_emb.weight[u[idx]] * self.item_emb.weight[ineg]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  BPR-MF ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        with torch.no_grad():
            uids_t = torch.LongTensor(uids)
            cand_t = torch.LongTensor(cand)
            dev = self.user_emb.weight.device
            u_e = self.user_emb.weight[uids_t.to(dev)]  # [B, d]
            i_e = self.item_emb.weight[cand_t.to(dev)]  # [B, N, d]
            scores = (u_e.unsqueeze(1) * i_e).sum(-1)  # [B, N]
        return scores.detach().cpu().numpy()


# ---------------- LightGCN (GPU, sparse adj) ----------------
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)
        self.norm_adj = None

    def _build_adj(self, train_pos, dev):
        n_u, n_i = self.user_emb.num_embeddings, self.item_emb.num_embeddings
        rows = np.concatenate([np.array([p[0] for p in train_pos], dtype=np.int32),
                               np.array([p[1] for p in train_pos], dtype=np.int32) + n_u])
        cols = np.concatenate([np.array([p[1] for p in train_pos], dtype=np.int32) + n_u,
                               np.array([p[0] for p in train_pos], dtype=np.int32)])
        data = np.ones(len(rows), dtype=np.float32)
        N = n_u + n_i
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        deg = np.array(A.sum(1)).flatten()
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0).astype(np.float32)
        D = sp.diags(dinv)
        Anorm = D @ A @ D
        coo = Anorm.tocoo()
        idx = torch.LongTensor([coo.row, coo.col])
        val = torch.FloatTensor(coo.data)
        self.norm_adj = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(dev)

    def _propagate(self):
        n_u = self.user_emb.num_embeddings
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], 0)
        e = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            e.append(x)
        return torch.stack(e, 0).mean(0), n_u

    def to(self, dev):
        self.user_emb = self.user_emb.to(dev); self.item_emb = self.item_emb.to(dev); return self

    def parameters(self):
        return list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, train_pos, train_neg, epochs=30, batch=256, lr=0.001, dev='cuda'):
        self._build_adj(train_pos, dev)
        u = torch.LongTensor([p[0] for p in train_pos]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pos]).to(dev)
        all_negs = [n[1] for n in train_neg]
        # LightGCN 原版不用 weight decay
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pos)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ineg = torch.LongTensor([
                    int(all_negs[j][np.random.randint(0, len(all_negs[j]))])
                    for j in idx.tolist()
                ]).to(dev)
                e, n_u = self._propagate()
                pos = (e[u[idx]] * e[ip[idx] + n_u]).sum(-1)
                neg = (e[u[idx]] * e[ineg + n_u]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  LightGCN ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        with torch.no_grad():
            uids_t = torch.LongTensor(uids)
            cand_t = torch.LongTensor(cand)
            dev = self.user_emb.weight.device
            e, n_u = self._propagate()
            u_e = e[uids_t.to(dev)]  # [B, d]
            cand_abs = cand_t.to(dev) + n_u  # [B, N]
            i_e = e[cand_abs]  # [B, N, d]
            scores = (u_e.unsqueeze(1) * i_e).sum(-1)  # [B, N]
        return scores.detach().cpu().numpy()


# ---------------- RippleNet (GPU, 多跳 KG) ----------------
class RippleNet(nn.Module):
    """RippleNet: 多跳知识图谱推荐.

    真正的 2-hop ripple set：在异构图上，从用户历史 item 出发，
    沿 rel1(item→category) hop-1，再沿 rel2(category→L2/L1) hop-2。
    每跳对当前节点集合的每个 node 取其邻居 embedding 均值作为该 node 的
    ripple 上下文；最后用注意力把 hop1/hop2 上下文聚合到 user_emb，
    再叠加 item_emb 得到 KG 增强的 item 表示，做 BPR 训练。

    性能：每个 rel 用 src→LongTensor[neighbor_ids] 字典存于 GPU，
    lookup 是 O(|seed_set|) GPU index_select（不走 Python 邻居枚举）。
    """

    def __init__(self, n_users, n_items, dim=64, n_hops=2):
        super().__init__()
        self.dim = dim
        self.n_hops = n_hops
        self.n_users, self.n_items = n_users, n_items
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        # hop1/hop2 关系投影（已知异构图 edge type: 0=user-item 不参与 ripple，
        # 1=item-category(hop1), 2=subcategory(hop2), 3=item-brand 可作 hop1 备用)
        self.rel_emb = nn.Embedding(4, dim)
        nn.init.normal_(self.rel_emb.weight, 0, 0.01)
        # hop-level MLP
        self.transform = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        # 用户历史 item 列表（GPU tensor 化）
        self._user_hist_flat = None   # LongTensor，所有用户历史 item 拼接
        self._user_hist_ptr = None    # LongTensor[N+1] CSR 风格指针

        # KG 节点嵌入 (用户/商品/类别/品牌全部统一表示):在 _build_kg 里按 n_total 分配
        self.kg_node_emb = None
        self._n_total = 0
        self._item_offset = 0  # item 节点在异构图中的起点 (= n_users)

    # ----- KG lookup 表构造（GPU 张量化）-----
    def _build_kg(self, edge_index, edge_type, n_total, dev):
        """对每个关系 r，建 src→LongTensor[dst_neighbors] 的字典（GPU 上）。"""
        ei = edge_index.to(dev)
        et = edge_type.to(dev)
        # 用 scatter 构建 per-src 的邻居列表：用 unique + cumcount 偏移
        src = ei[0]
        # 按 src 排序，相同 src 聚一起
        order = torch.argsort(src)
        src_sorted = src[order]
        et_sorted = et[order]
        # 拆分成 per-rel 段，对每段重新 scatter 到 src 上
        self._rel_neighbors = {}
        for r in range(4):
            mask = (et_sorted == r)
            r_src = src_sorted[mask]
            r_dst = ei[1][order][mask]
            # 建 dict：src id → tensor of dst ids（GPU）
            unique_src, inverse = torch.unique(r_src, return_inverse=True)
            # 把同一 src 的 dst 聚合：sort by inverse, split by unique
            order2 = torch.argsort(inverse)
            inv_sorted = inverse[order2]
            r_dst_sorted = r_dst[order2]
            counts = torch.bincount(inv_sorted, minlength=unique_src.numel())
            ptr = torch.cat([torch.zeros(1, dtype=counts.dtype, device=dev),
                             counts.cumsum(0)])
            splits = torch.split(r_dst_sorted, counts.tolist())
            tbl = {int(unique_src[i].item()): splits[i] for i in range(unique_src.numel())}
            self._rel_neighbors[r] = tbl
        # 分配 KG 节点嵌入:异构图中前 NI (=n_items) 个槽位为商品(与 item_emb 对齐),
        # 其余为类别/品牌等额外节点
        item_offset = self.n_users
        self._item_offset = item_offset
        self._n_total = n_total
        self.kg_node_emb = nn.Embedding(n_total, self.dim).to(dev)
        with torch.no_grad():
            # 拷贝 item_emb[:NI] 到 kg_node_emb 的 item 槽位
            n_copy = min(self.n_items, n_total - item_offset)
            if n_copy > 0:
                self.kg_node_emb.weight[item_offset:item_offset + n_copy] = self.item_emb.weight[:n_copy]
        logger.info(f"  RippleNet KG built: rel sizes "
                    f"{[(r, len(self._rel_neighbors[r])) for r in range(4)]}, n_total={n_total}")

    # ----- 用户历史表（CSR 风格）-----
    def _set_user_history(self, train_pos):
        """train_pos: list[(uid, iid)]，构建 GPU 上 (uid→item list) CSR。"""
        from collections import defaultdict
        per_user = defaultdict(list)
        for uid, iid in train_pos:
            per_user[uid].append(iid)
        # 限长：取每个用户前 50 个历史 item（防爆 GPU）
        MAX_H = 50
        flat = []
        ptr = [0]
        for uid in range(self.n_users):
            items = per_user.get(uid, [])[:MAX_H]
            flat.extend(items)
            ptr.append(len(flat))
        dev = self.user_emb.weight.device
        self._user_hist_flat = torch.tensor(flat, dtype=torch.long, device=dev)
        self._user_hist_ptr = torch.tensor(ptr, dtype=torch.long, device=dev)
        logger.info(f"  RippleNet user-history: avg={len(flat)/max(self.n_users,1):.1f} "
                    f"items/user, max_h={MAX_H}")

    def _user_history_items(self, uids):
        """返回 list[Tensor]，每个用户的 (≤MAX_H) 个历史 item 的 item_emb 均值 [d]."""
        out = []
        # GPU 上快速切片，避免每用户来回 python
        for u in uids.tolist():
            s = int(self._user_hist_ptr[u].item())
            e = int(self._user_hist_ptr[u + 1].item())
            if e > s:
                out.append(self.item_emb.weight[self._user_hist_flat[s:e]].mean(0))
            else:
                out.append(torch.zeros(self.dim, device=self.item_emb.weight.device))
        return torch.stack(out, 0)  # [B, d]

    # ----- Ripple 一步：对一组 seed node 一次性 GPU 聚合其 rel-r 邻居均值 -----
    def _hop_aggregate(self, node_ids, rel, dev):
        """node_ids: LongTensor[seed_size].
        返回 [seed_size, d]，第 i 行 = node_ids[i] 的 rel-r 邻居 emb 均值。

        实现：把 (seed, weight) 形式的 message 一次性 scatter 到邻居空间，
        再从邻居空间 gather 回 seed 侧。每个邻居被其 src 的 (1/deg_src) 系数加权。
        """
        n_seeds = node_ids.numel()
        if n_seeds == 0:
            return torch.zeros(0, self.dim, device=dev)
        tbl = self._rel_neighbors.get(rel, {})
        out = torch.zeros(n_seeds, self.dim, device=dev)
        # n_seeds 通常仅几百,python loop 可接受
        nid_list = node_ids.tolist()
        for i, n in enumerate(nid_list):
            nbrs = tbl.get(n, None)
            if nbrs is not None and nbrs.numel() > 0:
                out[i] = self.kg_node_emb.weight[nbrs].mean(0)
        return out
    def _ripple_user_context(self, uids, dev):
        """对 batch 用户:用其历史 item 集做 hop1(rel=1) + hop2(rel=2) 邻居均值聚合。
        注意力把两 hop 拼到 user_emb,再过 transform。"""
        u_e = self.user_emb.weight[uids]  # [B, d]
        B = uids.numel()
        # 把 batch 内所有 (user_pos, hist_item) 平铺到一个张量,避免 per-user python loop
        items_chunks = []
        rows_chunks = []
        for ui, u in enumerate(uids.tolist()):
            s = int(self._user_hist_ptr[u].item())
            e = int(self._user_hist_ptr[u + 1].item())
            if e > s:
                its = self._user_hist_flat[s:e]
                items_chunks.append(its)
                rows_chunks.append(torch.full((its.numel(),), ui,
                                              dtype=torch.long, device=dev))
        if not items_chunks:
            return self.transform(u_e)
        all_items = torch.cat(items_chunks, 0)
        all_rows = torch.cat(rows_chunks, 0)
        # hop1 / hop2: 对每个 hist item 取其 rel 邻居均值
        h1 = self._hop_aggregate(all_items, rel=1, dev=dev)  # [H, d]
        h2 = self._hop_aggregate(all_items, rel=2, dev=dev)  # [H, d]
        # 按用户聚合 (segment mean, all_rows in [0,B))
        u_hop1 = torch.zeros(B, self.dim, device=dev).index_add_(0, all_rows, h1)
        u_hop2 = torch.zeros(B, self.dim, device=dev).index_add_(0, all_rows, h2)
        cnt = torch.zeros(B, dtype=torch.float, device=dev).index_add_(
            0, all_rows, torch.ones(all_rows.numel(), device=dev))
        cnt = cnt.clamp(min=1).unsqueeze(-1)
        u_hop1 = u_hop1 / cnt
        u_hop2 = u_hop2 / cnt
        # 注意力聚合
        s1 = (u_e * u_hop1).sum(-1, keepdim=True)
        s2 = (u_e * u_hop2).sum(-1, keepdim=True)
        attn = torch.softmax(torch.cat([s1, s2], -1), -1)
        ctx = attn[:, 0:1] * u_hop1 + attn[:, 1:2] * u_hop2
        return self.transform(ctx + u_e)

    def _item_repr_with_kg(self, item_ids, uids_enhanced, dev):
        """item_ids: 1-D LongTensor (候选 item), uids_enhanced: [B, d] (已注入 ripple 上下文的 user)
        返回 item 增强表示（直接 = item_emb，因为 user 侧已包含 ripple 上下文）。
        （也可用 hop1 增强 item 本体，但用户已注入上下文后这一项增益小、还耗算。）"""
        return self.item_emb.weight[item_ids]

    def to(self, dev):
        self.user_emb = self.user_emb.to(dev)
        self.item_emb = self.item_emb.to(dev)
        self.rel_emb = self.rel_emb.to(dev)
        self.transform = self.transform.to(dev)
        return self

    def parameters(self):
        return (list(self.user_emb.parameters()) + list(self.item_emb.parameters()) +
                list(self.rel_emb.parameters()) + list(self.transform.parameters()))

    def fit(self, train_pos, train_neg, edge_index, edge_type, node_offsets, n_total,
            epochs=30, batch=256, lr=0.001, dev='cuda'):
        self.to(dev)
        self._build_kg(edge_index, edge_type, n_total, dev)
        self._set_user_history(train_pos)
        n = len(train_pos)
        u = torch.LongTensor([p[0] for p in train_pos]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pos]).to(dev)
        all_negs = [n_ for n_ in train_neg]
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ub = u[idx]; pb = ip[idx]
                # train_neg[i] = (uid, [neg_items]); 每 batch 随机抽一个
                nb = torch.LongTensor([
                    int(np.random.choice(all_negs[j][1]))
                    for j in idx.tolist()
                ]).to(dev)
                # 用 ripple 用户上下文评分（替代之前裸 user_emb）
                u_ctx = self._ripple_user_context(ub, dev)  # [B, d]
                pos_e = self.item_emb.weight[pb]  # [B, d]
                neg_e = self.item_emb.weight[nb]  # [B, d]
                pos = (u_ctx * pos_e).sum(-1)
                neg = (u_ctx * neg_e).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  RippleNet ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        """cand: [B, N] 候选 item。返回 [B, N] 分数。
        每个用户使用其 ripple 上下文（user_emb + hop1/hop2 transformer）评分。
        """
        with torch.no_grad():
            uids_t = torch.LongTensor(uids).to(self.user_emb.weight.device)
            cand_t = torch.LongTensor(cand).to(self.user_emb.weight.device)
            dev = self.user_emb.weight.device
            # ripple 用户上下文（per-user，独立、无 batch 依赖）
            u_ctx = self._ripple_user_context(uids_t, dev)  # [B, d]
            i_e = self.item_emb.weight[cand_t]  # [B, N, d]
            scores = (u_ctx.unsqueeze(1) * i_e).sum(-1)  # [B, N]
        return scores.detach().cpu().numpy()


# ---------------- KGCL (GPU, 图对比) ----------------
class KGCL(nn.Module):
    """KGCL: BPR-MF + 异构图对比正则化.

    在异构图上做 edge-dropout 双视图 InfoNCE，
    但不使用 KG embedding 分支和 adaptive gate（与本文方法对照）。
    """
    def __init__(self, n_users, n_items, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.dim = dim
        self.n_users, self.n_items = n_users, n_items
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        self.graph = None
        self.gat = None
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def _build_graph(self, edge_index, edge_type, n_total):
        from recommender.kgat.sparse_gat import SparseGAT
        self.graph = {'edge_index': edge_index, 'edge_type': edge_type,
                       'n_total': n_total, 'item_offset': self.n_users}
        self.node_emb = nn.Embedding(n_total, self.dim)
        nn.init.normal_(self.node_emb.weight, 0, 0.01)
        self.gat = SparseGAT(self.dim, self.dim, self.dim, n_heads=4, n_layers=2)

    def _drop_edges(self, ei, et, rates=(0.1, 0.2, 0.3)):
        """对 4 种边类型分别做 edge dropout。"""
        masks = []
        for t in range(4):
            r = rates[t] if t < len(rates) else 0.2
            tm = (et == t)
            if tm.sum() == 0:
                m = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
                masks.append(m); continue
            keep = torch.rand(tm.sum().item(), device=ei.device) >= r
            m = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
            m[tm] = keep
            masks.append(m)
        mask = masks[0]
        for m in masks[1:]:
            mask = mask & m
        return ei[:, mask]

    def to(self, dev):
        self.user_emb = self.user_emb.to(dev)
        self.item_emb = self.item_emb.to(dev)
        self.proj = self.proj.to(dev)
        if self.gat: self.gat = self.gat.to(dev)
        if hasattr(self, 'node_emb') and self.node_emb is not None:
            self.node_emb = self.node_emb.to(dev)
        return self

    def parameters(self):
        ps = list(self.user_emb.parameters()) + list(self.item_emb.parameters()) + list(self.proj.parameters())
        if self.gat: ps += list(self.gat.parameters())
        if hasattr(self, 'node_emb') and self.node_emb is not None:
            ps += list(self.node_emb.parameters())
        return ps

    def fit(self, train_pos, train_neg, edge_index, edge_type, n_total,
            epochs=30, batch=256, lr=0.001, dev='cuda', lambda_cl=0.1):
        self._build_graph(edge_index, edge_type, n_total)
        self.to(dev)
        ei = self.graph['edge_index'].to(dev)
        et = self.graph['edge_type'].to(dev)
        n = len(train_pos)
        u = torch.LongTensor([p[0] for p in train_pos]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pos]).to(dev)
        all_negs = [n_ for n_ in train_neg]
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ub = u[idx]; pb = ip[idx]
                # train_neg[i] = (uid, [neg_items]); 每 batch 随机抽一个
                nb = torch.LongTensor([
                    int(np.random.choice(all_negs[j][1]))
                    for j in idx.tolist()
                ]).to(dev)
                pos = (self.user_emb.weight[ub] * self.item_emb.weight[pb]).sum(-1)
                neg = (self.user_emb.weight[ub] * self.item_emb.weight[nb]).sum(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                # 图对比：两个 edge-dropout 视图
                v1 = self.gat(self.node_emb.weight, self._drop_edges(ei, et))
                v2 = self.gat(self.node_emb.weight, self._drop_edges(ei, et))
                nodes = torch.cat([ub, pb + self.n_users, nb + self.n_users])
                z1 = self.proj(v1[nodes]); z2 = self.proj(v2[nodes])
                z1 = torch.nn.functional.normalize(z1, dim=-1)
                z2 = torch.nn.functional.normalize(z2, dim=-1)
                sim = z1 @ z2.t() / 0.1
                pos_sim = torch.diagonal(sim).clone()
                sim.masked_fill_(torch.eye(len(nodes), device=dev, dtype=torch.bool), float('-inf'))
                cl = -pos_sim + torch.logsumexp(sim, dim=-1)
                cl = cl.mean()
                loss = bpr + lambda_cl * cl
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  KGCL ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        with torch.no_grad():
            uids_t = torch.LongTensor(uids).to(self.user_emb.weight.device)
            cand_t = torch.LongTensor(cand).to(self.user_emb.weight.device)
            u_e = self.user_emb.weight[uids_t]
            i_e = self.item_emb.weight[cand_t]
            scores = (u_e.unsqueeze(1) * i_e).sum(-1)
        return scores.detach().cpu().numpy()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--baselines', nargs='+',
                   default=['popular', 'usercf', 'itemcf', 'bprmf', 'lightgcn', 'ripplen', 'kgcl'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--usercf-topk', type=int, default=200)
    ap.add_argument('--output', default='experiments/results/baselines_real.json')
    ap.add_argument('--ripplen-hops', type=int, default=2, help='RippleNet hops')
    ap.add_argument('--kgcl-lambda', type=float, default=0.1, help='KGCL lambda_cl')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {dev}")

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pos = [(p[0], p[1]) for p, _ in data['train_data']]
    # data['train_data'][i] = ((uid, pos_item), [neg_items...])
    # 取整条负样本列表(n)当作 (uid, neg_list)
    train_neg = [(p[0], n) for p, n in data['train_data']]
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                                n_negative=args.n_negative, seed=args.seed)
    logger.info(f"data={args.data} train={len(train_pos)} test={len(test_data)} "
                f"n_u={n_u} n_i={n_i} epochs={args.epochs}")

    # 构建异构图（供 RippleNet 和 KGCL 使用）
    from recommender.kgat.hetero_graph import build_hetero_graph
    idx2user = {idx: uid for uid, idx in data['user2idx'].items()}
    idx2item = {idx: iid for iid, idx in data['item2idx'].items()}
    raw_pairs = [(idx2user[p[0]], idx2item[p[1]]) for p in train_pos]
    ei, et, n_total, node_offsets = build_hetero_graph(
        products, raw_pairs, data['user2idx'], data['item2idx']
    )
    logger.info(f"Hetero graph: {n_total} nodes, {ei.size(1)} edges")

    results = {}
    for name in args.baselines:
        logger.info(f"==== {name} ====")
        t0 = time.time()
        try:
            if name == 'popular':
                m = Popular(n_i); m.fit(train_pos)
            elif name == 'usercf':
                m = UserCF(n_u, n_i, top_k=args.usercf_topk); m.fit(train_pos)
            elif name == 'itemcf':
                m = ItemCF(n_u, n_i); m.fit(train_pos)
            elif name == 'bprmf':
                m = BPRMF(n_u, n_i).to(dev); m.fit(train_pos, train_neg, epochs=args.epochs, dev=dev)
            elif name == 'lightgcn':
                m = LightGCN(n_u, n_i).to(dev); m.fit(train_pos, train_neg, epochs=args.epochs, dev=dev)
            elif name == 'ripplen':
                m = RippleNet(n_u, n_i, n_hops=args.ripplen_hops).to(dev)
                m.fit(train_pos, train_neg, ei, et, node_offsets, n_total,
                      epochs=args.epochs, dev=dev)
            elif name == 'kgcl':
                m = KGCL(n_u, n_i).to(dev)
                m.fit(train_pos, train_neg, ei, et, n_total,
                      epochs=args.epochs, dev=dev, lambda_cl=args.kgcl_lambda)
            else:
                logger.warning(f"unknown {name}, skip"); continue
            logger.info(f"  {name} trained in {time.time()-t0:.0f}s, evaluating...")
            r = evaluate_batch(m.score_batch, test_data, [5, 10, 20])
            logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}  "
                        f"Precision@10={r['precision@10']:.4f} Recall@10={r['recall@10']:.4f}  "
                        f"({time.time()-t0:.0f}s total)")
            results[name] = r
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'data': args.data, 'args': vars(args), 'results': results}, f,
                  indent=2, ensure_ascii=False)
    logger.info(f"saved -> {out}")
    print("\n=== Baselines on real data ===")
    for k, v in results.items():
        if 'error' not in v:
            print(f"  {k:10s} HR@10={v['hr@10']:.4f}  NDCG@10={v['ndcg@10']:.4f}  "
                  f"HR@20={v['hr@20']:.4f}")


if __name__ == '__main__':
    main()
