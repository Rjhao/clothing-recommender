# -*- coding: utf-8 -*-
"""
KG-HybridRec 三项改进的参考实现
  (1) CAGF  置信度感知的自适应门控融合（模态级 + 分支级）
  (2) KG-CSL 知识图谱对比自监督学习（图增强 + InfoNCE）
  (3) AttnExplainPath 注意力加权的解释路径

依赖：仅需 torch（>=1.10）。无需 torch_geometric / torch_scatter，便于直接运行。
作者：供论文 4.3 节复现使用。所有符号与论文公式保持一致。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# (1) CAGF —— 置信度感知的自适应门控融合
# =====================================================================
class CAGFModal(nn.Module):
    """模态级门控融合：替代固定权重(图像0.6/文本0.4)。
       g = softmax(W_g · [h_img; h_txt; c] + b_g)
       h_mm = g_img * h_img + g_txt * h_txt
    其中上下文 c = [log(1+交互次数), ||h_img||, ||h_txt||]，体现各模态可靠性。
    """
    def __init__(self, dim, ctx_dim=3, hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),          # 输出两路(图像/文本)门控 logits
        )

    def forward(self, h_img, h_txt, inter_count):
        # h_img, h_txt: [B, dim]; inter_count: [B] 该商品交互次数
        ctx = torch.stack([
            torch.log1p(inter_count.float()),
            h_img.norm(dim=-1),
            h_txt.norm(dim=-1),
        ], dim=-1)                                   # [B, 3]
        logits = self.gate(torch.cat([h_img, h_txt, ctx], dim=-1))  # [B, 2]
        g = torch.softmax(logits, dim=-1)            # [B, 2]
        h_mm = g[:, 0:1] * h_img + g[:, 1:2] * h_txt
        return h_mm, g                               # 返回门控权重便于可视化/分析


class CAGFBranch(nn.Module):
    """分支级门控融合：替代 h = h_cf ‖ h_kg ‖ h_mm 的简单拼接。
       β = softmax(W · [h_cf; h_kg; h_mm; c] + b)
       h = β_cf*h_cf + β_kg*h_kg + β_mm*h_mm
    冷启动(交互稀疏)时，模型自动调低 β_cf、调高 β_kg/β_mm。
    """
    def __init__(self, dim, ctx_dim=1, hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, h_cf, h_kg, h_mm, inter_count):
        ctx = torch.log1p(inter_count.float()).unsqueeze(-1)        # [B,1] 可靠性上下文
        logits = self.gate(torch.cat([h_cf, h_kg, h_mm, ctx], dim=-1))
        beta = torch.softmax(logits, dim=-1)                        # [B,3]
        h = beta[:, 0:1] * h_cf + beta[:, 1:2] * h_kg + beta[:, 2:3] * h_mm
        return h, beta


# =====================================================================
# (2) KG-CSL —— 知识图谱对比自监督学习
# =====================================================================
def augment_graph(edge_index, x, edge_drop=0.2, feat_mask=0.2):
    """对异构图做两类结构扰动，生成一个增强视图。
       edge_index: [2, E]  (无需 PyG，纯张量)
       x:          [N, dim]
    """
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > edge_drop      # 随机边丢弃
    ei = edge_index[:, keep]
    mask = (torch.rand_like(x) > feat_mask).float()                 # 节点特征掩码
    return ei, x * mask


class LightGCNEncoder(nn.Module):
    """轻量图编码器(对称归一化邻居聚合)，仅用 torch 实现，用于产生对比视图表示。
       工程中可替换为论文使用的多层 GAT。"""
    def __init__(self, dim, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.lin = nn.Linear(dim, dim)

    def forward(self, edge_index, x):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=torch.float)) + 1e-6
        norm = deg.pow(-0.5)
        h = x
        out = x
        for _ in range(self.n_layers):
            msg = h[src] * (norm[src] * norm[dst]).unsqueeze(-1)
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = self.lin(agg)
            out = out + h
        return out / (self.n_layers + 1)


def info_nce(z1, z2, tau=0.2):
    """节点级 InfoNCE 对比损失（同一节点两视图为正样本，批内其余为负样本）。
       L_cl = -Σ_v log[ exp(sim(z'_v,z''_v)/τ) / Σ_k exp(sim(z'_v,z''_k)/τ) ]
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / tau                  # [N, N]
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class KGContrastiveSSL(nn.Module):
    """整合：两次增强 -> 编码 -> InfoNCE。返回对比损失，供 L = L_BPR + λ1*L_cl + λ2*||Θ||²。"""
    def __init__(self, dim, n_layers=2, tau=0.2, edge_drop=0.2, feat_mask=0.2):
        super().__init__()
        self.encoder = LightGCNEncoder(dim, n_layers)
        self.tau, self.edge_drop, self.feat_mask = tau, edge_drop, feat_mask

    def forward(self, edge_index, x):
        ei1, x1 = augment_graph(edge_index, x, self.edge_drop, self.feat_mask)
        ei2, x2 = augment_graph(edge_index, x, self.edge_drop, self.feat_mask)
        z1 = self.encoder(ei1, x1)
        z2 = self.encoder(ei2, x2)
        return info_nce(z1, z2, self.tau)


# =====================================================================
# (3) 注意力加权的解释路径
# =====================================================================
def attn_explanation_path(adj, src_items, target_item, max_hop=3):
    """从用户历史交互商品(src_items)出发，在 KG 中搜索通往 target_item 的路径，
       以 GAT 已学得的注意力权重沿路径累乘作为路径得分，返回得分最高的路径。
       adj: dict[node] -> list[(neighbor, relation, attn_weight)]，attn_weight∈(0,1]
       做法：以 -log(attn) 为边权的最短路(等价于注意力累乘最大路)，Dijkstra。
    """
    import heapq
    best = {s: (0.0, [s], []) for s in src_items}      # node -> (cost=-Σlog a, 路径节点, 关系序列)
    pq = [(0.0, s) for s in src_items]
    heapq.heapify(pq)
    visited = {}
    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited[u] = best[u]
        if u == target_item:
            break
        if len(best[u][1]) - 1 >= max_hop:
            continue
        for v, rel, a in adj.get(u, []):
            a = max(float(a), 1e-6)
            nc = cost - math.log(a)
            if v not in best or nc < best[v][0]:
                best[v] = (nc, best[u][1] + [v], best[u][2] + [rel])
                heapq.heappush(pq, (nc, v))
    if target_item not in visited:
        return None
    cost, nodes, rels = visited[target_item]
    return {"nodes": nodes, "relations": rels, "score": math.exp(-cost)}


_REL_TPL = {"属于": "同属{}", "搭配": "可与其搭配", "相似": "与其风格相似"}

def path_to_text(path, name_fn=lambda x: str(x)):
    """把解释路径转成自然语言理由。"""
    if path is None:
        return "未找到可解释路径"
    parts = [name_fn(path["nodes"][0])]
    for rel, nxt in zip(path["relations"], path["nodes"][1:]):
        parts.append(f"——[{rel}]——>{name_fn(nxt)}")
    chain = "".join(parts)
    return f"因为你曾关注「{name_fn(path['nodes'][0])}」，它{chain}（路径置信度{path['score']:.3f}），故为你推荐。"


# =====================================================================
# 集成示例（最小 KG-HybridRec 骨架）+ 合成数据冒烟测试
# =====================================================================
class KGHybridRec(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)   # 协同过滤分支
        self.item_cf  = nn.Embedding(n_items, dim)
        self.item_kg  = nn.Embedding(n_items, dim)    # 知识图谱分支(简化)
        self.img_proj = nn.Linear(128, dim)           # 多模态
        self.txt_proj = nn.Linear(768, dim)
        self.cagf_modal  = CAGFModal(dim)
        self.cagf_branch = CAGFBranch(dim)
        self.predict = nn.Linear(2 * dim, 1)

    def item_repr(self, item_ids, img, txt, inter_cnt):
        h_cf = self.item_cf(item_ids)
        h_kg = self.item_kg(item_ids)
        h_img = self.img_proj(img)
        h_txt = self.txt_proj(txt)
        h_mm, _ = self.cagf_modal(h_img, h_txt, inter_cnt)
        h_item, beta = self.cagf_branch(h_cf, h_kg, h_mm, inter_cnt)
        return h_item, beta

    def forward(self, users, items, img, txt, inter_cnt):
        hu = self.user_emb(users)
        hi, beta = self.item_repr(items, img, txt, inter_cnt)
        score = torch.sigmoid(self.predict(torch.cat([hu, hi], dim=-1))).squeeze(-1)
        return score, beta


def bpr_loss(pos, neg):
    return -F.logsigmoid(pos - neg).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    NU, NI, B, D = 200, 500, 64, 64
    model = KGHybridRec(NU, NI, D)
    ssl = KGContrastiveSSL(D)
    opt = torch.optim.Adam(list(model.parameters()) + list(ssl.parameters()), lr=1e-3)

    # 合成一批数据
    users = torch.randint(0, NU, (B,))
    pos_i = torch.randint(0, NI, (B,))
    neg_i = torch.randint(0, NI, (B,))
    img = torch.randn(B, 128); txt = torch.randn(B, 768)
    cnt_pos = torch.randint(0, 3, (B,))      # 冷启动商品(交互少)
    cnt_neg = torch.randint(0, 50, (B,))

    # 合成 KG 用于对比自监督
    N_node = NI
    x = model.item_kg.weight.detach()
    E = 4000
    edge_index = torch.randint(0, N_node, (2, E))

    lam1, lam2 = 0.1, 1e-4
    for step in range(3):
        opt.zero_grad()
        s_pos, beta = model(users, pos_i, img, txt, cnt_pos)
        s_neg, _    = model(users, neg_i, img, txt, cnt_neg)
        l_bpr = bpr_loss(s_pos, s_neg)
        l_cl  = ssl(edge_index, x)
        l_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss = l_bpr + lam1 * l_cl + lam2 * l_reg
        loss.backward(); opt.step()
        print(f"step{step}  L={loss.item():.4f}  BPR={l_bpr.item():.4f}  CL={l_cl.item():.4f}")

    # 验证冷启动门控行为：交互少 -> β_cf 应更小
    print("分支门控 β(均值) [cf, kg, mm] =", beta.mean(0).detach().numpy().round(3))

    # 解释路径示例
    adj = {0: [(7, "属于", 0.6), (3, "相似", 0.3)],
           7: [(42, "搭配", 0.8)], 3: [(42, "相似", 0.5)]}
    p = attn_explanation_path(adj, src_items=[0], target_item=42, max_hop=3)
    print(path_to_text(p))
    print("OK")
