# -*- coding: utf-8 -*-
"""
KG-HybridRec 真实实现（严格对齐论文 4.3 节四层架构）
  1) 协同过滤层  : BPR 矩阵分解（可学习 user/item 嵌入）
  2) 知识图谱层  : TransR 实体/关系嵌入 + 关系特定投影（含 KG 嵌入损失）
  3) 多模态融合层: CLIP/BERT 特征投影 + CAGF 模态级门控
  4) GAT 融合层  : 多层图注意力在异构图上聚合高阶邻居 + CAGF 分支级门控
  预测/解释      : MLP 打分 + 注意力加权解释路径（见 modules.py）

节点统一编号：users [0,NU) | items [NU,NU+NI) | 其它实体 [NU+NI,N)
异构图 edge_index: [2,E]（无向边请双向各存一次）
"""
import torch, torch.nn as nn, torch.nn.functional as F
from modules import CAGFModal, CAGFBranch, KGContrastiveSSL


class GATLayer(nn.Module):
    """手写单头 GAT（多头可堆叠多个实例）。"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * dim) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        N = x.size(0); src, dst = edge_index
        h = self.W(x)
        e = self.leaky((torch.cat([h[dst], h[src]], -1) * self.a).sum(-1))
        e = e - e.max()
        alpha = e.exp()
        denom = torch.zeros(N, device=x.device).index_add_(0, dst, alpha) + 1e-10
        alpha = alpha / denom[dst]
        out = torch.zeros_like(h).index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return F.elu(out)


class TransR(nn.Module):
    def __init__(self, n_ent, n_rel, d_e, d_r=None):
        super().__init__()
        d_r = d_r or d_e
        self.ent = nn.Embedding(n_ent, d_e)
        self.rel = nn.Embedding(n_rel, d_r)
        self.proj = nn.Embedding(n_rel, d_e * d_r)
        self.d_e, self.d_r = d_e, d_r
        for p in (self.ent, self.rel, self.proj):
            nn.init.xavier_uniform_(p.weight)

    def _project(self, e, r_id):
        M = self.proj(r_id).view(-1, self.d_e, self.d_r)
        return torch.bmm(e.unsqueeze(1), M).squeeze(1)

    def score(self, h, r, t):
        hr = self._project(self.ent(h), r); tr = self._project(self.ent(t), r)
        return (hr + self.rel(r) - tr).pow(2).sum(-1)

    def loss(self, h, r, t, t_neg, margin=1.0):
        return F.relu(margin + self.score(h, r, t) - self.score(h, r, t_neg)).mean()


class KGHybridRec(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_relations,
                 dim=64, img_dim=128, txt_dim=768, gat_layers=2):
        super().__init__()
        self.NU, self.NI = n_users, n_items
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.kg = TransR(n_entities, n_relations, dim)
        self.kg_proj = nn.Linear(dim, dim)
        self.img_proj = nn.Linear(img_dim, dim)
        self.txt_proj = nn.Linear(txt_dim, dim)
        self.cagf_modal = CAGFModal(dim)
        self.gats = nn.ModuleList([GATLayer(dim) for _ in range(gat_layers)])
        self.cagf_branch = CAGFBranch(dim)
        self.predict = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.ssl = KGContrastiveSSL(dim)
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.01)

    def build_x(self, edge_index):
        """构造整张异构图的节点初始特征 [N, dim]（用户/商品/实体）。
        统一编号：图节点 NU+eid 对应 TransR 实体 eid（商品实体 0..NI-1，外部实体 NI..）。
        KG-CSL 与 GAT 共用此特征，故节点数 N 必须一致。"""
        dim = self.user_emb.weight.size(1)
        N = int(edge_index.max().item()) + 1
        x = torch.zeros(N, dim, device=self.user_emb.weight.device)
        x[:self.NU] = self.user_emb.weight
        x[self.NU:self.NU + self.NI] = self.item_emb.weight
        n_ext = N - (self.NU + self.NI)
        if n_ext > 0:
            x[self.NU + self.NI:] = self.kg.ent.weight[self.NI:self.NI + n_ext]
        return x

    def encode(self, edge_index, item_entity_id, img, txt, inter_cnt,
               use_mm=True, use_gat=True, use_cagf=True):
        x = self.build_x(edge_index)
        z = x
        if use_gat:
            for gat in self.gats:
                z = gat(z, edge_index)
        z_user = z[:self.NU]
        h_cf_item = z[self.NU:self.NU + self.NI]
        h_kg_item = self.kg_proj(self.kg.ent(item_entity_id))
        if use_mm:
            h_mm_item, _ = self.cagf_modal(self.img_proj(img), self.txt_proj(txt), inter_cnt)
        else:
            h_mm_item = torch.zeros_like(h_cf_item)
        if use_cagf:
            h_item, beta = self.cagf_branch(h_cf_item, h_kg_item, h_mm_item, inter_cnt)
        else:
            h_item = (h_cf_item + h_kg_item + h_mm_item) / 3.0; beta = None
        return z_user, h_item, beta

    def score_pairs(self, z_user, h_item, users, items):
        return torch.sigmoid(self.predict(
            torch.cat([z_user[users], h_item[items]], -1))).squeeze(-1)


def bpr_loss(pos, neg):
    return -F.logsigmoid(pos - neg + 1e-8).mean()
