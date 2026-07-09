"""Baselines for KG-HybridRec comparison (stage 3).

Implements classic baselines under the SAME sampled-100 leave-one-out
protocol as the main model, so numbers are directly comparable.

Baselines:
    usercf   - User-based CF (cosine on user-item matrix)
    itemcf   - Item-based CF (cosine on item-item co-occurrence)
    bprmf    - BPR Matrix Factorization (the CF branch alone)
    lightgcn - LightGCN graph convolution + BPR
    popular  - Popularity baseline (lower bound)

All trained on the same train split as the main model (leave-one-out).
"""
import argparse
import json
import logging
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_scores(score_fn, test_data, k_values=(5, 10, 20)):
    """Evaluate a score_fn(user_idx, item_idx) -> float over sampled-100 test set."""
    results = {f'{m}@{k}': [] for m in ('hr', 'ndcg') for k in k_values}
    for user_idx, pos_items, neg_items in test_data:
        candidates = list(pos_items) + list(neg_items)
        scores = np.array([score_fn(user_idx, it) for it in candidates])
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        for k in k_values:
            results[f'hr@{k}'].append(compute_hr_at_k(pos_items, ranked, k))
            results[f'ndcg@{k}'].append(compute_ndcg_at_k(pos_items, ranked, k))
    return {k: float(np.mean(v)) for k, v in results.items()}


# ---------------- UserCF ----------------
class UserCF:
    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.sim = None
        self.ui = None

    def fit(self, train_pairs):
        ui = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for u, i in train_pairs:
            ui[u, i] = 1.0
        self.ui = ui
        norm = np.linalg.norm(ui, axis=1, keepdims=True)
        norm[norm == 0] = 1
        self.sim = (ui @ ui.T) / (norm @ norm.T)
        np.fill_diagonal(self.sim, 0)

    def score(self, user_idx, item_idx):
        if self.ui[user_idx, item_idx] > 0:
            return -1e9  # exclude seen
        return float(self.sim[user_idx] @ self.ui[:, item_idx])


# ---------------- ItemCF ----------------
class ItemCF:
    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.sim = None
        self.ui = None

    def fit(self, train_pairs):
        ui = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for u, i in train_pairs:
            ui[u, i] = 1.0
        self.ui = ui
        # co-occurrence based cosine
        co = ui.T @ ui
        norm = np.sqrt(np.diag(co))
        denom = np.outer(norm, norm)
        denom[denom == 0] = 1
        self.sim = co / denom
        np.fill_diagonal(self.sim, 0)

    def score(self, user_idx, item_idx):
        if self.ui[user_idx, item_idx] > 0:
            return -1e9
        # sum of sim to user's interacted items
        interacted = np.where(self.ui[user_idx] > 0)[0]
        if len(interacted) == 0:
            return 0.0
        return float(self.sim[item_idx, interacted].sum())


# ---------------- Popular ----------------
class Popular:
    def __init__(self, n_items):
        self.pop = np.zeros(n_items, dtype=np.float32)

    def fit(self, train_pairs):
        for u, i in train_pairs:
            self.pop[i] += 1

    def score(self, user_idx, item_idx):
        return float(self.pop[item_idx])


# ---------------- BPR-MF ----------------
class BPRMFTrainer:
    def __init__(self, n_users, n_items, dim=64):
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        self.n_items = n_items

    def to(self, dev):
        self.user_emb = self.user_emb.to(dev)
        self.item_emb = self.item_emb.to(dev)
        return self

    def parameters(self):
        return list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, train_pairs, neg_pairs, epochs=100, batch=256, lr=0.001, dev='cpu'):
        u = torch.LongTensor([p[0] for p in train_pairs]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pairs]).to(dev)
        ineg = torch.LongTensor([p[1] for p in neg_pairs]).to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(train_pairs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                pos = (self.user_emb.weight[u[idx]] * self.item_emb.weight[ip[idx]]).sum(-1)
                neg = (self.user_emb.weight[u[idx]] * self.item_emb.weight[ineg[idx]]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 25 == 0:
                logger.info(f"  BPR-MF epoch {ep+1}/{epochs} loss={tot/n:.4f}")

    def score(self, user_idx, item_idx):
        with torch.no_grad():
            return float((self.user_emb.weight[user_idx] * self.item_emb.weight[item_idx]).sum())


# ---------------- LightGCN ----------------
class LightGCN:
    """LightGCN: symmetric normalized adjacency, k-layer propagation, BPR."""
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        self.dim, self.n_layers = dim, n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        # 初始化 0.1(原 0.01 太小,稠密图传播后被稀释+weight_decay 衰减致 embedding 坍缩到 0)
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)
        self.norm_adj = None  # sparse

    def _build_adj(self, train_pairs, dev):
        from scipy.sparse import csr_matrix, coo_matrix
        n_u, n_i = self.user_emb.num_embeddings, self.item_emb.num_embeddings
        rows = np.concatenate([np.array([p[0] for p in train_pairs]),
                               np.array([p[1] for p in train_pairs]) + n_u])
        cols = np.concatenate([np.array([p[1] for p in train_pairs]) + n_u,
                               np.array([p[0] for p in train_pairs])])
        data = np.ones(len(rows), dtype=np.float32)
        N = n_u + n_i
        A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        deg = np.array(A.sum(1)).flatten()
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        D = coo_matrix((dinv, (np.arange(N), np.arange(N))), shape=(N, N))
        Anorm = D @ A @ D  # symmetric normalized
        # to torch sparse
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
        self.user_emb = self.user_emb.to(dev)
        self.item_emb = self.item_emb.to(dev)
        return self

    def parameters(self):
        return list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, train_pairs, neg_pairs, dev='cuda', epochs=100, batch=256, lr=0.001):
        self._build_adj(train_pairs, dev)
        u = torch.LongTensor([p[0] for p in train_pairs]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pairs]).to(dev)
        ineg = torch.LongTensor([p[1] for p in neg_pairs]).to(dev)
        # weight_decay=0(LightGCN 原论文不用权重衰减;原 1e-4 在稠密图上把 embedding 压到 0 致坍缩)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        n = len(train_pairs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                e, n_u = self._propagate()
                pos = (e[u[idx]] * e[ip[idx] + n_u]).sum(-1)
                neg = (e[u[idx]] * e[ineg[idx] + n_u]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                # L2 由 Adam 的 weight_decay=1e-4 承担(标准 LightGCN 做法);
                # 原手动全表平方和 L2 在稠密图上梯度过强导致 embedding 坍缩,已移除。
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 25 == 0:
                logger.info(f"  LightGCN epoch {ep+1}/{epochs} loss={tot/n:.4f}")

    def score(self, user_idx, item_idx):
        with torch.no_grad():
            e, n_u = self._propagate()
            return float((e[user_idx] * e[item_idx + n_u]).sum())


# ---------------- RippleNet (KG baseline) ----------------
class RippleNetBaseline:
    """RippleNet-style KG recommendation.

    Uses the heterogeneous graph as KG: for each user, propagates interest
    through item -> category/brand neighbors (ripple sets), aggregates with
    attention, BPR training. Simplified single-hop ripple over item attributes.
    """
    def __init__(self, n_users, n_items, dim=64, n_hops=2):
        self.dim, self.n_hops = dim, n_hops
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        # entity embeddings for category+brand entities (offset within graph)
        self.n_entities = 0
        self.entity_emb = None
        self.item_neighbors = None  # item_idx -> list of entity node offsets
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        self.n_items = n_items
        self.n_users = n_users
        self.transform = nn.Linear(dim, dim, bias=False)

    def _build_kg(self, products, item2idx):
        """Vectorized single-hop ripple over item category (each item has 1 cat)."""
        cats = sorted({p['category'] for p in products if p.get('category')})
        cat2e = {c: i for i, c in enumerate(cats)}
        self.n_entities = len(cats)
        self.entity_emb = nn.Embedding(self.n_entities, self.dim)
        nn.init.normal_(self.entity_emb.weight, 0, 0.01)
        # item_idx -> category entity idx (-1 if none); vectorizable lookup
        self.item_cat = torch.full((self.n_items,), -1, dtype=torch.long)
        for p in products:
            iid = item2idx.get(int(p['product_id']))
            if iid is not None and p.get('category') in cat2e:
                self.item_cat[iid] = cat2e[p['category']]

    def _item_repr(self, item_ids):
        """Item embedding + KG ripple (vectorized single-hop over category)."""
        e = self.item_emb.weight[item_ids]  # [B, d]
        cat_idx = self.item_cat[item_ids]  # [B], already on dev via to()
        valid = cat_idx >= 0
        if valid.any():
            cat_emb = self.entity_emb.weight[cat_idx.clamp(min=0)]  # [B, d]
            attn = torch.sigmoid((e * cat_emb).sum(-1, keepdim=True) / self.dim**0.5)  # [B,1]
            agg = self.transform(attn * cat_emb)
            agg = agg * valid.unsqueeze(-1).float().to(agg.dtype)
            return e + agg
        return e

    def to(self, dev):
        for m in [self.user_emb, self.item_emb, self.transform] + ([self.entity_emb] if self.entity_emb else []):
            m.to(dev)
        if hasattr(self, 'item_cat'):
            self.item_cat = self.item_cat.to(dev)
        return self

    def parameters(self):
        ps = list(self.user_emb.parameters()) + list(self.item_emb.parameters()) + list(self.transform.parameters())
        if self.entity_emb is not None:
            ps += list(self.entity_emb.parameters())
        return ps

    def fit(self, train_pairs, neg_pairs, products, item2idx, dev='cuda', epochs=100, batch=256, lr=0.001):
        self._build_kg(products, item2idx)
        self.to(dev)
        u = torch.LongTensor([p[0] for p in train_pairs]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pairs]).to(dev)
        ineg = torch.LongTensor([p[1] for p in neg_pairs]).to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(train_pairs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ub = u[idx]
                # item repr with KG ripple (loop is slow but batch small)
                pos_r = self._item_repr(ip[idx])
                neg_r = self._item_repr(ineg[idx])
                pos = (self.user_emb.weight[ub] * pos_r).sum(-1)
                neg = (self.user_emb.weight[ub] * neg_r).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 25 == 0:
                logger.info(f"  RippleNet epoch {ep+1}/{epochs} loss={tot/n:.4f}")

    def score(self, user_idx, item_idx):
        with torch.no_grad():
            r = self._item_repr(torch.LongTensor([item_idx]).to(self.user_emb.weight.device))
            return float((self.user_emb.weight[user_idx] * r).sum())


# ---------------- KGCL (graph contrastive baseline) ----------------
class KGCLBaseline:
    """KGCL-style: BPR-MF + graph contrastive regularization.

    Same heterogeneous graph + edge-dropout two-view InfoNCE as our
    improvement B, but WITHOUT the KG embedding branch and adaptive gate.
    This is the direct ablation isolating graph contrastive learning.
    """
    def __init__(self, n_users, n_items, dim=64):
        self.dim = dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        self.graph = None  # built in fit
        self.gat = None
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.n_users, self.n_items = n_users, n_items

    def _build_graph(self, products, train_data, item2idx):
        from recommender.kgat.hetero_graph import build_hetero_graph
        from recommender.kgat.sparse_gat import SparseGAT
        user2idx = {i: i for i in range(self.n_users)}  # identity mapping
        # train_data entries are (pos_tuple, neg_tuple); extract raw (user_idx, item_idx)
        pairs = [(pos[0], pos[1]) for pos, _ in train_data]
        ei, et, n_total, _ = build_hetero_graph(products, pairs, user2idx, item2idx)
        self.graph = {'edge_index': ei, 'edge_type': et, 'n_total': n_total,
                      'item_offset': self.n_users}
        self.node_emb = nn.Embedding(n_total, self.dim)
        nn.init.normal_(self.node_emb.weight, 0, 0.01)
        self.gat = SparseGAT(self.dim, self.dim, self.dim, n_heads=4, n_layers=2)

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

    def _drop_edges(self, ei, et, rates=(0.1, 0.2, 0.3)):
        masks = []
        for t in range(3):
            r = rates[t] if t < len(rates) else 0.2
            tm = (et == t)
            keep = (torch.rand(tm.sum().item(), device=ei.device) >= r) if tm.sum()>0 else torch.ones(0,dtype=torch.bool,device=ei.device)
            m = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
            m[tm] = keep
            masks.append(m)
        mask = masks[0] & masks[1] & masks[2]
        return ei[:, mask]

    def fit(self, train_data, products, item2idx, dev='cuda', epochs=100, batch=256, lr=0.001, lambda_cl=0.1):
        # train_data: list of (pos_tuple, neg_tuple)
        train_pairs = [(pos[0], pos[1]) for pos, _ in train_data]
        neg_pairs = [(neg[0], neg[1]) for _, neg in train_data]
        self._build_graph(products, train_data, item2idx)
        self.to(dev)
        ei = self.graph['edge_index'].to(dev); et = self.graph['edge_type'].to(dev)
        u = torch.LongTensor([p[0] for p in train_pairs]).to(dev)
        ip = torch.LongTensor([p[1] for p in train_pairs]).to(dev)
        ineg = torch.LongTensor([p[1] for p in neg_pairs]).to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(train_pairs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                ub = u[idx]; pb = ip[idx]; nb = ineg[idx]
                pos = (self.user_emb.weight[ub] * self.item_emb.weight[pb]).sum(-1)
                neg = (self.user_emb.weight[ub] * self.item_emb.weight[nb]).sum(-1)
                bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                # graph contrastive: two edge-dropout views
                v1 = self.gat(self.node_emb.weight, self._drop_edges(ei, et))
                v2 = self.gat(self.node_emb.weight, self._drop_edges(ei, et))
                nodes = torch.cat([ub, pb + self.n_users, nb + self.n_users])
                z1 = self.proj(v1[nodes]); z2 = self.proj(v2[nodes])
                z1 = torch.nn.functional.normalize(z1, dim=-1); z2 = torch.nn.functional.normalize(z2, dim=-1)
                sim = z1 @ z2.t() / 0.1
                pos_sim = torch.diagonal(sim).clone()
                sim.masked_fill_(torch.eye(len(nodes), device=dev, dtype=torch.bool), float('-inf'))
                cl = -pos_sim + torch.logsumexp(sim, dim=-1)
                cl = cl.mean()
                loss = bpr + lambda_cl * cl
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 25 == 0:
                logger.info(f"  KGCL epoch {ep+1}/{epochs} loss={tot/n:.4f}")

    def score(self, user_idx, item_idx):
        with torch.no_grad():
            return float((self.user_emb.weight[user_idx] * self.item_emb.weight[item_idx]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_expanded.json')
    ap.add_argument('--baselines', nargs='+', default=['popular', 'usercf', 'itemcf', 'bprmf', 'lightgcn'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='experiments/results/baselines.json')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']

    # train positive pairs + negative pairs
    train_pos = [(pos[0], pos[1]) for pos, _ in data['train_data']]
    train_neg = [(neg[0], neg[1]) for _, neg in data['train_data']]

    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'], 100, args.seed)
    logger.info(f"train={len(train_pos)} test_users={len(test_data)} dev={dev}")

    results = {}
    for name in args.baselines:
        logger.info(f"==== {name} ====")
        if name == 'popular':
            m = Popular(n_i); m.fit(train_pos)
            r = evaluate_scores(m.score, test_data)
        elif name == 'usercf':
            m = UserCF(n_u, n_i); m.fit(train_pos)
            r = evaluate_scores(m.score, test_data)
        elif name == 'itemcf':
            m = ItemCF(n_u, n_i); m.fit(train_pos)
            r = evaluate_scores(m.score, test_data)
        elif name == 'bprmf':
            m = BPRMFTrainer(n_u, n_i).to(dev)
            m.fit(train_pos, train_neg, epochs=args.epochs, dev=dev)
            r = evaluate_scores(m.score, test_data)
        elif name == 'lightgcn':
            m = LightGCN(n_u, n_i, dim=64, n_layers=3).to(dev)
            m.fit(train_pos, train_neg, dev=dev, epochs=args.epochs)
            r = evaluate_scores(m.score, test_data)
        elif name == 'ripplenet':
            m = RippleNetBaseline(n_u, n_i, dim=64)
            m.fit(train_pos, train_neg, products, data['item2idx'], dev=dev, epochs=args.epochs)
            r = evaluate_scores(m.score, test_data)
        elif name == 'kgcl':
            m = KGCLBaseline(n_u, n_i, dim=64)
            m.fit(data['train_data'], products, data['item2idx'], dev=dev, epochs=args.epochs)
            r = evaluate_scores(m.score, test_data)
        else:
            logger.warning(f"unknown baseline {name}, skip"); continue
        logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}")
        results[name] = r

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"saved to {out}")
    for k, v in results.items():
        logger.info(f"  {k:10s} HR@10={v['hr@10']:.4f} NDCG@10={v['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()
