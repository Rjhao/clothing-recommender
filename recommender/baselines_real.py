"""真实数据规模下的基线（sparse + 批量评估，64k 用户 / 21k 商品）。

原 baselines.py 在 64k² dense sim 矩阵下 OOM，本模块用 scipy.sparse +
top-K 邻居截断 + 分批评估。

基线（与原版口径一致）:
    popular   流行度
    usercf    User-based CF, sparse + top-K 邻居
    itemcf    Item-based CF, sparse item-item 共现
    bprmf     BPR-MF (GPU)
    lightgcn  LightGCN (GPU, sparse 邻接)

评估：sampled-100 leave-one-out（同 evaluate.py 协议），但按用户分批
计算候选分数，避免逐用户小 forward 的开销（与 evaluate_fast 一致）。

Usage:
    python -m recommender.baselines_real --data data/amazon_fashion_real.json \
        --baselines popular usercf itemcf bprmf lightgcn --epochs 30
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
    """score_fn(user_idxs[B], item_idxs[B, N]) -> scores[B, N]。"""
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
        # = (ui[u] @ sim.T)[i]  ->  [B, n_i] = ui[uids] @ sim.T
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
class BPRMF:
    def __init__(self, n_users, n_items, dim=64):
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
        ineg = torch.LongTensor([p[1] for p in train_neg]).to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(train_pos)
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
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  BPR-MF ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        uids_t = torch.LongTensor(uids)
        cand_t = torch.LongTensor(cand)
        dev = self.user_emb.weight.device
        u_e = self.user_emb.weight[uids_t.to(dev)]  # [B, d]
        i_e = self.item_emb.weight[cand_t.to(dev)]  # [B, N, d]
        scores = (u_e.unsqueeze(1) * i_e).sum(-1)  # [B, N]
        return scores.cpu().numpy()


# ---------------- LightGCN (GPU, sparse adj) ----------------
class LightGCN:
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        self.dim, self.n_layers = dim, n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
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
        ineg = torch.LongTensor([p[1] for p in train_neg]).to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(train_pos)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                e, n_u = self._propagate()
                pos = (e[u[idx]] * e[ip[idx] + n_u]).sum(-1)
                neg = (e[u[idx]] * e[ineg[idx] + n_u]).sum(-1)
                loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                loss = loss + 1e-4 * (self.user_emb.weight.pow(2).sum() + self.item_emb.weight.pow(2).sum())
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"  LightGCN ep {ep+1}/{epochs} loss={tot/n:.4f}")

    def score_batch(self, uids, cand):
        uids_t = torch.LongTensor(uids)
        cand_t = torch.LongTensor(cand)
        dev = self.user_emb.weight.device
        e, n_u = self._propagate()
        u_e = e[uids_t.to(dev)]  # [B, d]
        cand_abs = cand_t.to(dev) + n_u  # [B, N]
        i_e = e[cand_abs]  # [B, N, d]
        scores = (u_e.unsqueeze(1) * i_e).sum(-1)  # [B, N]
        return scores.cpu().numpy()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--baselines', nargs='+', default=['popular', 'usercf', 'itemcf', 'bprmf', 'lightgcn'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--usercf-topk', type=int, default=200)
    ap.add_argument('--output', default='experiments/results/baselines_real.json')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pos = [(p[0], p[1]) for p, _ in data['train_data']]
    train_neg = [(n[0], n[1]) for _, n in data['train_data']]
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                                n_negative=args.n_negative, seed=args.seed)
    logger.info(f"train={len(train_pos)} test={len(test_data)} dev={dev} n_u={n_u} n_i={n_i}")

    results = {}
    for name in args.baselines:
        logger.info(f"==== {name} ====")
        t0 = time.time()
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
        else:
            logger.warning(f"unknown {name}, skip"); continue
        logger.info(f"  {name} trained in {time.time()-t0:.0f}s, evaluating...")
        r = evaluate_batch(m.score_batch, test_data, [5, 10, 20])
        logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}  ({time.time()-t0:.0f}s)")
        results[name] = r

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'data': args.data, 'args': vars(args), 'results': results}, f, indent=2, ensure_ascii=False)
    logger.info(f"saved -> {out}")
    print("\n=== Baselines on real data ===")
    for k, v in results.items():
        print(f"  {k:10s} HR@10={v['hr@10']:.4f}  NDCG@10={v['ndcg@10']:.4f}  HR@20={v['hr@20']:.4f}")


if __name__ == '__main__':
    main()
