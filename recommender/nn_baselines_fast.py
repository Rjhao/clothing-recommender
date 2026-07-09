"""向量化版 NN 基线训练+评估:避免 per-user score 慢。

适用:BPR-MF / LightGCN / RippleNet / KGCL(都用 user_emb @ item_emb.T 或类似快速 score)。
策略:复用 baselines.py 的 .fit 训练,然后 vectorized eval(全 101 候选一次矩阵乘)。
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k
from recommender.baselines import BPRMFTrainer, LightGCN, RippleNetBaseline, KGCLBaseline
from recommender.hybrid_rec import KGHybridRec  # for import path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def vectorized_eval(score_matrix, test_data, k=10):
    """score_matrix: [n_u, n_i]. Mask seen in test_data (test item is fine to leave in)."""
    results = []
    for u_idx, pos_items, neg_items in test_data:
        candidates = list(pos_items) + list(neg_items)
        scores = score_matrix[u_idx, candidates]
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        results.append({
            'u': u_idx,
            'hr10': compute_hr_at_k(pos_items, ranked, k),
            'ndcg10': compute_ndcg_at_k(pos_items, ranked, k),
        })
    return results


def mask_train_seen(score_matrix, train_pairs, n_u, n_i):
    """把 train 集里见过的商品 score 设为 -1e9(不推荐用户已交互的)。"""
    seen_mask = np.zeros((n_u, n_i), dtype=bool)
    for u, i in train_pairs:
        seen_mask[u, i] = True
    score_matrix[seen_mask] = -1e9
    return score_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--baselines', nargs='+', default=['bprmf', 'lightgcn', 'ripplenet', 'kgcl'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    user2idx, item2idx = data['user2idx'], data['item2idx']
    train_pairs = [(p[0], p[1]) for p, _ in data['train_data']]

    test_data = build_loo_test(interactions, user2idx, item2idx, 100, args.seed)
    logger.info(f"n_users={n_u} n_items={n_i} train={len(train_pairs)} test_users={len(test_data)}")

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    for name in args.baselines:
        t1 = time.time()
        logger.info(f"==== {name} ====")
        if name == 'bprmf':
            m = BPRMFTrainer(n_u, n_i, dim=64).to(dev)
            # train_data[i] = ((u, pos), negs_list); 1:1 neg sampling 取 negs_list[0]
            train_neg = [(u, negs[0]) for (u, _), negs in data['train_data']]
            m.fit(train_pairs, train_neg, epochs=args.epochs, batch=256, lr=0.001, dev=dev)
            with torch.no_grad():
                score_matrix = (m.user_emb.weight @ m.item_emb.weight.T).cpu().numpy()
            score_matrix = mask_train_seen(score_matrix, train_pairs, n_u, n_i)

        elif name == 'lightgcn':
            m = LightGCN(n_u, n_i, dim=64, n_layers=3).to(dev)
            train_neg = [(u, negs[0]) for (u, _), negs in data['train_data']]
            m.fit(train_pairs, train_neg, dev=dev, epochs=args.epochs, batch=256, lr=0.001)
            with torch.no_grad():
                e, n_u_emb = m._propagate()
                # 正确:user_emb = e[:n_u], item_emb = e[n_u:]; score = user @ item.T
                user_e = e[:n_u]
                item_e = e[n_u:]
                score_matrix = (user_e @ item_e.T).cpu().numpy()
            score_matrix = mask_train_seen(score_matrix, train_pairs, n_u, n_i)

        elif name == 'ripplenet':
            m = RippleNetBaseline(n_u, n_i, dim=64, n_hops=2)
            train_neg = [(u, negs[0]) for (u, _), negs in data['train_data']]
            m.fit(train_pairs, train_neg, products, item2idx, dev=dev, epochs=args.epochs, batch=256, lr=0.001)
            m.to(dev)
            with torch.no_grad():
                all_items = torch.arange(n_i, device=dev)
                item_r = m._item_repr(all_items)  # [n_i, d]
                user_e = m.user_emb.weight  # [n_u, d]
                score_matrix = (user_e @ item_r.T).cpu().numpy()
            score_matrix = mask_train_seen(score_matrix, train_pairs, n_u, n_i)

        elif name == 'kgcl':
            m = KGCLBaseline(n_u, n_i, dim=64)
            m.fit(data['train_data'], products, item2idx, dev=dev, epochs=args.epochs, batch=256, lr=0.001)
            m.to(dev)
            with torch.no_grad():
                # KGCL score: user_emb @ item_emb.T (the simple BPR-MF score, no GAT fusion in predict)
                score_matrix = (m.user_emb.weight @ m.item_emb.weight.T).cpu().numpy()
            score_matrix = mask_train_seen(score_matrix, train_pairs, n_u, n_i)

        else:
            logger.warning(f"unknown: {name}"); continue

        # evaluate
        per_user = vectorized_eval(score_matrix, test_data)
        all_hr = np.mean([r['hr10'] for r in per_user])
        all_ndcg = np.mean([r['ndcg10'] for r in per_user])
        results[name] = {'hr@10': float(all_hr), 'ndcg@10': float(all_ndcg)}
        logger.info(f"  {name}: HR@10={all_hr:.4f}  NDCG@10={all_ndcg:.4f}  ({time.time()-t1:.0f}s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'data': args.data, 'results': results}, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved -> {out_path}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
