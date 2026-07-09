"""快速基线脚本:向量化 UserCF/ItemCF/Popular 的 evaluate,避免 per-user score 调用。

问题:baselines.py 用 evaluate_scores(score_fn, test_data) 逐 user 调用 score,UserCF 的 score
是 O(n_users) 的 sim @ ui,25000 用户 × 252 万 score calls = 63 亿 ops,纯 numpy 要 30+ 分钟。

本脚本:一次性计算所有 test users × all items 的分数矩阵,然后批量取 candidates 计算 HR/NDCG。
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def vectorized_eval(score_matrix, test_data, k_values=(5, 10, 20)):
    """score_matrix: [n_users, n_items], per-user 全物品分数。

    Returns dict of metric means.
    """
    results = {f'{m}@{k}': [] for m in ('hr', 'ndcg') for k in k_values}
    t0 = time.time()
    n_done = 0
    for user_idx, pos_items, neg_items in test_data:
        candidates = list(pos_items) + list(neg_items)
        scores = score_matrix[user_idx, candidates]  # [101]
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        for k in k_values:
            results[f'hr@{k}'].append(compute_hr_at_k(pos_items, ranked, k))
            results[f'ndcg@{k}'].append(compute_ndcg_at_k(pos_items, ranked, k))
        n_done += 1
        if n_done % 5000 == 0:
            logger.info(f"    eval {n_done}/{len(test_data)} ({time.time()-t0:.0f}s)")
    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--baselines', nargs='+', default=['popular', 'usercf', 'itemcf'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='experiments/results/baselines_fast.json')
    args = ap.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pairs = [(p[0], p[1]) for p, _ in data['train_data']]

    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'], 100, args.seed)
    logger.info(f"n_users={n_u} n_items={n_i} train={len(train_pairs)} test_users={len(test_data)}")

    # Build sparse ui matrix
    logger.info("Building sparse user-item matrix...")
    rows = np.array([u for u, i in train_pairs], dtype=np.int32)
    cols = np.array([i for u, i in train_pairs], dtype=np.int32)
    vals = np.ones(len(train_pairs), dtype=np.float32)
    from scipy.sparse import csr_matrix
    ui_sparse = csr_matrix((vals, (rows, cols)), shape=(n_u, n_i))
    # exclude seen items mask (1 if NOT seen)
    logger.info("Building seen mask (1 if item NOT in user's train set)...")
    # We don't want to recommend items user has already seen → set their score to -inf
    # Build dense mask on the fly per metric call (cheap since candidates is small)
    logger.info(f"  sparse nnz={ui_sparse.nnz:,}, density={ui_sparse.nnz/(n_u*n_i):.6f}")

    results = {}
    test_user_indices = np.array([u for u, _, _ in test_data])

    for name in args.baselines:
        t0 = time.time()
        logger.info(f"==== {name} ====")
        if name == 'popular':
            score_matrix = np.array(ui_sparse.sum(axis=0).flatten())  # [n_items]
            score_matrix = np.broadcast_to(score_matrix, (n_u, n_i)).copy()
            # mask seen
            score_matrix[ui_sparse.nonzero()] = -1e9
            t_fit = time.time() - t0
            logger.info(f"  {name} scored in {t_fit:.1f}s, evaluating...")
            r = vectorized_eval(score_matrix, test_data)
            logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}  ({time.time()-t0:.0f}s)")

        elif name == 'usercf':
            # Compute sim = ui @ ui.T / (||ui|| * ||ui||)
            logger.info("  computing user-user cosine sim (dense)...")
            ui_dense = ui_sparse.toarray()  # [n_u, n_i] - HEAVY but n_u=25k, n_i=34k → 3.4GB
            norm = np.linalg.norm(ui_dense, axis=1, keepdims=True)
            norm[norm == 0] = 1
            sim = (ui_dense @ ui_dense.T) / (norm @ norm.T)
            np.fill_diagonal(sim, 0)
            sim_sp = -1e9 * (sim == 0).astype(np.float32)  # mask zero-similarity users
            sim = sim + sim_sp
            del ui_dense  # free 3.4GB
            # Score: sim[all_u] @ ui → [n_u, n_items]
            # sim is dense ndarray, ui is csr → use scipy to_dense path
            # Convert to dense: 3.4GB, OK
            ui_dense_for_score = ui_sparse.toarray()  # [n_u, n_i]
            score_matrix = sim @ ui_dense_for_score  # [n_u, n_items] dense
            # mask seen items
            score_matrix[ui_dense_for_score > 0] = -1e9
            del sim, ui_dense_for_score
            t_fit = time.time() - t0
            logger.info(f"  {name} scored in {t_fit:.0f}s, evaluating...")
            r = vectorized_eval(score_matrix, test_data)
            logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}  ({time.time()-t0:.0f}s)")

        elif name == 'itemcf':
            logger.info("  computing item-item cosine sim...")
            ui_dense = ui_sparse.toarray().T  # [n_i, n_u]
            norm = np.linalg.norm(ui_dense, axis=1, keepdims=True)
            norm[norm == 0] = 1
            sim = (ui_dense @ ui_dense.T) / (norm @ norm.T)
            np.fill_diagonal(sim, 0)
            # scores[test_u, j] = sum_i sim[j, i] * ui[test_u, i] = ui_sparse[test_u] @ sim
            # Full: ui_sparse @ sim → [n_u, n_items]; csr @ dense → ndarray
            score_matrix = ui_sparse @ sim  # [n_u, n_items] ndarray
            score_matrix[(ui_sparse != 0).toarray()] = -1e9
            del sim, ui_dense
            t_fit = time.time() - t0
            logger.info(f"  {name} scored in {t_fit:.0f}s, evaluating...")
            r = vectorized_eval(score_matrix, test_data)
            logger.info(f"  {name}: HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}  ({time.time()-t0:.0f}s)")

        else:
            logger.warning(f"  unknown: {name}, skip")
            continue
        results[name] = r

    logger.info(f"\n=== TOTAL {time.time()-t_start:.0f}s ===")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'data': args.data, 'results': results}, f, indent=2)
    for k, v in results.items():
        logger.info(f"  {k:10s} HR@10={v['hr@10']:.4f} NDCG@10={v['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()
