"""冷启动子集评估:在 view_sparse 上跑 M3 / ItemCF / Popular 的 cold/warm/heavy user + cold/warm/hot item 分层 HR@10。

复用 baselines_fast 的向量化 score 计算 + 同一 test_data 切分,保证可比。
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
from recommender.hybrid_rec import KGHybridRec
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def vectorized_eval_per_user(score_matrix, test_data):
    """返回每个 test user 的 (hr@10, ndcg@10),用于冷启动子集分析。"""
    results = []
    for user_idx, pos_items, neg_items in test_data:
        candidates = list(pos_items) + list(neg_items)
        scores = score_matrix[user_idx, candidates]
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        results.append({
            'u': user_idx,
            'hr10': compute_hr_at_k(pos_items, ranked, 10),
            'ndcg10': compute_ndcg_at_k(pos_items, ranked, 10),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--methods', nargs='+', default=['popular', 'usercf', 'itemcf', 'm3'])
    ap.add_argument('--m3-checkpoint', default='checkpoints/view_sparse_m3_5ep/best_model.pt')
    ap.add_argument('--user-thresholds', type=float, nargs='+', default=[5, 20],
                    help='cold/warm/heavy 用户交互数分位阈值(默认 <5 cold, 5-20 warm, >20 heavy)')
    ap.add_argument('--item-thresholds', type=float, nargs='+', default=[10, 100],
                    help='cold/warm/hot 商品被交互数分位阈值')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='experiments/results/coldstart_view_sparse.json')
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    user2idx = data['user2idx']
    item2idx = data['item2idx']
    train_pairs = [(p[0], p[1]) for p, _ in data['train_data']]

    test_data = build_loo_test(interactions, user2idx, item2idx, 100, args.seed)
    logger.info(f"n_users={n_u} n_items={n_i} test_users={len(test_data)}")

    # user_count: 每个用户的 train 交互数(用于分层)
    user_count = np.zeros(n_u, dtype=np.int32)
    for u, i in train_pairs:
        user_count[u] += 1
    # item_count: 每个商品的 train 被交互数
    item_count = np.zeros(n_i, dtype=np.int32)
    for u, i in train_pairs:
        item_count[i] += 1

    # 分层函数
    def user_layer(c):
        if c < args.user_thresholds[0]:
            return 'cold'
        elif c < args.user_thresholds[1]:
            return 'warm'
        else:
            return 'heavy'

    def item_layer(c):
        if c < args.item_thresholds[0]:
            return 'cold_item'
        elif c < args.item_thresholds[1]:
            return 'warm_item'
        else:
            return 'hot_item'

    # 给每个 test user 打标签
    test_user_layers = {}
    for u, pos, neg in test_data:
        test_user_layers[u] = user_layer(user_count[u])

    layer_counts = {'cold': 0, 'warm': 0, 'heavy': 0}
    for lyr in test_user_layers.values():
        layer_counts[lyr] += 1
    logger.info(f"user layers: {layer_counts}")

    # Build sparse ui
    from scipy.sparse import csr_matrix
    rows = np.array([u for u, i in train_pairs], dtype=np.int32)
    cols = np.array([i for u, i in train_pairs], dtype=np.int32)
    vals = np.ones(len(train_pairs), dtype=np.float32)
    ui_sparse = csr_matrix((vals, (rows, cols)), shape=(n_u, n_i))

    final_results = {}

    for method in args.methods:
        t_method = time.time()
        logger.info(f"==== {method} ====")
        if method == 'popular':
            score = np.array(ui_sparse.sum(axis=0).flatten())
            score_matrix = np.broadcast_to(score, (n_u, n_i)).copy()
            score_matrix[ui_sparse.nonzero()] = -1e9
        elif method == 'usercf':
            ui_dense = ui_sparse.toarray()
            norm = np.linalg.norm(ui_dense, axis=1, keepdims=True)
            norm[norm == 0] = 1
            sim = (ui_dense @ ui_dense.T) / (norm @ norm.T)
            np.fill_diagonal(sim, 0)
            sim = sim + (-1e9) * (sim == 0).astype(np.float32)
            del ui_dense
            score_matrix = sim @ ui_sparse.toarray()
            score_matrix[ui_sparse.toarray() > 0] = -1e9
            del sim
        elif method == 'itemcf':
            ui_dense_t = ui_sparse.toarray().T
            norm = np.linalg.norm(ui_dense_t, axis=1, keepdims=True)
            norm[norm == 0] = 1
            sim = (ui_dense_t @ ui_dense_t.T) / (norm @ norm.T)
            np.fill_diagonal(sim, 0)
            score_matrix = ui_sparse @ sim
            score_matrix[ui_sparse.toarray() > 0] = -1e9
            del sim, ui_dense_t
        elif method == 'm3':
            logger.info(f"  loading M3 from {args.m3_checkpoint}")
            # Build graph + model (need products + train_data + user/item2idx)
            from recommender.kgat.hetero_graph import build_graph_for_model, build_kg_triplets
            graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
            kg_triplets_all = build_kg_triplets(products, item2idx, graph['node_offsets'])
            user_count_array = torch.zeros(n_u, dtype=torch.float32)
            for uid, cnt in data['user_count'].items():
                user_count_array[uid] = cnt
            n_entities = graph['n_total']
            n_relations = graph['n_relations']
            model = KGHybridRec(
                n_users=n_u, n_items=n_i, n_entities=n_entities, n_relations=n_relations,
                embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
                use_multimodal=False, use_improvement_a=True, use_improvement_b=True, use_improvement_c=True,
                lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1,
                graph=graph, n_interactions=user_count_array,
            ).to('cuda')
            model.load_state_dict(torch.load(args.m3_checkpoint, map_location='cuda', weights_only=True))
            model.eval()
            # Vectorized score: predict_score for all (u, candidates) at once
            score_matrix = np.full((n_u, n_i), -1e9, dtype=np.float32)
            batch = 1024
            with torch.no_grad():
                for u_start in range(0, n_u, batch):
                    u_end = min(u_start + batch, n_u)
                    uids = torch.arange(u_start, u_end, device='cuda')
                    iids = torch.arange(n_i, device='cuda')
                    uu, ii = torch.meshgrid(uids, iids, indexing='ij')
                    uu = uu.flatten(); ii = ii.flatten()
                    counts = torch.LongTensor([user_count[u] for u in range(u_start, u_end)]).to('cuda')
                    counts_rep = counts.repeat_interleave(n_i)
                    scores = model.predict_score(uu, ii, counts_rep).cpu().numpy()
                    score_matrix[u_start:u_end] = scores.reshape(-1, n_i)
                    if (u_start // batch) % 5 == 0:
                        logger.info(f"    score {u_end}/{n_u}")
            # mask seen
            score_matrix[ui_sparse.toarray() > 0] = -1e9
        else:
            logger.warning(f"unknown method {method}, skip")
            continue

        logger.info(f"  scored in {time.time()-t_method:.0f}s, evaluating...")
        per_user = vectorized_eval_per_user(score_matrix, test_data)

        # 按 user_layer 聚合
        by_user = {'cold': [], 'warm': [], 'heavy': []}
        for r in per_user:
            lyr = test_user_layers[r['u']]
            by_user[lyr].append(r)

        summary = {'overall': {}, 'by_user': {}, 'by_item': {}}
        all_hr = [r['hr10'] for r in per_user]
        all_ndcg = [r['ndcg10'] for r in per_user]
        summary['overall'] = {
            'n': len(per_user),
            'hr@10': float(np.mean(all_hr)),
            'ndcg@10': float(np.mean(all_ndcg)),
        }
        for lyr in ('cold', 'warm', 'heavy'):
            arr = by_user[lyr]
            if not arr: continue
            summary['by_user'][lyr] = {
                'n': len(arr),
                'hr@10': float(np.mean([r['hr10'] for r in arr])),
                'ndcg@10': float(np.mean([r['ndcg10'] for r in arr])),
            }

        # 按 test item 的 item_layer 聚合
        # 每个 test user 的 pos_item 分到 item layer
        # (item_count of test item 看被多少 train 用户交互)
        pos_item_layers = {'cold_item': 0, 'warm_item': 0, 'hot_item': 0}
        per_item_layer = {k: [] for k in pos_item_layers}
        for u, pos_items, neg_items in test_data:
            pos = pos_items[0] if pos_items else None
            if pos is None: continue
            lyr = item_layer(item_count[pos])
            per_item_layer[lyr].append({
                'hr10': next((r['hr10'] for r in per_user if r['u'] == u), 0.0),
                'ndcg10': next((r['ndcg10'] for r in per_user if r['u'] == u), 0.0),
            })
        for lyr in ('cold_item', 'warm_item', 'hot_item'):
            arr = per_item_layer[lyr]
            if not arr: continue
            summary['by_item'][lyr] = {
                'n': len(arr),
                'hr@10': float(np.mean([r['hr10'] for r in arr])),
                'ndcg@10': float(np.mean([r['ndcg10'] for r in arr])),
            }

        logger.info(f"  {method}: overall HR@10={summary['overall']['hr@10']:.4f}  ({time.time()-t_method:.0f}s)")
        for lyr in ('cold','warm','heavy'):
            if lyr in summary['by_user']:
                d = summary['by_user'][lyr]
                logger.info(f"    {lyr:5s} n={d['n']:6d}  HR@10={d['hr@10']:.4f}")
        for lyr in ('cold_item','warm_item','hot_item'):
            if lyr in summary['by_item']:
                d = summary['by_item'][lyr]
                logger.info(f"    {lyr:9s} n={d['n']:6d}  HR@10={d['hr@10']:.4f}")
        final_results[method] = summary

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'data': args.data, 'results': final_results}, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved -> {out}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
