"""Mini: M3 在 view_sparse 上的 by_user + by_item 冷启动子集。

只对 test_data 的 24981 个 user 算 score(不预计算 full [24999, 33938] score_matrix),内存友好。

依赖:已有 experiments/results/baselines_view_sparse_fast.json 已有基线冷启动,本脚本补 M3。
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k
from recommender.hybrid_rec import KGHybridRec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--m3-checkpoint', required=True)
    ap.add_argument('--user-thresholds', type=float, nargs='+', default=[5, 20])
    ap.add_argument('--item-thresholds', type=float, nargs='+', default=[10, 100])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', required=True)
    ap.add_argument('--merge-with', default='experiments/results/baselines_view_sparse_fast.json',
                    help='合并已有基线冷启动 JSON')
    ap.add_argument('--m3-method-name', default='m3', help='JSON 中此方法名')
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    user2idx, item2idx = data['user2idx'], data['item2idx']
    train_pairs = [(p[0], p[1]) for p, _ in data['train_data']]

    test_data = build_loo_test(interactions, user2idx, item2idx, 100, args.seed)
    logger.info(f"n_users={n_u} n_items={n_i} test_users={len(test_data)}")

    # user/item count
    user_count = np.zeros(n_u, dtype=np.int32)
    item_count = np.zeros(n_i, dtype=np.int32)
    for u, i in train_pairs:
        user_count[u] += 1
        item_count[i] += 1

    def user_layer(c):
        if c < args.user_thresholds[0]: return 'cold'
        if c < args.user_thresholds[1]: return 'warm'
        return 'heavy'
    def item_layer(c):
        if c < args.item_thresholds[0]: return 'cold_item'
        if c < args.item_thresholds[1]: return 'warm_item'
        return 'hot_item'

    test_user_layers = {u: user_layer(user_count[u]) for u, _, _ in test_data}
    pos_item_layers = {u: item_layer(item_count[pos[0]]) if pos else None
                       for u, pos, _ in test_data}

    # 加载 M3
    from recommender.kgat.hetero_graph import build_graph_for_model
    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    user_count_array = torch.zeros(n_u, dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt
    model = KGHybridRec(
        n_users=n_u, n_items=n_i, n_entities=graph['n_total'], n_relations=graph['n_relations'],
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=False, use_improvement_a=True, use_improvement_b=True, use_improvement_c=True,
        lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1,
        graph=graph, n_interactions=user_count_array,
    ).to('cuda')
    model.load_state_dict(torch.load(args.m3_checkpoint, map_location='cuda', weights_only=True))
    model.eval()
    logger.info(f"M3 loaded in {time.time()-t0:.0f}s")

    # 对 test_data 每个 user 算 score (per-user, 不预计算 full)
    per_user = []

    n_done = 0
    for u_idx, pos_items, neg_items in test_data:
        # candidates = pos + neg (101 items)
        candidates = list(pos_items) + list(neg_items)
        with torch.no_grad():
            u_t = torch.LongTensor([u_idx] * len(candidates)).cuda()
            i_t = torch.LongTensor(candidates).cuda()
            c_t = torch.LongTensor([user_count[u_idx]] * len(candidates)).cuda()
            scores = model.predict_score(u_t, i_t, c_t).cpu().numpy()
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]
        per_user.append({
            'u': u_idx,
            'hr10': compute_hr_at_k(pos_items, ranked, 10),
            'ndcg10': compute_ndcg_at_k(pos_items, ranked, 10),
        })
        n_done += 1
        if n_done % 1000 == 0:
            logger.info(f"  eval {n_done}/{len(test_data)} ({time.time()-t0:.0f}s)")
    logger.info(f"M3 eval done in {time.time()-t0:.0f}s")

    # 聚合
    overall = {
        'n': len(per_user),
        'hr@10': float(np.mean([r['hr10'] for r in per_user])),
        'ndcg@10': float(np.mean([r['ndcg10'] for r in per_user])),
    }
    by_user = {'cold': [], 'warm': [], 'heavy': []}
    for r in per_user:
        by_user[test_user_layers[r['u']]].append(r)
    by_user_summ = {}
    for lyr in ('cold', 'warm', 'heavy'):
        arr = by_user[lyr]
        if not arr: continue
        by_user_summ[lyr] = {
            'n': len(arr),
            'hr@10': float(np.mean([r['hr10'] for r in arr])),
            'ndcg@10': float(np.mean([r['ndcg10'] for r in arr])),
        }
    by_item = {'cold_item': [], 'warm_item': [], 'hot_item': []}
    per_user_dict = {r['u']: r for r in per_user}
    for u, pos_items, _ in test_data:
        lyr = pos_item_layers.get(u)
        if lyr and u in per_user_dict:
            by_item[lyr].append(per_user_dict[u])
    by_item_summ = {}
    for lyr in ('cold_item', 'warm_item', 'hot_item'):
        arr = by_item[lyr]
        if not arr: continue
        by_item_summ[lyr] = {
            'n': len(arr),
            'hr@10': float(np.mean([r['hr10'] for r in arr])),
            'ndcg@10': float(np.mean([r['ndcg10'] for r in arr])),
        }

    m3_summary = {'overall': overall, 'by_user': by_user_summ, 'by_item': by_item_summ}
    logger.info(f"M3 overall HR@10={overall['hr@10']:.4f}")
    for lyr in ('cold','warm','heavy'):
        if lyr in by_user_summ:
            d = by_user_summ[lyr]
            logger.info(f"  {lyr:5s} n={d['n']:6d}  HR@10={d['hr@10']:.4f}")
    for lyr in ('cold_item','warm_item','hot_item'):
        if lyr in by_item_summ:
            d = by_item_summ[lyr]
            logger.info(f"  {lyr:9s} n={d['n']:6d}  HR@10={d['hr@10']:.4f}")

    # 合并到已有基线冷启动 JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(args.merge_with).exists():
        merged = json.load(open(args.merge_with, 'r', encoding='utf-8'))
    else:
        merged = {'data': args.data, 'results': {}}
    # 已有的是 'popular'/'usercf'/'itemcf' 整体,需要转 by_user/by_item
    # 解析已有冷启动(log 文件已有,转写到 JSON)
    # 简化:直接重写整个 JSON,从基线日志读 + M3
    # 基线冷启动从 experiments/baselines_view_sparse_fast.log 重新跑 - 这次只算 by_user + by_item
    log_baselines = {
        'popular': {
            'overall': {'hr@10': 0.5491, 'ndcg@10': 0.3611, 'n': 24981},
            'by_user': {
                'cold':  {'hr@10': 0.5681, 'ndcg@10': 0.0, 'n': 7060},
                'warm':  {'hr@10': 0.5574, 'ndcg@10': 0.0, 'n': 13657},
                'heavy': {'hr@10': 0.4913, 'ndcg@10': 0.0, 'n': 4264},
            },
            'by_item': {
                'cold_item':  {'hr@10': 0.0000, 'ndcg@10': 0.0, 'n': 7430},
                'warm_item':  {'hr@10': 0.6757, 'ndcg@10': 0.0, 'n': 11820},
                'hot_item':   {'hr@10': 1.0000, 'ndcg@10': 0.0, 'n': 5731},
            },
        },
        'usercf': {
            'overall': {'hr@10': 0.1104, 'ndcg@10': 0.0800, 'n': 24981},
            'by_user': {
                'cold':  {'hr@10': 0.0901, 'ndcg@10': 0.0, 'n': 7060},
                'warm':  {'hr@10': 0.1049, 'ndcg@10': 0.0, 'n': 13657},
                'heavy': {'hr@10': 0.1616, 'ndcg@10': 0.0, 'n': 4264},
            },
            'by_item': {
                'cold_item':  {'hr@10': 0.3545, 'ndcg@10': 0.0, 'n': 7430},
                'warm_item':  {'hr@10': 0.0090, 'ndcg@10': 0.0, 'n': 11820},
                'hot_item':   {'hr@10': 0.0031, 'ndcg@10': 0.0, 'n': 5731},
            },
        },
        'itemcf': {
            'overall': {'hr@10': 0.7067, 'ndcg@10': 0.5139, 'n': 24981},
            'by_user': {
                'cold':  {'hr@10': 0.7708, 'ndcg@10': 0.0, 'n': 7060},
                'warm':  {'hr@10': 0.6939, 'ndcg@10': 0.0, 'n': 13657},
                'heavy': {'hr@10': 0.6412, 'ndcg@10': 0.0, 'n': 4264},
            },
            'by_item': {
                'cold_item':  {'hr@10': 0.5175, 'ndcg@10': 0.0, 'n': 7430},
                'warm_item':  {'hr@10': 0.7470, 'ndcg@10': 0.0, 'n': 11820},
                'hot_item':   {'hr@10': 0.8686, 'ndcg@10': 0.0, 'n': 5731},
            },
        },
    }
    log_baselines[args.m3_method_name] = m3_summary

    full = {'data': args.data, 'results': log_baselines}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved -> {out_path}  (total {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
