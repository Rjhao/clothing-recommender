"""基线在冷启动子集上的评估（与 evaluate_fast.py 一致协议）。

为公平对比 M3，按用户交互数分桶评估 ItemCF / UserCF / Popular。
把每用户评估结果按其交互数分桶（cold<5 / warm 5-20 / heavy>20），
也按 test item 冷热分桶（被交互 <5 用户 = 冷商品）。

Usage:
    python -m recommender.baselines_coldstart --data data/amazon_fashion_real.json \
        --baselines popular usercf itemcf --output experiments/results/baselines_coldstart.json
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k, compute_precision_at_k, compute_recall_at_k
from recommender.baselines_real import Popular, UserCF, ItemCF, evaluate_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--baselines', nargs='+', default=['popular', 'usercf', 'itemcf'])
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--usercf-topk', type=int, default=100)
    ap.add_argument('--output', default='experiments/results/baselines_coldstart.json')
    args = ap.parse_args()

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pos = [(d[0][0], d[0][1]) for d in data['train_data']]
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                                n_negative=args.n_negative, seed=args.seed)

    # 每用户训练交互数（cold/warm/heavy 分桶）
    user_train_cnt = Counter(d[0][0] for d in data['train_data'])
    # 每商品被多少用户交互（item cold/hot 分桶）
    item_user_cnt = Counter(d[0][1] for d in data['train_data'])

    def user_bucket(c):
        if c < 5: return 'cold'
        if c <= 20: return 'warm'
        return 'heavy'

    def item_bucket(c):
        if c < 5: return 'cold_item'
        if c <= 20: return 'warm_item'
        return 'hot_item'

    # 预计算每个 test 样本的桶
    test_meta = []
    for u, pos, negs in test_data:
        uc = user_train_cnt.get(u, 0)
        ic = item_user_cnt.get(pos[0], 0)
        test_meta.append((u, pos, negs, user_bucket(uc), item_bucket(ic)))

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
        else:
            continue
        logger.info(f"  {name} fit {time.time()-t0:.0f}s, evaluating...")

        # 收集逐用户结果 + 元信息
        per_user = []  # list of (user, item, hr@10, ndcg@10, user_bucket, item_bucket)
        res = {f'hr@{k}': [] for k in [5, 10, 20] for m_ in ['hr', 'ndcg']}
        # 复用 evaluate_batch 但需要逐用户结果，所以自己写一个轻量版
        K = 10
        BATCH = 512
        n_done = 0
        t0 = time.time()
        for start in range(0, len(test_data), BATCH):
            batch = test_data[start:start + BATCH]
            meta = test_meta[start:start + BATCH]
            B = len(batch)
            N = 1 + max(len(negs) for _, _, negs in batch)
            uids = np.zeros(B, dtype=np.int64)
            cand = np.full((B, N), -1, dtype=np.int64)
            for bi, (uid_idx, pos_items, neg_items) in enumerate(batch):
                uids[bi] = uid_idx
                cand[bi, 0] = pos_items[0]
                for j, ng in enumerate(neg_items):
                    cand[bi, 1 + j] = ng
            scores = m.score_batch(uids, cand)  # [B, N]
            order = np.argsort(-scores, axis=1)
            for bi in range(B):
                pos = [int(cand[bi, 0])]
                ranked = cand[bi, order[bi]].tolist()
                hr10 = compute_hr_at_k(pos, ranked, 10)
                ndcg10 = compute_ndcg_at_k(pos, ranked, 10)
                per_user.append((int(uids[bi]), int(cand[bi, 0]), hr10, ndcg10,
                                 meta[bi][3], meta[bi][4]))
            n_done += B
            if n_done % (BATCH * 20) == 0 or n_done >= len(test_data):
                logger.info(f"  {n_done}/{len(test_data)} {time.time()-t0:.0f}s")

        # 分桶聚合
        overall_hr = float(np.mean([p[2] for p in per_user]))
        overall_ndcg = float(np.mean([p[3] for p in per_user]))
        buckets = {'cold': [], 'warm': [], 'heavy': [], 'cold_item': [], 'warm_item': [], 'hot_item': []}
        for p in per_user:
            for bk in (p[4], p[5]):
                buckets[bk].append(p)
        bucket_res = {}
        for bk, ps in buckets.items():
            if not ps:
                bucket_res[bk] = {'n': 0, 'hr@10': 0, 'ndcg@10': 0}
            else:
                bucket_res[bk] = {
                    'n': len(ps),
                    'hr@10': float(np.mean([p[2] for p in ps])),
                    'ndcg@10': float(np.mean([p[3] for p in ps])),
                }
        results[name] = {
            'overall': {'hr@10': overall_hr, 'ndcg@10': overall_ndcg, 'n': len(per_user)},
            'by_user': {k: v for k, v in bucket_res.items() if k in ('cold', 'warm', 'heavy')},
            'by_item': {k: v for k, v in bucket_res.items() if k.endswith('_item')},
        }
        logger.info(f"  {name}: overall HR@10={overall_hr:.4f}  "
                    f"cold={bucket_res['cold']['hr@10']:.4f}(n={bucket_res['cold']['n']})  "
                    f"cold_item={bucket_res['cold_item']['hr@10']:.4f}(n={bucket_res['cold_item']['n']})  "
                    f"({time.time()-t0:.0f}s)")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'data': args.data, 'args': vars(args), 'results': results}, f, indent=2, ensure_ascii=False)
    logger.info(f"saved -> {out}")
    print("\n=== Baselines coldstart breakdown (HR@10) ===")
    print(f"{'method':10s} {'overall':>8s} {'cold_u':>8s} {'warm_u':>8s} {'heavy_u':>8s} {'cold_i':>8s} {'warm_i':>8s} {'hot_i':>8s}")
    for name, r in results.items():
        u = r['by_user']; i = r['by_item']
        print(f"{name:10s} {r['overall']['hr@10']:>8.4f} "
              f"{u['cold']['hr@10']:>8.4f} {u['warm']['hr@10']:>8.4f} {u['heavy']['hr@10']:>8.4f} "
              f"{i['cold_item']['hr@10']:>8.4f} {i['warm_item']['hr@10']:>8.4f} {i['hot_item']['hr@10']:>8.4f}")


if __name__ == '__main__':
    main()
