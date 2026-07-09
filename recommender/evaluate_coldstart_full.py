"""M3 完整分桶评估（用户 cold/warm/heavy + 商品 cold/warm/hot）。

补 evaluate_fast 缺失的 item 冷热分桶，使 M3 与 baselines_coldstart 同协议可比。
重新跑一次评估，逐用户记录结果 + 该用户/商品的桶。

Usage:
    python -m recommender.evaluate_coldstart_full \
        --model checkpoints/m3_real/best_model.pt \
        --data data/amazon_fashion_real.json --tag m3_real
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.hybrid_rec import KGHybridRec
from recommender.train import load_data, prepare_training_data
from recommender.kgat.hetero_graph import build_graph_for_model
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-users', type=int, default=512)
    ap.add_argument('--use-a', type=int, default=1)
    ap.add_argument('--use-b', type=int, default=1)
    ap.add_argument('--use-c', type=int, default=1)
    ap.add_argument('--use-multimodal', type=int, default=0)
    ap.add_argument('--embedding-dim', type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()
    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    user2idx, item2idx = data['user2idx'], data['item2idx']
    n_users, n_items = data['n_users'], data['n_items']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    n_entities, n_relations = graph['n_total'], graph['n_relations']
    user_count_array = torch.zeros(n_users, dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=n_entities, n_relations=n_relations,
        embedding_dim=args.embedding_dim, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=bool(args.use_multimodal),
        use_improvement_a=bool(args.use_a), use_improvement_b=bool(args.use_b),
        use_improvement_c=bool(args.use_c),
        graph=graph, n_interactions=user_count_array,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    logger.info(f"Loaded {args.model}")

    test_data = build_loo_test(interactions, user2idx, item2idx,
                               n_negative=args.n_negative, seed=args.seed)
    logger.info(f"Test users: {len(test_data)}")

    # 桶元信息
    user_train_cnt = Counter(d[0][0] for d in data['train_data'])
    item_user_cnt = Counter(d[0][1] for d in data['train_data'])
    def ub(c): return 'cold' if c < 5 else ('warm' if c <= 20 else 'heavy')
    def ib(c): return 'cold_item' if c < 5 else ('warm_item' if c <= 20 else 'hot_item')

    # 批量评估，逐用户记录
    per_user = []  # (uid, pos_item, hr10, ndcg10, ubucket, ibucket)
    N = 1 + args.n_negative
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_data), args.batch_users):
            batch = test_data[start:start + args.batch_users]
            B = len(batch)
            cand = torch.full((B, N), 0, dtype=torch.long)
            mask = torch.ones(B, N, dtype=torch.bool)
            uids = torch.zeros(B, dtype=torch.long)
            cnts = torch.zeros(B, dtype=torch.long)
            metas = []
            for bi, (uid_idx, pos_items, neg_items) in enumerate(batch):
                uids[bi] = uid_idx
                cnts[bi] = data['user_count'].get(uid_idx, 1)
                cand[bi, 0] = pos_items[0]
                for j, ng in enumerate(neg_items):
                    cand[bi, 1 + j] = ng
                if len(neg_items) < args.n_negative:
                    mask[bi, 1 + len(neg_items):] = False
                uc = user_train_cnt.get(uid_idx, 0)
                ic = item_user_cnt.get(pos_items[0], 0)
                metas.append((ub(uc), ib(ic)))
            uids, cnts, cand, mask = (uids.to(device), cnts.to(device),
                                       cand.to(device), mask.to(device))
            fu = uids.unsqueeze(1).expand(-1, N).reshape(-1)
            fi = cand.reshape(-1)
            fc = cnts.unsqueeze(1).expand(-1, N).reshape(-1)
            scores = model.predict_score(fu, fi, interaction_counts=fc).view(B, N)
            scores = scores.masked_fill(~mask, float('-inf'))
            order = torch.argsort(scores, descending=True, dim=1).cpu().tolist()
            cand_cpu = cand.cpu().tolist()
            for bi in range(B):
                pos = [cand_cpu[bi][0]]
                ranked = [cand_cpu[bi][i] for i in order[bi] if mask[bi][i].item()]
                per_user.append((int(uids[bi].item()), int(cand_cpu[bi][0]),
                                 compute_hr_at_k(pos, ranked, 10),
                                 compute_ndcg_at_k(pos, ranked, 10),
                                 metas[bi][0], metas[bi][1]))
            if (start // args.batch_users) % 20 == 0:
                logger.info(f"  {start+B}/{len(test_data)} {time.time()-t0:.0f}s")

    # 聚合
    def agg(ps):
        if not ps: return {'n': 0, 'hr@10': 0, 'ndcg@10': 0}
        return {'n': len(ps), 'hr@10': float(np.mean([p[2] for p in ps])),
                'ndcg@10': float(np.mean([p[3] for p in ps]))}
    overall = agg(per_user)
    by_user = {bk: agg([p for p in per_user if p[4] == bk]) for bk in ('cold', 'warm', 'heavy')}
    by_item = {bk: agg([p for p in per_user if p[5] == bk]) for bk in ('cold_item', 'warm_item', 'hot_item')}

    out = {
        'model': args.model, 'data': args.data, 'tag': args.tag or Path(args.model).parent.name,
        'protocol': f'sampled-{args.n_negative} leave-one-out',
        'n_test_users': len(test_data),
        'overall': overall, 'by_user': by_user, 'by_item': by_item,
    }
    out_path = Path('experiments/results') / f'eval_{out["tag"]}_coldstart_full.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("\n=== M3 coldstart full breakdown (HR@10) ===")
    logger.info(f"overall: {overall['hr@10']:.4f} (n={overall['n']})")
    for bk in ('cold', 'warm', 'heavy'):
        b = by_user[bk]; logger.info(f"  [{bk}_user] n={b['n']} hr@10={b['hr@10']:.4f} ndcg@10={b['ndcg@10']:.4f}")
    for bk in ('cold_item', 'warm_item', 'hot_item'):
        b = by_item[bk]; logger.info(f"  [{bk}] n={b['n']} hr@10={b['hr@10']:.4f} ndcg@10={b['ndcg@10']:.4f}")
    logger.info(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
