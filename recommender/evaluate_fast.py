"""批量 sampled-100 评估（向量化的 evaluate_model）。

原生 evaluate.py 逐用户调用 predict_score（64097 次小 forward），在真实数据上
极慢。本脚本把所有 test 用户 × 101 候选拼成大张量，分批过模型，快几十倍。

复用 evaluate.py 的 build_loo_test 生成 test_data（保证与训练协议一致），
复用其指标函数保证数值一致。

Usage:
    python -m recommender.evaluate_fast --model checkpoints/probe_real/best_model.pt \
        --data data/amazon_fashion_real.json --tag probe_real
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec
from recommender.train import load_data, prepare_training_data
from recommender.kgat.hetero_graph import build_graph_for_model
from recommender.evaluate import (
    build_loo_test, compute_hr_at_k, compute_ndcg_at_k,
    compute_precision_at_k, compute_recall_at_k,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--tag', type=str, default=None, help='output subdir name (defaults to model parent)')
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--k', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--batch-users', type=int, default=512, help='users per forward')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use-a', type=int, default=1)
    ap.add_argument('--use-b', type=int, default=1)
    ap.add_argument('--use-c', type=int, default=1)
    ap.add_argument('--use-multimodal', type=int, default=0)
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--subset', type=int, default=None, help='只评前 N 个用户(探针用)')
    ap.add_argument('--coldstart', action='store_true', help='同时算冷启动子集')
    return ap.parse_args()


def evaluate_fast(model, test_data, k_values, device, user_count, batch_users=512, subset=None):
    """批量评估：每 batch_users 个用户 × (1+N) 候选一次 forward。"""
    model.eval()
    results = {f'hr@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    results.update({f'precision@{k}': [] for k in k_values})
    results.update({f'recall@{k}': [] for k in k_values})

    # 为冷启动分桶：记录每用户交互数
    per_user_counts = []

    if subset:
        test_data = test_data[:subset]
        logger.info(f"subset -> first {len(test_data)} test users")

    n_neg_max = max(len(negs) for _, _, negs in test_data) if test_data else 0
    # 不同用户负样本数可能不同（候选不足时），统一 pad 到 max，用 mask
    N = n_neg_max  # 通常 = n_negative
    logger.info(f"Evaluating {len(test_data)} users, {N} negatives each, batch_users={batch_users}")

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for start in range(0, len(test_data), batch_users):
            batch = test_data[start:start + batch_users]
            B = len(batch)
            # 构造 [B, 1+N]：第 0 列正样本，其余负样本（不足的用 0 填充 + mask）
            cand = torch.zeros(B, 1 + N, dtype=torch.long)
            mask = torch.ones(B, 1 + N, dtype=torch.bool)  # True=有效
            uids = torch.zeros(B, dtype=torch.long)
            cnts = torch.zeros(B, dtype=torch.long)
            for bi, (uid_idx, pos_items, neg_items) in enumerate(batch):
                uids[bi] = uid_idx
                cnts[bi] = user_count.get(uid_idx, 1)
                cand[bi, 0] = pos_items[0]
                for j, ng in enumerate(neg_items):
                    cand[bi, 1 + j] = ng
                # 没填满的负样本位置标无效
                if len(neg_items) < N:
                    mask[bi, 1 + len(neg_items):] = False

            uids = uids.to(device)
            cnts = cnts.to(device)
            cand = cand.to(device)
            mask = mask.to(device)

            # 展平过模型
            flat_user = uids.unsqueeze(1).expand(-1, 1 + N).reshape(-1)
            flat_item = cand.reshape(-1)
            flat_cnt = cnts.unsqueeze(1).expand(-1, 1 + N).reshape(-1)
            scores = model.predict_score(flat_user, flat_item, interaction_counts=flat_cnt)
            scores = scores.view(B, 1 + N)
            # 无效候选分数置 -inf
            scores = scores.masked_fill(~mask, float('-inf'))

            # 排序
            order = torch.argsort(scores, descending=True, dim=1).cpu().tolist()
            cand_cpu = cand.cpu().tolist()

            for bi in range(B):
                pos = [cand_cpu[bi][0]]
                ranked = [cand_cpu[bi][i] for i in order[bi] if mask[bi][i]]
                for k in k_values:
                    results[f'hr@{k}'].append(compute_hr_at_k(pos, ranked, k))
                    results[f'ndcg@{k}'].append(compute_ndcg_at_k(pos, ranked, k))
                    results[f'precision@{k}'].append(compute_precision_at_k(pos, ranked, k))
                    results[f'recall@{k}'].append(compute_recall_at_k(pos, ranked, k))
                per_user_counts.append(int(cnts[bi].item()))

            n_done += B
            if n_done % (batch_users * 20) == 0 or n_done >= len(test_data):
                logger.info(f"  {n_done}/{len(test_data)} users, {time.time()-t0:.0f}s")

    overall = {k: float(np.mean(v)) for k, v in results.items()}
    return overall, per_user_counts


def coldstart_breakdown(per_user_counts, results_lists, k_values):
    """按用户交互数分桶：cold(<5), warm(5-20), heavy(>20)。"""
    arr = np.array(per_user_counts)
    buckets = {'cold': arr < 5, 'warm': (arr >= 5) & (arr <= 20), 'heavy': arr > 20}
    out = {}
    for name, sel in buckets.items():
        n = int(sel.sum())
        sub = {k: (float(np.mean(np.array(v)[sel])) if n > 0 else 0.0) for k, v in results_lists.items()}
        sub['n'] = n
        out[name] = sub
    return out


def main():
    args = parse_args()
    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    user2idx, item2idx = data['user2idx'], data['item2idx']
    n_users, n_items = data['n_users'], data['n_items']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    logger.info(f"Graph: {graph['n_total']} nodes, {graph['edge_index'].size(1)} edges, {graph['n_relations']} rels")
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
    logger.info(f"Loaded model: {args.model}")

    test_data = build_loo_test(interactions, user2idx, item2idx,
                               n_negative=args.n_negative, seed=args.seed)
    logger.info(f"Test users: {len(test_data)}")

    overall, per_user_counts, lists = evaluate_fast_with_lists(
        model, test_data, args.k, device, data['user_count'],
        batch_users=args.batch_users, subset=args.subset,
    )

    tag = args.tag or Path(args.model).parent.name
    out = {
        'model': args.model, 'data': args.data, 'tag': tag,
        'protocol': f'sampled-{args.n_negative} leave-one-out',
        'n_test_users': len(test_data) if not args.subset else args.subset,
        'overall': overall,
    }
    if args.coldstart and not args.subset:
        out['coldstart'] = coldstart_breakdown(per_user_counts, lists, args.k)

    out_path = Path('experiments/results') / f'eval_{tag}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 50)
    logger.info(f"Eval [{tag}] (n={out['n_test_users']})")
    logger.info("=" * 50)
    for m, v in sorted(overall.items()):
        logger.info(f"  {m}: {v:.4f}")
    if 'coldstart' in out:
        for bname, b in out['coldstart'].items():
            logger.info(f"  [{bname}] n={b['n']} hr@10={b['hr@10']:.4f} ndcg@10={b['ndcg@10']:.4f}")
    logger.info(f"Saved -> {out_path}")


def evaluate_fast_with_lists(model, test_data, k_values, device, user_count, batch_users=512, subset=None):
    """同 evaluate_fast 但返回逐用户 list（用于冷启动分桶）。"""
    model.eval()
    if subset:
        test_data = test_data[:subset]
    results = {f'hr@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    results.update({f'precision@{k}': [] for k in k_values})
    results.update({f'recall@{k}': [] for k in k_values})
    per_user_counts = []
    n_neg_max = max(len(negs) for _, _, negs in test_data) if test_data else 0
    N = n_neg_max
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(test_data), batch_users):
            batch = test_data[start:start + batch_users]
            B = len(batch)
            cand = torch.zeros(B, 1 + N, dtype=torch.long)
            mask = torch.ones(B, 1 + N, dtype=torch.bool)
            uids = torch.zeros(B, dtype=torch.long)
            cnts = torch.zeros(B, dtype=torch.long)
            for bi, (uid_idx, pos_items, neg_items) in enumerate(batch):
                uids[bi] = uid_idx
                cnts[bi] = user_count.get(uid_idx, 1)
                cand[bi, 0] = pos_items[0]
                for j, ng in enumerate(neg_items):
                    cand[bi, 1 + j] = ng
                if len(neg_items) < N:
                    mask[bi, 1 + len(neg_items):] = False
            uids, cnts, cand, mask = uids.to(device), cnts.to(device), cand.to(device), mask.to(device)
            flat_user = uids.unsqueeze(1).expand(-1, 1 + N).reshape(-1)
            flat_item = cand.reshape(-1)
            flat_cnt = cnts.unsqueeze(1).expand(-1, 1 + N).reshape(-1)
            scores = model.predict_score(flat_user, flat_item, interaction_counts=flat_cnt).view(B, 1 + N)
            scores = scores.masked_fill(~mask, float('-inf'))
            order = torch.argsort(scores, descending=True, dim=1).cpu().tolist()
            cand_cpu = cand.cpu().tolist()
            for bi in range(B):
                pos = [cand_cpu[bi][0]]
                ranked = [cand_cpu[bi][i] for i in order[bi] if mask[bi][i]]
                for k in k_values:
                    results[f'hr@{k}'].append(compute_hr_at_k(pos, ranked, k))
                    results[f'ndcg@{k}'].append(compute_ndcg_at_k(pos, ranked, k))
                    results[f'precision@{k}'].append(compute_precision_at_k(pos, ranked, k))
                    results[f'recall@{k}'].append(compute_recall_at_k(pos, ranked, k))
                per_user_counts.append(int(cnts[bi].item()))
            if (start // batch_users) % 20 == 0:
                logger.info(f"  [lists] {start+B}/{len(test_data)} {time.time()-t0:.0f}s")
    overall = {k: float(np.mean(v)) for k, v in results.items()}
    return overall, per_user_counts, results


if __name__ == '__main__':
    main()
