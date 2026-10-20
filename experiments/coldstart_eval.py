"""冷启动子集评估：把所有 baseline + KG-HybridRec 按 user bucket 切片。

支持 purchase + view_sparse 两场景。把每个 baseline 的 test 结果按用户训练交互数
cold/warm/heavy 切片，并跑两遍：
  1. 用 eval_batch.py 现有 baseline.score_batch
  2. 用 kg_hybrid_recommend.py 评估 KG-HybridRec

设计：
  - 不重新训练（已被 30 epoch baseline 训练好；KG-HybridRec checkpoint 已存在）
  - 一次性评估所有 baseline，按 user_bucket 切片
  - 输出 experiments/results/coldstart_eval.json
"""
import argparse, json, time
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k
from recommender.baselines_real import Popular, UserCF, ItemCF, BPRMF, LightGCN, RippleNet, KGCL
from recommender.kgat.hetero_graph import build_hetero_graph


def user_bucket(c):
    if c < 5: return 'cold'      # < 5 train interactions
    if c <= 20: return 'warm'    # 5-20
    return 'heavy'                # > 20


def slice_eval(score_fn, test_meta, K=10, batch_users=512, k_values=(5, 10, 20)):
    """score_fn(uids, cand) -> [B, N] scores. eval by user_bucket."""
    per_user = []
    for start in range(0, len(test_meta), batch_users):
        batch = test_meta[start:start + batch_users]
        B = len(batch)
        N = 1 + max(len(negs) for u, p, negs, _, _ in batch)
        uids = np.zeros(B, dtype=np.int64)
        cand = np.full((B, N), -1, dtype=np.int64)
        for bi, (uid, pos, negs, ub, ib) in enumerate(batch):
            uids[bi] = uid
            cand[bi, 0] = pos[0]
            for j, ng in enumerate(negs):
                cand[bi, 1 + j] = ng
        scores = score_fn(uids, cand)
        order = np.argsort(-scores, axis=1)
        for bi in range(B):
            pos = [int(cand[bi, 0])]
            ranked = cand[bi, order[bi]].tolist()
            r = {}
            for k in k_values:
                r[f'hr@{k}'] = compute_hr_at_k(pos, ranked, k)
                r[f'ndcg@{k}'] = compute_ndcg_at_k(pos, ranked, k)
            per_user.append((int(uids[bi]), int(cand[bi, 0]), r, batch[bi][3], batch[bi][4]))
    return per_user


def aggregate_by_bucket(per_user, k=10):
    by_ub = defaultdict(list)
    by_ib = defaultdict(list)
    by_cross = defaultdict(list)
    for u, i, r, ub, ib in per_user:
        by_ub[ub].append(r[f'hr@{k}'])
        by_ib[ib].append(r[f'hr@{k}'])
        by_cross[(ub, ib)].append(r[f'hr@{k}'])
    out = {
        'hr@10_by_user': {b: float(np.mean(v)) if v else None for b, v in by_ub.items()},
        'hr@10_by_item': {b: float(np.mean(v)) if v else None for b, v in by_ib.items()},
        'hr@10_by_cross': {f"{u}_{i}": float(np.mean(v)) if v else None
                          for (u, i), v in by_cross.items()},
        'n_user': {b: len(v) for b, v in by_ub.items()},
        'n_item': {b: len(v) for b, v in by_ib.items()},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--scene-name', default='purchase',
                   help='场景名,用于输出 json 字段名')
    ap.add_argument('--baselines', nargs='+',
                   default=['popular', 'usercf', 'itemcf', 'bprmf', 'lightgcn', 'ripplen', 'kgcl'])
    ap.add_argument('--kg-hybridrec', action='store_true', default=True,
                   help='include KG-HybridRec if checkpoint exists')
    ap.add_argument('--kg-hybridrec-checkpoint', default='checkpoints/m3_real')
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--usercf-topk', type=int, default=200)
    ap.add_argument('--output', default='experiments/results/coldstart_eval.json')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据
    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    train_pos = [(p[0], p[1]) for p, _ in data['train_data']]

    user_train_cnt = Counter(p[0] for p in train_pos)
    item_user_cnt = Counter(p[1] for p in train_pos)
    train_neg = [(p[0], n) for p, n in data['train_data']]
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                                n_negative=args.n_negative, seed=args.seed)

    # 异构图（ripplen / kgcl 需要）
    idx2user = {idx: uid for uid, idx in data['user2idx'].items()}
    idx2item = {idx: iid for iid, idx in data['item2idx'].items()}
    raw_pairs = [(idx2user[p[0]], idx2item[p[1]]) for p in train_pos]
    ei, et, n_total, _ = build_hetero_graph(products, raw_pairs, data['user2idx'], data['item2idx'])

    # 预计算 test_meta (每用户 user_bucket, item_bucket)
    def ub(c):
        if c < 5: return 'cold'
        if c <= 20: return 'warm'
        return 'heavy'
    def ib(c):
        if c < 5: return 'cold_item'
        if c <= 20: return 'warm_item'
        return 'hot_item'

    test_meta = []
    for u, pos, negs in test_data:
        uc = user_train_cnt.get(u, 0)
        ic = item_user_cnt.get(pos[0], 0)
        test_meta.append((u, pos, negs, ub(uc), ib(ic)))

    logger.info(f"Scene: {args.scene_name}  n_u={n_u}  n_i={n_i}  test={len(test_meta)}")
    logger.info(f"test buckets  by_user: {Counter(m[3] for m in test_meta)}")
    logger.info(f"test buckets  by_item: {Counter(m[4] for m in test_meta)}")

    results = {}
    for name in args.baselines:
        logger.info(f"==== {name} ====")
        t0 = time.time()
        try:
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
            elif name == 'ripplen':
                m = RippleNet(n_u, n_i, n_hops=2).to(dev)
                m.fit(train_pos, train_neg, ei, et, _, n_total, epochs=args.epochs, dev=dev)
            elif name == 'kgcl':
                m = KGCL(n_u, n_i).to(dev)
                m.fit(train_pos, train_neg, ei, et, n_total, epochs=args.epochs, dev=dev)
            else:
                logger.warning(f"unknown {name}, skip"); continue

            logger.info(f"  {name} trained in {time.time()-t0:.0f}s, eval...")
            per_user = slice_eval(m.score_batch, test_meta)
            agg = aggregate_by_bucket(per_user, k=10)
            results[name] = agg
            logger.info(f"  {name} cold={agg['hr@10_by_user'].get('cold',0):.4f} "
                        f"warm={agg['hr@10_by_user'].get('warm',0):.4f} "
                        f"heavy={agg['hr@10_by_user'].get('heavy',0):.4f}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'scene': args.scene_name,
        'data': args.data,
        'n_users': n_u, 'n_items': n_i,
        'n_test': len(test_meta),
        'results': results,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    logger.info(f"saved -> {out_path}")


if __name__ == '__main__':
    main()
