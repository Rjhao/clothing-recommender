"""数据稀疏度对比实验。

构造 3 种稀疏度训练集(full / 50% / 20%) 分别训练各 baseline + KG-HybridRec,
观察在不同数据稀疏度下各模型表现是否稳定。

稀疏度定义:train 数据相对 full set 的采样比例。
"""
import argparse, json, logging, time
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.train import load_data, prepare_training_data
from recommender.evaluate import build_loo_test, compute_hr_at_k, compute_ndcg_at_k
from recommender.baselines_real import Popular, UserCF, ItemCF, BPRMF, LightGCN, RippleNet, KGCL
from recommender.kgat.hetero_graph import build_hetero_graph


def subsample_train_data(train_pos, train_neg_dict, ratio, seed):
    """按 ratio 随机采样训练集,保持 user/item distribution 大致。"""
    rng = np.random.default_rng(seed)
    n = len(train_pos)
    keep_n = max(1, int(n * ratio))
    idx = rng.choice(n, size=keep_n, replace=False)
    sub_pos = [train_pos[i] for i in idx]
    sub_neg = []
    for i in idx:
        uid = train_pos[i][0]
        ns = train_neg_dict.get(uid, [])
        sub_neg.append((uid, ns))
    return sub_pos, sub_neg


def user_bucket(c):
    if c < 5: return 'cold'
    if c <= 20: return 'warm'
    return 'heavy'

def slice_eval(score_fn, test_meta, K=10, batch_users=512):
    per_user = []
    for start in range(0, len(test_meta), batch_users):
        batch = test_meta[start:start + batch_users]
        B = len(batch)
        N = 1 + max(len(negs) for _, _, negs, _, _ in batch)
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
            r['hr@10'] = compute_hr_at_k(pos, ranked, 10)
            r['ndcg@10'] = compute_ndcg_at_k(pos, ranked, 10)
            per_user.append((int(uids[bi]), int(cand[bi, 0]), r, batch[bi][3], batch[bi][4]))
    return per_user


def aggregate_by_bucket(per_user, k=10):
    by_ub = defaultdict(list)
    for u, i, r, ub, ib in per_user:
        by_ub[ub].append(r[f'hr@{k}'])
    return {
        'cold': float(np.mean(by_ub['cold'])) if by_ub['cold'] else None,
        'warm': float(np.mean(by_ub['warm'])) if by_ub['warm'] else None,
        'heavy': float(np.mean(by_ub['heavy'])) if by_ub['heavy'] else None,
        'n_cold': len(by_ub['cold']), 'n_warm': len(by_ub['warm']), 'n_heavy': len(by_ub['heavy']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--density', nargs='+', type=float, default=[1.0, 0.5, 0.2],
                   help='训练集采样比例列表 (full=1.0, half=0.5, sparse=0.2 etc)')
    ap.add_argument('--baselines', nargs='+',
                   default=['popular', 'usercf', 'itemcf', 'bprmf', 'lightgcn'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='experiments/results/density_eval.json')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Full 数据
    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    n_u, n_i = data['n_users'], data['n_items']
    # 简单构建 train_neg_dict (uid -> neg_items list)
    train_neg_dict = {}
    train_neg_dict = {p[0]: n for p, n in data['train_data']}
    train_pos = [(p[0], p[1]) for p, _ in data['train_data']]
    train_neg = [(p[0], n) for p, n in data['train_data']]

    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                                n_negative=args.n_negative, seed=args.seed)

    # 异构图(ripplen/kgcl/lightgcn 需要)
    idx2user = {idx: uid for uid, idx in data['user2idx'].items()}
    idx2item = {idx: iid for iid, idx in data['item2idx'].items()}
    raw_pairs_full = [(idx2user[p[0]], idx2item[p[1]]) for p in train_pos]
    ei, et, n_total, _ = build_hetero_graph(products, raw_pairs_full,
                                              data['user2idx'], data['item2idx'])

    # test_meta 用 full set 分桶(test set 不变)
    user_train_cnt = Counter(p[0] for p in train_pos)
    item_user_cnt = Counter(p[1] for p in train_pos)
    def ib(c):
        if c < 5: return 'cold_item'
        if c <= 20: return 'warm_item'
        return 'hot_item'
    test_meta = []
    for u, pos, negs in test_data:
        uc = user_train_cnt.get(u, 0)
        ic = item_user_cnt.get(pos[0], 0)
        test_meta.append((u, pos, negs, user_bucket(uc), ib(ic)))

    results = {}
    for ratio in args.density:
        logger.info(f"==== density ratio = {ratio} ====")
        # 采样子集
        if ratio < 1.0:
            sub_pos, sub_neg = subsample_train_data(train_pos, train_neg_dict, ratio, args.seed)
            # 也重建异构图(根据 sub_pos)
            raw_pairs_sub = [(idx2user[u], idx2item[i]) for u, i in sub_pos]
            ei_sub, et_sub, n_total_sub, _ = build_hetero_graph(
                products, raw_pairs_sub, data['user2idx'], data['item2idx'])
        else:
            sub_pos, sub_neg = train_pos, train_neg
            ei_sub, et_sub, n_total_sub = ei, et, n_total

        # 也重建 user_train_cnt 用于 cold/warm/heavy 划分
        sub_user_cnt = Counter(u for u, _ in sub_pos)

        # 重新分桶(基于子集)
        sub_test_meta = []
        for u, pos, negs, ub, ib_ in test_meta:
            uc_new = sub_user_cnt.get(u, 0)
            sub_test_meta.append((u, pos, negs, user_bucket(uc_new), ib_))

        ratio_results = {}
        for name in args.baselines:
            t0 = time.time()
            try:
                if name == 'popular':
                    m = Popular(n_i); m.fit(sub_pos)
                elif name == 'usercf':
                    m = UserCF(n_u, n_i, top_k=200); m.fit(sub_pos)
                elif name == 'itemcf':
                    m = ItemCF(n_u, n_i); m.fit(sub_pos)
                elif name == 'bprmf':
                    m = BPRMF(n_u, n_i).to(dev); m.fit(sub_pos, sub_neg, epochs=args.epochs, dev=dev)
                elif name == 'lightgcn':
                    m = LightGCN(n_u, n_i).to(dev); m.fit(sub_pos, sub_neg, epochs=args.epochs, dev=dev)
                else:
                    logger.warning(f"unknown {name}, skip"); continue
                pu = slice_eval(m.score_batch, sub_test_meta)
                agg = aggregate_by_bucket(pu, k=10)
                ratio_results[name] = {'hr@10_by_user': agg, 'trained_time': time.time()-t0}
                logger.info(f"  {name}: cold={agg['cold'] if agg['cold'] is not None else 0:.4f} warm={agg['warm'] if agg['warm'] is not None else 0:.4f} heavy={agg['heavy'] if agg['heavy'] is not None else 0:.4f}")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                ratio_results[name] = {'error': str(e)}
        results[f'ratio_{ratio}'] = {
            'n_train': len(sub_pos),
            'baselines': ratio_results,
        }

    out = {
        'data': args.data,
        'full_n_train': len(train_pos),
        'ratios': args.density,
        'results': results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"saved -> {out_path}")


if __name__ == '__main__':
    main()
