"""消融配置的推荐多样性/覆盖度评估 (真实 purchase 数据).

对比学习 (B) 的本职是 uniformity -> 多样性, 故 B 的价值可能不在 HR 而在推荐
分布的多样性/长尾覆盖. 本脚本对每个 test 用户在全量 item catalog 上打分
取 top-K, 统计:

  - Coverage@K: 所有用户 top-K 并集覆盖的不同 item 数 / 总 item 数 (越高越多样)
  - Gini@K:     item 被推荐次数的基尼系数 (越低越公平, 长尾受益)
  - Entropy@K:  推荐分布的香农熵 (越高越多样)
  - TailRatio@K: top-K 中长尾 item (训练交互<tail_thresh) 的比例 (越高长尾覆盖越好)
  - AggregateDiversity@K: 不同 item 总数 / (用户数*K), 即去重后占比

对 4 个消融 checkpoint (abl_m0/m1/m2/m3) 各跑一次, 汇总成多样性消融表.

用法 (单配置):
  PYTHONUTF8=1 python experiments/eval_diversity.py --tag abl_m0 --use-a 0 --use-b 0 --use-c 0
汇总:
  PYTHONUTF8=1 python experiments/eval_diversity.py --aggregate
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = 'data/amazon_fashion_real.json'
SEED = 42
TOP_K = 10          # 与 HR@10 对齐
TAIL_THRESH = 5     # 训练交互数 < 此值视为长尾 item
USER_BATCH = 32    # 每批用户数, 控制 (用户批 x 全量item) 矩阵显存. predict_score 内部
                   # KG ripple 邻居查表对大 batch 敏感 (256 会 OOM), 故取 32.

# (显示名, tag, A, B, C)
CONFIGS = [
    ('M0',            'abl_m0', 0, 0, 0),
    ('M0+A',          'abl_m1', 1, 0, 0),
    ('M0+A+B',        'abl_m2', 1, 1, 0),
    ('M0+A+B+C (M3)', 'abl_m3', 1, 1, 1),
]


def gini(arr):
    """基尼系数, arr 为各 item 被推荐次数. 0=完全均匀, 1=完全集中."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n


def eval_one(tag, use_a, use_b, use_c, top_k=TOP_K, tail_thresh=TAIL_THRESH, max_users=None):
    from recommender.train import load_data, prepare_training_data
    from recommender.kgat.hetero_graph import build_graph_for_model

    ckpt = Path('checkpoints') / tag / 'best_model.pt'
    if not ckpt.exists():
        logger.warning(f"[{tag}] checkpoint missing: {ckpt}")
        return None

    products, users, interactions = load_data(DATA)
    data = prepare_training_data(users, interactions, seed=SEED)
    user2idx = data['user2idx']; item2idx = data['item2idx']
    n_users = data['n_users']; n_items = data['n_items']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    user_count_array = torch.zeros(n_users, dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=graph['n_total'], n_relations=graph['n_relations'],
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=False,
        use_improvement_a=bool(use_a), use_improvement_b=bool(use_b),
        use_improvement_c=bool(use_c),
        lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1,
        graph=graph, n_interactions=user_count_array,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info(f"[{tag}] loaded, use-a/b/c={use_a}/{use_b}/{use_c}")

    # 每用户已交互 item (train) -> 推荐时排除
    user_train_items = {}
    for (uid_idx, iid), _ in data['train_data']:
        user_train_items.setdefault(uid_idx, set()).add(iid)

    # test 用户集合 (与 evaluate.build_loo_test 一致: 交互>=2)
    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], []).append(item2idx[iid])
    test_users = [u for u, items in user_items.items() if len(items) >= 2]
    # 采样降规模: 全量 64097 用户 x 21683 item 全打分太慢 (KG ripple 查表), 且多样性
    # 是聚合统计, 采样足够稳定. 固定 seed 采样保证各配置用同一批用户, 可比.
    if max_users is not None and len(test_users) > max_users:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(test_users), size=max_users, replace=False)
        test_users = [test_users[i] for i in idx]
        logger.info(f"[{tag}] sampled test_users to {len(test_users)} (from {len(user_items)})")

    # item 训练交互次数 (用于长尾判定)
    item_train_count = np.zeros(n_items, dtype=np.int64)
    for (uid_idx, iid), _ in data['train_data']:
        item_train_count[iid] += 1
    tail_items = set(np.where(item_train_count < tail_thresh)[0].tolist())
    logger.info(f"[{tag}] test_users={len(test_users)} n_items={n_items} tail_items(<{tail_thresh})={len(tail_items)}")

    all_item_ids = torch.arange(n_items, dtype=torch.long, device=device)
    recommend_counts = np.zeros(n_items, dtype=np.int64)   # 每个 item 被推荐(top-K)的次数
    unique_items_hit = set()
    total_tail_hits = 0
    total_rec = 0

    # 逐用户打分 (batch=1 用户): predict_score 内部 KG ripple 邻居查表对 (user,item)
    # 对构建多跳张量, 全量 21683 item x 多用户会撑爆 8GB 显存. 故逐用户: 每次只 1 用户
    # x 21683 item, 显存峰值最小 (21683 对的 KG 查表), 绝不 OOM. 采样用户保证总时长可控.
    with torch.no_grad():
        for u in tqdm(test_users, desc=f'[{tag}] div'):
            # 1 用户 x 全量 item 打分 -> [n_items]
            u_t = torch.full((n_items,), u, device=device, dtype=torch.long)
            c_t = torch.full((n_items,), data['user_count'].get(u, 1), device=device, dtype=torch.long)
            scores = model.predict_score(u_t, all_item_ids, interaction_counts=c_t)  # [n_items]

            # 排除训练已交互 item (置 -inf)
            tr = user_train_items.get(u)
            if tr:
                scores[list(tr)] = -1e9

            topk = torch.topk(scores, top_k).indices.cpu().numpy()  # [K]
            for it in topk:
                recommend_counts[it] += 1
                unique_items_hit.add(int(it))
                if int(it) in tail_items:
                    total_tail_hits += 1
                total_rec += 1

    # 指标
    coverage = len(unique_items_hit) / n_items
    g = gini(recommend_counts)
    # 熵: 推荐分布 p(item) = count/total
    p = recommend_counts / max(recommend_counts.sum(), 1)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * np.log(p_nz)).sum())
    max_entropy = float(np.log(n_items))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    tail_ratio = total_tail_hits / total_rec
    agg_div = len(unique_items_hit) / total_rec  # 去重后占比

    out = {
        'tag': tag, 'switches': {'a': use_a, 'b': use_b, 'c': use_c},
        'n_test_users': len(test_users), 'n_items': n_items, 'top_k': top_k,
        'tail_thresh': tail_thresh, 'n_tail_items': len(tail_items),
        'coverage@k': round(coverage, 6),
        'gini@k': round(g, 6),
        'entropy@k': round(entropy, 6),
        'norm_entropy@k': round(norm_entropy, 6),
        'tail_ratio@k': round(tail_ratio, 6),
        'agg_diversity@k': round(agg_div, 6),
    }
    logger.info(f"[{tag}] Coverage@{top_k}={coverage:.4f} Gini={g:.4f} "
                f"NormEntropy={norm_entropy:.4f} TailRatio={tail_ratio:.4f} AggDiv={agg_div:.4f}")
    out_path = Path('checkpoints') / tag / 'diversity_eval.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f"[{tag}] saved -> {out_path}")
    return out


def aggregate():
    rows = []
    for label, tag, a, b, c in CONFIGS:
        p = Path('checkpoints') / tag / 'diversity_eval.json'
        if not p.exists():
            rows.append(dict(label=label, tag=tag, A=a, B=b, C=c, metrics={}, status='MISSING'))
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        rows.append(dict(label=label, tag=tag, A=a, B=b, C=c, metrics=d, status='OK'))

    lines = [f'# 推荐多样性消融(真实 purchase, top-{TOP_K}, 全量 catalog 打分)\n']
    lines.append('| 配置 | A | B | C | Coverage | Gini↓ | NormEntropy↑ | TailRatio↑ | AggDiv↑ |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        a_s = '✓' if r['A'] else '✗'; b_s = '✓' if r['B'] else '✗'; c_s = '✓' if r['C'] else '✗'
        m = r['metrics']
        if m:
            cells = [f"{m['coverage@k']:.4f}", f"{m['gini@k']:.4f}",
                     f"{m['norm_entropy@k']:.4f}", f"{m['tail_ratio@k']:.4f}",
                     f"{m['agg_diversity@k']:.4f}"]
        else:
            cells = ['?'] * 5
        st = '' if r['status'] == 'OK' else f" ({r['status']})"
        lines.append(f"| {r['label']}{st} | {a_s} | {b_s} | {c_s} | " + " | ".join(cells) + " |")

    lines.append('\n> 指标解读:Coverage=推荐覆盖的不同商品占比;Gini↓=推荐分布越均匀;'
                 'NormEntropy↑=分布越多样;TailRatio↑=长尾商品(训练交互<5)命中占比;AggDiv↑=去重后占比.')
    lines.append('> 对比学习(B)本职是 uniformity->多样性, 预期 M0+A+B 在多样性指标上优于 M0+A.')

    md = '\n'.join(lines)
    print(md)
    out_json = Path('experiments/results/ablation_diversity_summary.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        'protocol': f'top-{TOP_K} over full catalog, leave-one-out train exclusion, seed={SEED}',
        'tail_thresh': TAIL_THRESH,
        'rows': rows,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n汇总 JSON: {out_json}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--use-a', type=int, default=0)
    ap.add_argument('--use-b', type=int, default=0)
    ap.add_argument('--use-c', type=int, default=0)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--max-users', type=int, default=2000,
                    help='采样用户数 (全量太慢), 默认 2000, 固定 seed 保证各配置可比')
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.tag is None:
        ap.error('need --tag or --aggregate')
    eval_one(args.tag, args.use_a, args.use_b, args.use_c, max_users=args.max_users)


if __name__ == '__main__':
    main()
