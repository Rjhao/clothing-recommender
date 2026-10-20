"""冷启动子集消融评估 (真实 purchase 数据).

对 4 个消融 checkpoint (abl_m0/m1/m2/m3) 在冷启动子集上评估, 产出「冷启动消融表」.
复用 evaluate.py 的建图 + sampled-100 协议, 保证 load_state_dict 不 mismatch
(不用 evaluate_coldstart.py 的硬编码 n_entities/n_relations 占位值).

子集定义 (与 evaluate_coldstart.py 一致):
  cold_user:  test 用户总交互数 < cold_user_max (默认 5)
  cold_item:  test 正样本商品总交互数 < cold_item_max (默认 10)
  cold_both:  两者同时满足

用法 (对单配置):
  PYTHONUTF8=1 python experiments/eval_ablation_coldstart.py \
    --tag abl_m0 --use-a 0 --use-b 0 --use-c 0
汇总 (4 配置全跑完):
  PYTHONUTF8=1 python experiments/eval_ablation_coldstart.py --aggregate
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec
from recommender.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = 'data/amazon_fashion_real.json'
SEED = 42
N_NEG = 100

# (显示名, tag, A, B, C)
CONFIGS = [
    ('M0',            'abl_m0', 0, 0, 0),
    ('M0+A',          'abl_m1', 1, 0, 0),
    ('M0+A+B',        'abl_m2', 1, 1, 0),
    ('M0+A+B+C (M3)', 'abl_m3', 1, 1, 1),
]


def build_loo_test_filtered(interactions, user2idx, item2idx, n_negative, seed,
                            user_filter=None, item_filter=None):
    """LOO test set, 过滤到指定子集. 逻辑同 evaluate_coldstart.build_loo_test_filtered."""
    rng = random.Random(seed)
    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], []).append(item2idx[iid])
    n_items = len(item2idx)
    all_item_set = set(range(n_items))

    user_total = Counter()
    item_total = Counter()
    for uid_idx, items in user_items.items():
        user_total[uid_idx] += len(items)
        for it in items:
            item_total[it] += 1

    test_data = []
    for uid_idx, items in user_items.items():
        if len(items) < 2:
            continue
        if user_filter is not None and uid_idx not in user_filter:
            continue
        pos_test = items[-1]
        if item_filter is not None and pos_test not in item_filter:
            continue
        train_items = set(items[:-1])
        candidates = list(all_item_set - train_items - {pos_test})
        if len(candidates) < n_negative:
            negs = candidates
        else:
            negs = rng.sample(candidates, n_negative)
        test_data.append((uid_idx, [pos_test], negs))
    return test_data, user_total, item_total


def eval_one(tag, use_a, use_b, use_c, cold_user_max=5, cold_item_max=10):
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
    # 与 evaluate.py 完全一致的建图 -> 保证 load 成功
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

    # cold 子集
    user_total = Counter()
    item_total = Counter()
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in user2idx: user_total[user2idx[uid]] += 1
        if iid in item2idx: item_total[item2idx[iid]] += 1
    cold_users = {u for u, c in user_total.items() if c < cold_user_max}
    cold_items = {it for it, c in item_total.items() if c < cold_item_max}
    logger.info(f"[{tag}] cold_user(<{cold_user_max})={len(cold_users)} cold_item(<{cold_item_max})={len(cold_items)}")

    subsets = {
        'all':        (None, None),
        'cold_user':  (cold_users, None),
        'cold_item':  (None, cold_items),
        'cold_both':  (cold_users, cold_items),
    }
    out = {}
    for name, (uf, itf) in subsets.items():
        td, _, _ = build_loo_test_filtered(interactions, user2idx, item2idx,
                                           N_NEG, SEED, user_filter=uf, item_filter=itf)
        if len(td) == 0:
            out[name] = {'n_test': 0}
            continue
        r = evaluate_model(model, td, [5, 10, 20], device=device,
                           user_count=data['user_count'])
        r['n_test'] = len(td)
        out[name] = r
        logger.info(f"[{tag}] [{name}] n={len(td):6d}  HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}")

    result = {
        'tag': tag, 'switches': {'a': use_a, 'b': use_b, 'c': use_c},
        'cold_user_max': cold_user_max, 'cold_item_max': cold_item_max,
        'subsets': out,
    }
    out_path = Path('checkpoints') / tag / 'coldstart_eval.json'
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f"[{tag}] saved -> {out_path}")
    return result


def aggregate():
    """汇总 4 配置冷启动结果成表."""
    rows = []
    for label, tag, a, b, c in CONFIGS:
        p = Path('checkpoints') / tag / 'coldstart_eval.json'
        if not p.exists():
            rows.append(dict(label=label, tag=tag, A=a, B=b, C=c,
                             subsets={}, status='MISSING'))
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        rows.append(dict(label=label, tag=tag, A=a, B=b, C=c,
                         subsets=d['subsets'], status='OK'))

    subsets = ['all', 'cold_user', 'cold_item', 'cold_both']
    lines = ['# 冷启动消融实验(真实 purchase 数据, sampled-100)\n']
    lines.append('## HR@10')
    lines.append('| 配置 | A | B | C | all | cold_user | cold_item | cold_both |')
    lines.append('|---|---|---|---|---|---|---|---|')
    m0_hr = {s: (rows[0]['subsets'].get(s, {}) or {}).get('hr@10') for s in subsets}
    for r in rows:
        a_s = '✓' if r['A'] else '✗'; b_s = '✓' if r['B'] else '✗'; c_s = '✓' if r['C'] else '✗'
        cells = []
        for s in subsets:
            sub = (r['subsets'].get(s, {}) or {})
            hr = sub.get('hr@10')
            cells.append(f"{hr:.4f}" if hr is not None else '?')
        st = '' if r['status'] == 'OK' else f" ({r['status']})"
        lines.append(f"| {r['label']}{st} | {a_s} | {b_s} | {c_s} | " + " | ".join(cells) + " |")

    lines.append('\n## NDCG@10')
    lines.append('| 配置 | A | B | C | all | cold_user | cold_item | cold_both |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        a_s = '✓' if r['A'] else '✗'; b_s = '✓' if r['B'] else '✗'; c_s = '✓' if r['C'] else '✗'
        cells = []
        for s in subsets:
            sub = (r['subsets'].get(s, {}) or {})
            nd = sub.get('ndcg@10')
            cells.append(f"{nd:.4f}" if nd is not None else '?')
        lines.append(f"| {r['label']} | {a_s} | {b_s} | {c_s} | " + " | ".join(cells) + " |")

    # vs M0 (cold_user 单项贡献)
    lines.append('\n## cold_user 上 vs M0 的 HR@10 增益(消融贡献核心)')
    lines.append('| 配置 | HR@10 | vs M0 | 单项贡献ΔHR |')
    lines.append('|---|---|---|---|')
    prev = None
    for r in rows:
        hr = (r['subsets'].get('cold_user', {}) or {}).get('hr@10')
        vs = f"{hr - m0_hr['cold_user']:+.4f}" if (hr is not None and m0_hr['cold_user'] is not None and r['label'] != 'M0') else '-'
        cd = f"{hr - prev:+.4f}" if (hr is not None and prev is not None) else '-'
        h_s = f"{hr:.4f}" if hr is not None else '?'
        lines.append(f"| {r['label']} | {h_s} | {vs} | {cd} |")
        if hr is not None:
            prev = hr

    md = '\n'.join(lines)
    print(md)
    out_json = Path('experiments/results/ablation_coldstart_summary.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        'protocol': 'leave-one-out + sampled-100, cold_user<5, cold_item<10, seed=42',
        'rows': rows,
        'note': '冷启动消融: A/B 主要为稀疏/冷启动设计, 预期 cold 子集上贡献 > 全量. '
                'C (attention_explain) 不参与 forward/predict_score, 故 M0+A+B 与 M0+A+B+C(M3) 的 HR 相同.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n汇总 JSON: {out_json}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default=None, help='单配置 tag, 如 abl_m0')
    ap.add_argument('--use-a', type=int, default=0)
    ap.add_argument('--use-b', type=int, default=0)
    ap.add_argument('--use-c', type=int, default=0)
    ap.add_argument('--cold-user-max', type=int, default=5)
    ap.add_argument('--cold-item-max', type=int, default=10)
    ap.add_argument('--aggregate', action='store_true', help='汇总 4 配置成表')
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return
    if args.tag is None:
        ap.error('need --tag or --aggregate')
    eval_one(args.tag, args.use_a, args.use_b, args.use_c,
             cold_user_max=args.cold_user_max, cold_item_max=args.cold_item_max)


if __name__ == '__main__':
    main()
