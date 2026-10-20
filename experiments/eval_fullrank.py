"""全量排序评估 (非 sampled, 对全量 catalog 排序).

回应"为何不全量评估"的审稿质疑. 在 sampled-100 评估掩盖下,
模型实际可能严重热门偏置. 本脚本在 500-2000 采样用户上对全 21683 item
排序, 输出 NDCG@10 / HR@10 / MRR / 推荐列表多样性.

与 sampled-100 不同: 不配 100 随机负样本, 而是把所有其他 item 作负,
1 正 + N-1 负 候选排序. HR@10 数值会比 sampled-100 低很多(更多候选),
但能更真实反映个性化能力.

用法 (单配置):
  PYTHONUTF8=1 python experiments/eval_fullrank.py --tag abl_m0 \\
    --use-a 0 --use-b 0 --use-c 0 [--max-users 1000]
汇总:
  PYTHONUTF8=1 python experiments/eval_fullrank.py --aggregate
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = 'data/amazon_fashion_real.json'
SEED = 42
TOP_K = 10
USER_BATCH = 16  # 全量 item 打分极费, 用极小 batch

CONFIGS = [
    ('M0',            'abl_m0', 0, 0, 0),
    ('M0+A',          'abl_m1', 1, 0, 0),
    ('M0+A+B',        'abl_m2', 1, 1, 0),
    ('M0+A+B+C (M3)', 'abl_m3', 1, 1, 1),
]


def build_fullrank_test(interactions, user2idx, item2idx, seed):
    """每用户最后 1 条交互作 test, 其余作 train, 留一法 (同 evaluate.build_loo_test)."""
    rng = random.Random(seed)
    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], []).append(item2idx[iid])
    test_data = []
    for u, items in user_items.items():
        if len(items) < 2: continue
        pos = items[-1]
        test_data.append((u, [pos]))
    return test_data


def eval_one(tag, use_a, use_b, use_c, top_k=TOP_K, max_users=None):
    from recommender.train import load_data, prepare_training_data
    from recommender.kgat.hetero_graph import build_graph_for_model

    ckpt = Path('checkpoints') / tag / 'best_model.pt'
    if not ckpt.exists():
        logger.warning(f"[{tag}] checkpoint missing"); return None
    products, users, interactions = load_data(DATA)
    data = prepare_training_data(users, interactions, seed=SEED)
    user2idx = data['user2idx']; item2idx = data['item2idx']
    n_users = data['n_users']; n_items = data['n_items']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    uca = torch.zeros(n_users, dtype=torch.float32)
    for uid, cnt in data['user_count'].items(): uca[uid] = cnt

    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=graph['n_total'], n_relations=graph['n_relations'],
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=False,
        use_improvement_a=bool(use_a), use_improvement_b=bool(use_b),
        use_improvement_c=bool(use_c),
        lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1,
        graph=graph, n_interactions=uca,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    user_train_items = {}
    for (uid, iid), _ in data['train_data']:
        user_train_items.setdefault(uid, set()).add(iid)

    test_data = build_fullrank_test(interactions, user2idx, item2idx, SEED)
    if max_users and len(test_data) > max_users:
        rng = random.Random(SEED)
        test_data = rng.sample(test_data, max_users)
    logger.info(f"[{tag}] fullrank test_users={len(test_data)} n_items={n_items}")

    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    hr10s = []; ndcg10s = []; mrrs = []
    unique_hit = set()

    with torch.no_grad():
        for u, pos_items in tqdm(test_data, desc=f'[{tag}] fullrank'):
            cnt = data['user_count'].get(u, 1)
            u_t = torch.full((n_items,), u, device=device, dtype=torch.long)
            c_t = torch.full((n_items,), cnt, device=device, dtype=torch.long)
            scores = model.predict_score(u_t, all_items, interaction_counts=c_t)
            tr = user_train_items.get(u)
            if tr: scores[list(tr)] = -1e9
            rank = torch.argsort(scores, descending=True).cpu().numpy()
            topk = rank[:top_k]
            pos = set(pos_items)
            # HR@10
            hr10s.append(1.0 if any(it in pos for it in topk) else 0.0)
            # NDCG@10
            dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(topk) if it in pos)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos), top_k)))
            ndcg10s.append(dcg / idcg if idcg > 0 else 0.0)
            # MRR
            for r, it in enumerate(rank):
                if it in pos:
                    mrrs.append(1.0 / (r + 1)); break
            else:
                mrrs.append(0.0)
            for it in topk: unique_hit.add(int(it))

    out = {
        'tag': tag, 'switches': {'a': use_a, 'b': use_b, 'c': use_c},
        'n_test_users': len(test_data), 'n_items': n_items, 'top_k': top_k,
        'fullrank_hr@10': round(float(np.mean(hr10s)), 6),
        'fullrank_ndcg@10': round(float(np.mean(ndcg10s)), 6),
        'fullrank_mrr': round(float(np.mean(mrrs)), 6),
        'fullrank_coverage@10': round(len(unique_hit) / n_items, 6),
    }
    logger.info(f"[{tag}] HR@10={out['fullrank_hr@10']:.4f} NDCG@10={out['fullrank_ndcg@10']:.4f} "
                f"MRR={out['fullrank_mrr']:.4f} Coverage={out['fullrank_coverage@10']:.4f}")
    p = Path('checkpoints') / tag / 'fullrank_eval.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def aggregate():
    rows = []
    for label, tag, a, b, c in CONFIGS:
        p = Path('checkpoints') / tag / 'fullrank_eval.json'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            rows.append({'label': label, 'tag': tag, 'A': a, 'B': b, 'C': c, **d, 'status': 'OK'})
        else:
            rows.append({'label': label, 'tag': tag, 'A': a, 'B': b, 'C': c, 'status': 'MISSING'})
    print('# 全量排序评估 (留一法, 全 21683 item 作候选, 无 sampled)')
    print(f'| 配置 | A | B | C | HR@10 | NDCG@10 | MRR | Coverage@10 |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['status'] == 'OK':
            st = ''
            print(f"| {r['label']}{st} | {'✓' if r['A'] else '✗'} | {'✓' if r['B'] else '✗'} | {'✓' if r['C'] else '✗'} | "
                  f"{r['fullrank_hr@10']:.4f} | {r['fullrank_ndcg@10']:.4f} | {r['fullrank_mrr']:.4f} | "
                  f"{r['fullrank_coverage@10']:.4f} |")
        else:
            print(f"| {r['label']} (MISSING) | -- | -- | -- | -- | -- | -- | -- |")
    out = Path('experiments/results/ablation_fullrank_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': rows,
        'note': '全量排序评估, 回应 sampled-100 协议对热门偏置的掩盖. 数值远低于 sampled-100 (候选基数大), 但更能反映真实个性化能力.'},
        ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str)
    ap.add_argument('--use-a', type=int, default=0)
    ap.add_argument('--use-b', type=int, default=0)
    ap.add_argument('--use-c', type=int, default=0)
    ap.add_argument('--max-users', type=int, default=1000)
    ap.add_argument('--aggregate', action='store_true')
    args = ap.parse_args()
    if args.aggregate:
        aggregate(); return
    eval_one(args.tag, args.use_a, args.use_b, args.use_c, max_users=args.max_users)


if __name__ == '__main__':
    main()
