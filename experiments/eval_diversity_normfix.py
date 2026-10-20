"""快速诊断: 不重训练, 测试 norm 控制能否救 diversity.

Plan D 思路:
  1. 删 item_bias/user_bias/global_bias (cf_module.py)  -> 只影响训练, 推理路径不用
  2. per-param L2 (只对 item/user embedding 加 weight_decay) -> 只影响训练

两个改动都不直接影响已训练 ckpt 的推理. 等价于在推理时把 cf/kg embedding
归一化到统一范数, 模拟"L2 应该做的事".

测试 4 个变体 (同一 ckpt, 仅改 inference):
  A) baseline       : 当前 (原样)
  B) cf_norm        : L2 normalize cf_emb
  C) cf+kg_norm     : L2 normalize cf_emb & kg_emb
  D) layer_norm     : cf_emb 过 LayerNorm 后再融合
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec
from recommender.train import load_data, prepare_training_data
from recommender.kgat.hetero_graph import build_graph_for_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = 'data/amazon_fashion_real.json'
SEED = 42
TOP_K = 10
TAIL_THRESH = 5
USER_BATCH = 32
MAX_USERS_DEFAULT = 100  # CPU 跑得快 (500 在 CPU 上要 30 分钟/模式)

CONFIGS = [
    ('abl_m1', 1, 0, 0),
    ('abl_m2', 1, 1, 0),
    ('abl_m3', 1, 1, 1),
    ('m3_normfix', 1, 1, 1),
]


def gini(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n


def load_model(tag, data, products, graph, device):
    user2idx = data['user2idx']; item2idx = data['item2idx']
    n_users = data['n_users']; n_items = data['n_items']
    user_count_array = torch.zeros(n_users, dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    ckpt = Path('checkpoints') / tag / 'best_model.pt'
    use_a, use_b, use_c, use_mm = {
        'abl_m1': (1, 0, 0, True),
        'abl_m2': (1, 1, 0, True),
        'abl_m3': (1, 1, 1, True),
        'm3_normfix': (1, 1, 1, False),
    }[tag]

    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=graph['n_total'], n_relations=graph['n_relations'],
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=use_mm,
        use_improvement_a=bool(use_a), use_improvement_b=bool(use_b),
        use_improvement_c=bool(use_c),
        lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1,
        graph=graph, n_interactions=user_count_array,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def make_patched_predict(model, mode, layer_norm=None):
    """mode: 'baseline' | 'cf_norm' | 'cf_kg_norm' | 'cf_ln'"""
    def patched(user_ids, item_ids, interaction_counts=None, images=None, texts=None):
        cf_emb = model.cf_module.get_item_representation(item_ids)
        kg_emb, _ = model.kg_module(user_ids, item_ids)
        if model.use_multimodal and images is not None:
            mm_emb, _ = model.mm_module(images, texts)
        else:
            mm_emb = torch.zeros_like(cf_emb)

        if mode == 'cf_norm':
            cf_emb = F.normalize(cf_emb, p=2, dim=-1)
        elif mode == 'cf_kg_norm':
            cf_emb = F.normalize(cf_emb, p=2, dim=-1)
            kg_emb = F.normalize(kg_emb, p=2, dim=-1)
        elif mode == 'cf_ln':
            cf_emb = layer_norm(cf_emb)

        if model.use_improvement_a:
            cf_confidence = model.cf_module.get_confidence(user_ids)
            fused, _ = model.adaptive_gate(
                cf_emb, kg_emb, mm_emb,
                interaction_counts=interaction_counts,
                cf_confidence=cf_confidence,
            )
        else:
            adj = torch.eye(user_ids.size(0), device=user_ids.device)
            fused, _ = model.gat_fusion(cf_emb, kg_emb, mm_emb, adj)
        scores = model.predict(fused).squeeze(-1)
        return torch.sigmoid(scores)
    return patched


def evaluate_with_mode(model, mode, data, n_items, tail_items, user_train_items, test_users, device, layer_norm=None):
    model.predict_score = make_patched_predict(model, mode, layer_norm)
    all_item_ids = torch.arange(n_items, dtype=torch.long, device=device)
    recommend_counts = np.zeros(n_items, dtype=np.int64)
    unique_items_hit = set()
    total_tail_hits = 0
    total_rec = 0

    with torch.no_grad():
        for u in tqdm(test_users, desc=f'[{mode}]', leave=False):
            u_t = torch.full((n_items,), u, device=device, dtype=torch.long)
            c_t = torch.full((n_items,), data['user_count'].get(u, 1), device=device, dtype=torch.long)
            scores = model.predict_score(u_t, all_item_ids, interaction_counts=c_t)
            tr = user_train_items.get(u)
            if tr:
                scores[list(tr)] = -1e9
            topk = torch.topk(scores, TOP_K).indices.cpu().numpy()
            for it in topk:
                recommend_counts[it] += 1
                unique_items_hit.add(int(it))
                if int(it) in tail_items:
                    total_tail_hits += 1
                total_rec += 1

    coverage = len(unique_items_hit) / n_items
    g = gini(recommend_counts)
    p = recommend_counts / max(recommend_counts.sum(), 1)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * np.log(p_nz)).sum())
    norm_entropy = entropy / float(np.log(n_items))
    tail_ratio = total_tail_hits / total_rec

    # cf_emb norm 分布 (调试用)
    return {
        'mode': mode,
        'coverage@k': round(coverage, 6),
        'gini@k': round(g, 6),
        'norm_entropy@k': round(norm_entropy, 6),
        'tail_ratio@k': round(tail_ratio, 6),
        'n_unique': len(unique_items_hit),
        'n_tail_hits': total_tail_hits,
    }


def diagnose_norms(model, device, n_items):
    """诊断: cf_emb 和 kg_emb 的范数分布"""
    item_ids = torch.arange(n_items, device=device)
    cf_emb = model.cf_module.get_item_representation(item_ids)
    kg_emb, _ = model.kg_module(item_ids[:1], item_ids)  # kg 是 per-user 的
    # 取一个 user 看 kg 范数
    u_t = torch.zeros(n_items, dtype=torch.long, device=device)
    kg_emb_full, _ = model.kg_module(u_t, item_ids)

    cf_norm = cf_emb.norm(dim=-1).detach().cpu().numpy()
    kg_norm = kg_emb_full.norm(dim=-1).detach().cpu().numpy()

    logger.info(f'cf_emb norm: mean={cf_norm.mean():.3f} std={cf_norm.std():.3f} '
                f'min={cf_norm.min():.3f} max={cf_norm.max():.3f}')
    logger.info(f'kg_emb norm: mean={kg_norm.mean():.3f} std={kg_norm.std():.3f} '
                f'min={kg_norm.min():.3f} max={kg_norm.max():.3f}')

    # top-10 热门 vs 冷门 item 的范数对比
    from recommender.train import load_data, prepare_training_data
    products, users, interactions = load_data(DATA)
    data = prepare_training_data(users, interactions, seed=SEED)
    item_count = np.zeros(n_items)
    for (uid, iid), _ in data['train_data']:
        item_count[iid] += 1

    top10_idx = np.argsort(item_count)[-10:]
    bot10_idx = np.argsort(item_count)[:10]

    logger.info(f'  Top-10 popular item cf_norm: {cf_norm[top10_idx].mean():.3f}')
    logger.info(f'  Bottom-10 cold item cf_norm: {cf_norm[bot10_idx].mean():.3f}')
    logger.info(f'  Ratio (popular/cold): {cf_norm[top10_idx].mean() / max(cf_norm[bot10_idx].mean(), 1e-6):.2f}x')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='abl_m3')
    ap.add_argument('--modes', nargs='+',
                    default=['baseline', 'cf_norm', 'cf_kg_norm', 'cf_ln'])
    ap.add_argument('--max-users', type=int, default=MAX_USERS_DEFAULT,
                    help='Number of users to evaluate (default 100)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    products, users, interactions = load_data(DATA)
    data = prepare_training_data(users, interactions, seed=SEED)
    n_users = data['n_users']; n_items = data['n_items']
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])

    # 准备测试集
    user_train_items = {}
    for (uid_idx, iid), _ in data['train_data']:
        user_train_items.setdefault(uid_idx, set()).add(iid)

    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in data['user2idx'] and iid in data['item2idx']:
            user_items.setdefault(data['user2idx'][uid], []).append(data['item2idx'][iid])
    test_users = [u for u, items in user_items.items() if len(items) >= 2]
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(test_users), size=min(args.max_users, len(test_users)), replace=False)
    test_users = [test_users[i] for i in idx]

    item_train_count = np.zeros(n_items, dtype=np.int64)
    for (uid_idx, iid), _ in data['train_data']:
        item_train_count[iid] += 1
    tail_items = set(np.where(item_train_count < TAIL_THRESH)[0].tolist())

    model = load_model(args.tag, data, products, graph, device)

    # 诊断
    logger.info('=' * 60)
    logger.info(f'[{args.tag}] Norm diagnostic')
    logger.info('=' * 60)
    diagnose_norms(model, device, n_items)

    # LayerNorm 准备
    layer_norm = nn.LayerNorm(64).to(device)
    layer_norm.eval()

    results = []
    for mode in args.modes:
        logger.info('=' * 60)
        logger.info(f'[{args.tag}] mode={mode}')
        r = evaluate_with_mode(
            model, mode, data, n_items, tail_items,
            user_train_items, test_users, device, layer_norm,
        )
        results.append(r)
        logger.info(f'  Coverage={r["coverage@k"]:.4f}  Gini={r["gini@k"]:.4f}  '
                    f'NormEnt={r["norm_entropy@k"]:.4f}  TailRatio={r["tail_ratio@k"]:.4f}  '
                    f'unique={r["n_unique"]}  tail_hits={r["n_tail_hits"]}')

    print('\n' + '=' * 80)
    print(f'[{args.tag}] 多样性 vs norm 控制')
    print('=' * 80)
    print(f'{"mode":<15} {"coverage":>10} {"gini":>8} {"norm_ent":>10} {"tail_ratio":>12} {"unique":>8} {"tail_hits":>10}')
    for r in results:
        print(f'{r["mode"]:<15} {r["coverage@k"]:>10.4f} {r["gini@k"]:>8.4f} '
              f'{r["norm_entropy@k"]:>10.4f} {r["tail_ratio@k"]:>12.4f} '
              f'{r["n_unique"]:>8d} {r["n_tail_hits"]:>10d}')

    out_path = Path(f'experiments/results/normfix_{args.tag}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()