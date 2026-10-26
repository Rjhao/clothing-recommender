"""Evaluation script for KG-HybridRec.

Computes HR@K, NDCG@K, Precision@K, Recall@K metrics.

评估协议（业界标准，与论文可比）:
  - Leave-one-out: 每用户最后 1 条交互作为 test 正样本，其余进 train
  - Sampled-100: 每 test 正样本配 100 个随机负样本，共 101 候选排序
  - HR@K / NDCG@K，K ∈ {5, 10, 20}

Usage:
    python -m recommender.evaluate --model checkpoints/best_model.pt \
        --data data/amazon_fashion_real.json
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate KG-HybridRec')

    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to data JSON')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20], help='K values')
    parser.add_argument('--n-negative', type=int, default=100, help='sampled negatives per positive')
    parser.add_argument('--seed', type=int, default=42)

    # 三项改进开关（须与训练时一致，否则权重对不上）
    parser.add_argument('--use-a', type=int, default=1)
    parser.add_argument('--use-b', type=int, default=1)
    parser.add_argument('--use-c', type=int, default=1)
    parser.add_argument('--use-multimodal', type=int, default=0)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--use-cold-adjust', type=int, default=1,
                        help='改进 A 的稀疏度调节项开关(须与训练时一致)。'
                             '置 0 用于评测"无稀疏度自适应"的对照片段。')

    return parser.parse_args()


def compute_hr_at_k(positive_items, recommended_items, k):
    recommended_k = set(recommended_items[:k])
    return 1 if any(item in recommended_k for item in positive_items) else 0


def compute_ndcg_at_k(positive_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in positive_items:
            dcg += 1 / np.log2(i + 2)
    n_positive = len(positive_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_positive, k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_precision_at_k(positive_items, recommended_items, k):
    recommended_k = set(recommended_items[:k])
    return len(recommended_k & set(positive_items)) / k


def compute_recall_at_k(positive_items, recommended_items, k):
    recommended_k = set(recommended_items[:k])
    n_positive = len(positive_items)
    if n_positive == 0:
        return 0.0
    return len(recommended_k & set(positive_items)) / n_positive


def build_loo_test(interactions, user2idx, item2idx, n_negative=100, seed=42):
    """Leave-one-out: 每用户最后 1 条交互作 test 正样本，配 sampled 负样本。

    Returns:
        test_data: list of (user_idx, [pos_item_idx], [neg_item_idx...])
    """
    rng = random.Random(seed)

    # 按 user 聚合交互（保留顺序）
    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id'])
        iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], []).append(item2idx[iid])

    n_items = len(item2idx)
    all_item_set = set(range(n_items))

    test_data = []
    for uid_idx, items in user_items.items():
        if len(items) < 2:
            # 交互不足 2 条的用户，leave-one-out 无法做，跳过
            continue
        pos_test = items[-1]  # 最后 1 条作 test
        train_items = set(items[:-1])

        # sampled negatives：从用户未交互的商品中抽 n_negative 个
        candidates = list(all_item_set - train_items - {pos_test})
        if len(candidates) < n_negative:
            # 不够就用全部
            negs = candidates
        else:
            negs = rng.sample(candidates, n_negative)

        test_data.append((uid_idx, [pos_test], negs))

    return test_data


def evaluate_model(model, test_data, k_values, device='cpu', user_count=None):
    """Sampled-100 评估：对每个用户，将 1 正 + N 负 共 N+1 候选排序算指标。"""
    model.eval()

    results = {f'hr@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    results.update({f'precision@{k}': [] for k in k_values})
    results.update({f'recall@{k}': [] for k in k_values})

    with torch.no_grad():
        for user_idx, pos_items, neg_items in tqdm(test_data, desc='Evaluating'):
            candidate_items = list(pos_items) + list(neg_items)
            user_ids = torch.LongTensor([user_idx] * len(candidate_items)).to(device)
            item_ids = torch.LongTensor(candidate_items).to(device)

            if user_count is not None:
                cnt = user_count.get(user_idx, 1)
                counts = torch.LongTensor([cnt] * len(candidate_items)).to(device)
                scores = model.predict_score(user_ids, item_ids, interaction_counts=counts)
            else:
                scores = model.predict_score(user_ids, item_ids)

            sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
            ranked_items = [candidate_items[i] for i in sorted_indices]

            for k in k_values:
                results[f'hr@{k}'].append(compute_hr_at_k(pos_items, ranked_items, k))
                results[f'ndcg@{k}'].append(compute_ndcg_at_k(pos_items, ranked_items, k))
                results[f'precision@{k}'].append(compute_precision_at_k(pos_items, ranked_items, k))
                results[f'recall@{k}'].append(compute_recall_at_k(pos_items, ranked_items, k))

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    args = parse_args()

    from recommender.train import load_data, prepare_training_data
    from recommender.kgat.hetero_graph import build_graph_for_model

    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    user2idx = data['user2idx']
    item2idx = data['item2idx']
    n_users = data['n_users']
    n_items = data['n_items']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Rebuild the same graph used at training (for KG branch + improvement B)
    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    logger.info(f"Built hetero graph: {graph['n_total']} nodes, {graph['edge_index'].size(1)} edges")

    n_entities = graph['n_total']
    n_relations = graph['n_relations']

    # Per-user interaction count tensor (real values for adaptive gate)
    user_count_array = torch.zeros(n_users, dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    model = KGHybridRec(
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities,
        n_relations=n_relations,
        embedding_dim=args.embedding_dim,
        n_hops=2,
        n_heads=4,
        n_layers=2,
        use_multimodal=bool(args.use_multimodal),
        use_improvement_a=bool(args.use_a),
        use_improvement_b=bool(args.use_b),
        use_improvement_c=bool(args.use_c),
        graph=graph,
        n_interactions=user_count_array,
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    if not args.use_cold_adjust:
        model.adaptive_gate.adaptive_gate.use_interaction_count = False
    logger.info("Model loaded")

    test_data = build_loo_test(
        interactions, user2idx, item2idx,
        n_negative=args.n_negative, seed=args.seed,
    )
    logger.info(f"Test users: {len(test_data)}, sampled negatives: {args.n_negative}")

    results = evaluate_model(model, test_data, args.k, device=device, user_count=data['user_count'])

    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results (sampled-{})".format(args.n_negative))
    logger.info("=" * 50)
    for metric, value in sorted(results.items()):
        logger.info(f"{metric}: {value:.4f}")

    output_path = Path(args.model).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
