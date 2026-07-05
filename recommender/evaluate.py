"""Evaluation script for KG-HybridRec.

Computes HR@K, NDCG@K, Precision@K, Recall@K metrics.

Usage:
    python -m recommender.evaluate --model checkpoints/best_model.pt --data data/amazon_fashion_processed.json
"""
import argparse
import json
import logging
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
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20], help='K values for HR@K, NDCG@K')

    return parser.parse_args()


def compute_hr_at_k(positive_items, recommended_items, k):
    """Compute Hit Rate at K.

    Args:
        positive_items: List of positive (relevant) items
        recommended_items: List of recommended items (sorted by score, top K)
        k: Cutoff

    Returns:
        1 if any positive item is in top-K recommendations, 0 otherwise
    """
    recommended_k = set(recommended_items[:k])
    return 1 if any(item in recommended_k for item in positive_items) else 0


def compute_ndcg_at_k(positive_items, recommended_items, k):
    """Compute NDCG at K.

    Args:
        positive_items: List of positive (relevant) items
        recommended_items: List of recommended items (sorted by score, top K)
        k: Cutoff

    Returns:
        NDCG@K score
    """
    recommended_k = recommended_items[:k]

    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in positive_items:
            dcg += 1 / np.log2(i + 2)  # i+2 because position starts at 1 and we want i+2 for formula

    # IDCG: ideal case where all positive items are at the top
    n_positive = len(positive_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_positive, k)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_precision_at_k(positive_items, recommended_items, k):
    """Compute Precision at K."""
    recommended_k = set(recommended_items[:k])
    return len(recommended_k & set(positive_items)) / k


def compute_recall_at_k(positive_items, recommended_items, k):
    """Compute Recall at K."""
    recommended_k = set(recommended_items[:k])
    n_positive = len(positive_items)
    if n_positive == 0:
        return 0.0
    return len(recommended_k & set(positive_items)) / n_positive


def evaluate_model(model, test_data, user2idx, item2idx, batch_size=256, k_values=[5, 10, 20]):
    """Evaluate model on test data.

    Args:
        model: KG-HybridRec model
        test_data: List of (user_idx, positive_items, negative_items) tuples
        user2idx: User ID to index mapping
        item2idx: Item ID to index mapping
        batch_size: Batch size for evaluation
        k_values: List of K values for metrics

    Returns:
        Dictionary of metrics
    """
    model.eval()

    results = {f'hr@{k}': [] for k in k_values}
    results.update({f'ndcg@{k}': [] for k in k_values})
    results.update({f'precision@{k}': [] for k in k_values})
    results.update({f'recall@{k}': [] for k in k_values})

    n_users = len(user2idx)
    n_items = len(item2idx)

    with torch.no_grad():
        for user_idx, pos_items, neg_items in tqdm(test_data, desc='Evaluating'):
            # Combine positive and negative items as candidates
            candidate_items = list(pos_items) + list(neg_items)
            candidate_set = set(candidate_items)

            # Get scores for all candidates
            user_ids = torch.LongTensor([user_idx] * len(candidate_items))
            item_ids = torch.LongTensor(candidate_items)

            scores = model.predict_score(user_ids, item_ids)

            # Sort by score (descending)
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            ranked_items = [candidate_items[i] for i in sorted_indices]

            # Compute metrics for each K
            for k in k_values:
                hr = compute_hr_at_k(pos_items, ranked_items, k)
                ndcg = compute_ndcg_at_k(pos_items, ranked_items, k)
                prec = compute_precision_at_k(pos_items, ranked_items, k)
                rec = compute_recall_at_k(pos_items, ranked_items, k)

                results[f'hr@{k}'].append(hr)
                results[f'ndcg@{k}'].append(ndcg)
                results[f'precision@{k}'].append(prec)
                results[f'recall@{k}'].append(rec)

    # Average metrics
    avg_results = {k: np.mean(v) for k, v in results.items()}

    return avg_results


def main():
    args = parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")

    # Load data to get dimensions
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    products = data['products']
    users = data['users']
    interactions = data['interactions']

    # Build mappings
    user_ids = sorted(set(int(i['user_id']) for i in interactions))
    item_ids = sorted(set(int(i['product_id']) for i in interactions))

    n_users = len(user_ids)
    n_items = len(item_ids)
    n_entities = n_items + n_users + 100
    n_relations = 10

    # Create model
    model = KGHybridRec(
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities,
        n_relations=n_relations,
        embedding_dim=64,
        n_hops=2,
        n_heads=4,
        n_layers=2,
        use_improvement_a=True,
        use_improvement_b=True,
        use_improvement_c=True,
    )

    # Load weights
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    logger.info("Model loaded successfully")

    # Prepare test data (use same logic as train.py)
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # Build user interactions
    user_pos_items = {}
    for interaction in interactions:
        uid = int(interaction['user_id'])
        iid = int(interaction['product_id'])
        if uid in user2idx and iid in item2idx:
            uid_idx = user2idx[uid]
            iid_idx = item2idx[iid]
            user_pos_items.setdefault(uid_idx, []).append(iid_idx)

    # Create test samples
    test_data = []
    for uid_idx, pos_items in user_pos_items.items():
        # For simplicity, use the positive items as both " positives to test"
        # and sample negatives from other items
        all_items = set(range(n_items))
        neg_items = list(all_items - set(pos_items))

        test_data.append((
            uid_idx,
            pos_items,
            neg_items[:len(pos_items)]  # Same number of negatives as positives
        ))

    logger.info(f"Test samples: {len(test_data)}")

    # Evaluate
    results = evaluate_model(
        model=model,
        test_data=test_data,
        user2idx=user2idx,
        item2idx=item2idx,
        batch_size=args.batch_size,
        k_values=args.k,
    )

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)

    for metric, value in sorted(results.items()):
        logger.info(f"{metric}: {value:.4f}")

    # Save results
    output_path = Path(args.model).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
