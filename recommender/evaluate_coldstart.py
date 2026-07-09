"""Cold-start evaluation for KG-HybridRec (stage 2.4).

Evaluates a trained model on cold-start subsets:
  - cold users: test users whose total interaction count < 5
  - cold items: test positives whose total interaction count (as an item) < 10
  - both: cold user AND cold item

Reuses the same sampled-100 leave-one-out protocol as evaluate.py, but filters
the test set to the cold-start subset. The 100 sampled negatives are kept
unchanged (they reflect the full item catalog, which is realistic).

Usage:
    python -m recommender.evaluate_coldstart --model checkpoints/m3/best_model.pt \
        --use-a 1 --use-b 1 --use-c 1
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_loo_test_filtered(interactions, user2idx, item2idx, n_negative=100, seed=42,
                            user_filter=None, item_filter=None):
    """Leave-one-out test set, optionally filtered to a cold-start subset.

    Args:
        user_filter: set of user_idx to keep (e.g. cold users). None = all.
        item_filter: set of item_idx whose positive test samples to keep. None = all.
    """
    rng = random.Random(seed)

    user_items = {}
    for inter in interactions:
        uid = int(inter['user_id'])
        iid = int(inter['product_id'])
        if uid in user2idx and iid in item2idx:
            user_items.setdefault(user2idx[uid], []).append(item2idx[iid])

    n_items = len(item2idx)
    all_item_set = set(range(n_items))

    # total interaction count per user/item (over the full dataset)
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

    return test_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, default='data/amazon_fashion_expanded.json')
    ap.add_argument('--use-a', type=int, default=1)
    ap.add_argument('--use-b', type=int, default=1)
    ap.add_argument('--use-c', type=int, default=1)
    ap.add_argument('--use-multimodal', type=int, default=0)
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--cold-user-max', type=int, default=5, help='cold user = total inter < this')
    ap.add_argument('--cold-item-max', type=int, default=10, help='cold item = total inter < this')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    from recommender.train import load_data, prepare_training_data
    from recommender.kgat.hetero_graph import build_graph_for_model
    products, users, interactions = load_data(args.data)
    data = prepare_training_data(users, interactions, seed=args.seed)
    user2idx = data['user2idx']
    item2idx = data['item2idx']
    n_users = data['n_users']
    n_items = data['n_items']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = build_graph_for_model(products, data['train_data'], user2idx, item2idx)
    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=n_items + n_users + 100, n_relations=10,
        embedding_dim=args.embedding_dim, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=bool(args.use_multimodal),
        use_improvement_a=bool(args.use_a), use_improvement_b=bool(args.use_b),
        use_improvement_c=bool(args.use_c),
        graph=graph,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # compute cold subsets
    user_total = Counter()
    item_total = Counter()
    for inter in interactions:
        uid = int(inter['user_id']); iid = int(inter['product_id'])
        if uid in user2idx: user_total[user2idx[uid]] += 1
        if iid in item2idx: item_total[item2idx[iid]] += 1
    cold_users = {u for u, c in user_total.items() if c < args.cold_user_max}
    cold_items = {it for it, c in item_total.items() if c < args.cold_item_max}
    logger.info(f"cold users (<{args.cold_user_max}): {len(cold_users)} | cold items (<{args.cold_item_max}): {len(cold_items)}")

    subsets = {
        'all': (None, None),
        'cold_user': (cold_users, None),
        'cold_item': (None, cold_items),
        'cold_both': (cold_users, cold_items),
    }

    results = {}
    for name, (uf, itf) in subsets.items():
        td = build_loo_test_filtered(interactions, user2idx, item2idx,
                                     n_negative=args.n_negative, seed=args.seed,
                                     user_filter=uf, item_filter=itf)
        if len(td) == 0:
            logger.info(f"[{name}] no test samples, skip")
            results[name] = {'n_test': 0}
            continue
        r = evaluate_model(model, td, [5, 10, 20], device=device)
        r['n_test'] = len(td)
        results[name] = r
        logger.info(f"[{name}] n={len(td):5d}  HR@10={r['hr@10']:.4f} NDCG@10={r['ndcg@10']:.4f}")

    out = args.output or str(Path(args.model).parent / 'coldstart_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"saved to {out}")


if __name__ == '__main__':
    main()
