"""5-seed repetition for M3b (real-graph improvement B), stage 3.7.

Trains the best model (M3b: A+B+C with real heterogeneous graph) under
5 random seeds and reports mean±std of HR@10 / NDCG@10 (paper standard).

Seed varies negative sampling + model init; the leave-one-out split is
deterministic (by interaction order), so all seeds share the same test set.

Usage:
    python -m recommender.run_seeds --seeds 42 123 456 789 2024
"""
import argparse
import json
import logging
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec, KGHybridRecTrainer
from recommender.train import load_data, prepare_training_data, sample_kg_triplets
from recommender.kgat.hetero_graph import build_graph_for_model, build_kg_triplets
from recommender.evaluate import build_loo_test, evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_one(seed, products, interactions, epochs=100, n_neg=4, lambda_kg=0.1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data = prepare_training_data(None, interactions, n_neg=n_neg, seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])
    kg_triplets_all = build_kg_triplets(products, data['item2idx'], graph['node_offsets'])
    n_entities = graph['n_total']
    n_relations = graph['n_relations']

    user_count_array = torch.zeros(data['n_users'], dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    model = KGHybridRec(
        n_users=data['n_users'], n_items=data['n_items'],
        n_entities=n_entities, n_relations=n_relations,
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=False,
        use_improvement_a=True, use_improvement_b=True, use_improvement_c=True,
        lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=lambda_kg,
        graph=graph, n_interactions=user_count_array,
    ).to(device)

    trainer = KGHybridRecTrainer(model, learning_rate=0.001, weight_decay=1e-4,
                                 lambda_cl=0.1, lambda_reg=1e-4)
    out_dir = Path('checkpoints') / f'm3b_seed{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    for epoch in range(epochs):
        td = data['train_data']; np.random.shuffle(td)
        for i in range(0, len(td), 256):
            batch = td[i:i+256]
            u = torch.LongTensor([d[0][0] for d in batch]).to(device)
            p = torch.LongTensor([d[0][1] for d in batch]).to(device)
            n = torch.LongTensor([d[1] for d in batch]).to(device)  # [B, n_neg]
            c = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)
            kg = sample_kg_triplets(kg_triplets_all, len(batch), n_entities, device)
            trainer.train_epoch(u, p, n, interaction_counts=c, kg_triplets=kg)
        # val
        vlosses = []
        for i in range(0, len(data['val_data']), 256):
            batch = data['val_data'][i:i+256]
            u = torch.LongTensor([d[0][0] for d in batch]).to(device)
            p = torch.LongTensor([d[0][1] for d in batch]).to(device)
            n = torch.LongTensor([d[1] for d in batch]).to(device)
            c = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)
            m = trainer.evaluate(u, p, n, interaction_counts=c)
            vlosses.append(m['total_loss'])
        avg_v = float(np.mean(vlosses))
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        trainer.scheduler.step(avg_v)
        if (epoch+1) % 25 == 0:
            logger.info(f"[seed {seed}] epoch {epoch+1}/{epochs} val={avg_v:.4f}")

    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=device))
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'], 100, seed)
    results = evaluate_model(model, test_data, [5, 10, 20], device=str(device),
                             user_count=data['user_count'])
    logger.info(f"[seed {seed}] HR@10={results['hr@10']:.4f} NDCG@10={results['ndcg@10']:.4f} best_val={best_val:.4f}")
    return results, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 2024])
    ap.add_argument('--data', default='data/amazon_fashion_real.json')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--n-neg', type=int, default=4)
    ap.add_argument('--lambda-kg', type=float, default=0.1)
    ap.add_argument('--output', default='experiments/results/m3b_5seeds.json')
    args = ap.parse_args()

    products, users, interactions = load_data(args.data)
    all_results = {}
    for seed in args.seeds:
        t0 = time.time()
        r, bv = run_one(seed, products, interactions, epochs=args.epochs,
                        n_neg=args.n_neg, lambda_kg=args.lambda_kg)
        all_results[seed] = {'metrics': r, 'best_val_loss': bv, 'time': round(time.time()-t0)}

    # aggregate mean±std
    hr10 = [all_results[s]['metrics']['hr@10'] for s in args.seeds]
    ndcg10 = [all_results[s]['metrics']['ndcg@10'] for s in args.seeds]
    hr5 = [all_results[s]['metrics']['hr@5'] for s in args.seeds]
    summary = {
        'seeds': args.seeds,
        'per_seed': all_results,
        'hr@10_mean': float(np.mean(hr10)), 'hr@10_std': float(np.std(hr10)),
        'ndcg@10_mean': float(np.mean(ndcg10)), 'ndcg@10_std': float(np.std(ndcg10)),
        'hr@5_mean': float(np.mean(hr5)), 'hr@5_std': float(np.std(hr5)),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved to {out}")
    logger.info(f"HR@10 = {summary['hr@10_mean']:.4f} ± {summary['hr@10_std']:.4f}")
    logger.info(f"NDCG@10 = {summary['ndcg@10_mean']:.4f} ± {summary['ndcg@10_std']:.4f}")


if __name__ == '__main__':
    main()
