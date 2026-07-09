"""KG-HybridRec ablation runner (stage 2).

Runs a set of improvement-switch configurations, training + sampled-100
evaluating each, and aggregates results into one JSON.

Configs (improvement switches A=adaptive_gate, B=graph_contrastive,
C=attention_explain; multimodal always off in main experiments):
    M0    A0 B0 C0   (baseline, already trained — skipped here)
    M1    A1 B0 C0   (+A)
    M2    A1 B1 C0   (+A+B)
    M3    A1 B1 C1   (+A+B+C, full model)
    M3-A  A0 B1 C1   (leave-one-out: drop A)
    M3-B  A1 B0 C1   (leave-one-out: drop B)

Note: C (attention_explain) only constructs its module; it does NOT
participate in forward/predict_score, so M2 and M3 produce identical HR.
This is by design (C is interpretability, not accuracy) and is reported
as-is.

Usage:
    python -m recommender.run_ablation            # run M1,M2,M3,M3-A,M3-B
    python -m recommender.run_ablation --only m1  # run a single config
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec, KGHybridRecTrainer
from recommender.train import load_data, prepare_training_data, sample_kg_triplets
from recommender.kgat.hetero_graph import build_graph_for_model, build_kg_triplets
from recommender.evaluate import build_loo_test, evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# (tag, use_a, use_b, use_c)
CONFIGS = [
    ('m1',   1, 0, 0),
    ('m2',   1, 1, 0),
    ('m3',   1, 1, 1),
    ('m3-a', 0, 1, 1),
    ('m3-b', 1, 0, 1),
]

DEFAULT_ARGS = dict(
    embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
    epochs=100, batch_size=256, lr=0.001, weight_decay=1e-4,
    lambda_cl=0.1, lambda_reg=1e-4, lambda_kg=0.1, n_neg=4, seed=42,
)


def train_one(tag, use_a, use_b, use_c, products, interactions, args):
    """Train one config, return (model, history_dict, data_dict, device)."""
    data = prepare_training_data(None, interactions, n_neg=args['n_neg'], seed=args['seed'])
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
        embedding_dim=args['embedding_dim'], n_hops=args['n_hops'],
        n_heads=args['n_heads'], n_layers=args['n_layers'],
        use_multimodal=False,
        use_improvement_a=bool(use_a), use_improvement_b=bool(use_b),
        use_improvement_c=bool(use_c),
        lambda_cl=args['lambda_cl'], lambda_reg=args['lambda_reg'],
        lambda_kg=args['lambda_kg'],
        graph=graph, n_interactions=user_count_array,
    ).to(device)

    trainer = KGHybridRecTrainer(model, args['lr'], args['weight_decay'],
                                 args['lambda_cl'], args['lambda_reg'])

    output_dir = Path('checkpoints') / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    train_history, val_history = [], []
    bs = args['batch_size']

    for epoch in range(args['epochs']):
        train_data = data['train_data']
        np.random.shuffle(train_data)

        train_losses = []
        for i in range(0, len(train_data), bs):
            batch = train_data[i:i + bs]
            user_ids = torch.LongTensor([d[0][0] for d in batch]).to(device)
            pos_ids = torch.LongTensor([d[0][1] for d in batch]).to(device)
            neg_ids = torch.LongTensor([d[1] for d in batch]).to(device)  # [B, n_neg]
            counts = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)
            kg_triplets = sample_kg_triplets(kg_triplets_all, len(batch), n_entities, device)
            m = trainer.train_epoch(user_ids, pos_ids, neg_ids,
                                    interaction_counts=counts, kg_triplets=kg_triplets)
            train_losses.append(m['total_loss'])
        avg_train = float(np.mean(train_losses))

        val_losses = []
        for i in range(0, len(data['val_data']), bs):
            batch = data['val_data'][i:i + bs]
            user_ids = torch.LongTensor([d[0][0] for d in batch]).to(device)
            pos_ids = torch.LongTensor([d[0][1] for d in batch]).to(device)
            neg_ids = torch.LongTensor([d[1] for d in batch]).to(device)
            counts = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)
            m = trainer.evaluate(user_ids, pos_ids, neg_ids, interaction_counts=counts)
            val_losses.append(m['total_loss'])
        avg_val = float(np.mean(val_losses))

        train_history.append(avg_train)
        val_history.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_dir / 'best_model.pt')

        trainer.scheduler.step(avg_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"[{tag}] epoch {epoch+1}/{args['epochs']} train={avg_train:.4f} val={avg_val:.4f}")

    history = {'train_loss': train_history, 'val_loss': val_history,
               'best_val_loss': best_val_loss,
               'switches': {'a': use_a, 'b': use_b, 'c': use_c}}
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"[{tag}] done, best_val_loss={best_val_loss:.4f}")

    # reload best for evaluation
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    return model, history, data, device


def evaluate_one(model, interactions, data, device, n_negative=100, seed=42):
    test_data = build_loo_test(interactions, data['user2idx'], data['item2idx'],
                               n_negative=n_negative, seed=seed)
    results = evaluate_model(model, test_data, [5, 10, 20], device=str(device),
                             user_count=data['user_count'])
    return results, len(test_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default=None, help='run single config tag')
    ap.add_argument('--data', type=str, default='data/amazon_fashion_expanded.json')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--n-negative', type=int, default=100)
    ap.add_argument('--output', type=str, default='experiments/results/ablation_summary.json')
    a = ap.parse_args()

    args = dict(DEFAULT_ARGS)
    args['epochs'] = a.epochs

    products, users, interactions = load_data(a.data)

    configs = [c for c in CONFIGS if (a.only is None or c[0] == a.only)]
    summary = {}
    for tag, use_a, use_b, use_c in configs:
        logger.info(f"==== TRAIN {tag} (A={use_a} B={use_b} C={use_c}) ====")
        t0 = time.time()
        model, history, data, device = train_one(tag, use_a, use_b, use_c, products, interactions, args)
        logger.info(f"==== EVAL {tag} ====")
        results, n_test = evaluate_one(model, interactions, data, device, a.n_negative)
        logger.info(f"[{tag}] HR@10={results['hr@10']:.4f} NDCG@10={results['ndcg@10']:.4f} (n_test={n_test}, {time.time()-t0:.0f}s)")
        summary[tag] = {
            'switches': {'a': use_a, 'b': use_b, 'c': use_c},
            'best_val_loss': history['best_val_loss'],
            'n_test_users': n_test,
            'metrics': results,
        }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # merge with existing (e.g. M0) if present
    if out.exists():
        existing = json.load(open(out))
        existing.update(summary)
        summary = existing
    summary['_meta'] = {'protocol': 'sampled-100 leave-one-out', 'device': str(device),
                        'args': args, 'note': 'C does not affect HR (explanations only); M2==M3 HR'}
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary saved to {out}")
    for tag, r in summary.items():
        if tag.startswith('_'):
            continue
        m = r['metrics']
        logger.info(f"  {tag:6s} A={r['switches']['a']} B={r['switches']['b']} C={r['switches']['c']}  HR@10={m['hr@10']:.4f} NDCG@10={m['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()
