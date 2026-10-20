"""Training script for KG-HybridRec.

Usage:
    python -m recommender.train --epochs 100 --batch_size 256 --lr 0.001
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.hybrid_rec import KGHybridRec, KGHybridRecTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """Dataset for recommendation training."""

    def __init__(
        self,
        user_ids,
        pos_item_ids,
        neg_item_ids,
        interaction_counts=None,
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.pos_item_ids = torch.LongTensor(pos_item_ids)
        self.neg_item_ids = torch.LongTensor(neg_item_ids)
        self.interaction_counts = (
            torch.LongTensor(interaction_counts) if interaction_counts is not None else None
        )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        item = {
            'user_ids': self.user_ids[idx],
            'pos_item_ids': self.pos_item_ids[idx],
            'neg_item_ids': self.neg_item_ids[idx],
        }
        if self.interaction_counts is not None:
            item['interaction_counts'] = self.interaction_counts[idx]
        return item


def parse_args():
    parser = argparse.ArgumentParser(description='Train KG-HybridRec')

    # Model
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--n-hops', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Uniform L2 on all params (legacy). Used when '
                             '--cf-weight-decay/--kg-weight-decay/'
                             '--other-weight-decay are all unset.')
    parser.add_argument('--cf-weight-decay', type=float, default=None,
                        help='Per-group L2 on cf_module.bpr_mf (user/item '
                             'embeddings). Default: same as --weight-decay.')
    parser.add_argument('--kg-weight-decay', type=float, default=None,
                        help='Per-group L2 on kg_module (KG + TransR + '
                             'RippleNet). Default: same as --weight-decay.')
    parser.add_argument('--other-weight-decay', type=float, default=None,
                        help='Per-group L2 on everything else (gates, GAT, '
                             'contrastive, predict MLP, multimodal). '
                             'Default: same as --weight-decay. Set 0 to '
                             'redirect L2 from non-embedding modules onto '
                             'cf/kg only (popular-bias fix).')
    parser.add_argument('--lambda-cl', type=float, default=0.1)
    parser.add_argument('--lambda-reg', type=float, default=1e-4)
    parser.add_argument('--lambda-kg', type=float, default=0.1, help='TransR kg_loss weight')
    parser.add_argument('--n-neg', type=int, default=4, help='negatives per positive (paper 1:4)')

    # Data
    parser.add_argument('--data', type=str, default='data/amazon_fashion_expanded.json')
    parser.add_argument('--seed', type=int, default=42)

    # Improvements switches (for ablation)
    parser.add_argument('--use-a', type=int, default=1, help='adaptive gate')
    parser.add_argument('--use-b', type=int, default=1, help='graph contrastive')
    parser.add_argument('--use-c', type=int, default=1, help='attention explain')
    parser.add_argument('--use-multimodal', type=int, default=0)

    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--tag', type=str, default='full', help='run tag for output subdir')

    return parser.parse_args()


def load_data(data_path):
    """Load processed Amazon Fashion data."""
    logger.info(f"Loading data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    products = data['products']
    users = data['users']
    interactions = data['interactions']

    logger.info(f"Loaded {len(products)} products, {len(users)} users, {len(interactions)} interactions")

    return products, users, interactions


def prepare_training_data(users, interactions, n_neg=4, seed=42):
    """Prepare training data with unified leave-one-out split + 1:n_neg sampling.

    Uses recommender.split.leave_one_out_split so train and eval agree on which
    item is test (fixes the 2-interaction val/test collision). Negative sampling
    excludes the user's FULL interaction set (incl. test item) so test items are
    never trained as negatives. Per-user counts are REAL (for the adaptive gate).
    """
    from recommender.split import (
        leave_one_out_split, build_user_interacted_set, build_user_train_count
    )
    rng = np.random.RandomState(seed)

    user_ids = sorted(set(int(i['user_id']) for i in interactions))
    item_ids = sorted(set(int(i['product_id']) for i in interactions))
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    train_pairs, val_pairs, test_pairs = leave_one_out_split(
        interactions, user2idx, item2idx
    )
    user_items = build_user_interacted_set(interactions, user2idx, item2idx)
    user_count = build_user_train_count(interactions, user2idx, item2idx)

    n_users = len(user_ids)
    n_items = len(item_ids)

    def make_neg(pairs):
        """For each (uid, item) sample n_neg negatives excluding user's full set."""
        out = []
        for uid_idx, iid in pairs:
            excluded = user_items.get(uid_idx, set())
            negs = []
            while len(negs) < n_neg:
                cand = int(rng.randint(0, n_items))
                if cand not in excluded:
                    negs.append(cand)
            out.append(negs)
        return out

    train_neg = make_neg(train_pairs)
    val_neg = make_neg(val_pairs)

    # train_data / val_data: list of (pos, neg) where pos=(uid,item), neg=[n_neg items]
    train_data = [((u, i), neg) for (u, i), neg in zip(train_pairs, train_neg)]
    val_data = [((u, i), neg) for (u, i), neg in zip(val_pairs, val_neg)]

    logger.info(f"Users: {n_users}, Items: {n_items}")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_pairs)}, n_neg: {n_neg}")

    return {
        'n_users': n_users,
        'n_items': n_items,
        'train_data': train_data,
        'val_data': val_data,
        'test_pairs': test_pairs,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'user_count': user_count,  # {uid_idx: count} — real per-user counts
    }


def sample_kg_triplets(kg_triplets_all, batch_size, n_entities, device):
    """Sample a batch of (head, rel, tail, neg_tail) for TransR kg_loss."""
    if len(kg_triplets_all) == 0:
        return None
    tidx = np.random.randint(0, len(kg_triplets_all), size=batch_size)
    t = kg_triplets_all[tidx]
    head = t[:, 0].to(device)
    rel = t[:, 1].to(device)
    tail = t[:, 2].to(device)
    neg_tail = torch.randint(0, n_entities, (batch_size,), device=device)
    return (head, rel, tail, neg_tail)


def train(args):
    """Main training function."""
    # Reproducibility: seed python/numpy/torch (CPU + CUDA) + deterministic cudnn.
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    products, users, interactions = load_data(args.data)

    # Prepare training data (unified LOO split + 1:n_neg sampling + real counts)
    data = prepare_training_data(
        users, interactions,
        n_neg=args.n_neg,
        seed=args.seed,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build heterogeneous graph (used by KG branch + improvement B)
    from recommender.kgat.hetero_graph import build_graph_for_model, build_kg_triplets
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])
    logger.info(f"Built hetero graph: {graph['n_total']} nodes, {graph['edge_index'].size(1)} edges, {graph['n_relations']} relations")

    # KG triplets for TransR kg_loss
    kg_triplets_all = build_kg_triplets(products, data['item2idx'], graph['node_offsets'])
    logger.info(f"Built {len(kg_triplets_all)} KG triplets for TransR")

    # Per-user interaction count tensor (real values, not the old label=1)
    user_count_array = torch.zeros(data['n_users'], dtype=torch.float32)
    for uid, cnt in data['user_count'].items():
        user_count_array[uid] = cnt

    n_entities = graph['n_total']
    n_relations = graph['n_relations']

    # Create model
    logger.info("Creating KG-HybridRec model...")
    logger.info(f"Improvements: A={bool(args.use_a)} B={bool(args.use_b)} C={bool(args.use_c)} MM={bool(args.use_multimodal)}")
    model = KGHybridRec(
        n_users=data['n_users'],
        n_items=data['n_items'],
        n_entities=n_entities,
        n_relations=n_relations,
        embedding_dim=args.embedding_dim,
        n_hops=args.n_hops,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_multimodal=bool(args.use_multimodal),
        use_improvement_a=bool(args.use_a),
        use_improvement_b=bool(args.use_b),
        use_improvement_c=bool(args.use_c),
        lambda_cl=args.lambda_cl,
        lambda_reg=args.lambda_reg,
        lambda_kg=args.lambda_kg,
        graph=graph,
        n_interactions=user_count_array,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = KGHybridRecTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_cl=args.lambda_cl,
        lambda_reg=args.lambda_reg,
        cf_weight_decay=args.cf_weight_decay,
        kg_weight_decay=args.kg_weight_decay,
        other_weight_decay=args.other_weight_decay,
    )

    # Create output directory
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    train_history = []
    val_history = []

    for epoch in range(args.epochs):
        # Train
        train_losses = []
        train_metrics = []

        # Shuffle training data
        train_data = data['train_data']
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]

            user_ids = torch.LongTensor([d[0][0] for d in batch]).to(device)
            pos_ids = torch.LongTensor([d[0][1] for d in batch]).to(device)
            neg_ids = torch.LongTensor([d[1] for d in batch]).to(device)  # [B, n_neg]
            counts = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)

            kg_triplets = sample_kg_triplets(kg_triplets_all, len(batch), n_entities, device)

            metrics = trainer.train_epoch(
                user_ids=user_ids,
                pos_item_ids=pos_ids,
                neg_item_ids=neg_ids,
                interaction_counts=counts,
                kg_triplets=kg_triplets,
            )

            train_losses.append(metrics['total_loss'])
            train_metrics.append(metrics)

        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {
            k: np.mean([m[k] for m in train_metrics])
            for k in train_metrics[0].keys()
        }

        # Validate
        val_losses = []
        val_metrics = []

        for i in range(0, len(data['val_data']), args.batch_size):
            batch = data['val_data'][i:i + args.batch_size]

            user_ids = torch.LongTensor([d[0][0] for d in batch]).to(device)
            pos_ids = torch.LongTensor([d[0][1] for d in batch]).to(device)
            neg_ids = torch.LongTensor([d[1] for d in batch]).to(device)  # [B, n_neg]
            counts = torch.LongTensor([data['user_count'][d[0][0]] for d in batch]).to(device)

            metrics = trainer.evaluate(
                user_ids=user_ids,
                pos_item_ids=pos_ids,
                neg_item_ids=neg_ids,
                interaction_counts=counts,
            )

            val_losses.append(metrics['total_loss'])
            val_metrics.append(metrics)

        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {
            k: np.mean([m[k] for m in val_metrics])
            for k in val_metrics[0].keys()
        }

        train_history.append(avg_train_metrics)
        val_history.append(avg_val_metrics)

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        logger.info(
            f"  Train BPR: {avg_train_metrics.get('bpr_loss', 0):.4f}, "
            f"Val BPR: {avg_val_metrics.get('bpr_loss', 0):.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Update scheduler
        trainer.scheduler.step(avg_val_loss)

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')

    # Save training history
    history = {
        'train': train_history,
        'val': val_history,
        'args': vars(args),
        'best_val_loss': best_val_loss,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to {output_dir}")

    return model, history


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
