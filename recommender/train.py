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
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda-cl', type=float, default=0.1)
    parser.add_argument('--lambda-reg', type=float, default=1e-4)

    # Data
    parser.add_argument('--data', type=str, default='data/amazon_fashion_processed.json')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--save-every', type=int, default=10)

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


def prepare_training_data(users, interactions, train_ratio=0.7, val_ratio=0.1):
    """Prepare training and validation data."""
    # Create user and item ID mappings
    user_ids = sorted(set(int(i['user_id']) for i in interactions))
    item_ids = sorted(set(int(i['product_id']) for i in interactions))

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}
    idx2item = {idx: iid for iid, idx in item2idx.items()}

    # Map interactions to indices
    interaction_list = []
    for interaction in interactions:
        uid = int(interaction['user_id'])
        iid = int(interaction['product_id'])
        if uid in user2idx and iid in item2idx:
            interaction_list.append({
                'user_idx': user2idx[uid],
                'item_idx': item2idx[iid],
                'count': interaction.get('interaction_count', 1),
            })

    logger.info(f"Valid interactions: {len(interaction_list)}")

    # Create positive interactions
    pos_data = [(d['user_idx'], d['item_idx'], d['count']) for d in interaction_list]

    # Create negative samples (random items not in user's interactions)
    n_users = len(user_ids)
    n_items = len(item_ids)

    neg_data = []
    user_interacted = {}
    for d in interaction_list:
        uid = d['user_idx']
        user_interacted.setdefault(uid, set()).add(d['item_idx'])

    for d in interaction_list:
        uid = d['user_idx']
        # Sample negative item
        neg_item = np.random.randint(0, n_items)
        while neg_item in user_interacted.get(uid, set()):
            neg_item = np.random.randint(0, n_items)
        neg_data.append((uid, neg_item, d['count']))

    # Split train/val/test
    n_samples = len(pos_data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = [pos_data[i] for i in train_idx]
    train_neg_data = [neg_data[i] for i in train_idx]
    val_data = [pos_data[i] for i in val_idx]
    val_neg_data = [neg_data[i] for i in val_idx]
    test_data = [pos_data[i] for i in test_idx]
    test_neg_data = [neg_data[i] for i in test_idx]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_entities': n_items + n_users + 100,  # Approximate
        'n_relations': 10,
        'train_data': list(zip(train_data, train_neg_data)),
        'val_data': list(zip(val_data, val_neg_data)),
        'test_data': list(zip(test_data, test_neg_data)),
        'user2idx': user2idx,
        'item2idx': item2idx,
    }


def train(args):
    """Main training function."""
    # Load data
    products, users, interactions = load_data(args.data)

    # Prepare training data
    data = prepare_training_data(
        users, interactions,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Create model
    logger.info("Creating KG-HybridRec model...")
    model = KGHybridRec(
        n_users=data['n_users'],
        n_items=data['n_items'],
        n_entities=data['n_entities'],
        n_relations=data['n_relations'],
        embedding_dim=args.embedding_dim,
        n_hops=args.n_hops,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_multimodal=False,  # Disable for now (needs images/text)
        use_improvement_a=True,
        use_improvement_b=True,
        use_improvement_c=True,
        lambda_cl=args.lambda_cl,
        lambda_reg=args.lambda_reg,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = KGHybridRecTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_cl=args.lambda_cl,
        lambda_reg=args.lambda_reg,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
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

            user_ids = torch.LongTensor([d[0][0] for d in batch])
            pos_ids = torch.LongTensor([d[0][1] for d in batch])
            neg_ids = torch.LongTensor([d[1][1] for d in batch])
            counts = torch.LongTensor([d[0][2] for d in batch])

            metrics = trainer.train_epoch(
                user_ids=user_ids,
                pos_item_ids=pos_ids,
                neg_item_ids=neg_ids,
                interaction_counts=counts,
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

            user_ids = torch.LongTensor([d[0][0] for d in batch])
            pos_ids = torch.LongTensor([d[0][1] for d in batch])
            neg_ids = torch.LongTensor([d[1][1] for d in batch])
            counts = torch.LongTensor([d[0][2] for d in batch])

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
