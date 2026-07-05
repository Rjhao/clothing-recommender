"""Knowledge Graph Embedding implementations."""
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.

    paper: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        norm: int = 1
    ):
        """
        Initialize TransE model.

        Args:
            num_entities: Number of unique entities
            num_relations: Number of unique relations
            embedding_dim: Embedding dimension
            margin: Margin for margin-based ranking loss
            norm: L1 or L2 norm for distance calculation
        """
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(
        self,
        positive_triplets: torch.Tensor,
        negative_triplets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss for positive and negative triplets.

        Args:
            positive_triplets: (batch_size, 3) tensor of (head, relation, tail)
            negative_triplets: (batch_size, 3) tensor of (head, relation, tail)

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        # Get embeddings
        head_pos = self.entity_embeddings(positive_triplets[:, 0])
        relation_pos = self.relation_embeddings(positive_triplets[:, 1])
        tail_pos = self.entity_embeddings(positive_triplets[:, 2])

        head_neg = self.entity_embeddings(negative_triplets[:, 0])
        relation_neg = self.relation_embeddings(negative_triplets[:, 1])
        tail_neg = self.entity_embeddings(negative_triplets[:, 2])

        # TransE scoring: h + r ≈ t
        # Distance = ||h + r - t||
        positive_scores = torch.norm(head_pos + relation_pos - tail_pos, p=self.norm, dim=1)
        negative_scores = torch.norm(head_neg + relation_neg - tail_neg, p=self.norm, dim=1)

        return positive_scores, negative_scores

    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """Get embedding for a specific entity."""
        with torch.no_grad():
            emb = self.entity_embeddings(torch.tensor(entity_id))
            return emb.cpu().numpy()

    def get_relation_embedding(self, relation_id: int) -> np.ndarray:
        """Get embedding for a specific relation."""
        with torch.no_grad():
            emb = self.relation_embeddings(torch.tensor(relation_id))
            return emb.cpu().numpy()

    def compute_similarity(
        self,
        entity1_id: int,
        entity2_id: int
    ) -> float:
        """Compute cosine similarity between two entities."""
        emb1 = self.get_entity_embedding(entity1_id)
        emb2 = self.get_entity_embedding(entity2_id)

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot_product / (norm1 * norm2 + 1e-8)


class KGEmbeddingTrainer:
    """Trainer for knowledge graph embedding models."""

    def __init__(
        self,
        model: TransE,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100
    ):
        """Initialize trainer."""
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MarginRankingLoss(margin=model.margin)

    def train(self, dataset: "KGTripletDataset") -> List[float]:
        """
        Train the model on the dataset.

        Args:
            dataset: KG triplet dataset

        Returns:
            List of loss values per epoch
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for positive_batch, negative_batch in dataloader:
                self.optimizer.zero_grad()

                pos_scores, neg_scores = self.model(positive_batch, negative_batch)

                # Margin ranking loss
                loss = self.criterion(pos_scores, neg_scores, torch.ones_like(pos_scores))

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return losses


class KGTripletDataset(Dataset):
    """Dataset for KG triplets."""

    def __init__(
        self,
        positive_triplets: List[Tuple[int, int, int]],
        entity_to_id: Dict[str, int],
        relation_to_id: Dict[str, int],
        num_negative_samples: int = 1
    ):
        """
        Initialize dataset.

        Args:
            positive_triplets: List of (head, relation, tail) triplets
            entity_to_id: Mapping from entity string to ID
            relation_to_id: Mapping from relation string to ID
            num_negative_samples: Number of negative samples per positive
        """
        self.positive_triplets = positive_triplets
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.num_negative_samples = num_negative_samples

        self.num_entities = max(entity_to_id.values()) + 1
        self.num_relations = max(relation_to_id.values()) + 1

    def __len__(self) -> int:
        return len(self.positive_triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of positive and negative triplets."""
        positive = self.positive_triplets[idx]

        # Generate negative samples by corrupting head or tail
        head, relation, tail = positive

        if np.random.random() > 0.5:
            # Corrupt head
            negative_head = np.random.randint(0, self.num_entities)
            negative = (negative_head, relation, tail)
        else:
            # Corrupt tail
            negative_tail = np.random.randint(0, self.num_entities)
            negative = (head, relation, negative_tail)

        positive_tensor = torch.LongTensor(list(positive))
        negative_tensor = torch.LongTensor(list(negative))

        return positive_tensor, negative_tensor


def build_entity_relation_mappings(triplets: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build mappings from entity/relation names to IDs."""
    entities = set()
    relations = set()

    for head, relation, tail in triplets:
        entities.add(head)
        entities.add(tail)
        relations.add(relation)

    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation_to_id = {relation: idx for idx, relation in enumerate(sorted(relations))}

    return entity_to_id, relation_to_id