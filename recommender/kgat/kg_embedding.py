"""Knowledge Graph Embedding Module - TransR + RippleNet.

This module implements:
1. TransR: Relational projection in entity-relation specific space
2. RippleNet: Multi-hop interest propagation on knowledge graph

Key formulas:
    TransR: h_r = M_r · h, t_r = M_r · t
    RippleNet: For each hop, compute attention over neighbors and aggregate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class TransR(nn.Module):
    """TransR: Entity-Relation Specific Projection.

    Projects entities and relations into relation-specific spaces.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        entity_dim: int = 64,
        relation_dim: int = 64,
    ):
        """
        Initialize TransR.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            entity_dim: Entity embedding dimension
            relation_dim: Relation embedding dimension
        """
        super().__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        # Entity embeddings
        self.entity_embedding = nn.Embedding(n_entities, entity_dim)

        # Relation embeddings and projection matrices
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)

        # For each relation, create a projection matrix M_r: entity_dim -> relation_dim
        # We implement this as a linear layer that shares weights across entities
        self.projection = nn.Linear(entity_dim, relation_dim, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.relation_embedding.weight, mean=0, std=0.01)
        nn.init.eye_(self.projection.weight)

    def forward(
        self,
        head_entities: torch.Tensor,
        relations: torch.Tensor,
        tail_entities: torch.Tensor
    ) -> torch.Tensor:
        """
        TransR forward: h_r = M_r · h, t_r = M_r · t.

        Args:
            head_entities: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tail_entities: Tail entity indices [batch_size]

        Returns:
            Scores for (h, r, t) triplets [batch_size]
        """
        # Get embeddings
        h = self.entity_embedding(head_entities)  # [batch_size, entity_dim]
        r = self.relation_embedding(relations)   # [batch_size, relation_dim]
        t = self.entity_embedding(tail_entities) # [batch_size, entity_dim]

        # Project to relation-specific space
        # Note: In full TransR, each relation has its own M_r
        # Here we use a simplified version where M_r is a linear projection
        # shared across entities but modulated by relation embedding
        h_r = self.projection(h)  # [batch_size, relation_dim]
        t_r = self.projection(t)  # [batch_size, relation_dim]

        # Score: ||h_r + r - t_r||^2
        score = h_r + r - t_r
        score = -torch.sum(score ** 2, dim=-1)  # Negative L2 distance

        return score

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings."""
        return self.entity_embedding(entity_ids)

    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Get relation embeddings."""
        return self.relation_embedding(relation_ids)


class RippleNet(nn.Module):
    """RippleNet: Multi-hop Interest Propagation.

    Simulates user interest propagation on knowledge graph like ripples.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_hops: int = 2,
        memory_dim: int = 64,
        item_update_mode: str = "plus_transform",
        using_all_hops: bool = True,
    ):
        """
        Initialize RippleNet.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            embedding_dim: Embedding dimension
            n_hops: Number of hops (ripples) to propagate
            memory_dim: Dimension of memory (hidden state)
            item_update_mode: How to update item embeddings ('plus_transform' or 'replace')
            using_all_hops: Whether to use all hops or just the last one
        """
        super().__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.n_hops = n_hops
        self.memory_dim = memory_dim
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops

        # Entity embeddings (shared with TransR)
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)

        # Relation embeddings
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)

        # Projections for each hop
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, memory_dim, bias=False)
            for _ in range(n_hops)
        ])

        # Attention for aggregation
        self.attention = nn.Linear(memory_dim * 2, 1, bias=False)

        # Item update transformation
        if item_update_mode == "plus_transform":
            self.item_update = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.item_update = nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.relation_embedding.weight, mean=0, std=0.01)

    def get_ripples(
        self,
        user_embeddings: torch.Tensor,
        n_neighbors: int = 8
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get ripple sets for each user.

        Args:
            user_embeddings: User embeddings [batch_size, embedding_dim]
            n_neighbors: Number of neighbors to sample per hop

        Returns:
            List of ripple sets, one per hop
        """
        # This would typically query the knowledge graph
        # For simplicity, we return mock data here
        # In practice, this should query Neo4j or similar
        batch_size = user_embeddings.size(0)
        device = user_embeddings.device

        ripple_sets = []
        for hop in range(self.n_hops):
            # Mock: create random neighbors
            # In real implementation, query KG for h-hop neighbors
            n_entities_available = min(n_neighbors, self.n_entities)

            ripple_sets.append({
                'entities': torch.randint(
                    0, self.n_entities,
                    (batch_size, n_entities_available),
                    device=device
                ),
                'relations': torch.randint(
                    0, self.n_relations,
                    (batch_size, n_entities_available),
                    device=device
                ),
                'weights': torch.rand(batch_size, n_entities_available, device=device),
            })

        return ripple_sets

    def forward(
        self,
        user_ids: torch.Tensor,
        items: torch.Tensor,
        ripple_sets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        RippleNet forward pass.

        Args:
            user_ids: User indices [batch_size]
            items: Item entity indices [batch_size]
            ripple_sets: Pre-computed ripple sets (optional)

        Returns:
            Tuple of (updated_item_embeddings, hop_embeddings)
        """
        # Get embeddings
        user_emb = self.entity_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.entity_embedding(items)        # [batch_size, embedding_dim]

        # Get ripple sets if not provided
        if ripple_sets is None:
            ripple_sets = self.get_ripples(user_emb)

        # Propagate through hops
        hop_representations = []
        current_item = item_emb

        for hop in range(self.n_hops):
            ripple = ripple_sets[hop]

            # Get neighbor embeddings
            neighbor_emb = self.entity_embedding(ripple['entities'])  # [batch, n_neighbors, dim]
            relation_emb = self.relation_embedding(ripple['relations'])  # [batch, n_neighbors, dim]

            # Compute attention weights
            # attention(e_i, r_i, h) = softmax(v^T · ReLU(W · [h; e_i; r_i]))
            h_expanded = user_emb.unsqueeze(1).expand_as(neighbor_emb)  # [batch, n_neighbors, dim]

            # Compute attention scores
            concat = torch.cat([h_expanded, neighbor_emb, relation_emb], dim=-1)
            # Simplified attention
            attention_scores = torch.sum(h_expanded * neighbor_emb, dim=-1)  # [batch, n_neighbors]

            # Apply relation type importance
            relation_weights = torch.sum(relation_emb ** 2, dim=-1)  # [batch, n_neighbors]
            attention_scores = attention_scores * torch.softmax(relation_weights, dim=-1)

            # Normalize
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, n_neighbors]

            # Aggregate neighbor information
            aggregated = torch.sum(
                attention_weights.unsqueeze(-1) * neighbor_emb,
                dim=1
            )  # [batch, dim]

            # Project to memory dimension
            memory = F.relu(self.projections[hop](aggregated))  # [batch, memory_dim]

            hop_representations.append(memory)

            # Update item embedding
            if self.item_update_mode == "plus_transform":
                current_item = current_item + self.item_update(memory)
            else:
                current_item = self.item_update(memory)

        # Use all hops or just the last
        if self.using_all_hops:
            # Concatenate all hop representations
            item_representation = torch.cat(hop_representations, dim=-1)
        else:
            item_representation = hop_representations[-1]

        return item_representation, hop_representations

    def get_attention_weights(
        self,
        user_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        relation_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights for explanation (used in improvement C)."""
        # Expand user_emb ([batch, dim]) to match neighbor_emb's
        # batch-of-neighbors shape ([batch, n_neighbors, dim]) so the cat below
        # does not fail on a dim-1 mismatch (1 vs n_neighbors).
        h_expanded = user_emb.unsqueeze(1).expand_as(neighbor_emb)
        concat = torch.cat([h_expanded, neighbor_emb, relation_emb], dim=-1)

        # Simplified attention computation
        scores = torch.sum(h_expanded * neighbor_emb, dim=-1)
        return F.softmax(scores, dim=-1)


class KGEmbeddingModule(nn.Module):
    """Knowledge Graph Embedding Module combining TransR and RippleNet.

    This is the KG branch in KG-HybridRec.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_hops: int = 2,
    ):
        """
        Initialize KG Embedding Module.

        Args:
            n_entities: Number of entities
            n_relations: Number of relations
            embedding_dim: Embedding dimension
            n_hops: Number of hops for RippleNet
        """
        super().__init__()

        self.transR = TransR(n_entities, n_relations, embedding_dim, embedding_dim)
        self.rippleNet = RippleNet(
            n_entities, n_relations,
            embedding_dim=embedding_dim,
            n_hops=n_hops,
            memory_dim=embedding_dim,
        )

        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        items: torch.Tensor,
        ripple_sets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning item representation and hop attention weights."""
        return self.rippleNet(user_ids, items, ripple_sets)

    def compute_kg_loss(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        neg_tail: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute TransR embedding loss.

        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Positive tail entity indices
            neg_tail: Negative tail entity indices

        Returns:
            Tuple of (loss, metrics)
        """
        pos_scores = self.transR(head, relation, tail)
        neg_scores = self.transR(head, relation, neg_tail)

        # Margin-based ranking loss
        margin = 1.0
        loss = F.relu(margin - pos_scores + neg_scores).mean()

        metrics = {
            'kg_loss': loss.item(),
            'pos_score': pos_scores.mean().item(),
            'neg_score': neg_scores.mean().item(),
        }

        return loss, metrics

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings."""
        return self.transR.get_entity_embedding(entity_ids)

    def get_attention_for_explanation(
        self,
        user_emb: torch.Tensor,
        ripple_sets: List[Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Get attention weights for explanation path generation.

        Used by improvement C (attention-based explanation paths).

        Returns:
            List of attention weights per hop
        """
        attention_weights = []

        for hop, ripple in enumerate(ripple_sets):
            neighbor_emb = self.transR.entity_embedding(ripple['entities'])
            relation_emb = self.transR.relation_embedding(ripple['relations'])

            weights = self.rippleNet.get_attention_weights(
                user_emb, neighbor_emb, relation_emb
            )
            attention_weights.append(weights)

        return attention_weights
