"""Collaborative Filtering Module - BPR Matrix Factorization.

This module implements Bayesian Personalized Ranking (BPR) for collaborative filtering,
which is the foundation of the CF branch in KG-HybridRec.

Key formula:
    L_BPR = -Σ log σ(x_ui - x_uj)
    x_ui = h_u^T · v_i
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization.

    Learns user and item embeddings from implicit feedback data.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        user_dropout: float = 0.2,
        item_dropout: float = 0.2,
    ):
        """
        Initialize BPR-MF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            user_dropout: Dropout rate for user embeddings
            item_dropout: Dropout rate for item embeddings
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Dropout
        self.user_dropout = nn.Dropout(user_dropout)
        self.item_dropout = nn.Dropout(item_dropout)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for BPR-MF.

        Args:
            user_ids: User indices [batch_size]
            pos_item_ids: Positive item indices [batch_size]
            neg_item_ids: Negative item indices [batch_size]

        Returns:
            Tuple of (positive_scores, negative_scores, user_embeddings)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        pos_item_emb = self.item_embedding(pos_item_ids)
        neg_item_emb = self.item_embedding(neg_item_ids)

        # Apply dropout
        if self.training:
            user_emb = self.user_dropout(user_emb)
            pos_item_emb = self.item_dropout(pos_item_emb)
            neg_item_emb = self.item_dropout(neg_item_emb)

        # Get biases
        user_b = self.user_bias(user_ids).squeeze(-1)  # [batch_size]
        pos_item_b = self.item_bias(pos_item_ids).squeeze(-1)
        neg_item_b = self.item_bias(neg_item_ids).squeeze(-1)

        # Compute scores: x_ui = h_u^T · v_i + b_u + b_i + b_global
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1) + user_b + pos_item_b + self.global_bias
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1) + user_b + neg_item_b + self.global_bias

        return pos_scores, neg_scores, user_emb

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for downstream tasks."""
        return self.user_embedding(user_ids)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings for downstream tasks."""
        return self.item_embedding(item_ids)

    def compute_bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute BPR loss.

        L_BPR = -Σ log σ(x_ui - x_uj)

        Args:
            user_ids: User indices
            pos_item_ids: Positive item indices
            neg_item_ids: Negative item indices

        Returns:
            Tuple of (loss, metrics_dict)
        """
        pos_scores, neg_scores, _ = self.forward(user_ids, pos_item_ids, neg_item_ids)

        # BPR loss: -log σ(pos_score - neg_score)
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        # Metrics
        metrics = {
            'bpr_loss': loss.item(),
            'pos_score_mean': pos_scores.mean().item(),
            'neg_score_mean': neg_scores.mean().item(),
            'diff_mean': diff.mean().item(),
            'auc': (diff > 0).float().mean().item(),  # AUC approximation
        }

        return loss, metrics


class CFModule(nn.Module):
    """Collaborative Filtering Module wrapper that combines BPR with interactions.

    This is the CF branch in KG-HybridRec that learns user-item interactions.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_interactions: Optional[torch.Tensor] = None,
    ):
        """
        Initialize CF Module.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            n_interactions: Optional tensor of interaction counts per user for confidence
        """
        super().__init__()

        self.bpr_mf = BPRMF(n_users, n_items, embedding_dim)
        self.embedding_dim = embedding_dim

        # Interaction counts for confidence estimation (used in adaptive gate)
        if n_interactions is not None:
            self.register_buffer('interaction_counts', n_interactions.float())
        else:
            self.interaction_counts = None

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass returning loss and metrics."""
        return self.bpr_mf.compute_bpr_loss(user_ids, pos_item_ids, neg_item_ids)

    def get_user_representation(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user representation vector."""
        return self.bpr_mf.get_user_embedding(user_ids)

    def get_item_representation(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item representation vector."""
        return self.bpr_mf.get_item_embedding(item_ids)

    def get_confidence(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Get confidence score for each user based on interaction count.

        This is used by the adaptive gate to determine how much to trust CF signals.

        Args:
            user_ids: User indices

        Returns:
            Confidence scores [batch_size] in range [0, 1]
        """
        if self.interaction_counts is None:
            # Uniform confidence if no interaction data
            return torch.ones_like(user_ids, dtype=torch.float32)

        # Normalize interaction counts to [0, 1]
        counts = self.interaction_counts[user_ids]
        # Apply some normalization (e.g., tanh to bound values)
        confidence = torch.tanh(counts / 10.0)  # Assume 10 interactions is "high confidence"
        return confidence
