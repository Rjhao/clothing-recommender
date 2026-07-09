"""Improvement B: Self-Supervised Graph Contrastive Learning.

This module implements self-supervised graph contrastive learning to alleviate
the data sparsity problem by learning from unlabeled graph structure.

Key formula:
    L_CL = -log Σ exp(sim(z_i, z_j)/τ) / Σ exp(...)
    where positive pairs are same node in two augmented views

Edge dropout rates by edge type:
    - User-item interactions: 0.1
    - Item-attribute edges: 0.2
    - Entity-entity relations: 0.3

This addresses Research Question 2: Data Sparsity Problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math

from recommender.kgat.sparse_gat import SparseGAT


class EdgeDropout(nn.Module):
    """Edge Dropout with modality-aware rates.

    Applies different dropout rates to different edge types in the knowledge graph.
    """

    def __init__(
        self,
        edge_type_rates: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Edge Dropout.

        Args:
            edge_type_rates: Dict mapping edge type names to dropout rates
        """
        super().__init__()

        # Default rates by edge type
        self.edge_type_rates = edge_type_rates or {
            'user_item': 0.1,      # User-item interactions
            'item_attribute': 0.2,  # Item-attribute edges
            'entity_relation': 0.3,  # Entity-entity relations
        }

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        edge_type_names: Optional[List[str]] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply edge dropout to adjacency.

        Args:
            edge_index: Edge indices [2, n_edges]
            edge_type: Edge type indices [n_edges] (if using type-based dropout)
            edge_type_names: List of edge type names
            training: Whether in training mode

        Returns:
            Mask tensor [n_edges] with 1 for kept edges, 0 for dropped
        """
        if not training:
            return torch.ones(edge_index.size(1), device=edge_index.device)

        n_edges = edge_index.size(1)
        mask = torch.ones(n_edges, device=edge_index.device, dtype=torch.float32)

        if edge_type is not None:
            # Apply type-specific dropout rates
            for i, rate in enumerate(self.edge_type_rates.values()):
                type_mask = (edge_type == i)
                # type_mask.sum() is a 0-dim tensor; cast to int — passing a
                # 0-dim tensor as the size arg to torch.rand is fragile across
                # PyTorch versions.
                n_typed = int(type_mask.sum().item())
                if n_typed > 0:
                    drop_mask = torch.rand(n_typed, device=edge_index.device) < rate
                    mask[type_mask] = mask[type_mask] * (~drop_mask).float()
        else:
            # Uniform random dropout
            dropout_rate = 0.2  # Default rate
            drop_mask = torch.rand(n_edges, device=edge_index.device) < dropout_rate
            mask = mask * (~drop_mask).float()

        return mask


class GraphContrastiveLoss(nn.Module):
    """InfoNCE-based Contrastive Loss for Graph.

    Implements the contrastive loss described in the paper:
    L_CL = -log Σ exp(sim(z_i, z_j)/τ) / Σ exp(...)

    With properties:
    1. Lower bound on mutual information between views
    2. Decomposes into alignment and uniformity terms
    """

    def __init__(
        self,
        temperature: float = 0.1,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize Contrastive Loss.

        Args:
            temperature: Temperature parameter for softmax
            batch_size: Batch size for negative sampling
        """
        super().__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute contrastive loss.

        Args:
            z_i: First view embeddings [batch, dim]
            z_j: Second view embeddings [batch, dim]
            mask: Optional mask for valid nodes

        Returns:
            Tuple of (loss, metrics)
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature  # [batch, batch]

        # Positive pairs are on the diagonal (z_i[i] with z_j[i]).
        pos_sim = torch.diagonal(sim_matrix)  # [batch]
        mask_diag = torch.eye(batch_size, device=device, dtype=torch.bool)

        # InfoNCE (paper eq. 3-2): the positive STAYS in the denominator.
        #   L = -log exp(pos) / (exp(pos) + Σ exp(neg)) = -pos + logsumexp(all)
        # Previously the diagonal was masked to -inf (decoupled CL, unbounded
        # below and not the paper's formula) — restored here. Symmetric form:
        # average both directions (z_i->z_j and z_j->z_i).
        loss_i2j = -pos_sim + torch.logsumexp(sim_matrix, dim=-1)
        loss_j2i = -pos_sim + torch.logsumexp(sim_matrix, dim=0)
        loss = (loss_i2j + loss_j2i) / 2

        # Apply mask if provided
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        # Metrics
        metrics = {
            'contrastive_loss': loss.item(),
            'pos_sim_mean': pos_sim.mean().item(),
            'neg_sim_mean': sim_matrix[~mask_diag].mean().item(),
        }

        # Compute alignment and uniformity (Proposition 2)
        alignment = (z_i - z_j).norm(dim=-1).pow(2).mean()
        # Uniformity: log of average pairwise Gaussian kernel
        # Approximated as: -log mean(exp(-||z_i - z_j||^2 / 2))
        pairwise_dist = torch.cdist(z_i, z_j, p=2).pow(2)
        uniformity = -torch.log(torch.exp(-pairwise_dist / 2).mean() + 1e-8)

        metrics['alignment'] = alignment.item()
        metrics['uniformity'] = uniformity.item()

        return loss, metrics


class GraphContrastiveModule(nn.Module):
    """Graph Contrastive Learning Module.

    This is Improvement B in KG-HybridRec.

    When n_total_nodes is provided, uses REAL graph contrastive learning:
    two edge-dropout augmented views of the heterogeneous graph are passed
    through a shared sparse GAT to produce two-view node embeddings, then
    InfoNCE is computed with same-node-across-views as positives.

    When n_total_nodes is None, falls back to feature-dropout approximation
    (used by older M0-M3 runs without a graph).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        temperature: float = 0.1,
        edge_type_rates: Optional[Dict[str, float]] = None,
        n_total_nodes: Optional[int] = None,
        gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Initialize Graph Contrastive Module.

        Args:
            embedding_dim: Embedding dimension
            temperature: Temperature for contrastive loss
            edge_type_rates: Dropout rates by edge type
            n_total_nodes: If given, enable real graph contrastive learning
            gat_layers / gat_heads: Sparse GAT config (real-graph mode only)
            dropout: GAT dropout
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Edge dropout with modality-aware rates
        self.edge_dropout = EdgeDropout(edge_type_rates)

        # Contrastive loss
        self.contrastive_loss = GraphContrastiveLoss(temperature=temperature)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Real graph contrastive learning (improvement B done right)
        self.use_graph = n_total_nodes is not None
        if self.use_graph:
            self.node_embedding = nn.Embedding(n_total_nodes, embedding_dim)
            self.gat = SparseGAT(
                in_dim=embedding_dim, hidden_dim=embedding_dim,
                out_dim=embedding_dim, n_heads=gat_heads,
                n_layers=gat_layers, dropout=dropout,
            )
            nn.init.normal_(self.node_embedding.weight, mean=0, std=0.01)

    def _drop_edges(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Apply type-aware edge dropout, return filtered edge_index."""
        mask = self.edge_dropout(edge_index, edge_type=edge_type, training=True)
        return edge_index[:, mask.bool()]

    def forward_graph(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_node_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Real improvement B: two edge-dropout views over the full graph.

        Args:
            edge_index: [2, E] full graph edges
            edge_type: [E] edge type per edge
            batch_node_ids: [B] node ids in the batch to compute CL over
        Returns:
            (cl_loss, metrics)
        """
        ei1 = self._drop_edges(edge_index, edge_type)
        ei2 = self._drop_edges(edge_index, edge_type)

        v1 = self.gat(self.node_embedding.weight, ei1)
        v2 = self.gat(self.node_embedding.weight, ei2)

        z_i = self.projection_head(v1[batch_node_ids])
        z_j = self.projection_head(v2[batch_node_ids])

        return self.contrastive_loss(z_i, z_j)

    def create_augmented_views(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of the graph via edge dropout.

        Args:
            features: Node features [n_nodes, dim]
            edge_index: Edge indices [2, n_edges]
            edge_type: Edge type indices [n_edges]

        Returns:
            Tuple of (view1_features, view2_features, mask)
        """
        # Get dropout masks for each view
        mask1 = self.edge_dropout(edge_index, edge_type, training=True)
        mask2 = self.edge_dropout(edge_index, edge_type, training=True)

        # In practice, you would apply these masks to the adjacency
        # Here we return the masks for the caller to apply

        return features, features, mask1  # Simplified: return same features with different masks

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass computing contrastive loss.

        Args:
            z_i: First view embeddings
            z_j: Second view embeddings
            mask: Optional mask for valid nodes

        Returns:
            Tuple of (loss, metrics)
        """
        # Project embeddings
        z_i_proj = self.projection_head(z_i)
        z_j_proj = self.projection_head(z_j)

        # Compute contrastive loss
        loss, metrics = self.contrastive_loss(z_i_proj, z_j_proj, mask)

        return loss, metrics

    def get_alignment_uniformity(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> dict:
        """
        Compute alignment and uniformity metrics for analysis.

        From Proposition 2: L_CL can be decomposed into these terms.
        """
        alignment = (z_i - z_j).norm(dim=-1).pow(2).mean().item()

        pairwise_dist = torch.cdist(z_i, z_j, p=2).pow(2)
        uniformity = -torch.log(torch.exp(-pairwise_dist / 2).mean() + 1e-8).item()

        return {
            'alignment': alignment,
            'uniformity': uniformity,
        }
