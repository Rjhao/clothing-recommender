"""Graph Attention Network (GAT) Fusion Module.

This module implements the GAT-based fusion layer that combines:
- Collaborative Filtering features
- Knowledge Graph features
- Multimodal features

Key formula:
    α_ij = softmax(LeakyReLU(a^T · [W·h_i || W·h_j]))
    h_i' = σ(Σ α_ij · W·h_j)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer.

    Implements single-head attention for graph neural networks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        """
        Initialize Graph Attention Layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU slope
            concat: Whether to concatenate heads or average
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear projection
        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)

        # Attention mechanism: a^T · [Wh_i || Wh_j]
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        # Activation
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Output projection (for residual)
        if in_features != n_heads * out_features:
            self.residual_proj = nn.Linear(in_features, n_heads * out_features)
        else:
            self.residual_proj = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            features: Node features [n_nodes, in_features]
            adj: Adjacency matrix [n_nodes, n_nodes] (binary or weighted)
            return_attention: Whether to return attention coefficients

        Returns:
            Tuple of (output features, attention coefficients if return_attention=True)
        """
        n_nodes = features.size(0)

        # Add self-loops so every node has at least itself as a neighbor;
        # otherwise isolated nodes produce all-(-inf) rows after masking, and
        # the subsequent softmax yields NaN for those rows.
        adj = adj + torch.eye(n_nodes, device=adj.device)

        # Project to multi-head space
        Wh = self.W(features)  # [n_nodes, n_heads * out_features]
        Wh = Wh.view(n_nodes, self.n_heads, self.out_features)  # [n_nodes, n_heads, out_features]
        Wh = Wh.transpose(0, 1)  # [n_heads, n_nodes, out_features]

        # Compute attention scores
        # a^T · [Wh_i || Wh_j] for all pairs
        a_input = self._prepare_attention_input(Wh)  # [n_heads, n_nodes, n_nodes, 2*out_features]

        e = self.a(a_input).squeeze(-1)  # [n_heads, n_nodes, n_nodes]

        # LeakyReLU activation
        e = self.leaky_relu(e)

        # Mask attention scores on non-edges
        attention_mask = adj.unsqueeze(0).expand_as(e)  # [n_heads, n_nodes, n_nodes]
        e = e.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax to get attention coefficients
        attention = F.softmax(e, dim=-1)  # [n_heads, n_nodes, n_nodes]
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to values
        # attention: [n_heads, n_nodes, n_nodes], Wh: [n_heads, n_nodes, out_features]
        # (previously this transposed Wh a second time, causing a shape mismatch)
        h_prime = torch.matmul(attention, Wh)  # [n_heads, n_nodes, out_features]

        # Concatenate or average heads
        if self.concat:
            h_prime = h_prime.transpose(0, 1).contiguous()
            h_prime = h_prime.view(n_nodes, self.n_heads * self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)  # [n_nodes, out_features]

        # Residual connection
        if self.residual_proj is not None:
            h_prime = h_prime + self.residual_proj(features)
        else:
            h_prime = h_prime + features

        if return_attention:
            # Return average attention across heads
            return h_prime, attention.mean(dim=0)
        return h_prime, None

    def _prepare_attention_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare attention input for all node pairs.

        Args:
            Wh: [n_heads, n_nodes, out_features]

        Returns:
            Concatenated [Wh_i || Wh_j] for all pairs
        """
        n_heads, n_nodes, out_features = Wh.shape

        # Wh_i (self)
        Wh_i = Wh.unsqueeze(2)  # [n_heads, n_nodes, 1, out_features]
        Wh_i = Wh_i.expand(n_heads, n_nodes, n_nodes, out_features)  # [n_heads, n_nodes, n_nodes, out_features]

        # Wh_j (neighbors)
        Wh_j = Wh.unsqueeze(1)  # [n_heads, 1, n_nodes, out_features]
        Wh_j = Wh_j.expand(n_heads, n_nodes, n_nodes, out_features)  # [n_heads, n_nodes, n_nodes, out_features]

        # Concatenate
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # [n_heads, n_nodes, n_nodes, 2*out_features]

        return a_input


class GATFusion(nn.Module):
    """Graph Attention Network Fusion Module.

    Fuses CF, KG, and MM features using graph attention.
    """

    def __init__(
        self,
        cf_dim: int = 64,
        kg_dim: int = 64,
        mm_dim: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize GAT Fusion.

        Args:
            cf_dim: CF feature dimension
            kg_dim: KG feature dimension
            mm_dim: MM feature dimension
            hidden_dim: Hidden dimension for fusion
            n_heads: Number of attention heads
            n_layers: Number of GAT layers
            dropout: Dropout rate
        """
        super().__init__()

        self.cf_dim = cf_dim
        self.kg_dim = kg_dim
        self.mm_dim = mm_dim
        self.hidden_dim = hidden_dim

        # Input dimension is sum of all features
        total_dim = cf_dim + kg_dim + mm_dim

        # Projection to hidden dimension
        self.input_proj = nn.Linear(total_dim, hidden_dim)

        # Multiple GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim // n_heads,
                n_heads=n_heads,
                dropout=dropout,
                concat=True,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            cf_features: CF embeddings [n_nodes, cf_dim]
            kg_features: KG embeddings [n_nodes, kg_dim]
            mm_features: MM embeddings [n_nodes, mm_dim]
            adj: Adjacency matrix [n_nodes, n_nodes]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (fused_features, attention_weights if return_attention=True)
        """
        # Concatenate all features
        all_features = torch.cat([cf_features, kg_features, mm_features], dim=-1)

        # Project to hidden dimension
        h = self.input_proj(all_features)
        h = self.dropout(h)
        h = F.elu(h)

        # Apply GAT layers
        attention_weights = [] if return_attention else None

        for layer in self.gat_layers:
            h, attn = layer(h, adj, return_attention=True)
            h = F.elu(h)
            h = self.dropout(h)

            if return_attention and attn is not None:
                attention_weights.append(attn)

        # Output projection and residual
        h = self.output_proj(h)
        h = self.layer_norm(h + self.input_proj(all_features))  # Residual

        return h, attention_weights


class GATFusionModule(nn.Module):
    """GAT Fusion Module wrapper for KG-HybridRec.

    Combines CF, KG, and MM branches with attention.
    """

    def __init__(
        self,
        cf_dim: int = 64,
        kg_dim: int = 64,
        mm_dim: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize GAT Fusion Module.
        """
        super().__init__()

        self.gat_fusion = GATFusion(
            cf_dim=cf_dim,
            kg_dim=kg_dim,
            mm_dim=mm_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            cf_features: CF embeddings [batch, cf_dim]
            kg_features: KG embeddings [batch, kg_dim]
            mm_features: MM embeddings [batch, mm_dim]
            adj: Adjacency matrix
            return_attention: Return attention weights for explanation

        Returns:
            Tuple of (fused_features, attention_weights)
        """
        return self.gat_fusion(
            cf_features, kg_features, mm_features,
            adj, return_attention
        )

    def get_attention_for_explanation(
        self,
        attention_weights: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate attention weights for explanation path scoring.

        Used by improvement C: attention-based ranking for explanation paths.

        Args:
            attention_weights: List of attention matrices per layer

        Returns:
            Aggregated attention scores
        """
        # Average attention across layers
        avg_attention = torch.stack(attention_weights).mean(dim=0)

        # Compute path scores as sum of log attention along paths
        # s(path) = Σ_e log α_e
        # We approximate this by summing log(avg_attention)
        eps = 1e-8
        path_scores = torch.log(avg_attention + eps)

        return path_scores
