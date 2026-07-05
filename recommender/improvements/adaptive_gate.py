"""Improvement A: Adaptive Gated Fusion.

This module implements the adaptive gated fusion mechanism that dynamically
adjusts fusion weights based on user interaction count and modality confidence.

Key formula:
    gate = sigmoid(W_g @ [h_cf ⊕ h_kg ⊕ h_mm] + b_g)
    h_fusion = gate[0]·h_cf + gate[1]·h_kg + gate[2]·h_mm

When interaction count is low (cold start):
    - Automatically reduces CF weight
    - Increases KG and MM weights

This addresses Research Question 1: Cold Start Problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveGate(nn.Module):
    """Adaptive Gated Fusion Network.

    Dynamically generates fusion weights for CF, KG, and MM features
    based on interaction count and modality confidence.
    """

    def __init__(
        self,
        cf_dim: int = 64,
        kg_dim: int = 64,
        mm_dim: int = 64,
        hidden_dim: int = 32,
        use_confidence: bool = True,
        use_interaction_count: bool = True,
    ):
        """
        Initialize Adaptive Gate.

        Args:
            cf_dim: CF feature dimension
            kg_dim: KG feature dimension
            mm_dim: MM feature dimension
            hidden_dim: Hidden dimension for gate network
            use_confidence: Whether to use confidence scores
            use_interaction_count: Whether to use interaction counts
        """
        super().__init__()

        self.cf_dim = cf_dim
        self.kg_dim = kg_dim
        self.mm_dim = mm_dim
        self.hidden_dim = hidden_dim
        self.use_confidence = use_confidence
        self.use_interaction_count = use_interaction_count

        total_dim = cf_dim + kg_dim + mm_dim

        # Gate network: computes weight for each modality
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),  # 3 gates: CF, KG, MM
        )

        # Modality-specific processors
        self.cf_processor = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cf_dim),
        )

        self.kg_processor = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, kg_dim),
        )

        self.mm_processor = nn.Sequential(
            nn.Linear(mm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mm_dim),
        )

        # Interaction count embedding (for cold start detection)
        if use_interaction_count:
            self.interaction_embedding = nn.Embedding(1000, hidden_dim)  # Max 1000 interactions

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with adaptive gated fusion.

        Args:
            cf_features: CF embeddings [batch, cf_dim]
            kg_features: KG embeddings [batch, kg_dim]
            mm_features: MM embeddings [batch, mm_dim]
            interaction_counts: User interaction counts [batch] (used for cold start detection)
            cf_confidence: CF confidence scores [batch] (from CF module)

        Returns:
            Tuple of (fused_features, gate_metrics)
        """
        batch_size = cf_features.size(0)

        # Concatenate all features
        concat_features = torch.cat([cf_features, kg_features, mm_features], dim=-1)

        # Compute base gates
        gates = self.gate_network(concat_features)  # [batch, 3]

        # Modulate gates based on cold start signals
        if self.use_interaction_count and interaction_counts is not None:
            # Normalize interaction counts
            interaction_norm = torch.clamp(interaction_counts.float() / 10.0, 0, 1)

            # Create cold start mask (1 when cold, 0 when warm)
            cold_mask = 1 - interaction_norm

            # Cold start adjustment: reduce CF weight, increase KG/MM weights
            # When cold_mask is high (few interactions), reduce CF gate
            cold_adjustment = cold_mask.unsqueeze(-1)  # [batch, 1]

            # Adjust gates: CF down, KG/MM up when cold
            gate_adjustment = torch.tensor([
                -0.5,  # Reduce CF
                0.3,   # Increase KG
                0.3,   # Increase MM
            ], device=gates.device)

            gates = gates + cold_adjustment * gate_adjustment

        # Apply confidence modulation
        if self.use_confidence and cf_confidence is not None:
            # Reduce CF gate based on confidence
            cf_confidence_adj = cf_confidence.unsqueeze(-1).clamp(0.1, 1.0)
            gates[:, 0] = gates[:, 0] * cf_confidence_adj.squeeze(-1)

        # Apply softmax to get valid probability weights
        gate_weights = F.softmax(gates, dim=-1)  # [batch, 3]

        # Process each modality
        cf_processed = self.cf_processor(cf_features)
        kg_processed = self.kg_processor(kg_features)
        mm_processed = self.mm_processor(mm_features)

        # Weighted fusion
        fused = (
            gate_weights[:, 0:1] * cf_processed +
            gate_weights[:, 1:2] * kg_processed +
            gate_weights[:, 2:3] * mm_processed
        )

        # Metrics for monitoring
        metrics = {
            'gate_cf_mean': gate_weights[:, 0].mean().item(),
            'gate_kg_mean': gate_weights[:, 1].mean().item(),
            'gate_mm_mean': gate_weights[:, 2].mean().item(),
            'gate_cf_std': gate_weights[:, 0].std().item(),
        }

        if interaction_counts is not None:
            metrics['interaction_mean'] = interaction_counts.float().mean().item()
            metrics['cold_ratio'] = (interaction_counts < 5).float().mean().item()

        return fused, metrics

    def get_gate_weights(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get raw gate weights without fusion (for analysis)."""
        concat_features = torch.cat([cf_features, kg_features, mm_features], dim=-1)
        gates = self.gate_network(concat_features)
        return F.softmax(gates, dim=-1)


class AdaptiveGateModule(nn.Module):
    """Adaptive Gated Fusion Module wrapper.

    This is Improvement A in KG-HybridRec.
    """

    def __init__(
        self,
        cf_dim: int = 64,
        kg_dim: int = 64,
        mm_dim: int = 64,
        hidden_dim: int = 32,
    ):
        """
        Initialize Adaptive Gate Module.

        Args:
            cf_dim: CF feature dimension
            kg_dim: KG feature dimension
            mm_dim: MM feature dimension
            hidden_dim: Hidden dimension for gate network
        """
        super().__init__()

        self.adaptive_gate = AdaptiveGate(
            cf_dim=cf_dim,
            kg_dim=kg_dim,
            mm_dim=mm_dim,
            hidden_dim=hidden_dim,
            use_confidence=True,
            use_interaction_count=True,
        )

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Returns:
            Tuple of (fused_features, metrics)
        """
        return self.adaptive_gate(
            cf_features, kg_features, mm_features,
            interaction_counts, cf_confidence
        )
