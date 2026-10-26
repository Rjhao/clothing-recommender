"""Improvement A: Adaptive Gated Fusion (Fixed).

This module implements the adaptive gated fusion mechanism that dynamically
adjusts fusion weights based on user interaction count and modality confidence.

Key formula:
    gate = sigmoid(W_g @ [h_cf ⊕ h_kg ⊕ h_mm] + b_g)
    h_fusion = gate[0]·h_cf + gate[1]·h_kg + gate[2]·h_mm

When interaction count is low (cold start):
    - Automatically reduces CF weight
    - Increases KG and MM weights

This addresses Research Question 1: Cold Start Problem.

FIX: When MM is unavailable (use_mm=False), gate only uses CF+KG 2-way softmax
to prevent wasting capacity on zero-vector MM features.
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
        output_dim: Optional[int] = None,
        use_confidence: bool = True,
        use_interaction_count: bool = True,
        use_mm: bool = True,
        use_processors: bool = False,
    ):
        """
        Initialize Adaptive Gate.

        Args:
            cf_dim: CF feature dimension
            kg_dim: KG feature dimension (runtime = embedding_dim * n_hops)
            mm_dim: MM feature dimension (always embedding_dim; zeros when off)
            hidden_dim: Hidden dimension for gate network
            output_dim: Unified output dimension for fused representation.
                Defaults to cf_dim. All three modality processors project to
                this dimension so the weighted sum is well-defined.
            use_confidence: Whether to use confidence scores
            use_interaction_count: Whether to use interaction counts
            use_mm: Whether MM channel is available. If False, only CF+KG 2-way gate.
            use_processors: If True, create cf/kg/mm_processor MLPs (legacy
                checkpoints trained before 2026-07-29). Default False (B-route:
                gate directly on raw features). Must be True to load abl_m1/m2/m3
                checkpoints verbatim with strict=True.
        """
        super().__init__()

        self.cf_dim = cf_dim
        self.kg_dim = kg_dim
        self.mm_dim = mm_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else cf_dim
        self.use_confidence = use_confidence
        self.use_interaction_count = use_interaction_count
        self.use_mm = use_mm

        # Number of gates: 3 if MM, else 2
        self.n_gates = 3 if use_mm else 2

        # The gate input dim
        if use_mm:
            total_dim = cf_dim + kg_dim + mm_dim
        else:
            total_dim = cf_dim + kg_dim

        # Gate network: computes weight for each modality
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.n_gates),  # dynamic output size
        )

        # Modality-specific processors — all project to output_dim so the
        # weighted fusion sums tensors of the same shape.
        # B-route (use_processors=False): skip these MLPs, gate directly on raw features.
        # Reason: cf_processor / kg_processor add 2 nonlinear MLPs between the
        # signal and the BPR loss, which empirically causes the signal to weaken
        # (see §4.4.6 algorithm review). With raw cf_emb/kg_emb the gate
        # directly learns which channel to trust.
        # Legacy abl_m1/m2/m3 (pre-2026-07-29) checkpoints were trained with
        # use_processors=True; loading them with strict=True requires the same.
        self.use_processors = use_processors

        # When use_processors=False, gate output needs cf_dim and kg_dim to be equal.
        # KG path dims (kg_dim = embedding_dim * n_hops) may differ from cf_dim.
        # We project the KG branch into cf_dim only when use_processors=False.
        if not self.use_processors:
            # Project kg features to cf_dim for unified fusion
            self.kg_align = nn.Linear(kg_dim, cf_dim) if kg_dim != cf_dim else nn.Identity()
            # If MM enabled, also project mm to cf_dim
            if use_mm:
                self.mm_align = nn.Linear(mm_dim, cf_dim) if mm_dim != cf_dim else nn.Identity()
        else:
            self.cf_processor = nn.Sequential(
                nn.Linear(cf_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_dim),
            )
            self.kg_processor = nn.Sequential(
                nn.Linear(kg_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_dim),
            )
            if use_mm:
                self.mm_processor = nn.Sequential(
                    nn.Linear(mm_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_dim),
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

    def compute_gate_weights(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: Optional[torch.Tensor] = None,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算**实际生效**的门控权重(冷启动调整 + 置信度调制 + softmax 之后)。

        与 `forward` 走完全相同的计算路径,只返回权重而不做融合,
        供门控行为分析使用(论文 §4.4 门控权重的稀疏度自适应实测)。

        Returns:
            gate_weights: [batch, n_gates],每样本的 CF / KG(/MM) 融合权重
        """
        if self.use_mm:
            if mm_features is None:
                mm_features = torch.zeros_like(cf_features)
            concat_features = torch.cat([cf_features, kg_features, mm_features], dim=-1)
        else:
            concat_features = torch.cat([cf_features, kg_features], dim=-1)

        # Compute base gates
        gates = self.gate_network(concat_features)  # [batch, n_gates]

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
            if self.use_mm:
                gate_adjustment = torch.tensor([
                    -0.5,  # Reduce CF
                    0.3,   # Increase KG
                    0.3,   # Increase MM
                ], device=gates.device)
            else:
                gate_adjustment = torch.tensor([
                    -0.5,  # Reduce CF
                    0.5,   # Increase KG (heavier since MM not available)
                ], device=gates.device)
            gates = gates + cold_adjustment * gate_adjustment

        # Apply confidence modulation
        if self.use_confidence and cf_confidence is not None:
            # Reduce CF gate based on confidence
            cf_confidence_adj = cf_confidence.unsqueeze(-1).clamp(0.1, 1.0)
            gates[:, 0] = gates[:, 0] * cf_confidence_adj.squeeze(-1)

        # Apply softmax to get valid probability weights
        return F.softmax(gates, dim=-1)  # [batch, n_gates]

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: Optional[torch.Tensor] = None,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with adaptive gated fusion.

        Args:
            cf_features: CF embeddings [batch, cf_dim]
            kg_features: KG embeddings [batch, kg_dim]
            mm_features: MM embeddings [batch, mm_dim] (optional, ignored if use_mm=False)
            interaction_counts: User interaction counts [batch] (used for cold start detection)
            cf_confidence: CF confidence scores [batch] (from CF module)

        Returns:
            Tuple of (fused_features, gate_metrics)
        """
        batch_size = cf_features.size(0)

        # 实际生效的门控权重(冷启动调整 + 置信度调制 + softmax)
        gate_weights = self.compute_gate_weights(
            cf_features, kg_features, mm_features,
            interaction_counts=interaction_counts,
            cf_confidence=cf_confidence,
        )

        # Process each modality (B-route: skip processors, project for dim alignment)
        if not self.use_processors:
            cf_processed = cf_features  # raw cf embedding (64-dim)
            kg_processed = self.kg_align(kg_features)  # project to 64-dim
            if self.use_mm:
                mm_processed = self.mm_align(mm_features) if mm_features is not None else torch.zeros_like(cf_features)
                fused = (
                    gate_weights[:, 0:1] * cf_processed +
                    gate_weights[:, 1:2] * kg_processed +
                    gate_weights[:, 2:3] * mm_processed
                )
            else:
                fused = (
                    gate_weights[:, 0:1] * cf_processed +
                    gate_weights[:, 1:2] * kg_processed
                )
        else:
            cf_processed = self.cf_processor(cf_features)
            kg_processed = self.kg_processor(kg_features)
            if self.use_mm:
                mm_processed = self.mm_processor(mm_features) if mm_features is not None else torch.zeros_like(cf_processed)
                fused = (
                    gate_weights[:, 0:1] * cf_processed +
                    gate_weights[:, 1:2] * kg_processed +
                    gate_weights[:, 2:3] * mm_processed
                )
            else:
                fused = (
                    gate_weights[:, 0:1] * cf_processed +
                    gate_weights[:, 1:2] * kg_processed
                )

        # Metrics for monitoring
        metrics = {
            'gate_cf_mean': gate_weights[:, 0].mean().item(),
            'gate_kg_mean': gate_weights[:, 1].mean().item(),
        }
        if self.use_mm:
            metrics['gate_mm_mean'] = gate_weights[:, 2].mean().item()
        metrics['gate_cf_std'] = gate_weights[:, 0].std().item()

        if interaction_counts is not None:
            metrics['interaction_mean'] = interaction_counts.float().mean().item()
            metrics['cold_ratio'] = (interaction_counts < 5).float().mean().item()

        return fused, metrics

    def get_gate_weights(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: Optional[torch.Tensor] = None,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the effective gate weights (after cold-start adjustment and
        confidence modulation) without performing the fusion — for analysis
        of how the gate shifts with user interaction sparsity."""
        return self.compute_gate_weights(
            cf_features, kg_features, mm_features,
            interaction_counts=interaction_counts,
            cf_confidence=cf_confidence,
        )


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
        output_dim: Optional[int] = None,
        use_mm: bool = True,
        use_processors: bool = False,
    ):
        """
        Initialize Adaptive Gate Module.

        Args:
            cf_dim: CF feature dimension
            kg_dim: KG feature dimension
            mm_dim: MM feature dimension
            hidden_dim: Hidden dimension for gate network
            output_dim: Unified output dimension (defaults to cf_dim)
            use_mm: Whether MM channel is active. If False, gate uses 2-way softmax.
            use_processors: legacy — True for abl_m1/m2/m3 checkpoints (pre-2026-07-29).
        """
        super().__init__()

        self.adaptive_gate = AdaptiveGate(
            cf_dim=cf_dim,
            kg_dim=kg_dim,
            mm_dim=mm_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_confidence=True,
            use_interaction_count=True,
            use_mm=use_mm,
            use_processors=use_processors,
        )

    def forward(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: Optional[torch.Tensor] = None,
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

    def get_gate_weights(
        self,
        cf_features: torch.Tensor,
        kg_features: torch.Tensor,
        mm_features: Optional[torch.Tensor] = None,
        interaction_counts: Optional[torch.Tensor] = None,
        cf_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """透传 AdaptiveGate.compute_gate_weights:返回实际生效的门控权重。"""
        return self.adaptive_gate.compute_gate_weights(
            cf_features, kg_features, mm_features,
            interaction_counts=interaction_counts,
            cf_confidence=cf_confidence,
        )
