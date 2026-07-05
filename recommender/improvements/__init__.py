"""KG-HybridRec Improvements."""

from .adaptive_gate import AdaptiveGateModule
from .graph_contrastive import GraphContrastiveModule
from .attention_explain import AttentionExplanationModuleWrapper

__all__ = [
    'AdaptiveGateModule',
    'GraphContrastiveModule',
    'AttentionExplanationModuleWrapper',
]
