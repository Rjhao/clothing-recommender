"""Improvement C: Attention-based Ranking for Explanation Paths.

This module generates model-faithful explanations by reusing learned
attention weights to score reasoning paths.

Key formula:
    s(path) = Σ_e log α_e

Where α_e is the GAT attention coefficient for edge e.
Paths are scored by the sum of log attention weights along the path,
making explanations faithful to the model's actual decision process.

This addresses Research Question 3: Interpretability Problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import heapq


class AttentionPathScorer(nn.Module):
    """Scores reasoning paths using attention weights.

    s(path) = Σ_e log α_e

    Higher scores indicate paths that the model considers more important.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        """
        Initialize Attention Path Scorer.

        Args:
            temperature: Temperature for softening attention scores
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        attention_weights: List[torch.Tensor],
        paths: List[List[int]],
    ) -> torch.Tensor:
        """
        Score paths using attention weights.

        Args:
            attention_weights: List of attention matrices per GAT layer [n_layers, n_nodes, n_nodes]
            paths: List of paths, each path is a list of node indices

        Returns:
            Path scores [batch]
        """
        scores = []

        for path in paths:
            if len(path) < 2:
                scores.append(0.0)
                continue

            # Compute score as sum of log attention along path
            path_score = 0.0
            valid = True

            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]

                # Get attention from src to dst across all layers
                edge_attentions = []
                for attn in attention_weights:
                    if src < attn.size(0) and dst < attn.size(1):
                        edge_attentions.append(attn[src, dst].item())

                if not edge_attentions:
                    valid = False
                    break

                # Average attention across layers
                avg_attn = sum(edge_attentions) / len(edge_attentions)
                # Add log attention to score
                path_score += torch.log(torch.tensor(avg_attn + 1e-8)).item()

            scores.append(path_score if valid else float('-inf'))

        return torch.tensor(scores)


class AttentionExplanationModule(nn.Module):
    """Attention-based Explanation Generator.

    Generates explanations for recommendations by finding and ranking
    reasoning paths using learned attention weights.

    This is Improvement C in KG-HybridRec.
    """

    def __init__(
        self,
        n_relation_types: int = 10,
        max_path_length: int = 3,
        n_paths_per_node: int = 5,
    ):
        """
        Initialize Attention Explanation Module.

        Args:
            n_relation_types: Number of relation types in the knowledge graph
            max_path_length: Maximum path length to consider
            n_paths_per_node: Number of top paths to return per node
        """
        super().__init__()

        self.n_relation_types = n_relation_types
        self.max_path_length = max_path_length
        self.n_paths_per_node = n_paths_per_node

        # Relation type embeddings for generating readable explanations
        self.relation_embeddings = nn.Embedding(n_relation_types, 32)

        # Path scorer
        self.path_scorer = AttentionPathScorer()

        # Language generation (simplified - could use a small LM)
        self.explanation_generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def find_paths(
        self,
        user_id: int,
        target_item: int,
        kg_adjacency: Dict[int, List[Tuple[int, int]]],
        max_length: int = 3,
    ) -> List[List[int]]:
        """
        Find candidate paths from user to target item.

        Uses BFS to find all paths up to max_length.

        Args:
            user_id: User node ID
            target_item: Target item node ID
            kg_adjacency: Adjacency list {node: [(neighbor, relation_type), ...]}
            max_length: Maximum path length

        Returns:
            List of paths, each path is [user_id, ..., target_item]
        """
        paths = []
        stack = [(user_id, [user_id], 0)]

        while stack:
            node, path, length = stack.pop()

            if node == target_item and len(path) > 1:
                paths.append(path)
                continue

            if length >= max_length:
                continue

            neighbors = kg_adjacency.get(node, [])
            for neighbor, rel_type in neighbors:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path, length + 1))

        return paths

    def score_paths(
        self,
        paths: List[List[int]],
        attention_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Score paths using attention weights.

        Args:
            paths: List of paths
            attention_weights: Attention weights from GAT

        Returns:
            Path scores
        """
        return self.path_scorer(attention_weights, paths)

    def generate_explanation(
        self,
        user_id: int,
        item_id: int,
        path: List[int],
        relation_types: List[int],
        user_preferences: Optional[Dict] = None,
    ) -> str:
        """
        Generate natural language explanation for a path.

        Args:
            user_id: User node ID
            item_id: Item node ID
            path: Node sequence
            relation_types: Relation types along the path
            user_preferences: Optional user preference dict

        Returns:
            Natural language explanation
        """
        # Map relation types to readable names
        relation_names = {
            0: "浏览",
            1: "购买",
            2: "属于",
            3: "相似",
            4: "搭配",
            5: "偏好",
            6: "有",
        }

        # Build explanation
        if len(path) == 2:
            # Direct edge
            rel = relation_types[0] if relation_types else 0
            rel_name = relation_names.get(rel, "相关")
            explanation = f"根据您{relation_names.get(0, '的浏览历史')}，推荐这件商品"
        else:
            # Multi-hop path
            parts = []
            for i, (node, rel) in enumerate(zip(path[1:-1], relation_types[:-1])):
                rel_name = relation_names.get(rel, "相关")
                parts.append(f"通过{rel_name}关系发现")

            explanation = "推荐原因："
            explanation += " -> ".join(parts)
            explanation += f"，最终推荐这件商品"

        # Add personalization if available
        if user_preferences:
            style = user_preferences.get('preferred_style', '')
            if style:
                explanation += f"，符合您喜欢的{style}风格"

        return explanation

    def forward(
        self,
        user_id: int,
        target_item_id: int,
        kg_adjacency: Dict[int, List[Tuple[int, int]]],
        attention_weights: List[torch.Tensor],
        top_k: int = 3,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate top-k explanations for a recommendation.

        Args:
            user_id: User node ID
            target_item_id: Target item node ID
            kg_adjacency: Knowledge graph adjacency list
            attention_weights: GAT attention weights
            top_k: Number of top explanations to return

        Returns:
            Tuple of (explanations, path_scores)
        """
        # Find candidate paths
        paths = self.find_paths(
            user_id, target_item_id,
            kg_adjacency, self.max_path_length
        )

        if not paths:
            return ["根据您的偏好推荐"], torch.tensor([0.0])

        # Score paths
        scores = self.score_paths(paths, attention_weights)

        # Get top-k paths
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        # Generate explanations
        explanations = []
        for idx in top_indices:
            path = paths[idx.item()]
            # Extract relation types (simplified - would need actual KG query)
            relation_types = [0] * (len(path) - 1)

            explanation = self.generate_explanation(
                user_id, target_item_id, path, relation_types
            )
            explanations.append(explanation)

        return explanations, scores[top_indices]


class AttentionExplanationModuleWrapper(nn.Module):
    """Wrapper for Attention Explanation Module.

    This is Improvement C in KG-HybridRec.
    """

    def __init__(
        self,
        n_relation_types: int = 10,
        max_path_length: int = 3,
    ):
        """
        Initialize Attention Explanation Wrapper.

        Args:
            n_relation_types: Number of relation types
            max_path_length: Maximum path length for explanations
        """
        super().__init__()

        self.explainer = AttentionExplanationModule(
            n_relation_types=n_relation_types,
            max_path_length=max_path_length,
        )

    def forward(
        self,
        user_id: int,
        item_id: int,
        kg_adjacency: Dict[int, List[Tuple[int, int]]],
        attention_weights: List[torch.Tensor],
        top_k: int = 3,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate explanations.

        Returns:
            Tuple of (explanations, scores)
        """
        return self.explainer(
            user_id, item_id,
            kg_adjacency, attention_weights,
            top_k
        )

    def get_faithful_explanation(
        self,
        path: List[int],
        attention_weights: List[torch.Tensor],
    ) -> Dict:
        """
        Get a faithful explanation for a specific path.

        Faithful means the explanation is consistent with the model's
        actual decision process (attention weights).

        Args:
            path: Node path
            attention_weights: GAT attention weights

        Returns:
            Explanation dict with score breakdown
        """
        edge_scores = []

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]

            # Get attention from each layer
            layer_attentions = []
            for layer_idx, attn in enumerate(attention_weights):
                if src < attn.size(0) and dst < attn.size(1):
                    score = attn[src, dst].item()
                    layer_attentions.append(score)

            avg_attn = sum(layer_attentions) / len(layer_attentions) if layer_attentions else 0
            log_attn = torch.log(torch.tensor(avg_attn + 1e-8)).item()

            edge_scores.append({
                'from': src,
                'to': dst,
                'avg_attention': avg_attn,
                'log_attention': log_attn,
            })

        total_score = sum(e['log_attention'] for e in edge_scores)

        return {
            'path': path,
            'edge_scores': edge_scores,
            'total_score': total_score,
            'faithful': True,  # Because we use actual model attention
        }
