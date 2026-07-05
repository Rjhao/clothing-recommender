"""KG-HybridRec: Hybrid Recommendation with Knowledge Graph Attention Network.

This is the complete KG-HybridRec model combining:
1. KGAT Backbone (CF + KG Embedding + Multimodal + GAT)
2. Improvement A: Adaptive Gated Fusion
3. Improvement B: Self-Supervised Graph Contrastive Learning
4. Improvement C: Attention-based Explanation Paths

Training Objective:
    L_total = L_BPR + λ_CL · L_CL + λ_Θ · ||Θ||²
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math

# Import KGAT modules
from recommender.kgat.cf_module import CFModule, BPRMF
from recommender.kgat.kg_embedding import KGEmbeddingModule
from recommender.kgat.multimodal import MultimodalModule
from recommender.kgat.gat_fusion import GATFusionModule

# Import improvements
from recommender.improvements.adaptive_gate import AdaptiveGateModule
from recommender.improvements.graph_contrastive import GraphContrastiveModule
from recommender.improvements.attention_explain import AttentionExplanationModuleWrapper


class KGHybridRec(nn.Module):
    """KG-HybridRec: Knowledge Graph Hybrid Recommendation Model.

    A hybrid recommendation model combining:
    - Collaborative Filtering (BPR-MF)
    - Knowledge Graph Embeddings (TransR + RippleNet)
    - Multimodal Features (CLIP + BERT)
    - Graph Attention Network Fusion

    With three improvements:
    - A: Adaptive Gated Fusion (cold start)
    - B: Self-Supervised Graph Contrastive Learning (data sparsity)
    - C: Attention-based Explanation Paths (interpretability)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_hops: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        use_multimodal: bool = True,
        use_improvement_a: bool = True,
        use_improvement_b: bool = True,
        use_improvement_c: bool = True,
        lambda_cl: float = 0.1,
        lambda_reg: float = 1e-4,
    ):
        """
        Initialize KG-HybridRec.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_entities: Number of entities in knowledge graph
            n_relations: Number of relation types
            embedding_dim: Embedding dimension
            n_hops: Number of hops for RippleNet
            n_heads: Number of attention heads
            n_layers: Number of GAT layers
            use_multimodal: Whether to use multimodal features
            use_improvement_a: Whether to use adaptive gated fusion
            use_improvement_b: Whether to use graph contrastive learning
            use_improvement_c: Whether to use attention explanation
            lambda_cl: Weight for contrastive loss
            lambda_reg: L2 regularization coefficient
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim

        self.use_multimodal = use_multimodal
        self.use_improvement_a = use_improvement_a
        self.use_improvement_b = use_improvement_b
        self.use_improvement_c = use_improvement_c
        self.lambda_cl = lambda_cl
        self.lambda_reg = lambda_reg

        # ============ KGAT Backbone ============

        # CF Module (BPR-MF)
        self.cf_module = CFModule(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
        )

        # KG Embedding Module (TransR + RippleNet)
        self.kg_module = KGEmbeddingModule(
            n_entities=n_entities,
            n_relations=n_relations,
            embedding_dim=embedding_dim,
            n_hops=n_hops,
        )

        # Multimodal Module (CLIP + BERT)
        if use_multimodal:
            self.mm_module = MultimodalModule(
                embedding_dim=embedding_dim,
                fusion_method="weighted",
            )

        # GAT Fusion Module
        self.gat_fusion = GATFusionModule(
            cf_dim=embedding_dim,
            kg_dim=embedding_dim * (n_hops if True else 1),  # RippleNet output dim
            mm_dim=embedding_dim if use_multimodal else 0,
            hidden_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        # ============ Improvements ============

        # Improvement A: Adaptive Gated Fusion
        if use_improvement_a:
            self.adaptive_gate = AdaptiveGateModule(
                cf_dim=embedding_dim,
                kg_dim=embedding_dim,
                mm_dim=embedding_dim if use_multimodal else 0,
                hidden_dim=embedding_dim // 2,
            )

        # Improvement B: Graph Contrastive Learning
        if use_improvement_b:
            self.graph_contrastive = GraphContrastiveModule(
                embedding_dim=embedding_dim,
                temperature=0.1,
            )

        # Improvement C: Attention Explanation
        if use_improvement_c:
            self.explanation_module = AttentionExplanationModuleWrapper(
                n_relation_types=n_relations,
                max_path_length=3,
            )

        # ============ Prediction Layer ============

        # Final prediction layer
        self.predict = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        kg_triplets: Optional[Tuple] = None,
        interaction_counts: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        return_explanations: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            user_ids: User indices [batch_size]
            pos_item_ids: Positive item indices [batch_size]
            neg_item_ids: Negative item indices [batch_size]
            kg_triplets: Optional (head, relation, tail) tuples for KG loss
            interaction_counts: User interaction counts for adaptive gate
            images: Optional image tensors for multimodal
            texts: Optional text list for multimodal
            return_explanations: Whether to return explanations

        Returns:
            Tuple of (loss_dict, metrics_dict)
        """
        metrics = {}
        losses = {}

        # ============ 1. CF Branch ============
        cf_loss, cf_metrics = self.cf_module(
            user_ids, pos_item_ids, neg_item_ids
        )
        losses['cf_loss'] = cf_loss
        metrics.update({f'cf_{k}': v for k, v in cf_metrics.items()})

        # Get CF representations
        cf_user_emb = self.cf_module.get_user_representation(user_ids)
        cf_pos_emb = self.cf_module.get_item_representation(pos_item_ids)
        cf_neg_emb = self.cf_module.get_item_representation(neg_item_ids)

        # CF confidence for adaptive gate
        cf_confidence = self.cf_module.get_confidence(user_ids)

        # ============ 2. KG Branch ============
        # Get KG representations (using item IDs as entity IDs)
        kg_pos_emb, hop_attentions = self.kg_module(user_ids, pos_item_ids)
        kg_neg_emb, _ = self.kg_module(user_ids, neg_item_ids)

        # KG loss (TransR)
        if kg_triplets is not None:
            head, relation, tail_pos, tail_neg = kg_triplets
            kg_loss, kg_metrics = self.kg_module.compute_kg_loss(
                head, relation, tail_pos, tail_neg
            )
            losses['kg_loss'] = kg_loss
            metrics.update({f'kg_{k}': v for k, v in kg_metrics.items()})

        # ============ 3. Multimodal Branch ============
        if self.use_multimodal and images is not None and texts is not None:
            mm_pos_emb, mm_metrics = self.mm_module(images, texts)
            # NOTE: proper negative MM embeddings require negative-item
            # images/texts, which forward() does not currently receive. Using
            # zeros (not mm_pos_emb) so the MM branch gets a non-zero BPR
            # gradient instead of identical pos/neg (which gave zero gradient).
            mm_neg_emb = torch.zeros_like(mm_pos_emb)
            metrics.update({f'mm_{k}': v for k, v in mm_metrics.items()})
        else:
            mm_pos_emb = torch.zeros_like(cf_pos_emb)
            mm_neg_emb = torch.zeros_like(cf_neg_emb)

        # ============ 4. Improvement A: Adaptive Gate ============
        if self.use_improvement_a:
            # Use adaptive gated fusion
            fused_pos, gate_metrics = self.adaptive_gate(
                cf_pos_emb,
                kg_pos_emb,
                mm_pos_emb,
                interaction_counts=interaction_counts,
                cf_confidence=cf_confidence,
            )
            fused_neg, _ = self.adaptive_gate(
                cf_neg_emb,
                kg_neg_emb,
                mm_neg_emb,
                interaction_counts=interaction_counts,
                cf_confidence=cf_confidence,
            )
            metrics.update(gate_metrics)
        else:
            # Use simple GAT fusion
            # Create dummy adjacency for GAT
            batch_size = user_ids.size(0)
            adj = torch.eye(batch_size * 3, device=user_ids.device)  # Simplified

            fused_pos,gat_attention = self.gat_fusion(
                cf_pos_emb, kg_pos_emb, mm_pos_emb, adj
            )
            fused_neg, _ = self.gat_fusion(
                cf_neg_emb, kg_neg_emb, mm_neg_emb, adj
            )

        # ============ 5. Prediction ============
        # Compute scores using fused representations
        pos_scores = self.predict(fused_pos).squeeze(-1)
        neg_scores = self.predict(fused_neg).squeeze(-1)

        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        losses['bpr_loss'] = bpr_loss
        metrics['bpr_loss'] = bpr_loss.item()
        metrics['pos_score_mean'] = pos_scores.mean().item()
        metrics['neg_score_mean'] = neg_scores.mean().item()

        # ============ 6. Improvement B: Contrastive Loss ============
        if self.use_improvement_b:
            # Two augmented views of the SAME items via feature dropout.
            # (Previously torch.roll paired item i with item i+1 — a different
            #  item — giving wrong contrastive gradients. Feature-space dropout
            #  is a valid SimCLR-style augmentation when graph structure is not
            #  available in this forward pass.)
            z_i = fused_pos
            z_j = F.dropout(fused_pos, p=0.2, training=self.training)

            cl_loss, cl_metrics = self.graph_contrastive(z_i, z_j)
            losses['cl_loss'] = cl_loss
            metrics.update(cl_metrics)

        # ============ 7. Total Loss ============
        total_loss = (
            losses.get('cf_loss', 0) +
            losses.get('bpr_loss', 0) +
            losses.get('kg_loss', torch.tensor(0.0)) +
            self.lambda_cl * losses.get('cl_loss', torch.tensor(0.0))
        )

        # L2 regularization
        if self.lambda_reg > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            total_loss = total_loss + self.lambda_reg * l2_reg
            metrics['l2_reg'] = l2_reg.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def predict_score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
            interaction_counts: User interaction counts
            images: Optional images
            texts: Optional texts

        Returns:
            Predicted scores [batch_size]
        """
        # Get representations
        cf_emb = self.cf_module.get_item_representation(item_ids)
        kg_emb, _ = self.kg_module(user_ids, item_ids)

        if self.use_multimodal and images is not None:
            mm_emb, _ = self.mm_module(images, texts)
        else:
            mm_emb = torch.zeros_like(cf_emb)

        # Adaptive fusion
        if self.use_improvement_a:
            fused, _ = self.adaptive_gate(
                cf_emb, kg_emb, mm_emb,
                interaction_counts=interaction_counts,
            )
        else:
            batch_size = user_ids.size(0)
            adj = torch.eye(batch_size, device=user_ids.device)
            fused, _ = self.gat_fusion(cf_emb, kg_emb, mm_emb, adj)

        # Predict
        scores = self.predict(fused).squeeze(-1)
        return torch.sigmoid(scores)

    def get_recommendations(
        self,
        user_id: int,
        candidate_items: List[int],
        interaction_count: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float, Optional[str]]]:
        """
        Get top-k recommendations for a user.

        Args:
            user_id: User ID
            candidate_items: List of candidate item IDs
            interaction_count: User's interaction count (for cold start)
            top_k: Number of items to return

        Returns:
            List of (item_id, score, explanation) tuples
        """
        self.eval()
        with torch.no_grad():
            user_ids = torch.tensor([user_id] * len(candidate_items))
            item_ids = torch.tensor(candidate_items)
            interaction_counts = torch.tensor([interaction_count] * len(candidate_items))

            scores = self.predict_score(
                user_ids, item_ids, interaction_counts
            )

            # Sort by score
            sorted_indices = torch.argsort(scores, descending=True)[:top_k]

            recommendations = []
            for idx in sorted_indices:
                item_id = candidate_items[idx.item()]
                score = scores[idx].item()
                recommendations.append((item_id, score, None))  # explanation optional

        return recommendations

    def get_explanations(
        self,
        user_id: int,
        item_id: int,
        kg_adjacency: Dict[int, List[Tuple[int, int]]],
    ) -> List[str]:
        """
        Generate explanations for a recommendation.

        Args:
            user_id: User ID
            item_id: Item ID
            kg_adjacency: Knowledge graph adjacency list

        Returns:
            List of explanation strings
        """
        if not self.use_improvement_c:
            return ["基于您的偏好推荐"]

        self.eval()
        with torch.no_grad():
            # Get attention weights (would need to be passed from forward pass)
            # Simplified: return default explanation
            return [f"推荐这件商品因为它符合您的风格偏好"]


class KGHybridRecTrainer:
    """Trainer for KG-HybridRec model."""

    def __init__(
        self,
        model: KGHybridRec,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        lambda_cl: float = 0.1,
        lambda_reg: float = 1e-4,
    ):
        """
        Initialize trainer.

        Args:
            model: KG-HybridRec model
            learning_rate: Learning rate
            weight_decay: Weight decay
            lambda_cl: Contrastive loss weight
            lambda_reg: L2 regularization weight
        """
        self.model = model
        self.lambda_cl = lambda_cl
        self.lambda_reg = lambda_reg

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

    def train_epoch(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
        kg_triplets: Optional[Tuple] = None,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train for one epoch.

        Returns:
            Metrics dictionary
        """
        self.model.train()

        self.optimizer.zero_grad()

        loss, metrics = self.model(
            user_ids=user_ids,
            pos_item_ids=pos_item_ids,
            neg_item_ids=neg_item_ids,
            kg_triplets=kg_triplets,
            interaction_counts=interaction_counts,
            images=images,
            texts=texts,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return metrics

    def evaluate(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        interaction_counts: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Evaluate model.

        Returns:
            Metrics dictionary
        """
        self.model.eval()
        with torch.no_grad():
            _, metrics = self.model(
                user_ids=user_ids,
                pos_item_ids=pos_item_ids,
                neg_item_ids=neg_item_ids,
                interaction_counts=interaction_counts,
            )

        return metrics
