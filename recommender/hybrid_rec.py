"""KG-HybridRec: Hybrid Recommendation with Knowledge Graph Attention Network.

This is the complete KG-HybridRec model combining:
1. KGAT Backbone (CF + KG Embedding + Multimodal + GAT)
2. Improvement A: Adaptive Gated Fusion
3. Improvement B: Self-Supervised Graph Contrastive Learning
4. Improvement C: Attention-based Explanation Paths

Training Objective (paper):
    L_total = L_BPR + λ_KG · L_KG + λ_CL · L_CL + λ_Θ · ||Θ||²

Bug fixes applied (see algorithm review):
  - KG branch now uses the REAL heterogeneous graph (RippleNet samples real
    neighbors, not torch.randint noise). graph passed through to KGEmbeddingModule.
  - Adaptive gate receives REAL per-user interaction_counts and cf_confidence
    (CFModule constructed with n_interactions). predict_score passes cf_confidence.
  - Single BPR on the fused score (cf_loss is monitored but NOT added — the
    paper's objective has no auxiliary CF term; CF trains through the fused BPR).
  - Negative sampling supports 1:n_neg (neg_item_ids [B, n_neg]) per the paper's
    1:4 protocol; pos/user are expanded so BPR is computed over all pairs.
  - KG triplets (TransR margin loss) are now actually trained when provided.
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
    """KG-HybridRec: Knowledge Graph Hybrid Recommendation Model."""

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
        use_processors: bool = False,
        inject_cl_signal: bool = True,
        lambda_cl: float = 0.1,
        lambda_reg: float = 1e-4,
        lambda_kg: float = 0.1,
        graph: Optional[Dict] = None,
        n_interactions: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            n_users, n_items, n_entities, n_relations: graph dimensions
            embedding_dim: Embedding dimension
            n_hops: RippleNet hops
            n_heads, n_layers: GAT config
            use_improvement_a/b/c, use_multimodal: switches
            use_processors: legacy flag — True loads pre-2026-07-29 ablation
                checkpoints (abl_m1/m2/m3) which used cf/kg/mm_processor MLPs
                in the adaptive gate. Default False (B-route: raw-feature gate).
            inject_cl_signal: if False, predict_score skips the
                `fused += 0.1 * cl_signal` injection added in 2026-08-17.
                Used by eval scripts that want bit-for-bit reproduction of
                pre-Step-2 ablation numbers (13/14/14).
            lambda_cl, lambda_reg, lambda_kg: loss weights
            graph: heterogeneous graph dict (edge_index, edge_type, n_total,
                n_relations, item_offset, node_offsets) for KG + improvement B.
                None -> random-neighbor fallback (not recommended).
            n_interactions: per-user interaction count tensor [n_users] for
                CFModule confidence + adaptive gate cold-start signal.
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
        self.lambda_kg = lambda_kg
        self.inject_cl_signal = inject_cl_signal
        self.graph = graph

        # ============ KGAT Backbone ============

        # CF Module (BPR-MF) — receives per-user interaction counts so
        # get_confidence returns real values (drives the adaptive gate).
        self.cf_module = CFModule(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_interactions=n_interactions,
        )

        # KG Embedding Module (TransR + RippleNet) — wired to the REAL graph.
        self.kg_module = KGEmbeddingModule(
            n_entities=n_entities,
            n_relations=n_relations,
            embedding_dim=embedding_dim,
            n_hops=n_hops,
            graph=graph,
        )

        if use_multimodal:
            self.mm_module = MultimodalModule(
                embedding_dim=embedding_dim,
                fusion_method="weighted",
            )

        # GAT Fusion (used when improvement A is OFF)
        # kg_dim = embedding_dim * n_hops (RippleNet concat); mm_dim always embedding_dim.
        self.gat_fusion = GATFusionModule(
            cf_dim=embedding_dim,
            kg_dim=embedding_dim * n_hops,
            mm_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        # ============ Improvements ============

        if use_improvement_a:
            self.adaptive_gate = AdaptiveGateModule(
                cf_dim=embedding_dim,
                kg_dim=embedding_dim * n_hops,
                mm_dim=embedding_dim,
                hidden_dim=embedding_dim // 2,
                output_dim=embedding_dim,
                use_mm=use_multimodal,  # FIX: skip MM gating when MM is disabled
                use_processors=use_processors,  # legacy: True to load abl_* ckpts
            )

        if use_improvement_b:
            n_total_nodes = graph['n_total'] if graph is not None else None
            self.graph_contrastive = GraphContrastiveModule(
                embedding_dim=embedding_dim,
                temperature=0.1,
                n_total_nodes=n_total_nodes,
            )

        if use_improvement_c:
            self.explanation_module = AttentionExplanationModuleWrapper(
                n_relation_types=n_relations,
                max_path_length=3,
            )

        # ============ Prediction Layer ============
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
        """Forward pass.

        Args:
            user_ids: [B]
            pos_item_ids: [B]
            neg_item_ids: [B] or [B, n_neg] (1:n_neg negative sampling)
            kg_triplets: optional (head, relation, tail_pos, tail_neg) for TransR
            interaction_counts: [B] per-user counts for adaptive gate
        Returns:
            (total_loss, metrics)
        """
        metrics: Dict = {}
        losses: Dict = {}

        B = user_ids.size(0)
        device = user_ids.device

        # ---- expand to 1:n_neg granularity ----
        # neg may be [B] (n_neg=1) or [B, n_neg]. Flatten and expand pos/user.
        if neg_item_ids.dim() == 1:
            n_neg = 1
            neg_flat = neg_item_ids
            u_ids = user_ids
            p_ids = pos_item_ids
        else:
            n_neg = neg_item_ids.size(1)
            neg_flat = neg_item_ids.reshape(-1)
            u_ids = user_ids.unsqueeze(1).expand(B, n_neg).reshape(-1)
            p_ids = pos_item_ids.unsqueeze(1).expand(B, n_neg).reshape(-1)
        M = u_ids.size(0)  # B * n_neg

        # expand interaction_counts to [M]
        if interaction_counts is not None:
            if interaction_counts.size(0) == B and n_neg > 1:
                ic = interaction_counts.unsqueeze(1).expand(B, n_neg).reshape(-1)
            else:
                ic = interaction_counts
        else:
            ic = None

        # ============ 1. CF Branch ============
        cf_loss, cf_metrics = self.cf_module(u_ids, p_ids, neg_flat)
        losses['cf_loss'] = cf_loss  # monitored, NOT added to total (paper has no aux CF term)
        metrics.update({f'cf_{k}': v for k, v in cf_metrics.items()})

        cf_pos_emb = self.cf_module.get_item_representation(p_ids)    # [M, dim]
        cf_neg_emb = self.cf_module.get_item_representation(neg_flat)  # [M, dim]
        cf_confidence = self.cf_module.get_confidence(u_ids)          # [M]

        # ============ 2. KG Branch (real graph) ============
        kg_pos_emb, hop_attentions = self.kg_module(u_ids, p_ids)      # [M, dim*n_hops]
        kg_neg_emb, _ = self.kg_module(u_ids, neg_flat)

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
            mm_neg_emb = torch.zeros_like(mm_pos_emb)
            metrics.update({f'mm_{k}': v for k, v in mm_metrics.items()})
        else:
            mm_pos_emb = torch.zeros_like(cf_pos_emb)
            mm_neg_emb = torch.zeros_like(cf_neg_emb)

        # ============ 4. Improvement A: Adaptive Gate ============
        if self.use_improvement_a:
            fused_pos, gate_metrics = self.adaptive_gate(
                cf_pos_emb, kg_pos_emb, mm_pos_emb,
                interaction_counts=ic, cf_confidence=cf_confidence,
            )
            fused_neg, _ = self.adaptive_gate(
                cf_neg_emb, kg_neg_emb, mm_neg_emb,
                interaction_counts=ic, cf_confidence=cf_confidence,
            )
            metrics.update(gate_metrics)
        else:
            # GAT fusion with identity adjacency (cf/kg/mm are feature channels
            # of the same batch nodes, so adj is [M, M]).
            adj = torch.eye(M, device=device)
            fused_pos, _ = self.gat_fusion(cf_pos_emb, kg_pos_emb, mm_pos_emb, adj)
            fused_neg, _ = self.gat_fusion(cf_neg_emb, kg_neg_emb, mm_neg_emb, adj)

        # ============ 5. Prediction ============
        pos_scores = self.predict(fused_pos).squeeze(-1)  # [M]
        neg_scores = self.predict(fused_neg).squeeze(-1)  # [M]
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        losses['bpr_loss'] = bpr_loss
        metrics['bpr_loss'] = bpr_loss.item()
        metrics['pos_score_mean'] = pos_scores.mean().item()
        metrics['neg_score_mean'] = neg_scores.mean().item()

        # ============ 6. Improvement B: Contrastive Loss ============
        if self.use_improvement_b:
            if self.graph is not None:
                item_off = self.graph['item_offset']
                batch_nodes = torch.cat([
                    user_ids,
                    item_off + pos_item_ids.reshape(-1),
                    item_off + neg_flat,
                ])
                ei = self.graph['edge_index'].to(device)
                et = self.graph['edge_type'].to(device)
                cl_loss, cl_metrics = self.graph_contrastive.forward_graph(ei, et, batch_nodes)
            else:
                z_i = fused_pos
                z_j = F.dropout(fused_pos, p=0.2, training=self.training)
                cl_loss, cl_metrics = self.graph_contrastive(z_i, z_j)
            losses['cl_loss'] = cl_loss
            metrics.update(cl_metrics)

        # ============ 7. Total Loss ============
        # Paper: L_BPR + λ_KG·L_KG + λ_CL·L_CL + λ_Θ·||Θ||²
        total_loss = losses.get('bpr_loss', torch.tensor(0.0, device=device))
        if 'kg_loss' in losses:
            total_loss = total_loss + self.lambda_kg * losses['kg_loss']
        if 'cl_loss' in losses:
            total_loss = total_loss + self.lambda_cl * losses['cl_loss']

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
        """Predict scores for user-item pairs [batch_size]."""
        cf_emb = self.cf_module.get_item_representation(item_ids)
        kg_emb, _ = self.kg_module(user_ids, item_ids)

        if self.use_multimodal and images is not None:
            mm_emb, _ = self.mm_module(images, texts)
        else:
            mm_emb = torch.zeros_like(cf_emb)

        if self.use_improvement_a:
            cf_confidence = self.cf_module.get_confidence(user_ids)
            fused, _ = self.adaptive_gate(
                cf_emb, kg_emb, mm_emb,
                interaction_counts=interaction_counts,
                cf_confidence=cf_confidence,
            )
        else:
            # A=0 path uses GAT fusion over an identity adjacency. Chunking the
            # batch axis is mathematically exact IF kg_emb/cf_emb are computed
            # once in full (RippleNet's RNG consumption is order-sensitive, so
            # we must not call kg_module per-chunk). Only gat_fusion itself is
            # chunked; cf_module is a pure lookup, no RNG.
            chunk_size = 2048
            n = user_ids.size(0)
            fused_chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                adj = torch.eye(end - start, device=user_ids.device)
                chunk_fused, _ = self.gat_fusion(
                    cf_emb[start:end], kg_emb[start:end], mm_emb[start:end], adj,
                )
                fused_chunks.append(chunk_fused)
            fused = torch.cat(fused_chunks, dim=0)

        # Improvement B: weighted mix of contrastive node repr into fused.
        # alpha=0.1 keeps A's adaptive-gate signal dominant (A drives HR@10).
        # B's contribution is the auxiliary uniformization signal from
        # independent GAT encoder + InfoNCE — see §3.3 and Step 2 plan.
        if self.use_improvement_b and self.graph is not None and self.inject_cl_signal:
            entity_ids = item_ids + self.graph['item_offset']
            fused = fused + 0.1 * self.graph_contrastive.get_node_repr(entity_ids)

        scores = self.predict(fused).squeeze(-1)
        return torch.sigmoid(scores)

    def get_recommendations(
        self,
        user_id: int,
        candidate_items: List[int],
        interaction_count: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float, Optional[str]]]:
        """Get top-k recommendations for a user."""
        self.eval()
        with torch.no_grad():
            user_ids = torch.tensor([user_id] * len(candidate_items))
            item_ids = torch.tensor(candidate_items)
            interaction_counts = torch.tensor([interaction_count] * len(candidate_items))

            scores = self.predict_score(user_ids, item_ids, interaction_counts)
            sorted_indices = torch.argsort(scores, descending=True)[:top_k]

            recommendations = []
            for idx in sorted_indices:
                item_id = candidate_items[idx.item()]
                score = scores[idx].item()
                recommendations.append((item_id, score, None))

        return recommendations

    def get_explanations(
        self,
        user_id: int,
        item_id: int,
        kg_adjacency: Dict[int, List[Tuple[int, int]]],
    ) -> List[str]:
        """Generate explanations for a recommendation."""
        if not self.use_improvement_c:
            return ["基于您的偏好推荐"]

        self.eval()
        with torch.no_grad():
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
        cf_weight_decay: Optional[float] = None,
        kg_weight_decay: Optional[float] = None,
        other_weight_decay: Optional[float] = None,
    ):
        """Trainer for KG-HybridRec.

        Per-param L2 (cf_weight_decay / kg_weight_decay / other_weight_decay):
        when all three are None, falls back to a single uniform ``weight_decay``
        over every parameter (legacy behaviour). When any of them is set, the
        optimizer is built with three param groups:

          - cf_module.bpr_mf   -> cf_weight_decay    (BPR user/item embeddings)
          - kg_module          -> kg_weight_decay    (KG + TransR + RippleNet)
          - everything else    -> other_weight_decay (gates, GAT, contrastive,
                                                     prediction MLP, multimodal)

        Why split: uniform wd=1e-4 lets cf/kg embedding norms drift (popular
        items inflate ~15x over cold items, see ``experiments/eval_diversity_
        normfix.py``). Concentrating L2 on the embedding groups and zeroing it
        on the gating/fusion modules gives popular-bias a brake without
        crushing the adaptive gate's capacity. Lambda-reg in the loss still
        applies to all params as a global floor.
        """
        self.model = model
        self.lambda_cl = lambda_cl
        self.lambda_reg = lambda_reg

        # Resolve per-group wd: None -> use the legacy uniform weight_decay.
        cf_wd = weight_decay if cf_weight_decay is None else cf_weight_decay
        kg_wd = weight_decay if kg_weight_decay is None else kg_weight_decay
        other_wd = weight_decay if other_weight_decay is None else other_weight_decay

        # If all three collapse to the same value, keep the legacy single-group
        # optimizer (avoids touching optimizer state when the caller doesn't
        # opt into the fix).
        per_group_split = (
            cf_weight_decay is not None
            or kg_weight_decay is not None
            or other_weight_decay is not None
        )

        if per_group_split:
            cf_params = list(model.cf_module.bpr_mf.parameters())
            kg_params = list(model.kg_module.parameters())
            cf_param_ids = {id(p) for p in cf_params}
            kg_param_ids = {id(p) for p in kg_params}
            other_params = [
                p for p in model.parameters()
                if id(p) not in cf_param_ids and id(p) not in kg_param_ids
            ]
            param_groups = [
                {'params': cf_params, 'weight_decay': cf_wd},
                {'params': kg_params, 'weight_decay': kg_wd},
                {'params': other_params, 'weight_decay': other_wd},
            ]
            self.optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
        else:
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
        """Train for one epoch."""
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
        """Evaluate model."""
        self.model.eval()
        with torch.no_grad():
            _, metrics = self.model(
                user_ids=user_ids,
                pos_item_ids=pos_item_ids,
                neg_item_ids=neg_item_ids,
                interaction_counts=interaction_counts,
            )

        return metrics
