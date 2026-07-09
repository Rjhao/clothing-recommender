"""Knowledge Graph Embedding Module - TransR + RippleNet.

This module implements:
1. TransR: Relational projection in entity-relation specific space
2. RippleNet: Multi-hop interest propagation on the REAL knowledge graph

Entity ID space is the same as the heterogeneous graph (hetero_graph.py):
    users   [0, n_users)
    items   [item_offset, item_offset + n_items)
    L3/L2/L1 categories, brands  (after items)
So `forward` offsets item ids by `item_offset` before lookup — this fixes the
old bug where user_ids and items both indexed from 0 into the same embedding
table (user 5 and item 5 shared an embedding).

TransR is now truly relation-specific: each relation r has its own projection
matrix M_r (previously a single shared Linear, which is not TransR).

RippleNet's get_ripples now samples REAL 1/2-hop neighbors from edge_index
(previously torch.randint — pure noise, which is why UserCF beat the model).
Falls back to random neighbors only when no graph is provided.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class TransR(nn.Module):
    """TransR: Entity-Relation Specific Projection.

    Projects entities into relation-specific spaces via a per-relation
    matrix M_r: h_r = M_r · h, t_r = M_r · t.
    Score(h,r,t) = -||h_r + r - t_r||^2
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        entity_dim: int = 64,
        relation_dim: int = 64,
    ):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.entity_embedding = nn.Embedding(n_entities, entity_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)

        # Per-relation projection matrix M_r: [entity_dim, relation_dim].
        # This is the actual TransR (previously a single shared Linear).
        self.M = nn.Parameter(
            torch.eye(entity_dim).unsqueeze(0).repeat(n_relations, 1, 1)
        )
        if entity_dim != relation_dim:
            # eye only valid when square; for non-square use random init
            with torch.no_grad():
                for r in range(n_relations):
                    nn.init.xavier_uniform_(self.M.data[r])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.relation_embedding.weight, mean=0, std=0.01)

    def forward(
        self,
        head_entities: torch.Tensor,
        relations: torch.Tensor,
        tail_entities: torch.Tensor,
    ) -> torch.Tensor:
        """TransR forward: per-relation projection + negative L2 distance."""
        h = self.entity_embedding(head_entities)  # [B, edim]
        r = self.relation_embedding(relations)    # [B, rdim]
        t = self.entity_embedding(tail_entities)  # [B, edim]

        M_rel = self.M[relations]  # [B, edim, rdim]
        h_r = torch.bmm(h.unsqueeze(1), M_rel).squeeze(1)  # [B, rdim]
        t_r = torch.bmm(t.unsqueeze(1), M_rel).squeeze(1)  # [B, rdim]

        score = -((h_r + r - t_r) ** 2).sum(dim=-1)  # [B]
        return score

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.entity_embedding(entity_ids)

    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.relation_embedding(relation_ids)


class RippleNet(nn.Module):
    """RippleNet: Multi-hop Interest Propagation on the real KG.

    Item-centric: for each item, sample K-hop neighbors from edge_index and
    attention-aggregate them into the item representation (KGAT-style). The
    seed for hop k+1 is the aggregated memory of hop k.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_hops: int = 2,
        memory_dim: int = 64,
        n_neighbors: int = 8,
        item_update_mode: str = "plus_transform",
        using_all_hops: bool = True,
        graph: Optional[Dict] = None,
    ):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.n_hops = n_hops
        self.memory_dim = memory_dim
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops
        self.n_neighbors = n_neighbors
        self.graph = graph
        self.item_offset = graph['item_offset'] if graph is not None else 0

        # Entity/relation embeddings — shared with TransR via KGEmbeddingModule.
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)

        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, memory_dim, bias=False) for _ in range(n_hops)
        ])
        self.attention = nn.Linear(memory_dim * 2, 1, bias=False)

        if item_update_mode == "plus_transform":
            self.item_update = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.item_update = nn.Identity()

        self._init_weights()
        self._adj_built = False
        self._adj_nodes: List[torch.Tensor] = []
        self._adj_types: List[torch.Tensor] = []

    def _init_weights(self):
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.relation_embedding.weight, mean=0, std=0.01)

    def _build_adj(self):
        """Build per-node adjacency from edge_index (CPU, once)."""
        ei = self.graph['edge_index']
        et = self.graph['edge_type']
        n = self.graph['n_total']
        adj_nodes = [[] for _ in range(n)]
        adj_types = [[] for _ in range(n)]
        src_list = ei[0].tolist()
        dst_list = ei[1].tolist()
        et_list = et.tolist()
        for s, d, t in zip(src_list, dst_list, et_list):
            adj_nodes[s].append(d)
            adj_types[s].append(t)
        self._adj_nodes = [
            torch.tensor(x, dtype=torch.long) if x else torch.empty(0, dtype=torch.long)
            for x in adj_nodes
        ]
        self._adj_types = [
            torch.tensor(x, dtype=torch.long) if x else torch.empty(0, dtype=torch.long)
            for x in adj_types
        ]
        self._adj_built = True

    def get_ripples(
        self,
        seed_nodes: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Sample K-hop ripple sets from the real graph.

        Args:
            seed_nodes: [B] item ENTITY ids (already offset by item_offset)
            device: target device
        Returns:
            list (per hop) of {'entities': [B, n_neighbors], 'relations': [B, n_neighbors]}
        """
        B = seed_nodes.size(0)

        # Fallback: no graph -> random neighbors (kept for safety; not used
        # in normal training where graph is always provided).
        if self.graph is None:
            ripple_sets = []
            for _ in range(self.n_hops):
                ripple_sets.append({
                    'entities': torch.randint(0, self.n_entities, (B, self.n_neighbors), device=device),
                    'relations': torch.randint(0, self.n_relations, (B, self.n_neighbors), device=device),
                })
            return ripple_sets

        if not self._adj_built:
            self._build_adj()

        ripple_sets = []
        current_seeds = seed_nodes  # [B]
        for hop in range(self.n_hops):
            ent_col, rel_col, next_seeds = [], [], []
            seeds_list = current_seeds.tolist()
            for node in seeds_list:
                neighbors = self._adj_nodes[node]
                if neighbors.numel() == 0:
                    # isolated node: degenerate to self-loop
                    ent = torch.full((self.n_neighbors,), node, dtype=torch.long)
                    rel = torch.zeros(self.n_neighbors, dtype=torch.long)
                elif neighbors.numel() <= self.n_neighbors:
                    idx = torch.randint(0, neighbors.numel(), (self.n_neighbors,))
                    ent = neighbors[idx]
                    rel = self._adj_types[node][idx]
                else:
                    perm = torch.randperm(neighbors.numel())[:self.n_neighbors]
                    ent = neighbors[perm]
                    rel = self._adj_types[node][perm]
                ent_col.append(ent)
                rel_col.append(rel)
                # seed next hop from the first sampled neighbor
                next_seeds.append(int(ent[0]))
            entities = torch.stack(ent_col).to(device)   # [B, n_neighbors]
            relations = torch.stack(rel_col).to(device)  # [B, n_neighbors]
            ripple_sets.append({'entities': entities, 'relations': relations})
            current_seeds = torch.tensor(next_seeds, dtype=torch.long, device=device)
        return ripple_sets

    def forward(
        self,
        user_ids: torch.Tensor,
        items: torch.Tensor,
        ripple_sets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Item-centric KG propagation.

        Args:
            user_ids: [B] (entity ids in [0, n_users); used only for shape compat)
            items: [B] item INDICES (0-based); offset to entity ids internally
            ripple_sets: pre-computed (optional)
        Returns:
            (item_representation [B, embedding_dim*n_hops], hop_representations)
        """
        # item entity ids: offset so items don't collide with users
        item_entity_ids = items + self.item_offset
        item_emb = self.entity_embedding(item_entity_ids)  # [B, dim]

        if ripple_sets is None:
            ripple_sets = self.get_ripples(item_entity_ids, device=item_emb.device)

        hop_representations = []
        current_rep = item_emb
        for hop in range(self.n_hops):
            ripple = ripple_sets[hop]
            neighbor_emb = self.entity_embedding(ripple['entities'])   # [B, n_neigh, dim]
            relation_emb = self.relation_embedding(ripple['relations'])  # [B, n_neigh, dim]

            # Attention: current seed representation vs neighbors (KGAT-style).
            h_expanded = current_rep.unsqueeze(1).expand_as(neighbor_emb)
            attention_scores = torch.sum(h_expanded * neighbor_emb, dim=-1)  # [B, n_neigh]
            attention_weights = F.softmax(attention_scores, dim=-1)

            aggregated = torch.sum(
                attention_weights.unsqueeze(-1) * neighbor_emb, dim=1
            )  # [B, dim]
            memory = F.relu(self.projections[hop](aggregated))  # [B, memory_dim]
            hop_representations.append(memory)
            current_rep = memory  # seed next hop with aggregated memory

        item_representation = (
            torch.cat(hop_representations, dim=-1) if self.using_all_hops
            else hop_representations[-1]
        )
        return item_representation, hop_representations

    def get_attention_weights(
        self,
        user_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        relation_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention weights for explanation (used in improvement C)."""
        h_expanded = user_emb.unsqueeze(1).expand_as(neighbor_emb)
        scores = torch.sum(h_expanded * neighbor_emb, dim=-1)
        return F.softmax(scores, dim=-1)


class KGEmbeddingModule(nn.Module):
    """Knowledge Graph Embedding Module combining TransR and RippleNet.

    This is the KG branch in KG-HybridRec. TransR and RippleNet SHARE the
    entity/relation embeddings so the KG signal is unified.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_hops: int = 2,
        graph: Optional[Dict] = None,
    ):
        super().__init__()
        # If a graph is provided, the entity table must cover all graph nodes.
        if graph is not None:
            n_entities = max(n_entities, graph['n_total'])
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim

        self.transR = TransR(n_entities, n_relations, embedding_dim, embedding_dim)
        self.rippleNet = RippleNet(
            n_entities, n_relations,
            embedding_dim=embedding_dim,
            n_hops=n_hops,
            memory_dim=embedding_dim,
            graph=graph,
        )
        # Share embeddings so TransR-learned entities are what RippleNet propagates.
        self.rippleNet.entity_embedding = self.transR.entity_embedding
        self.rippleNet.relation_embedding = self.transR.relation_embedding

    def forward(
        self,
        user_ids: torch.Tensor,
        items: torch.Tensor,
        ripple_sets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning item representation and hop attention weights."""
        return self.rippleNet(user_ids, items, ripple_sets)

    def compute_kg_loss(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        neg_tail: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """TransR margin ranking loss over KG triplets."""
        pos_scores = self.transR(head, relation, tail)
        neg_scores = self.transR(head, relation, neg_tail)
        margin = 1.0
        loss = F.relu(margin - pos_scores + neg_scores).mean()
        metrics = {
            'kg_loss': loss.item(),
            'pos_score': pos_scores.mean().item(),
            'neg_score': neg_scores.mean().item(),
        }
        return loss, metrics

    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.transR.get_entity_embedding(entity_ids)

    def get_attention_for_explanation(
        self,
        user_emb: torch.Tensor,
        ripple_sets: List[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Get attention weights for explanation path generation (improvement C)."""
        attention_weights = []
        for hop, ripple in enumerate(ripple_sets):
            neighbor_emb = self.transR.entity_embedding(ripple['entities'])
            relation_emb = self.transR.relation_embedding(ripple['relations'])
            weights = self.rippleNet.get_attention_weights(
                user_emb, neighbor_emb, relation_emb
            )
            attention_weights.append(weights)
        return attention_weights
