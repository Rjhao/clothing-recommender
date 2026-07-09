"""Heterogeneous graph builder for KG-HybridRec.

Builds an in-memory heterogeneous graph from the Amazon data JSON. Used by:
  - improvement B (graph contrastive learning, edge-dropout two-view)
  - improvement A's KG branch (RippleNet real-neighbor lookup)
  - TransR kg_loss (KG triplets)

Neo4j is NOT used at training time (too slow); this builds the same
structure in-memory.

Node ID space (contiguous):
    [0, n_users)                          -> users
    [n_users, n_users+n_items)            -> items        (item_offset)
    [+n_items, +n_l3)                     -> L3 categories (finest, e.g. slipons)
    [+n_l3, +n_l2)                        -> L2 categories (e.g. shoes)
    [+n_l2, +n_l1)                        -> L1 categories (e.g. apparel)
    [+n_l1, +n_brand)                     -> brands

category_code is hierarchical ("apparel.shoes.slipons"); we split on '.' into
up to 3 levels (L1/L2/L3) and add has_subcategory edges between levels — this
matches the paper's "类别-子类别-属性" three-level KG. Products lacking
category_code fall back to the flat `category` field as a single L1.

Edge types (undirected, stored as both directions):
    0: user-item     (training interactions only; test/val excluded by caller)
    1: item-category (item -> L3, or item -> L2/L1 if fewer levels)
    2: subcategory   (L3->L2, L2->L1)
    3: item-brand

n_relations = 4 (one per edge type). TransR relation ids align with these.
"""
import torch
from typing import Tuple, Dict, List


def _split_category_code(product: dict) -> List[str]:
    """Split category_code into up to 3 hierarchical levels.

    "apparel.shoes.slipons" -> ["apparel", "shoes", "slipons"]
    Falls back to flat `category` as a single L1 when category_code is missing.
    """
    cc = product.get('category_code')
    if cc and isinstance(cc, str) and '.' in cc:
        parts = [s for s in cc.split('.') if s]
        if parts:
            return parts
    c = product.get('category')
    return [c] if c else []


def build_hetero_graph(
    products: List[dict],
    train_pairs: List[Tuple[int, int]],
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
    """Build heterogeneous graph edge_index and edge_type.

    Args:
        products: list of product dicts (with category_code, brand)
        train_pairs: list of (user_id, item_id) training interactions
            (test/val already excluded by leave-one-out — caller's job)
        user2idx, item2idx: id -> index mappings

    Returns:
        edge_index: LongTensor [2, E] (both directions)
        edge_type:  LongTensor [E]
        n_total_nodes: int
        node_offsets: dict with offsets/sizes per node type + idx maps
    """
    n_users = len(user2idx)
    n_items = len(item2idx)
    item_offset = n_users  # item node id = n_users + item_idx

    # ---- collect hierarchical category entities ----
    l1_set, l2_set, l3_set = set(), set(), set()
    for p in products:
        parts = _split_category_code(p)
        if len(parts) >= 1:
            l1_set.add(parts[0])
        if len(parts) >= 2:
            l2_set.add(parts[1])
        if len(parts) >= 3:
            l3_set.add(parts[2])

    l1s = sorted(l1_set)
    l2s = sorted(l2_set)
    l3s = sorted(l3_set)
    l1_idx = {c: i for i, c in enumerate(l1s)}
    l2_idx = {c: i for i, c in enumerate(l2s)}
    l3_idx = {c: i for i, c in enumerate(l3s)}

    l3_offset = item_offset + n_items
    l2_offset = l3_offset + len(l3s)
    l1_offset = l2_offset + len(l2s)

    brands = sorted({p['brand'] for p in products if p.get('brand')})
    brand2idx = {b: i for i, b in enumerate(brands)}
    brand_offset = l1_offset + len(l1s)

    n_total = brand_offset + len(brands)

    idx2item = {idx: pid for pid, idx in item2idx.items()}
    pid2prod = {int(p['product_id']): p for p in products}

    src, dst, etype = [], [], []

    def add_edge(a, b, t):
        src.append(a); dst.append(b); etype.append(t)
        src.append(b); dst.append(a); etype.append(t)

    # edge type 0: user-item (training only)
    for uid, iid in train_pairs:
        if uid in user2idx and iid in item2idx:
            add_edge(user2idx[uid], item_offset + item2idx[iid], 0)

    # edge type 1: item-category; edge type 2: subcategory; edge type 3: item-brand
    for idx, pid in idx2item.items():
        prod = pid2prod.get(pid)
        if not prod:
            continue
        item_node = item_offset + idx
        parts = _split_category_code(prod)

        if len(parts) >= 3:
            l3_node = l3_offset + l3_idx[parts[2]]
            add_edge(item_node, l3_node, 1)
            l2_node = l2_offset + l2_idx[parts[1]]
            add_edge(l3_node, l2_node, 2)
            l1_node = l1_offset + l1_idx[parts[0]]
            add_edge(l2_node, l1_node, 2)
        elif len(parts) == 2:
            l2_node = l2_offset + l2_idx[parts[1]]
            add_edge(item_node, l2_node, 1)
            l1_node = l1_offset + l1_idx[parts[0]]
            add_edge(l2_node, l1_node, 2)
        elif len(parts) == 1:
            l1_node = l1_offset + l1_idx[parts[0]]
            add_edge(item_node, l1_node, 1)

        brand = prod.get('brand')
        if brand in brand2idx:
            add_edge(item_node, brand_offset + brand2idx[brand], 3)

    edge_index = torch.LongTensor([src, dst])
    edge_type = torch.LongTensor(etype)

    node_offsets = {
        'user': 0, 'item': item_offset,
        'l3': l3_offset, 'l2': l2_offset, 'l1': l1_offset,
        'brand': brand_offset,
        'n_total': n_total, 'n_users': n_users, 'n_items': n_items,
        'n_l1': len(l1s), 'n_l2': len(l2s), 'n_l3': len(l3s),
        'n_brand': len(brands),
        'item_offset': item_offset,
        'l1_idx': l1_idx, 'l2_idx': l2_idx, 'l3_idx': l3_idx,
        'brand2idx': brand2idx,
    }
    return edge_index, edge_type, n_total, node_offsets


# Number of relation types (edge types). TransR relation ids align with these.
N_RELATIONS = 4


def build_graph_for_model(
    products: List[dict],
    train_data: List[Tuple[int, int, int]],
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
) -> Dict:
    """Build the graph dict consumed by KGHybridRec.

    Args:
        products: product dicts (with category_code, brand)
        train_data: list of (user_idx, item_idx, _) from prepare_training_data
            (test/val already excluded by leave-one-out)
        user2idx, item2idx: id -> index mappings
    Returns:
        dict with edge_index, edge_type, n_total, n_relations, item_offset,
        node_offsets (for RippleNet neighbor lookup + KG triplet building)
    """
    idx2user = {idx: uid for uid, idx in user2idx.items()}
    idx2item = {idx: iid for iid, idx in item2idx.items()}
    # train_data entries are (pos_tuple, neg_tuple) from prepare_training_data;
    # build_hetero_graph expects raw (user_id, item_id) pairs.
    train_pairs = [(idx2user[pos[0]], idx2item[pos[1]]) for pos, _ in train_data]
    ei, et, n_total, off = build_hetero_graph(products, train_pairs, user2idx, item2idx)
    return {
        'edge_index': ei,
        'edge_type': et,
        'n_total': n_total,
        'n_relations': N_RELATIONS,
        'item_offset': off['item'],
        'node_offsets': off,
    }


def build_kg_triplets(
    products: List[dict],
    item2idx: Dict[int, int],
    node_offsets: Dict,
) -> torch.Tensor:
    """Build KG triplets (head, relation, tail) for TransR kg_loss.

    Relations:
        1: has_category    (item -> L3/L2/L1)
        2: has_subcategory (L3->L2, L2->L1)
        3: has_brand       (item -> brand)

    Returns:
        LongTensor [N, 3] of (head_node, relation, tail_node)
    """
    item_offset = node_offsets['item_offset']
    l3_offset = node_offsets['l3']
    l2_offset = node_offsets['l2']
    l1_offset = node_offsets['l1']
    brand_offset = node_offsets['brand']
    l1_idx = node_offsets['l1_idx']
    l2_idx = node_offsets['l2_idx']
    l3_idx = node_offsets['l3_idx']
    brand2idx = node_offsets['brand2idx']

    pid2prod = {int(p['product_id']): p for p in products}
    triplets = []
    for iid, idx in item2idx.items():
        prod = pid2prod.get(int(iid))
        if not prod:
            continue
        item_node = item_offset + idx
        parts = _split_category_code(prod)
        if len(parts) >= 3:
            l3_node = l3_offset + l3_idx[parts[2]]
            l2_node = l2_offset + l2_idx[parts[1]]
            l1_node = l1_offset + l1_idx[parts[0]]
            triplets.append((item_node, 1, l3_node))
            triplets.append((l3_node, 2, l2_node))
            triplets.append((l2_node, 2, l1_node))
        elif len(parts) == 2:
            l2_node = l2_offset + l2_idx[parts[1]]
            l1_node = l1_offset + l1_idx[parts[0]]
            triplets.append((item_node, 1, l2_node))
            triplets.append((l2_node, 2, l1_node))
        elif len(parts) == 1:
            l1_node = l1_offset + l1_idx[parts[0]]
            triplets.append((item_node, 1, l1_node))
        brand = prod.get('brand')
        if brand in brand2idx:
            triplets.append((item_node, 3, brand_offset + brand2idx[brand]))

    if not triplets:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.LongTensor(triplets)
