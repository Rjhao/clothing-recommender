"""Quick normfix-ablation: 测 M0/M1/M2/M3/m2_new 在 cf_kg_norm 模式下的 Coverage@10.

对比 abl_m2_new 的 883 件 (B+归一化协同) vs 其他 ablation 是否也能接近 883 件,
从而判断 883 件是 B 的真贡献还是 normfix 单独就能做到。

用法: PYTHONUTF8=1 /c/Python311/python.exe -u experiments/quick_normfix_ablation.py
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.hybrid_rec import KGHybridRec
from recommender.train import load_data, prepare_training_data
from recommender.kgat.hetero_graph import build_graph_for_model


# Per-tag config: (tag, use_multimodal, use_a, use_b, use_c, use_processors, inject_cl_signal)
# Skip abl_m0: A=0 path uses gat_fusion which can't easily chunk inside this normfix
# predict. abl_m0 has no adaptive gate anyway, so normfix diversity gain is moot.
CONFIGS = [
    ('abl_m1',     True,  1, 0, 0, True,  False),  # 3-way gate, no CL injection
    ('abl_m2',     True,  1, 1, 0, True,  False),  # 3-way gate, graph CL but no projection/use_graph=False
    ('abl_m3',     True,  1, 1, 1, True,  False),  # 3-way gate, full graph CL
    ('abl_m2_new', True,  1, 1, 0, False, True),   # B-route + CL injection (projection trained)
]


def make_normfix_predict(model):
    def patched(user_ids, item_ids, interaction_counts=None, images=None, texts=None):
        cf_emb = model.cf_module.get_item_representation(item_ids)
        kg_emb, _ = model.kg_module(user_ids, item_ids)
        mm_emb = torch.zeros_like(cf_emb)
        cf_emb = F.normalize(cf_emb, p=2, dim=-1)
        kg_emb = F.normalize(kg_emb, p=2, dim=-1)
        if model.use_improvement_a:
            cf_confidence = model.cf_module.get_confidence(user_ids)
            fused, _ = model.adaptive_gate(cf_emb, kg_emb, mm_emb,
                                           interaction_counts=interaction_counts,
                                           cf_confidence=cf_confidence)
        else:
            adj = torch.eye(user_ids.size(0), device=user_ids.device)
            fused, _ = model.gat_fusion(cf_emb, kg_emb, mm_emb, adj)
        if model.use_improvement_b and model.graph is not None and model.inject_cl_signal:
            entity_ids = item_ids + model.graph['item_offset']
            fused = fused + 0.1 * model.graph_contrastive.get_node_repr(entity_ids)
        scores = model.predict(fused).squeeze(-1)
        return torch.sigmoid(scores)
    return patched


def score_test_users(model, test_users, n_items, user_count, user_train_items, device):
    model.predict_score = make_normfix_predict(model)
    item_t = torch.arange(n_items, device=device, dtype=torch.long)
    uniq = set()
    with torch.no_grad():
        for u in test_users:
            u_t = torch.full((n_items,), u, device=device, dtype=torch.long)
            c_t = torch.full((n_items,), user_count.get(u, 1), device=device, dtype=torch.long)
            scores = model.predict_score(u_t, item_t, interaction_counts=c_t)
            tr = user_train_items.get(u)
            if tr:
                scores[list(tr)] = -1e9
            topk = torch.topk(scores, 10).indices.cpu().numpy()
            for it in topk:
                uniq.add(int(it))
    return len(uniq)


def main():
    device = 'cuda'
    products, users, interactions = load_data('data/amazon_fashion_real.json')
    data = prepare_training_data(users, interactions, seed=42)
    n_users, n_items = data['n_users'], data['n_items']
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])
    uca = torch.zeros(n_users, dtype=torch.float32)
    for u, c in data['user_count'].items():
        uca[u] = c
    rng = np.random.RandomState(42)
    user_items = {}
    for it in interactions:
        uid, iid = int(it['user_id']), int(it['product_id'])
        if uid in data['user2idx'] and iid in data['item2idx']:
            user_items.setdefault(data['user2idx'][uid], []).append(data['item2idx'][iid])
    test_users_all = [u for u, items in user_items.items() if len(items) >= 2]
    test_users = [test_users_all[i] for i in rng.choice(len(test_users_all), size=500, replace=False)]
    user_train_items = {}
    for (u, i), _ in data['train_data']:
        user_train_items.setdefault(u, set()).add(i)
    user_count = {u: int(uca[u].item()) for u in range(n_users)}

    results = []
    for tag, mm, a, b, c, use_proc, inject_cl in CONFIGS:
        ckpt = Path('checkpoints') / tag / 'best_model.pt'
        if not ckpt.exists():
            print(f'[skip] {tag} no ckpt', flush=True)
            continue
        m = KGHybridRec(
            n_users=n_users, n_items=n_items,
            n_entities=graph['n_total'], n_relations=graph['n_relations'],
            embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
            use_multimodal=mm,
            use_improvement_a=bool(a), use_improvement_b=bool(b), use_improvement_c=bool(c),
            use_processors=use_proc,
            inject_cl_signal=inject_cl,
            graph=graph, n_interactions=uca,
        ).to(device)
        sd = torch.load(ckpt, map_location=device)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if 'interaction_counts' not in k]
        print(f'\n[{tag}] mm={mm} a={a} b={b} c={c} proc={use_proc} inject={inject_cl}', flush=True)
        print(f'  load: real_missing={len(real_missing)} unexpected={len(unexpected)}', flush=True)
        m.eval()

        t0 = time.time()
        cov = score_test_users(m, test_users, n_items, user_count, user_train_items, device)
        elapsed = time.time() - t0
        cov_ratio = cov / n_items
        print(f'  normfix cf_kg_norm Coverage@10 = {cov} items = {cov_ratio:.4f}  time={elapsed:.0f}s',
              flush=True)
        results.append({'tag': tag, 'normfix_cov_items': cov,
                        'normfix_cov_ratio': cov_ratio,
                        'config': {'mm': mm, 'a': a, 'b': b, 'c': c,
                                   'proc': use_proc, 'inject_cl': inject_cl}})

    out = Path('experiments/results/normfix_ablation_summary.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'\n[saved] {out}', flush=True)


if __name__ == '__main__':
    main()