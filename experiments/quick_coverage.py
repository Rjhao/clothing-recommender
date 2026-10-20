"""Quick diversity check for abl_m2_new — skip HR@10, only Coverage baseline + normfix.

用法: PYTHONUTF8=1 /c/Python311/python.exe -u experiments/quick_coverage.py
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


TAG = 'abl_m2_new'
USE_A, USE_B, USE_C = 1, 1, 0


def make_patched_predict(model, mode='baseline'):
    def patched(user_ids, item_ids, interaction_counts=None, images=None, texts=None):
        cf_emb = model.cf_module.get_item_representation(item_ids)
        kg_emb, _ = model.kg_module(user_ids, item_ids)
        mm_emb = torch.zeros_like(cf_emb)
        if mode == 'cf_kg_norm':
            cf_emb = F.normalize(cf_emb, p=2, dim=-1)
            kg_emb = F.normalize(kg_emb, p=2, dim=-1)
        cf_confidence = model.cf_module.get_confidence(user_ids)
        fused, _ = model.adaptive_gate(cf_emb, kg_emb, mm_emb,
                                       interaction_counts=interaction_counts,
                                       cf_confidence=cf_confidence)
        if model.use_improvement_b and model.graph is not None and model.inject_cl_signal:
            entity_ids = item_ids + model.graph['item_offset']
            fused = fused + 0.1 * model.graph_contrastive.get_node_repr(entity_ids)
        scores = model.predict(fused).squeeze(-1)
        return torch.sigmoid(scores)
    return patched


def score_test_users(model, test_users, n_items, user_count, user_train_items, device, mode):
    model.predict_score = make_patched_predict(model, mode)
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

    ckpt = Path('checkpoints') / TAG / 'best_model.pt'
    model = KGHybridRec(
        n_users=n_users, n_items=n_items,
        n_entities=graph['n_total'], n_relations=graph['n_relations'],
        embedding_dim=64, n_hops=2, n_heads=4, n_layers=2,
        use_multimodal=True,
        use_improvement_a=bool(USE_A), use_improvement_b=bool(USE_B), use_improvement_c=bool(USE_C),
        use_processors=False,
        inject_cl_signal=True,
        graph=graph, n_interactions=uca,
    ).to(device)
    sd = torch.load(ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if 'interaction_counts' not in k]
    print(f'[load] real_missing={len(real_missing)} unexpected={len(unexpected)}', flush=True)
    model.eval()

    print('[coverage@10] baseline ...', flush=True)
    t0 = time.time()
    cov = score_test_users(model, test_users, n_items, user_count, user_train_items, device,
                           mode='baseline')
    print(f'[coverage] baseline = {cov} items  time={time.time()-t0:.0f}s', flush=True)

    print('[coverage@10] normfix cf_kg_norm ...', flush=True)
    t0 = time.time()
    cov_norm = score_test_users(model, test_users, n_items, user_count, user_train_items, device,
                                mode='cf_kg_norm')
    cov_norm_ratio = cov_norm / n_items
    print(f'[coverage] cf_kg_norm = {cov_norm} items = {cov_norm_ratio:.4f}  '
          f'time={time.time()-t0:.0f}s', flush=True)

    out = {
        'tag': TAG,
        'coverage@10_items': cov,
        'coverage@10_ratio': cov / n_items,
        'normfix_cf_kg_norm_items': cov_norm,
        'normfix_cf_kg_norm_ratio': cov_norm_ratio,
    }
    p = Path('experiments/results') / f'{TAG}_coverage.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'[saved] {p}', flush=True)


if __name__ == '__main__':
    main()