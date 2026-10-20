"""验收 2: 修复加载后, abl_m1/m2/m3 应复现 13/14/14 件 (500 用户协议).

用法: PYTHONUTF8=1 /c/Python311/python.exe -u experiments/verify_repro.py
"""
import sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender.hybrid_rec import KGHybridRec
from recommender.train import load_data, prepare_training_data
from recommender.kgat.hetero_graph import build_graph_for_model


def main():
    device = 'cuda'
    print('[load data] ...', flush=True)
    products, users, interactions = load_data('data/amazon_fashion_real.json')
    data = prepare_training_data(users, interactions, seed=42)
    n_users = data['n_users']
    n_items = data['n_items']
    graph = build_graph_for_model(products, data['train_data'], data['user2idx'], data['item2idx'])
    uca = torch.zeros(n_users, dtype=torch.float32)
    for u, c in data['user_count'].items():
        uca[u] = c

    CONFIGS = [
        ('abl_m1', True, 1, 0, 0, 13),
        ('abl_m2', True, 1, 1, 0, 14),
        ('abl_m3', True, 1, 1, 1, 14),
    ]

    rng = np.random.RandomState(42)
    user_items = {}
    for it in interactions:
        uid = int(it['user_id'])
        iid = int(it['product_id'])
        if uid in data['user2idx'] and iid in data['item2idx']:
            user_items.setdefault(data['user2idx'][uid], []).append(data['item2idx'][iid])
    test_users_all = [u for u, items in user_items.items() if len(items) >= 2]
    test_users = [test_users_all[i] for i in rng.choice(len(test_users_all), size=500, replace=False)]
    user_train_items = {}
    for (u, i), _ in data['train_data']:
        user_train_items.setdefault(u, set()).add(i)
    item_t = torch.arange(n_items, device=device, dtype=torch.long)

    for tag, mm, a, b, c, expected in CONFIGS:
        t0 = time.time()
        m = KGHybridRec(
            n_users=n_users, n_items=n_items,
            n_entities=graph['n_total'], n_relations=graph['n_relations'],
            use_multimodal=mm,
            use_improvement_a=bool(a), use_improvement_b=bool(b), use_improvement_c=bool(c),
            use_processors=True,
            inject_cl_signal=False,  # 复现 7-20 数字, abl_m3 不被 cl_signal 干扰
            graph=graph, n_interactions=uca,
        ).to(device)
        sd = torch.load(f'checkpoints/{tag}/best_model.pt', map_location=device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        uniq = set()
        with torch.no_grad():
            for u in test_users:
                u_t = torch.full((n_items,), u, device=device, dtype=torch.long)
                c_t = torch.full((n_items,), data['user_count'].get(u, 1), device=device, dtype=torch.long)
                scores = m.predict_score(u_t, item_t, interaction_counts=c_t)
                tr = user_train_items.get(u)
                if tr:
                    scores[list(tr)] = -1e9
                topk = torch.topk(scores, 10).indices.cpu().numpy()
                for it in topk:
                    uniq.add(int(it))
        elapsed = time.time() - t0
        mark = 'OK ' if len(uniq) == expected else 'DIFF'
        print(f'{mark} {tag:8s} unique={len(uniq):4d} expected={expected} time={elapsed:.0f}s', flush=True)


if __name__ == '__main__':
    main()