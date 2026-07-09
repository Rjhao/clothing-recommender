"""对已生成的 view json 做二次 k-core 过滤，压到可训练规模。

view 全量 143万用户/9.4万商品/2228万交互太大，M3 跑不动。
本脚本在内存里重新 k-core，输出更小但仍稀疏的子集。

策略：逐步收紧 user-core / item-core，目标用户 10-15万、商品 2-3万、
交互 100-200万（M3 每 epoch 5-10min 可接受）。

Usage:
    python -m data.refilter_view \
        --in data/amazon_fashion_view.json \
        --out data/amazon_fashion_view_filtered.json \
        --user-core 20 --item-core 10
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--user-core', type=int, default=20)
    ap.add_argument('--item-core', type=int, default=10)
    ap.add_argument('--max-users', type=int, default=150000, help='若用户超过此数，自动提高 user-core')
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading {args.inp}...", flush=True)
    with open(args.inp) as f:
        d = json.load(f)
    print(f"  loaded in {time.time()-t0:.0f}s: {len(d['users'])} users, "
          f"{len(d['products'])} products, {len(d['interactions'])} interactions", flush=True)

    interactions = d['interactions']
    products = d['products']
    users = d['users']

    # 建索引
    pid_to_prod = {int(p['product_id']): p for p in products}

    def do_kcore(uc, ic):
        ui = defaultdict(set)
        iu = defaultdict(set)
        for inter in interactions:
            uid = int(inter['user_id'])
            iid = int(inter['product_id'])
            ui[uid].add(iid)
            iu[iid].add(uid)
        it = 0
        while True:
            it += 1
            bad_u = [u for u, ps in ui.items() if len(ps) < uc]
            bad_i = [i for i, us in iu.items() if len(us) < ic]
            if not bad_u and not bad_i:
                break
            for u in bad_u:
                for p in ui[u]:
                    iu[p].discard(u)
                del ui[u]
            for p in bad_i:
                for u in iu[p]:
                    ui[u].discard(p)
                del iu[p]
        return ui, iu, it

    uc, ic = args.user_core, args.item_core
    while True:
        print(f"\n--- k-core user≥{uc} item≥{ic} ---", flush=True)
        ui, iu, it = do_kcore(uc, ic)
        n_u, n_i, n_int = len(ui), len(iu), sum(len(v) for v in ui.values())
        print(f"  iters={it} users={n_u:,} items={n_i:,} interactions={n_int:,} "
              f"density={n_int/(n_u*n_i) if n_u and n_i else 0:.6f}", flush=True)
        if n_u <= args.max_users or uc >= 100:
            break
        uc = int(uc * 1.5) + 5
        print(f"  too many users, raising user-core to {uc}", flush=True)

    kept_users = set(ui.keys())
    kept_items = set(iu.keys())

    # 重建 products/users/interactions
    new_products = [pid_to_prod[p] for p in sorted(kept_items)]
    for p in new_products:
        p['popularity'] = len(iu[int(p['product_id'])])

    uid_set = {int(u['user_id']) for u in users}
    new_users = []
    for u in users:
        uid = int(u['user_id'])
        if uid in kept_users:
            new_users.append({**u, 'interaction_count': len(ui[uid])})

    new_interactions = [i for i in interactions
                        if int(i['user_id']) in kept_users
                        and int(i['product_id']) in kept_items]
    # 保持时序排序（原已按 (uid, last_time) 排）
    new_interactions.sort(key=lambda x: (int(x['user_id']), x['last_time']))

    ucnts = np.array([len(ui[u]) for u in kept_users])
    icnts = np.array([len(iu[p]) for p in kept_items])

    out = {
        'timestamp': d.get('timestamp'),
        'source': d.get('source') + f' [refiltered user≥{uc} item≥{ic}]',
        'files': d.get('files'),
        'events': d.get('events'),
        'event_weights': d.get('event_weights'),
        'k_core': {'user': uc, 'item': ic},
        'stats': {
            'n_users': len(new_users),
            'n_products': len(new_products),
            'n_interactions': len(new_interactions),
            'density': round(len(new_interactions) / (len(new_users) * len(new_products)), 6),
            'user_interactions': {
                'mean': float(ucnts.mean()), 'median': float(np.median(ucnts)),
                'p90': float(np.percentile(ucnts, 90)), 'max': int(ucnts.max()),
                'min': int(ucnts.min()),
            },
            'item_interactions': {
                'mean': float(icnts.mean()), 'median': float(np.median(icnts)),
                'p90': float(np.percentile(icnts, 90)), 'max': int(icnts.max()),
                'min': int(icnts.min()),
            },
        },
        'products': new_products,
        'users': new_users,
        'interactions': new_interactions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\n=== DONE ({time.time()-t0:.0f}s) ===")
    print(json.dumps(out['stats'], indent=2, ensure_ascii=False))
    print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()
