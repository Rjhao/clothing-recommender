"""从 6 个月真实 Amazon 数据生成 view+purchase 混合训练 JSON。

与 process_real_interactions.py 的区别：
1. 支持 view 事件（稀疏+噪声场景，让简单 CF 下降，给 KG/M3 空间）
2. 事件权重：view=1, cart=3, purchase=5（购买是更强意图）
3. 内存安全：同一 user-product 只存聚合统计（count/last_time/weighted_score），
   不存每次事件的时间戳列表（view 数据量大，存列表会爆内存）
4. 严格 k-core：用户≥5 交互（view 数据丰富，可严格过滤）

输出 schema 同 amazon_fashion_real.json，train.py / evaluate.py 直接可用。
interaction_count 字段 = 该 user-product 的加权交互强度（event 权重之和）。

Usage:
    python -m data.process_real_view \
        --files data/2019-Nov.csv.gz ... data/2020-Apr.csv.gz \
        --user-core 5 --item-core 2 \
        --out data/amazon_fashion_view.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.import_amazon_fashion import (
    categorize_product, extract_style_from_text, extract_season,
)
from data.name_utils import build_unique_name

USECOLS = ['event_type', 'product_id', 'category_code', 'brand', 'price', 'user_id', 'event_time']
FASHION_KW = [
    'apparel', 'clothing', 'fashion', 'shoes', 'footwear',
    'accessories', 'jewelry', 'watch', 'bag', 'purse',
    'underwear', 'swimwear', 'lingerie', 'sportswear',
]
# 事件权重：购买最强，加购次之，浏览最弱
EVENT_WEIGHTS = {'view': 1, 'cart': 3, 'purchase': 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--events', nargs='+', default=['view', 'cart', 'purchase'])
    ap.add_argument('--user-core', type=int, default=5)
    ap.add_argument('--item-core', type=int, default=2)
    ap.add_argument('--out', type=str, default='data/amazon_fashion_view.json')
    ap.add_argument('--stats-out', type=str, default='data/view_data_stats.json')
    args = ap.parse_args()

    pat = '|'.join(FASHION_KW)
    event_set = set(args.events)
    t0 = time.time()

    # 累积器：内存安全版
    # user_item[uid][pid] = {'w': 加权强度, 'cnt': 事件数, 'last': 最近时间(str), 'prices': [float]}
    # 注意 prices 仍用列表（商品数有限，21k 量级，OK）；user-item 对数会很大
    user_item = defaultdict(dict)  # uid -> pid -> dict
    prod_meta = {}
    user_cat_count = defaultdict(lambda: defaultdict(int))

    total_rows = 0
    fashion_rows = 0
    kept_rows = 0
    event_counts = defaultdict(int)

    for path in args.files:
        print(f"\n=== Reading {path} ===", flush=True)
        ft = time.time()
        reader = pd.read_csv(
            path, chunksize=1_000_000, usecols=USECOLS,
            dtype={'category_code': str, 'brand': str, 'event_type': str},
        )
        for ci, chunk in enumerate(reader):
            total_rows += len(chunk)
            cc = chunk['category_code'].fillna('').str.lower()
            fmask = cc.str.contains(pat, regex=True, na=False)
            fc = chunk[fmask]
            fashion_rows += len(fc)
            if len(fc) == 0:
                continue
            fc = fc[fc['event_type'].isin(event_set)]
            if len(fc) == 0:
                continue
            fc = fc.dropna(subset=['user_id', 'product_id'])
            kept_rows += len(fc)

            for et in fc['event_type']:
                event_counts[et] += 1

            uids = fc['user_id'].astype('int64').values
            pids = fc['product_id'].astype('int64').values
            times = fc['event_time'].astype(str).values
            prices = fc['price'].astype(float).values
            ccs = fc['category_code'].fillna('').values
            brands = fc['brand'].fillna('').values
            ets = fc['event_type'].values

            for uid, pid, tm, pr, ccv, br, et in zip(
                uids, pids, times, prices, ccs, brands, ets
            ):
                uid = int(uid); pid = int(pid)
                w = EVENT_WEIGHTS.get(et, 1)
                d = user_item[uid]
                if pid in d:
                    s = d[pid]
                    s['w'] += w
                    s['cnt'] += 1
                    if tm > s['last']:
                        s['last'] = str(tm)
                    if pr == pr and pr > 0:
                        s['ps'].append(float(pr))
                else:
                    d[pid] = {
                        'w': w, 'cnt': 1, 'last': str(tm),
                        'ps': [float(pr)] if (pr == pr and pr > 0) else [],
                    }
                if pid not in prod_meta:
                    cat = categorize_product(ccv)
                    prod_meta[pid] = {
                        'category_code': str(ccv) if ccv else '',
                        'brand': str(br) if br else '',
                        'category': cat if cat != 'unknown' else None,
                        'style': extract_style_from_text(ccv, br),
                        'season': extract_season(ccv),
                    }
                user_cat_count[uid][str(ccv)] += 1

            if (ci + 1) % 5 == 0:
                # 估算内存
                import os
                mem_mb = 0
                print(f"  chunk {ci+1}: total={total_rows:,} fashion={fashion_rows:,} "
                      f"kept={kept_rows:,} users={len(user_item):,} "
                      f"pairs={sum(len(v) for v in user_item.values()):,} "
                      f"elapsed={time.time()-ft:.0f}s", flush=True)
        print(f"  done {path} in {time.time()-ft:.0f}s", flush=True)
        # 每个月后报告内存
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"  [mem] {mem:.0f} MB", flush=True)
        except Exception:
            pass

    print(f"\n=== Raw accumulation done ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"total_rows={total_rows:,} fashion_rows={fashion_rows:,} kept_rows={kept_rows:,}")
    print(f"event_counts={dict(event_counts)}")
    print(f"raw users={len(user_item):,} raw user-item pairs={sum(len(v) for v in user_item.values()):,}")
    print(f"raw products={len(prod_meta):,}")

    # ---- k-core 过滤 ----
    ui = {u: set(d.keys()) for u, d in user_item.items()}
    iu = defaultdict(set)
    for u, pids in ui.items():
        for p in pids:
            iu[p].add(u)

    print(f"\n--- k-core (user≥{args.user_core}, item≥{args.item_core}) ---", flush=True)
    iteration = 0
    while True:
        iteration += 1
        bad_users = [u for u, pids in ui.items() if len(pids) < args.user_core]
        bad_items = [p for p, usrs in iu.items() if len(usrs) < args.item_core]
        if not bad_users and not bad_items:
            break
        for u in bad_users:
            for p in ui[u]:
                iu[p].discard(u)
            del ui[u]
        for p in bad_items:
            for u in iu[p]:
                ui[u].discard(p)
            del iu[p]
        print(f"  iter {iteration}: removed {len(bad_users)} users, {len(bad_items)} items; "
              f"remaining users={len(ui):,} items={len(iu):,}", flush=True)

    kept_users = set(ui.keys())
    kept_items = set(iu.keys())
    print(f"\nk-core done: users={len(kept_users):,} items={len(kept_items):,} "
          f"interactions={sum(len(v) for v in ui.values()):,}")

    # ---- 构造 products / users / interactions ----
    products = []
    pid_to_obj = {}
    for pid in sorted(kept_items):
        m = prod_meta.get(pid, {})
        cc = m.get('category_code', '')
        base_name = cc.replace('.', ' ').title() if cc else str(pid)
        brand = m.get('brand', '')
        obj = {
            'product_id': str(pid),
            'name': build_unique_name(base_name, brand, pid),
            'category': m.get('category'),
            'category_code': cc,
            'brand': brand,
            'color': None, 'size': None, 'material': None,
            'price': 0.0,
            'platform': 'AMAZON',
            'source': f"https://www.amazon.com/dp/{pid}",
            'style': m.get('style', 'casual'),
            'season': m.get('season', 'all_season'),
            'styles': [m.get('style', 'casual')] if m.get('style') else [],
            'popularity': len(iu[pid]),
            '_original_name': cc,
            '_original_category': m.get('category'),
        }
        products.append(obj)
        pid_to_obj[pid] = obj

    pid_prices = defaultdict(list)
    interactions_raw = []
    for uid in kept_users:
        d = user_item[uid]
        for pid, s in d.items():
            if pid not in kept_items:
                continue
            interactions_raw.append((uid, pid, s['w'], s['last'], s['ps']))
            pid_prices[pid].extend(s['ps'])

    for pid, obj in pid_to_obj.items():
        ps = pid_prices.get(pid, [])
        obj['price'] = round(sum(ps) / len(ps), 2) if ps else 0.0

    users = []
    for uid in sorted(kept_users):
        cc_counts = user_cat_count.get(uid, {})
        top_cc = max(cc_counts.items(), key=lambda x: x[1])[0] if cc_counts else ''
        style = extract_style_from_text(top_cc)
        users.append({
            'user_id': str(uid),
            'preferred_style': style,
            'interaction_count': len(user_item[uid]),
        })

    # 按 (uid, last_time) 排序，保证 split 时序
    interactions_raw.sort(key=lambda x: (x[0], x[3]))
    interactions = []
    for uid, pid, w, last_time, ps in interactions_raw:
        interactions.append({
            'user_id': uid,
            'product_id': pid,
            'interaction_count': w,  # 加权强度（view=1, cart=3, purchase=5 之和）
            'price': round(sum(ps) / len(ps), 2) if ps else 0.0,
            'last_time': last_time,
        })

    import numpy as np
    ucnts = np.array([len(user_item[u]) for u in kept_users])
    icnts = np.array([len(iu[p]) for p in kept_items])
    wcnts = np.array([i['interaction_count'] for i in interactions])

    out = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'source': 'real Amazon 2019-Nov ~ 2020-Apr (view+cart+purchase, weighted)',
        'files': args.files,
        'events': args.events,
        'event_weights': EVENT_WEIGHTS,
        'k_core': {'user': args.user_core, 'item': args.item_core},
        'stats': {
            'total_rows_scanned': total_rows,
            'fashion_rows': fashion_rows,
            'kept_event_rows': kept_rows,
            'event_counts': dict(event_counts),
            'n_users': len(users),
            'n_products': len(products),
            'n_interactions': len(interactions),
            'density': round(len(interactions) / (len(users) * len(products)), 6),
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
            'weighted_strength': {
                'mean': float(wcnts.mean()), 'median': float(np.median(wcnts)),
                'p90': float(np.percentile(wcnts, 90)), 'max': int(wcnts.max()),
            },
        },
        'products': products,
        'users': users,
        'interactions': interactions,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    with open(args.stats_out, 'w', encoding='utf-8') as f:
        json.dump(out['stats'], f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ({time.time()-t0:.0f}s) ===")
    print(json.dumps(out['stats'], indent=2, ensure_ascii=False))
    print(f"Saved -> {args.out}")
    print(f"Stats -> {args.stats_out}")


if __name__ == '__main__':
    main()
