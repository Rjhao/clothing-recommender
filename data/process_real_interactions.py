"""从 6 个月真实 Amazon gz 数据生成训练用 JSON（替换合成版 expanded.json）。

口径：purchase + cart 作为隐式正反馈（电商推荐标准强信号）。
跨月累积同一 user-product，去重为一条交互，记 interaction_count 与 last_time。
按 (user_id, last_time) 排序输出，使 split.py / evaluate.py 的 leave-one-out
取 items[-1] 是真实时序最后的交互（真实数据相比合成数据的核心优势）。
k-core 过滤：用户 ≥3 交互（留一法需 train+val+test），商品 ≥2 被交互（去噪）。

输出为 data/amazon_fashion_real.json，train.py / evaluate.py /
hetero_graph 直接可用。

Usage:
    python -m data.process_real_interactions \
        --files data/2019-Nov.csv.gz data/2019-Dec.csv.gz data/2020-Jan.csv.gz \
                data/2020-Feb.csv.gz data/2020-Mar.csv.gz data/2020-Apr.csv.gz \
        --user-core 3 --item-core 2 \
        --out data/amazon_fashion_real.json
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
    is_fashion, categorize_product, extract_style_from_text,
    extract_season,
)
from data.name_utils import build_unique_name

PAT = None  # 向量化用
USECOLS = ['event_type', 'product_id', 'category_code', 'brand', 'price', 'user_id', 'event_time']
FASHION_KW = [
    'apparel', 'clothing', 'fashion', 'shoes', 'footwear',
    'accessories', 'jewelry', 'watch', 'bag', 'purse',
    'underwear', 'swimwear', 'lingerie', 'sportswear',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True, help='gz/csv 文件路径（按时间顺序）')
    ap.add_argument('--events', nargs='+', default=['purchase', 'cart'],
                    help='取哪些 event_type 作为正反馈')
    ap.add_argument('--user-core', type=int, default=3)
    ap.add_argument('--item-core', type=int, default=2)
    ap.add_argument('--out', type=str, default='data/amazon_fashion_real.json')
    ap.add_argument('--stats-out', type=str, default='data/real_data_stats.json')
    args = ap.parse_args()

    pat = '|'.join(FASHION_KW)
    event_set = set(args.events)
    t0 = time.time()

    # 累积器（跨月合并）
    # uid -> pid -> {'times':[str], 'prices':[float]}
    user_item = defaultdict(lambda: defaultdict(lambda: {'times': [], 'prices': []}))
    # pid -> meta
    prod_meta = {}  # pid(int) -> {category_code, brand, category, style, season}
    # uid -> category_code 计数（推断 preferred_style）
    user_cat_count = defaultdict(lambda: defaultdict(int))
    # uid -> pid set（用于 k-core，= user_item 的 key 视图，下面直接用 user_item）

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
            # 只保留目标 event_type
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

            for uid, pid, tm, pr, ccv, br in zip(uids, pids, times, prices, ccs, brands):
                uid = int(uid); pid = int(pid)
                slot = user_item[uid][pid]
                slot['times'].append(str(tm))
                slot['prices'].append(float(pr) if pr == pr else 0.0)  # NaN 检查
                # product meta（首次见到的字段为准，category_code/brand 一致）
                if pid not in prod_meta:
                    cat = categorize_product(ccv)
                    prod_meta[pid] = {
                        'category_code': str(ccv) if ccv else '',
                        'brand': str(br) if br else '',
                        'category': cat if cat != 'unknown' else None,
                        'style': extract_style_from_text(ccv, br),
                        'season': extract_season(ccv),
                    }
                # user 偏好
                user_cat_count[uid][str(ccv)] += 1

            if (ci + 1) % 5 == 0:
                print(f"  chunk {ci+1}: total={total_rows:,} fashion={fashion_rows:,} "
                      f"kept={kept_rows:,} users={len(user_item):,} "
                      f"elapsed={time.time()-ft:.0f}s", flush=True)
        print(f"  done {path} in {time.time()-ft:.0f}s", flush=True)

    print(f"\n=== Raw accumulation done ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"total_rows={total_rows:,} fashion_rows={fashion_rows:,} kept_rows={kept_rows:,}")
    print(f"event_counts={dict(event_counts)}")
    print(f"raw users={len(user_item):,} raw user-item pairs={sum(len(v) for v in user_item.values()):,}")
    print(f"raw products={len(prod_meta):,}")

    # ---- k-core 过滤 ----
    # user-item 邻接：uid -> set(pid), pid -> set(uid)
    def build_adj():
        ui = {u: set(d.keys()) for u, d in user_item.items()}
        iu = defaultdict(set)
        for u, pids in ui.items():
            for p in pids:
                iu[p].add(u)
        return ui, iu

    ui, iu = build_adj()
    print(f"\n--- k-core (user≥{args.user_core}, item≥{args.item_core}) ---", flush=True)
    iteration = 0
    while True:
        iteration += 1
        # 去掉交互数不足的用户
        bad_users = [u for u, pids in ui.items() if len(pids) < args.user_core]
        # 去掉被交互数不足的商品
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
            'color': None,
            'size': None,
            'material': None,
            'price': 0.0,
            'platform': 'AMAZON',
            'source': f"https://www.amazon.com/dp/{pid}",
            'style': m.get('style', 'casual'),
            'season': m.get('season', 'all_season'),
            'styles': [m.get('style', 'casual')] if m.get('style') else [],
            'popularity': len(iu[pid]),  # 被多少用户交互
            '_original_name': cc,
            '_original_category': m.get('category'),
        }
        products.append(obj)
        pid_to_obj[pid] = obj

    # 填商品 price（用其所有交互的均值）
    pid_prices = defaultdict(list)
    interactions_raw = []  # (uid, pid, count, last_time, price_mean)
    for uid in kept_users:
        d = user_item[uid]
        for pid, slot in d.items():
            if pid not in kept_items:
                continue
            times = slot['times']
            prices = [p for p in slot['prices'] if p > 0]
            last_time = max(times)  # ISO 字符串比较
            price_mean = sum(prices) / len(prices) if prices else 0.0
            interactions_raw.append((uid, pid, len(times), last_time, price_mean))
            pid_prices[pid].extend(prices)

    for pid, obj in pid_to_obj.items():
        ps = pid_prices.get(pid, [])
        obj['price'] = round(sum(ps) / len(ps), 2) if ps else 0.0

    # users
    users = []
    for uid in sorted(kept_users):
        # preferred_style：用其最频繁 category_code 推断
        cc_counts = user_cat_count.get(uid, {})
        top_cc = max(cc_counts.items(), key=lambda x: x[1])[0] if cc_counts else ''
        style = extract_style_from_text(top_cc)
        users.append({
            'user_id': str(uid),
            'preferred_style': style,
            'interaction_count': len(user_item[uid]),  # 去重商品数
        })

    # interactions：按 (uid, last_time) 排序，保证 split 时序
    interactions_raw.sort(key=lambda x: (x[0], x[3]))
    interactions = []
    for uid, pid, cnt, last_time, price_mean in interactions_raw:
        interactions.append({
            'user_id': uid,
            'product_id': pid,
            'interaction_count': cnt,
            'price': round(price_mean, 2),
            'last_time': last_time,
        })

    # ---- 用户交互数分布 ----
    import numpy as np
    ucnts = np.array([len(user_item[u]) for u in kept_users])
    icnts = np.array([len(iu[p]) for p in kept_items])

    out = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'source': 'real Amazon 2019-Nov ~ 2020-Apr (purchase+cart)',
        'files': args.files,
        'events': args.events,
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
