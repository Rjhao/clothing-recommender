"""流式探查 Amazon 月度 gz 数据的 fashion 规模。

低内存：pandas chunk 流式读，只取必要列，向量化过滤。
输出：每月 + 汇总统计（总行数 / fashion 行数 / event_type 分布 /
purchase+cart 去重 user-product 交互 / 用户数 / 商品数 / 用户交互数分位）。

Usage:
    python -m data.scan_amazon data/2019-Nov.csv.gz
    python -m data.scan_amazon data/2019-Nov.csv.gz data/2019-Dec.csv.gz   # 多月
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

FASHION = [
    'apparel', 'clothing', 'fashion', 'shoes', 'footwear',
    'accessories', 'jewelry', 'watch', 'bag', 'purse',
    'underwear', 'swimwear', 'lingerie', 'sportswear',
]
PAT = '|'.join(FASHION)
USECOLS = ['event_type', 'product_id', 'category_code', 'brand', 'price', 'user_id', 'event_time']


def scan_one(path: str) -> dict:
    t0 = time.time()
    total = 0
    fashion_total = 0
    fashion_event = defaultdict(int)
    pc_user_items = defaultdict(set)      # purchase+cart 去重
    pur_user_items = defaultdict(set)     # purchase 去重
    view_user_items = defaultdict(set)    # view 去重（参考）
    all_fashion_users = set()
    all_fashion_items = set()

    reader = pd.read_csv(
        path, chunksize=1_000_000, usecols=USECOLS,
        dtype={'category_code': str, 'brand': str, 'event_type': str},
    )
    for i, chunk in enumerate(reader):
        total += len(chunk)
        cc = chunk['category_code'].fillna('').str.lower()
        fmask = cc.str.contains(PAT, regex=True, na=False)
        fc = chunk[fmask]
        fashion_total += len(fc)
        if len(fc):
            for et, c in fc['event_type'].value_counts().items():
                fashion_event[et] += int(c)
            # 去 NaN
            fc = fc.dropna(subset=['user_id', 'product_id'])
            fc_uid = fc['user_id'].astype('int64')
            fc_pid = fc['product_id'].astype('int64')
            all_fashion_users.update(fc_uid.tolist())
            all_fashion_items.update(fc_pid.tolist())
            et_arr = fc['event_type'].values
            for uid, pid, et in zip(fc_uid.values, fc_pid.values, et_arr):
                if et in ('purchase', 'cart'):
                    pc_user_items[int(uid)].add(int(pid))
                if et == 'purchase':
                    pur_user_items[int(uid)].add(int(pid))
                if et == 'view':
                    view_user_items[int(uid)].add(int(pid))
        if (i + 1) % 5 == 0:
            print(f"  [{Path(path).name}] chunk {i+1}: total={total:,} fashion={fashion_total:,} "
                  f"pc_users={len(pc_user_items):,} elapsed={time.time()-t0:.0f}s", flush=True)

    def dist(user_items):
        if not user_items:
            return None
        counts = sorted(len(v) for v in user_items.values())
        n = len(counts)
        import numpy as np
        a = np.array(counts)
        return {
            'n_users': n,
            'n_interactions': int(a.sum()),
            'mean': float(a.mean()),
            'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)),
            'max': int(a.max()),
            'min': int(a.min()),
            'n_users_ge3': int((a >= 3).sum()),
            'n_users_ge5': int((a >= 5).sum()),
        }

    res = {
        'file': Path(path).name,
        'total_rows': total,
        'fashion_rows': fashion_total,
        'fashion_event_type': dict(fashion_event),
        'fashion_unique_users': len(all_fashion_users),
        'fashion_unique_products': len(all_fashion_items),
        'purchase_cart': dist(pc_user_items),
        'purchase_only': dist(pur_user_items),
        'view_only': dist(view_user_items),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    return res


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(1)
    results = []
    for f in files:
        print(f"\n=== Scanning {f} ===", flush=True)
        results.append(scan_one(f))
        print(json.dumps(results[-1], indent=2, ensure_ascii=False), flush=True)

    # 汇总（跨月合并 user-item 集合需另算，这里只报每月）
    out_path = Path('data/scan_amazon_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary -> {out_path}")


if __name__ == '__main__':
    main()
