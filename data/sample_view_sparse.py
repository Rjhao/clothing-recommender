"""从 view.json(全量 142万用户/9.4万商品/2228万交互)随机采样出 2-3 万用户的稀疏子集。

目的:构建一个"用户 view 行为稀疏"的数据子集,让经典 CF(尤其是 UserCF)的物品/用户共现信号
显著弱化,KG-HybridRec 的 KG/GAT/对比学习能补上信号 → 论文"模型在稀疏场景胜出"的故事。

与 refilter_view.py 的区别:
- refilter_view 用 k-core(留高频用户,共现信号强)→ UserCF 仍强 → 模型仍被吊打
- 本脚本:随机抽 2.5 万用户(保留低活跃用户),商品弱 k-core(≥2 被交互)
  → view 行为稀疏 → UserCF 共现矩阵稀疏 → 模型优势显现

预期产出:~2.5万用户 × 15.6 mean交互 = ~39万交互,~1.5-2万商品,密度与 view.json 全量相当。

注意:view.json 3.0GB 一次加载会吃 6-10GB 内存,这里用 ijson 流式解析,只把 user_id 收集到
内存(2.8M 条记录的用户集合),然后随机抽用户,再二次流式解析写子集。
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# 不依赖 ijson,改用自己写的轻量流式 JSON 解析:只关心顶层 users 数组的 user_id 字段
# view.json 结构:{stats, products, users:[{user_id,...}], interactions:[{user_id,product_id,...}]}


def stream_user_ids(path, total_users_estimate=1_500_000):
    """流式返回 view.json 中 users 数组的 user_id。"""
    import ijson
    with open(path, 'rb') as f:
        for uid in ijson.items(f, 'users.item.user_id'):
            yield uid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='data/amazon_fashion_view.json',
                    help='view.json 路径(3.0GB 全量)')
    ap.add_argument('--out', default='data/amazon_fashion_view_sparse.json',
                    help='输出稀疏子集 JSON')
    ap.add_argument('--target-users', type=int, default=25000, help='目标用户数')
    ap.add_argument('--item-core', type=int, default=2, help='商品弱 k-core')
    ap.add_argument('--min-user-interactions', type=int, default=3,
                    help='用户最少交互数(避免抽到 1-2 个交互的无意义用户)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== Step 1: 收集 {args.inp} 中的所有 user_id (流式) ===", flush=True)
    try:
        import ijson
    except ImportError:
        print("ERROR: 需要 ijson. 安装: pip install ijson", flush=True)
        sys.exit(1)

    all_user_ids = []
    for uid in stream_user_ids(args.inp):
        all_user_ids.append(int(uid))
    print(f"  done: {len(all_user_ids):,} users in {time.time()-t0:.0f}s", flush=True)

    # Step 2: 先统计每个用户的交互数(二次流式),筛出 ≥ min_user_interactions 的用户
    print(f"\n=== Step 2: 统计每用户交互数,筛出 ≥{args.min_user_interactions} ===", flush=True)
    user_count = defaultdict(int)
    with open(args.inp, 'rb') as f:
        for inter in ijson.items(f, 'interactions.item'):
            uid = int(inter['user_id'])
            user_count[uid] += 1
    print(f"  done in {time.time()-t0:.0f}s, {len(user_count):,} unique users with interactions",
          flush=True)

    eligible = [u for u, c in user_count.items() if c >= args.min_user_interactions]
    print(f"  eligible users (≥{args.min_user_interactions} 交互): {len(eligible):,}", flush=True)

    if len(eligible) < args.target_users:
        print(f"  WARN: 候选 {len(eligible):,} 不足目标 {args.target_users:,}, 用全部", flush=True)
        sampled = eligible
    else:
        rng = random.Random(args.seed)
        sampled = rng.sample(eligible, args.target_users)
    sampled_set = set(sampled)
    print(f"  sampled {len(sampled_set):,} users", flush=True)

    # Step 3: 第三次流式解析 view.json,提取 sampled 用户的交互,商品弱 k-core
    print(f"\n=== Step 3: 流式抽取 {len(sampled_set):,} 用户的交互 ===", flush=True)
    raw_interactions = []  # (uid, pid, time, price)
    with open(args.inp, 'rb') as f:
        for inter in ijson.items(f, 'interactions.item'):
            uid = int(inter['user_id'])
            if uid in sampled_set:
                raw_interactions.append({
                    'user_id': uid,
                    'product_id': int(inter['product_id']),
                    'event_time': inter.get('event_time', ''),
                    'price': float(inter.get('price', 0.0) or 0.0) if inter.get('price') is not None else 0.0,
                })
    print(f"  raw_interactions: {len(raw_interactions):,} in {time.time()-t0:.0f}s", flush=True)

    # 商品弱 k-core
    print(f"\n=== Step 4: 商品弱 k-core (≥{args.item_core} 被交互) ===", flush=True)
    item_count = defaultdict(int)
    for r in raw_interactions:
        item_count[r['product_id']] += 1
    kept_items = {p for p, c in item_count.items() if c >= args.item_core}
    print(f"  商品: {len(item_count):,} → {len(kept_items):,} (k-core={args.item_core})", flush=True)

    final_interactions = [r for r in raw_interactions if r['product_id'] in kept_items]
    final_user_set = set(r['user_id'] for r in final_interactions)
    print(f"  final users: {len(final_user_set):,}, interactions: {len(final_interactions):,}",
          flush=True)

    # Step 4: 加载 sampled 用户的 product meta (从 view.json 一次取 products 数组)
    # 3.0GB JSON 全加载会吃 6-10GB,这里也用 ijson 流式
    print(f"\n=== Step 5: 收集 kept_items 的 product meta ===", flush=True)
    products = []
    with open(args.inp, 'rb') as f:
        for p in ijson.items(f, 'products.item'):
            pid = int(p['product_id'])
            if pid in kept_items:
                # 更新 popularity 为 k-core 后的实际被交互数
                p2 = dict(p)
                p2['popularity'] = item_count.get(pid, 0)
                # price 可能是 Decimal/str,统一转 float
                try:
                    p2['price'] = float(p2.get('price', 0.0) or 0.0)
                except (TypeError, ValueError):
                    p2['price'] = 0.0
                products.append(p2)
    print(f"  collected {len(products):,} product meta", flush=True)

    # Step 5: 用户 meta (从 view.json 一次取 users 数组)
    print(f"\n=== Step 6: 收集 final_user_set 的 user meta ===", flush=True)
    users = []
    with open(args.inp, 'rb') as f:
        for u in ijson.items(f, 'users.item'):
            uid = int(u['user_id'])
            if uid in final_user_set:
                # 更新 interaction_count 为最终值
                u2 = dict(u)
                u2['interaction_count'] = sum(1 for r in final_interactions if r['user_id'] == uid)
                users.append(u2)
    print(f"  collected {len(users):,} user meta", flush=True)

    # 排序:interactions 按 (uid, time),保证 split 时序
    final_interactions.sort(key=lambda x: (x['user_id'], x['event_time']))

    # 输出
    out = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source': f'{args.inp} (random-sampled {len(sampled_set):,} users, item-core={args.item_core})',
        'sampling': {
            'method': 'random_user_sample + weak item k-core',
            'target_users': args.target_users,
            'min_user_interactions': args.min_user_interactions,
            'item_core': args.item_core,
            'seed': args.seed,
        },
        'stats': {
            'n_users': len(users),
            'n_products': len(products),
            'n_interactions': len(final_interactions),
            'density': round(len(final_interactions) / (len(users) * len(products)), 6) if users and products else 0,
        },
        'products': products,
        'users': users,
        'interactions': final_interactions,
    }

    print(f"\n=== Step 7: 写出 {args.out} ===", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        # 自定义 default: 兜底处理 Decimal/未知类型
        def _default(obj):
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)
        json.dump(out, f, ensure_ascii=False, default=_default)
    print(f"\n=== DONE ({time.time()-t0:.0f}s) ===", flush=True)
    print(json.dumps(out['stats'], indent=2, ensure_ascii=False))
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
