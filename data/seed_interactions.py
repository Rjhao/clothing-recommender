"""把 amazon_fashion_real.json 的 User + 交互灌到 Neo4j,用于前端 demo。

只取高频用户/商品(减少 MERGE 数量,5 分钟内完成):
- 用户:interaction_count >= 5 的高频用户
- 商品:popularity >= 10 的热门商品(已被 import_to_kg 灌过)
- 交互:这些用户-热门商品的配对(限 200 用户 × 50 商品,演示够用)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_neo4j_driver

t0 = time.time()
print("Loading real.json ...", flush=True)
d = json.load(open('data/amazon_fashion_real.json', encoding='utf-8'))
users = d['users']
interactions = d['interactions']
products = d['products']
print(f"  users: {len(users)}, interactions: {len(interactions)}, products: {len(products)}", flush=True)

# 1) 高频用户: interaction_count >= 8(确保有推荐价值)
high_freq_users = [u for u in users if u.get('interaction_count', 0) >= 8]
print(f"  high-freq users (>=8): {len(high_freq_users)}", flush=True)

# 2) 热门商品: popularity >= 5
hot_products = {int(p['product_id']): p for p in products if p.get('popularity', 0) >= 5}
print(f"  hot products (>=5 popularity): {len(hot_products)}", flush=True)

# 3) 配对: 每个高频用户随机 30 个交互(覆盖热门商品)
import random
random.seed(42)
top_users = random.sample(high_freq_users, min(200, len(high_freq_users)))
hot_pids = list(hot_products.keys())
print(f"  sampled users: {len(top_users)}, hot pids: {len(hot_pids)}", flush=True)

pairs = []
for u in top_users:
    uid = int(u['user_id'])
    # 随机选 30 个热门商品
    chosen = random.sample(hot_pids, min(30, len(hot_pids)))
    for pid in chosen:
        pairs.append((uid, pid))
print(f"  total user-product pairs: {len(pairs)}", flush=True)

# 4) 灌 Neo4j
driver = get_neo4j_driver()
print("Seeding Neo4j ...", flush=True)
with driver.session() as s:
    # 4.1 创建 User 节点 + User 的 preferred_style 关系
    user_q = """
    UNWIND $batch AS row
    MERGE (u:User {id: row.uid})
    SET u.preferred_style = row.style
    """
    batch = [{"uid": int(u['user_id']), "style": u.get('preferred_style', 'casual')} for u in top_users]
    s.run(user_q, batch=batch)
    print(f"  users created", flush=True)

    # 4.2 创建 PURCHASED 关系
    rel_q = """
    UNWIND $batch AS row
    MATCH (u:User {id: row.uid})
    MATCH (p:Clothing {id: row.pid})
    MERGE (u)-[r:PURCHASED]->(p)
    SET r.interaction_count = row.ic, r.last_time = row.lt
    """
    # 注意:Clothing 用属性 'id'(int),不是 product_id(string)
    rel_batch = []
    for uid, pid in pairs:
        rel_batch.append({
            "uid": uid,
            "pid": pid,  # int
            "ic": 1,
            "lt": "2020-01-01 00:00:00 UTC",
        })
    # 分批,每批 1000
    BATCH = 1000
    for i in range(0, len(rel_batch), BATCH):
        s.run(rel_q, batch=rel_batch[i:i+BATCH])
    print(f"  PURCHASED relationships created: {len(rel_batch)}", flush=True)

    # 4.3 验证
    cnt = s.run("MATCH (u:User) RETURN count(u) AS c").single()
    print(f"  User count: {cnt['c']}", flush=True)
    cnt = s.run("MATCH ()-[r:PURCHASED]->() RETURN count(r) AS c").single()
    print(f"  PURCHASED count: {cnt['c']}", flush=True)
driver.close()
print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
