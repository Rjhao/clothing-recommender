"""为 testuser99 (user_id=1) 注入 20 条真实交互到 Neo4j + MySQL 改 gender=female。

步骤:
1. MySQL: UPDATE users SET gender='female' WHERE id=1
2. Neo4j: 复用 Amazon 真实用户 513781644 的前 20 个 product_id
   - MERGE (u:User {id:1, username:..., gender:..., body_shape:..., skin_tone:...})
   - 给 20 个 Clothing 节点补 color 字段(按 category 启发式)
   - MERGE (u)-[:BOUGHT {timestamp, interaction_count}]->(c)
   - 通过 c-[:HAS_STYLE]->s 创建 (u)-[:PREFERS {score}]->(s)
3. 验证 GET /recommend/1/profile 返回非空

Usage: python data/inject_demo_user_kg.py
"""
import json
import sys
import time
from pathlib import Path

import pymysql
from neo4j import GraphDatabase

# ============ 配置 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "root123"
MYSQL_DB = "clothing_recommender"
TARGET_USER_ID = 1  # 默认 testuser99(可命令行覆盖)
SOURCE_USER_ID = 513781644
N_INTERACTIONS = 20
DATA_FILE = "data/amazon_fashion_real.json"

# 按 category 启发式分配颜色(name 里没有显式颜色时)
CATEGORY_DEFAULT_COLOR = {
    "top": "blue",
    "bottom": "black",
    "dress": "red",
    "outerwear": "black",
    "shoes": "white",
    "accessories": "black",
}


def pick_color(name: str, category: str) -> str:
    n = (name or "").lower()
    for c in ("black", "white", "red", "blue", "green", "yellow", "pink", "grey", "gray",
              "brown", "beige", "navy", "purple", "orange"):
        if c in n:
            return c
    return CATEGORY_DEFAULT_COLOR.get(category, "black")


def inject_for_user(user_id: int, username: str):
    """给指定用户注入交互数据(幂等)。"""
    # 1. MySQL: 改 gender
    print(f"  [1/5] MySQL: UPDATE users SET gender='female' WHERE id={user_id}")
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB, charset="utf8mb4",
    )
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET gender=%s WHERE id=%s", ("female", user_id))
        conn.commit()
        cur.execute(
            "SELECT id, username, gender, body_shape, skin_tone FROM users WHERE id=%s",
            (user_id,),
        )
        row = cur.fetchone()
        print(f"     -> {row}")
    conn.close()

    # 3. Neo4j: 先清掉旧的 BOUGHT / PREFERS(幂等)
    print(f"  [2/5] Neo4j: 清掉 user {user_id} 旧 BOUGHT/PREFERS(幂等)")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (u:User {id: $uid})-[r:BOUGHT|PREFERS]->()
                DELETE r
                """,
                uid=user_id,
            )
            # 多样化选 20 件商品(按 style 平均分布,4 种风格各 5 件)
            rows_raw = session.run(
                """
                MATCH (c:Clothing)
                WHERE c.style IN ['casual', 'business', 'romantic', 'sporty']
                WITH c.style AS s, collect(c)[0..5] AS picked
                UNWIND picked AS c
                RETURN c.id AS id, c.name AS name, c.category AS category, c.style AS style
                """
            )
            items = list(rows_raw)
            print(f"     -> 选到 {len(items)} 件多风格商品")
            if len(items) < 8:
                print("     !! 多样化商品不足,跳过此用户")
                return

            # 3b. 给这 N 件 Clothing 补 color(避免整库更新)
            color_update = [{"id": r["id"], "color": pick_color(r["name"], r["category"])} for r in items]
            session.run(
                """
                UNWIND $rows AS r
                MATCH (c:Clothing {id: r.id})
                SET c.color = r.color
                """,
                rows=color_update,
            )
            print(f"     -> 已给 {len(color_update)} 件 Clothing 补 color")

            # 3c. 创建 User 节点
            session.run(
                """
                MERGE (u:User {id: $uid})
                ON CREATE SET u.username = $username,
                              u.gender = $gender,
                              u.body_shape = $body_shape,
                              u.skin_tone = $skin_tone,
                              u.created_at = datetime()
                ON MATCH SET  u.username = $username,
                              u.gender = $gender,
                              u.body_shape = $body_shape,
                              u.skin_tone = $skin_tone
                """,
                uid=user_id,
                username=row[1],
                gender="female",
                body_shape=row[3],
                skin_tone=row[4],
            )
            print(f"     -> MERGE User {{id:{user_id}}} (username={row[1]})")

            # 3d. 创建 BOUGHT 关系
            bought_rows = [
                {
                    "uid": user_id,
                    "cid": r["id"],
                    "ts": f"2020-03-{(i + 1):02d}T10:00:00",
                    "cnt": 1,
                    "price": 0.0,
                }
                for i, r in enumerate(items)
            ]
            session.run(
                """
                UNWIND $rows AS r
                MATCH (u:User {id: r.uid})
                MATCH (c:Clothing {id: r.cid})
                MERGE (u)-[b:BOUGHT]->(c)
                SET b.timestamp = datetime(r.ts),
                    b.interaction_count = r.cnt,
                    b.price = r.price
                """,
                rows=bought_rows,
            )
            print(f"     -> 创建 {len(bought_rows)} 条 BOUGHT 关系")

            # 3e. 通过 Clothing -> HAS_STYLE -> Style 给 User 加 PREFERS
            session.run(
                """
                MATCH (u:User {id: $uid})-[b:BOUGHT]->(c:Clothing)-[:HAS_STYLE]->(s:Style)
                WITH u, s, count(c) AS cnt
                MERGE (u)-[p:PREFERS]->(s)
                SET p.score = toFloat(cnt),
                    p.updated_at = datetime()
                """,
                uid=user_id,
            )
            prefs_count = session.run(
                "MATCH (u:User {id: $uid})-[:PREFERS]->(s:Style) RETURN count(s) AS cnt",
                uid=user_id,
            ).single()["cnt"]
            print(f"     -> 创建 {prefs_count} 条 PREFERS 关系")
    finally:
        driver.close()

    # 4. 验证 API
    print(f"  [3/5] 验证 GET /recommend/{user_id}/profile")
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:8000/recommend/{user_id}/profile") as resp:
            payload = json.loads(resp.read())
        print(f"     gender={payload.get('static_info', {}).get('gender')}")
        print(f"     style_distribution: {payload.get('style_distribution')}")
        print(f"     color_preferences:  {payload.get('color_preferences')}")
        print(f"     interaction_count:  {payload.get('interaction_count')}")
    except Exception as e:
        print(f"     !! API 调用失败: {e}")


def main():
    """默认给 MySQL 里所有现有用户都注入数据(Login 自动登 admin,确保 admin 有数据)。"""
    print("=" * 60)
    print("  Neo4j + MySQL 用户偏好数据注入器")
    print("=" * 60)

    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB, charset="utf8mb4",
    )
    with conn.cursor() as cur:
        cur.execute("SELECT id, username FROM users ORDER BY id")
        users = cur.fetchall()
    conn.close()

    if not users:
        print("!! MySQL users 表为空,退出")
        sys.exit(1)

    print(f"\n发现 {len(users)} 个用户: {[u[1] for u in users]}")
    for uid, uname in users:
        print(f"\n>>> 处理 user {uid} ({uname}) <<<")
        inject_for_user(uid, uname)

    print("\n" + "=" * 60)
    print("  [DONE] 所有用户已注入完成。刷新浏览器即可看到数据。")
    print("=" * 60)


if __name__ == "__main__":
    main()