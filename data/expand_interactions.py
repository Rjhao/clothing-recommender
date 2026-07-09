"""扩充交互数据。

策略：
1. 保留现有真实交互（1509 条）
2. 每个 user 按 interaction_count 补齐到应有数量
3. 缺失交互的商品按 popularity 加权采样（流行商品更可能被交互）
4. 加入品类多样性约束（每个用户至少跨 2 个品类，避免只买鞋）

输出: data/amazon_fashion_expanded.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent
SRC = DATA_DIR / "amazon_fashion_processed.json"
DST = DATA_DIR / "amazon_fashion_expanded.json"


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    users = data["users"]
    products = data["products"]
    interactions = data["interactions"]

    # 商品索引与 popularity 权重
    prod_ids = [str(p["product_id"]) for p in products]
    prod_id_to_obj = {str(p["product_id"]): p for p in products}
    # popularity 至少为 1，避免零权重；做轻度平滑
    weights = []
    for pid in prod_ids:
        pop = int(prod_id_to_obj[pid].get("popularity", 0))
        weights.append(max(pop, 1))
    # 按品类分组，用于多样性约束
    cat_groups = defaultdict(list)
    for idx, pid in enumerate(prod_ids):
        cat = prod_id_to_obj[pid].get("category") or "unknown"
        cat_groups[cat].append(idx)

    # 现有交互按 user 分组
    user_existing = defaultdict(set)  # user_id(str) -> set(item_id str)
    for inter in interactions:
        uid = str(inter["user_id"])
        iid = str(inter["product_id"])
        user_existing[uid].add(iid)

    # 按品类分组的商品 id（用于多样性采样）
    cat_to_pids = defaultdict(list)
    for p in products:
        cat = p.get("category") or "unknown"
        cat_to_pids[cat].append(str(p["product_id"]))

    all_new_interactions = list(interactions)  # 保留真实交互
    stats = {"expanded_users": 0, "synthesized": 0, "kept_real": len(interactions)}

    for user in users:
        uid = str(user["user_id"])
        target_count = int(user["interaction_count"])
        existing = user_existing.get(uid, set())
        need = target_count - len(existing)
        if need <= 0:
            continue

        stats["expanded_users"] += 1

        # 多样性：若现有交互品类单一，强制从其他品类补一部分
        existing_cats = set()
        for iid in existing:
            obj = prod_id_to_obj.get(iid)
            if obj:
                existing_cats.add(obj.get("category") or "unknown")

        # 70% 按 popularity 加权采样，30% 强制跨品类（如果当前品类<2）
        n_weighted = int(need * 0.7)
        n_diverse = need - n_weighted

        chosen = set()

        # 加权采样
        if n_weighted > 0:
            sampled = random.choices(prod_ids, weights=weights, k=n_weighted * 3)
            for iid in sampled:
                if iid not in existing and iid not in chosen:
                    chosen.add(iid)
                    if len(chosen) >= n_weighted:
                        break

        # 跨品类采样：从用户未涉及品类补
        if n_diverse > 0:
            other_cats = [c for c in cat_to_pids if c not in existing_cats]
            if not other_cats:
                other_cats = list(cat_to_pids.keys())
            for _ in range(n_diverse * 5):
                cat = random.choice(other_cats)
                pool = cat_to_pids[cat]
                iid = random.choice(pool)
                if iid not in existing and iid not in chosen:
                    chosen.add(iid)
                    if len(chosen) - n_weighted >= n_diverse:
                        break

        # 若还不够，继续加权补
        while len(chosen) < need:
            iid = random.choices(prod_ids, weights=weights, k=1)[0]
            if iid not in existing and iid not in chosen:
                chosen.add(iid)

        for iid in chosen:
            obj = prod_id_to_obj[iid]
            all_new_interactions.append({
                "user_id": int(uid),
                "product_id": int(iid),
                "interaction_count": 1,
                "price": obj.get("price", 0),
                "last_time": "2020-04-01 00:00:00+00:00",
                "_synthesized": True,
            })
            stats["synthesized"] += 1

    # 写出
    out = dict(data)
    out["interactions"] = all_new_interactions
    out["stats"] = {
        "n_users": len(users),
        "n_products": len(products),
        "n_interactions_real": len(interactions),
        "n_interactions_total": len(all_new_interactions),
        "n_synthesized": stats["synthesized"],
    }

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"真实交互: {len(interactions)}")
    print(f"合成交互: {stats['synthesized']}")
    print(f"总交互: {len(all_new_interactions)}")
    print(f"扩充用户数: {stats['expanded_users']}")
    print(f"输出: {DST}")


if __name__ == "__main__":
    main()
