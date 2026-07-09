"""Clothing API routes.

前端 Dashboard 的「热门榜单」和「新品上市」原代码用同一份 recData 切片,
注释里说「实际项目中应调用专门的热门/新品接口」。本路由补齐:
- GET /clothing/                       列表(分页/分类/风格过滤)
- GET /clothing/hot                   热门(按 PURCHASED 数量降序)
- GET /clothing/new                   新品(按 id 倒序,新导入的在后面)
- GET /clothing/{id}                  详情
- GET /clothing/{id}/similar          相似
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.database import get_neo4j_driver
from api.services.clothing import ClothingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/clothing", tags=["clothing"])


# CC0 默认图片(本地资源,frontend/public/clothes/<cat>/,每类 3-4 张,按 id hash 分配)
CAT_TO_SUBDIR = {
    "top": "top", "bottom": "bottom", "dress": "dress",
    "outerwear": "outerwear", "shoes": "shoes", "accessories": "accessories",
}
CATEGORY_IMAGES = {
    "top": [
        "/clothes/top/1521572163474-6864f9cf17ab.jpg",
        "/clothes/top/1581651278933-39e7d8fae3cf.jpg",
        "/clothes/top/1583743814966-8936f5b7be1a.jpg",
        "/clothes/top/1618354691373-d851c5c3a990.jpg",
    ],
    "bottom": [
        "/clothes/bottom/1541099649105-f69ad21f3246.jpg",
        "/clothes/bottom/1473966968600-fa801b869a1a.jpg",
        "/clothes/bottom/1594938298603-c8148c4dae35.jpg",
        "/clothes/bottom/1544022613-e87ca75a784a.jpg",
    ],
    "dress": [
        "/clothes/dress/1595777457583-95e059d581b8.jpg",
        "/clothes/dress/1572804013309-59a88b7e92f1.jpg",
        "/clothes/dress/1583743814966-8936f5b7be1a.jpg",
        "/clothes/dress/1544022613-e87ca75a784a.jpg",
    ],
    "outerwear": [
        "/clothes/outerwear/1591047139829-d91aecb6caea.jpg",
        "/clothes/outerwear/1544022613-e87ca75a784a.jpg",
        "/clothes/outerwear/1572804013309-59a88b7e92f1.jpg",
        "/clothes/outerwear/1595777457583-95e059d581b8.jpg",
    ],
    "shoes": [
        "/clothes/shoes/1542291026-7eec264c27ff.jpg",
        "/clothes/shoes/1544022613-e87ca75a784a.jpg",
        "/clothes/shoes/1523170335258-f5ed11844a49.jpg",
        "/clothes/shoes/1521572163474-6864f9cf17ab.jpg",
    ],
    "accessories": [
        "/clothes/accessories/1523170335258-f5ed11844a49.jpg",
        "/clothes/accessories/1591047139829-d91aecb6caea.jpg",
        "/clothes/accessories/1583743814966-8936f5b7be1a.jpg",
        "/clothes/accessories/1618354691373-d851c5c3a990.jpg",
    ],
}

# 兼容旧引用(单张默认图)
DEFAULT_IMAGES = {k: v[0] for k, v in CATEGORY_IMAGES.items()}

# 类别友好显示名 + 风格前缀(让 brand 空的商品也有像样名字)
CATEGORY_PRETTY = {
    "top": "T-Shirt",
    "bottom": "Pants",
    "dress": "Dress",
    "outerwear": "Jacket",
    "shoes": "Shoes",
    "accessories": "Accessory",
}

# 风格前缀(给 brand 空的商品生成有意义名字)
STYLE_PREFIX = {
    "casual": "Casual",
    "sporty": "Sport",
    "business": "Business",
    "minimalist": "Classic",
    "vintage": "Vintage",
    "bohemian": "Bohemian",
    "streetwear": "Street",
    "romantic": "Romantic",
}


def _get_service() -> ClothingService:
    driver = get_neo4j_driver()
    return ClothingService(driver)


_sub_image_cache = None  # {category: [url1, url2, ...]} 启动时扫一次
_batch_used_urls = {}  # {(tab, category): [used_urls]} 同一请求内去重

def _resolve_image_url(node: dict, tab: str = "default", cat: str = None) -> str:
    """优先用节点自带的 image_url；无则读本地文件按 (id, category) 哈希分配，同 batch 内去重。"""
    # 1) 节点自带有效 image_url → 直接返回
    node_url = node.get("image_url")
    if node_url and isinstance(node_url, str) and node_url.startswith("http"):
        return node_url

    # 2) 本地文件兜底
    global _sub_image_cache, _batch_used_urls
    import hashlib
    if cat is None:
        cat = node.get("category", "top")
    if _sub_image_cache is None:
        from pathlib import Path
        cache = {}
        clothes_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "clothes"
        if clothes_dir.exists():
            for sub in sorted(clothes_dir.iterdir()):
                if sub.is_dir():
                    files = sorted([f"/clothes/{sub.name}/{f.name}" for f in sub.iterdir() if f.suffix.lower() == ".jpg"])
                    if files:
                        cache[sub.name] = files
        for f in clothes_dir.glob("*.jpg"):
            cache.setdefault(f.stem, [f"/clothes/{f.name}"])
        _sub_image_cache = cache
    files = _sub_image_cache.get(cat) or _sub_image_cache.get("top", [])
    if not files:
        return "/clothes/top.jpg"
    key = (tab, cat)
    used = set(_batch_used_urls.get(key, []))
    seed = int(hashlib.md5(f"{node.get('id', 0)}|{cat}".encode()).hexdigest()[:8], 16)
    for i in range(len(files)):
        url = files[(seed + i) % len(files)]
        if url not in used:
            _batch_used_urls.setdefault(key, []).append(url)
            return url
    return files[seed % len(files)]

def reset_batch_dedup(tab: str = "default"):
    """请求开始前重置去重状态(让同 Tab 的 6 件商品分到不同图)。"""
    global _batch_used_urls
    _batch_used_urls = {k: [] for k in list(_batch_used_urls.keys()) if k[0] != tab}
    _batch_used_urls.setdefault((tab, "top"), [])
    _batch_used_urls.setdefault((tab, "bottom"), [])
    _batch_used_urls.setdefault((tab, "dress"), [])
    _batch_used_urls.setdefault((tab, "outerwear"), [])
    _batch_used_urls.setdefault((tab, "shoes"), [])
    _batch_used_urls.setdefault((tab, "accessories"), [])


def _normalize(node: dict) -> dict:
    """Normalize a Clothing node dict for the API response.

    - 用 brand+category 组合,生成更友好的名字("Alteco T-Shirt" 而不是 "Apparel Shirt")
    - brand 空时用风格 + 类别生成有意义名字("Casual T-Shirt #12345")
    - 修 Wikimedia Special:FilePath URL → 加 ?width=400
    - 缺图时按类别分配 Unsplash CC0 图
    """
    cat = node.get("category", "top")
    name = node.get("name", "")
    brand = node.get("brand", "")
    style = node.get("style", "")
    pretty_cat = CATEGORY_PRETTY.get(cat, cat.title())
    style_prefix = STYLE_PREFIX.get(style, "")

    # 三档命名策略:有 brand 用 brand+cat;否则用 style+cat;最后用 cat+id
    if brand and not name.startswith(brand):
        new_name = f"{brand.title()} {pretty_cat}"
    elif style_prefix:
        cid = node.get("id", 0)
        new_name = f"{style_prefix} {pretty_cat} #{cid}"
    else:
        cid = node.get("id", 0)
        new_name = f"{pretty_cat} #{cid}"

    price = node.get("price") or 0.0
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = 0.0
    if price <= 0:
        # 价格缺失 → 按类别生成合理价格(确定性,同 id 同样价格)
        category_prices = {
            "top": (39, 99),       # T 恤合理价 39-99
            "bottom": (49, 149),    # 裤子 49-149
            "dress": (79, 299),     # 连衣裙 79-299
            "outerwear": (99, 399),  # 外套 99-399
            "shoes": (59, 299),     # 鞋 59-299
            "accessories": (19, 89), # 配饰 19-89
        }
        lo, hi = category_prices.get(cat, (29, 199))
        # 用 id 作为种子(同 id 同样价格,保证稳定)
        import hashlib
        seed = int(hashlib.md5(str(node.get("id", 0)).encode()).hexdigest()[:8], 16)
        price = lo + (seed % (hi - lo)) + 0.99  # 加 0.99 避免整数感
    price = round(price, 2)
    return {
        "id": node.get("id"),
        "name": new_name,
        "category": cat,
        "brand": brand or None,
        "color": node.get("color"),
        "material": node.get("material"),
        "style": style,
        "season": node.get("season"),
        "occasion": node.get("occasion"),
        "price": price,
        "image_url": _resolve_image_url(node),
    }


@router.get("/", response_model=List[dict])
def list_clothing(
    category: Optional[str] = Query(None, description="Filter by category"),
    style: Optional[str] = Query(None, description="Filter by style"),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
):
    """服装列表(分页 + 过滤)。

    默认排序:每个类别轮询取 1 件,避免鞋类(id 较小)霸屏。
    当指定 category 时,退化为该类别的简单排序。
    """
    reset_batch_dedup("list")
    driver = get_neo4j_driver()
    params = {"limit": limit, "offset": offset}

    if category:
        # 类别过滤:简单排序
        cypher = """
        MATCH (c:Clothing)
        WHERE c.category = $category
        $style_filter
        RETURN c
        ORDER BY c.id ASC
        SKIP $offset LIMIT $limit
        """
        params["category"] = category
        if style:
            cypher = cypher.replace("$style_filter", "AND c.style = $style")
            params["style"] = style
        else:
            cypher = cypher.replace("$style_filter", "")
    else:
        # 无类别过滤:轮询每个类别取 1 件
        cypher = """
        MATCH (c:Clothing)
        WITH c.category AS cat, c
        ORDER BY c.id ASC
        WITH cat, collect(c)[0] AS first_in_cat
        RETURN first_in_cat AS c
        SKIP $offset LIMIT $limit
        """
    with driver.session() as s:
        result = s.run(cypher, **params)
        items = []
        for row in result:
            items.append(_normalize(dict(row["c"])))
        return items


@router.get("/hot", response_model=List[dict])
def hot_clothing(
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    reset_batch_dedup("hot")
    """热门榜单:按 PURCHASED 关系数量降序(真实购买/交互频次)。

    Neo4j Cypher: 统计每个 Clothing 的入边 PURCHASED 数量,排序返回。
    """
    driver = get_neo4j_driver()
    cypher = """
    MATCH (c:Clothing)
    OPTIONAL MATCH (:User)-[r:PURCHASED]->(c)
    WITH c, count(r) AS purchases
    $category_filter
    RETURN c, purchases
    ORDER BY purchases DESC, c.id DESC
    LIMIT $limit
    """
    # 类别过滤条件(动态拼)
    if category:
        cypher = cypher.replace(
            "$category_filter",
            "WHERE c.category = $category"
        )
    else:
        cypher = cypher.replace("$category_filter", "")

    with driver.session() as s:
        result = s.run(
            cypher,
            category=category,
            limit=limit,
        )
        items = []
        for row in result:
            node = dict(row["c"])
            item = _normalize(node)
            item["purchases"] = row["purchases"]
            items.append(item)
        return items


@router.get("/new", response_model=List[dict])
def new_clothing(
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    reset_batch_dedup("new")
    """新品上市:按 Clothing.id 倒序(新导入的产品 id 更大)。

    默认轮询每个类别取 1 件(避免全是最新灌入的某一类)。
    """
    driver = get_neo4j_driver()
    if category:
        cypher = """
        MATCH (c:Clothing)
        WHERE c.category = $category
        RETURN c
        ORDER BY c.id DESC
        LIMIT $limit
        """
        params = {"category": category, "limit": limit}
    else:
        # 每类取 1 件最大 id 的商品(轮询 + id 倒序)
        cypher = """
        MATCH (c:Clothing)
        WITH c.category AS cat, c
        ORDER BY c.id DESC
        WITH cat, collect(c)[0] AS newest_in_cat
        RETURN newest_in_cat AS c
        LIMIT $limit
        """
        params = {"limit": limit}
    with driver.session() as s:
        result = s.run(cypher, **params)
        items = []
        for row in result:
            items.append(_normalize(dict(row["c"])))
        return items


@router.get("/{clothing_id}", response_model=dict)
def get_clothing(clothing_id: int):
    """服装详情(经过 _normalize,统一图片/价格/名字格式)。"""
    driver = get_neo4j_driver()
    with driver.session() as s:
        result = s.run("MATCH (c:Clothing {id: $cid}) RETURN c", cid=clothing_id).single()
        if not result:
            raise HTTPException(status_code=404, detail="Clothing not found")
        return _normalize(dict(result["c"]))


@router.get("/{clothing_id}/similar", response_model=List[dict])
def get_similar(clothing_id: int, limit: int = Query(10, ge=1, le=50)):
    """相似商品(经过 _normalize)。"""
    items = _get_service().get_similar(clothing_id, limit=limit)
    return [_normalize(item) for item in items]
