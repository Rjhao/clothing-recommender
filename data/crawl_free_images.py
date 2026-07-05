"""
从 Wikimedia Commons 爬取时尚图片
免费，无需API Key
"""
import json
import logging
import time
import random
from typing import List, Dict

import requests
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    "input_file": "D:/clothing-recommender/data/amazon_fashion_processed.json",
    "output_file": "D:/clothing-recommender/data/amazon_fashion_with_images.json",

    # 每个类别获取的图片总数
    "images_per_category": 150,

    # 每个搜索词获取的图片数
    "images_per_search": 30,

    # 请求间隔（秒）- Wikipedia限流较严
    "delay": 5.0,
    "timeout": 20,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USER_AGENT = 'FashionApp/1.0 (https://github.com/example/fashion-recommender; contact@example.com)'
WIKIMEDIA_API = 'https://commons.wikimedia.org/w/api.php'


def search_wikimedia_images(search_term: str, limit: int = 30, retries: int = 3) -> List[str]:
    """搜索 Wikimedia Commons 图片"""
    images = []

    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search_term + ' fashion clothing',
        'srnamespace': 6,  # File namespace
        'srlimit': limit,
        'format': 'json'
    }

    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIMEDIA_API,
                params=params,
                headers={'User-Agent': USER_AGENT},
                timeout=CONFIG['timeout']
            )

            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for '{search_term}', retry {attempt + 1}")
                time.sleep(2)
                continue

            data = resp.json()

            for item in data.get('query', {}).get('search', []):
                title = item['title']
                # 从 File:Title 获取文件名
                filename = title.replace('File:', '').replace(' ', '_')
                # 使用 Special:FilePath 获取实际URL（会重定向到实际图片）
                file_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
                images.append(file_url)

            return images  # 成功就返回

        except Exception as e:
            logger.warning(f"Search failed for '{search_term}' (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return images


def fetch_category_images(category: str) -> List[str]:
    """获取某个类别的所有图片"""
    # 搜索词列表
    search_terms = {
        "top": ["t-shirt", "shirt", "blouse", "sweater", "tank top", "polo shirt", "hoodie", "cardigan"],
        "bottom": ["jeans", "trousers", "pants", "shorts", "skirt", "leggings", "chino pants", "jogger pants"],
        "dress": ["dress", "evening gown", "maxi dress", "mini dress", "cocktail dress", "summer dress"],
        "outerwear": ["jacket", "coat", "blazer", "parka", "windbreaker", "leather jacket", "denim jacket", "vest"],
        "shoes": ["shoes", "sneakers", "boots", "high heels", "sandals", "loafers", " flats", "pumps"],
        "accessories": ["handbag", "purse", "wallet", "backpack", "belt", "watch", "sunglasses", "hat", "scarf"],
    }

    terms = search_terms.get(category, [category])
    all_images = []
    seen = set()

    for term in terms:
        if len(all_images) >= CONFIG['images_per_category']:
            break

        images = search_wikimedia_images(term, CONFIG['images_per_search'])

        for img_url in images:
            if img_url not in seen:
                seen.add(img_url)
                all_images.append(img_url)

        time.sleep(CONFIG['delay'])

        if len(all_images) >= CONFIG['images_per_category']:
            break

    return all_images[:CONFIG['images_per_category']]


def load_products(filepath: str) -> list:
    """加载商品数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('products', [])


def build_image_pool() -> Dict[str, List[str]]:
    """构建所有类别的图片池"""
    categories = ["top", "bottom", "dress", "outerwear", "shoes", "accessories"]
    image_pool = {}

    logger.info("Fetching fashion images from Wikimedia Commons...")

    for cat in categories:
        logger.info(f"Fetching images for: {cat}")
        images = fetch_category_images(cat)
        image_pool[cat] = images
        logger.info(f"  Got {len(images)} images for {cat}")

    return image_pool


def assign_images_to_products(products: list, image_pool: Dict[str, List[str]]) -> list:
    """为每个商品分配图片"""
    result = []
    category_indices = {cat: 0 for cat in image_pool.keys()}

    for product in tqdm(products, desc="Assigning images"):
        category = product.get('category', 'top')
        pool = image_pool.get(category, [])

        if pool:
            idx = category_indices[category] % len(pool)
            product['image_url'] = pool[idx]
            category_indices[category] += 1
        else:
            product['image_url'] = None

        result.append(product)

    return result


def save_results(products: list, output_file: str):
    """保存结果"""
    success_count = sum(1 for p in products if p.get('image_url'))
    failed_count = sum(1 for p in products if not p.get('image_url'))

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(products),
        "success_count": success_count,
        "failed_count": failed_count,
        "products": products
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved: {success_count}/{len(products)} products with images")


def main():
    logger.info("Starting Wikimedia Commons image crawler...")
    logger.info(f"Will fetch ~{CONFIG['images_per_category']} images per category")

    # 加载商品
    products = load_products(CONFIG["input_file"])
    logger.info(f"Loaded {len(products)} products")

    # 构建图片池
    image_pool = build_image_pool()

    if not any(image_pool.values()):
        logger.error("Failed to fetch any images!")
        return

    # 分配图片
    products_with_images = assign_images_to_products(products, image_pool)

    # 保存
    save_results(products_with_images, CONFIG["output_file"])

    # 打印统计
    total_fetched = sum(len(imgs) for imgs in image_pool.values())
    logger.info(f"\nTotal images fetched: {total_fetched}")
    for cat, imgs in image_pool.items():
        logger.info(f"  {cat}: {len(imgs)} images")

    logger.info("Done!")


if __name__ == "__main__":
    main()
