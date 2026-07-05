"""
从 Wikipedia 爬取时尚图片 - 慢速版本
设置较长延迟避免限流
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

    # 每个类别获取的图片数
    "images_per_category": 300,

    # Wikipedia API 间隔（秒）- 设长一点避免限流
    "delay": 25,
    "timeout": 20,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USER_AGENT = 'FashionApp/1.0 (contact@example.com)'
WIKIMEDIA_API = 'https://commons.wikimedia.org/w/api.php'


def search_wikimedia_images(search_term: str, limit: int = 50, retries: int = 3) -> List[str]:
    """搜索 Wikimedia Commons 图片"""
    images = []

    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search_term + ' fashion',
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

            if resp.status_code == 429:
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for '{search_term}'")
                time.sleep(5)
                continue

            data = resp.json()
            results = data.get('query', {}).get('search', [])

            for item in results:
                title = item['title']
                filename = title.replace('File:', '').replace(' ', '_')
                file_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
                images.append(file_url)

            return images

        except Exception as e:
            logger.warning(f"Search failed for '{search_term}' (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(10)

    return images


def fetch_category_images(category: str) -> List[str]:
    """获取某个类别的所有图片"""
    search_terms = {
        "top": ["t-shirt", "shirt", "blouse", "sweater", "top clothing"],
        "bottom": ["jeans", "trousers", "pants", "shorts", "skirt"],
        "dress": ["dress", "gown", "maxi dress"],
        "outerwear": ["jacket", "coat", "blazer"],
        "shoes": ["shoes", "sneakers", "boots", "high heels"],
        "accessories": ["handbag", "purse", "wallet", "backpack"],
    }

    terms = search_terms.get(category, [category])
    all_images = []
    seen = set()

    logger.info(f"  Fetching from Wikipedia for {category}...")

    for i, term in enumerate(terms):
        if len(all_images) >= CONFIG['images_per_category']:
            break

        logger.info(f"    Searching: {term}")
        images = search_wikimedia_images(term, 50)

        for img_url in images:
            if img_url not in seen:
                seen.add(img_url)
                all_images.append(img_url)

        logger.info(f"    Got {len(images)} images, total {len(all_images)}")
        time.sleep(CONFIG['delay'])

        if len(all_images) >= CONFIG['images_per_category']:
            break

    return all_images[:CONFIG['images_per_category']]


def load_products(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('products', [])


def build_image_pool() -> Dict[str, List[str]]:
    categories = ["top", "bottom", "dress", "outerwear", "shoes", "accessories"]
    image_pool = {}

    logger.info("Fetching fashion images from Wikimedia Commons (slow mode)...")
    logger.info(f"Estimated time: ~{len(categories) * 6 * CONFIG['delay'] / 60:.0f} minutes")

    for cat in categories:
        logger.info(f"Fetching category: {cat}")
        images = fetch_category_images(cat)
        image_pool[cat] = images
        logger.info(f"  Total for {cat}: {len(images)} images")
        time.sleep(CONFIG['delay'])

    return image_pool


def assign_images_to_products(products: list, image_pool: Dict[str, List[str]]) -> list:
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
    success_count = sum(1 for p in products if p.get('image_url'))

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(products),
        "success_count": success_count,
        "products": products
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved: {success_count}/{len(products)} products with images")


def main():
    logger.info("Starting slow Wikimedia image crawler...")
    logger.info(f"Delay between requests: {CONFIG['delay']}s")

    products = load_products(CONFIG["input_file"])
    logger.info(f"Loaded {len(products)} products")

    image_pool = build_image_pool()

    if not any(image_pool.values()):
        logger.error("Failed to fetch any images!")
        return

    products_with_images = assign_images_to_products(products, image_pool)
    save_results(products_with_images, CONFIG["output_file"])

    total = sum(len(imgs) for imgs in image_pool.values())
    logger.info(f"\nTotal images fetched: {total}")
    for cat, imgs in image_pool.items():
        logger.info(f"  {cat}: {len(imgs)} images")

    logger.info("Done!")


if __name__ == "__main__":
    main()
