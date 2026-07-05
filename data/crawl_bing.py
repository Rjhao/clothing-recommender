"""
使用 Bing 免费图片搜索获取时尚图片
"""
import json
import logging
import time
import random
import re
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    "input_file": "D:/clothing-recommender/data/amazon_fashion_processed.json",
    "output_file": "D:/clothing-recommender/data/amazon_fashion_with_images.json",

    # 每个类别获取的图片数
    "images_per_category": 200,

    # 请求间隔（秒）
    "delay": 3.0,
    "timeout": 15,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def search_bing_images(query: str, max_images: int = 50) -> List[str]:
    """使用Bing图片搜索获取图片URL"""
    images = []

    try:
        # Bing图片搜索URL
        url = f"https://www.bing.com/images/search?q={requests.utils.quote(query + ' fashion clothing')}&first=0&count={max_images}"

        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        resp = requests.get(url, headers=headers, timeout=CONFIG['timeout'])
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 查找图片
        for img in soup.find_all('img'):
            src = img.get('src', '') or img.get('data-src', '')
            if src and ('.jpg' in src.lower() or '.jpeg' in src.lower() or '.png' in src.lower()):
                # 跳过缩略图和图标
                if 'thumb' not in src.lower() and 'icon' not in src.lower():
                    if src not in images:
                        images.append(src)

            if len(images) >= max_images:
                break

        # 也查找 linkshim 链接
        for a in soup.find_all('a', href=re.compile(r'/images/search.*')):
            href = a.get('href', '')
            if 'imgurl=' in href:
                # 提取实际图片URL
                match = re.search(r'imgurl=(https?://[^&]+)', href)
                if match:
                    img_url = match.group(1)
                    if img_url not in images:
                        images.append(img_url)

        logger.debug(f"Bing search for '{query}': found {len(images)} images")

    except Exception as e:
        logger.warning(f"Bing search failed for '{query}': {e}")

    return images[:max_images]


def search_google_images(query: str, max_images: int = 50) -> List[str]:
    """使用 Google 图片搜索获取图片URL"""
    images = []

    try:
        url = f"https://www.google.com/search?q={requests.utils.quote(query + ' fashion clothing')}&tbm=isch"

        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        resp = requests.get(url, headers=headers, timeout=CONFIG['timeout'])
        soup = BeautifulSoup(resp.text, 'html.parser')

        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and ('.jpg' in src.lower() or '.jpeg' in src.lower() or '.png' in src.lower()):
                if 'google.com' not in src and 'gstatic.com' not in src:
                    if src not in images:
                        images.append(src)

            if len(images) >= max_images:
                break

        logger.debug(f"Google search for '{query}': found {len(images)} images")

    except Exception as e:
        logger.warning(f"Google search failed for '{query}': {e}")

    return images[:max_images]


def fetch_category_images(category: str) -> List[str]:
    """获取某个类别的所有图片"""
    search_terms = {
        "top": ["t-shirt", "shirt", "blouse", "sweater"],
        "bottom": ["jeans", "pants", "trousers", "shorts", "skirt"],
        "dress": ["dress", "gown", "maxi dress"],
        "outerwear": ["jacket", "coat", "blazer"],
        "shoes": ["shoes", "sneakers", "boots", "high heels", "sandals"],
        "accessories": ["handbag", "purse", "wallet", "backpack"],
    }

    terms = search_terms.get(category, [category])
    all_images = []
    seen = set()

    for term in terms:
        if len(all_images) >= CONFIG['images_per_category']:
            break

        # 尝试Bing
        images = search_bing_images(term, 30)
        for img_url in images:
            if img_url not in seen and img_url.startswith('http'):
                seen.add(img_url)
                all_images.append(img_url)

        time.sleep(CONFIG['delay'])

        if len(all_images) >= CONFIG['images_per_category']:
            break

        # 如果Bing不够，用Google
        if len(images) < 10:
            images = search_google_images(term, 30)
            for img_url in images:
                if img_url not in seen and img_url.startswith('http'):
                    seen.add(img_url)
                    all_images.append(img_url)
            time.sleep(CONFIG['delay'])

    return all_images[:CONFIG['images_per_category']]


def load_products(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('products', [])


def build_image_pool() -> Dict[str, List[str]]:
    categories = ["top", "bottom", "dress", "outerwear", "shoes", "accessories"]
    image_pool = {}

    logger.info("Fetching fashion images from Bing/Google...")

    for cat in categories:
        logger.info(f"Fetching images for: {cat}")
        images = fetch_category_images(cat)
        image_pool[cat] = images
        logger.info(f"  Got {len(images)} images for {cat}")

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
    logger.info("Starting Bing/Google image crawler...")

    products = load_products(CONFIG["input_file"])
    logger.info(f"Loaded {len(products)} products")

    image_pool = build_image_pool()

    if not any(image_pool.values()):
        logger.error("Failed to fetch any images!")
        return

    products_with_images = assign_images_to_products(products, image_pool)
    save_results(products_with_images, CONFIG["output_file"])

    logger.info("Done!")


if __name__ == "__main__":
    main()
