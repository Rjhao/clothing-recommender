"""
使用 Pexels API 获取商品图片
按商品类别搜索时尚图片
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    "input_file": "D:/clothing-recommender/data/amazon_fashion_processed.json",
    "output_file": "D:/clothing-recommender/data/amazon_fashion_with_pexels_images.json",
    "failed_log": "D:/clothing-recommender/data/failed_pexels.txt",

    # Pexels API Key - 需要替换成你自己的
    "pexels_api_key": "YOUR_PEXELS_API_KEY",

    # 每个类别获取的图片数量
    "images_per_category": 20,

    # 类别到搜索关键词的映射
    "category_search_terms": {
        "top": ["t-shirt", "shirt", "blouse", "top"],
        "bottom": ["pants", "jeans", "shorts", "skirt"],
        "dress": ["dress", "gown", "maxi dress"],
        "outerwear": ["jacket", "coat", "blazer", "cardigan"],
        "shoes": ["shoes", "sneakers", "heels", "boots"],
        "accessories": ["bag", "handbag", "wallet", "accessories"],
    },

    "per_page": 15,  # Pexels每次最多15张
    "max_retries": 3,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PexelsImageFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/v1/search"
        self.headers = {
            "Authorization": api_key
        }
        # 缓存：category -> list of image URLs
        self.image_cache = {}

    def search_images(self, query: str, per_page: int = 15) -> list:
        """搜索图片，返回URL列表"""
        params = {
            "query": query,
            "per_page": per_page,
            "page": 1,
        }

        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                photos = data.get("photos", [])
                urls = [photo["src"]["large"] for photo in photos]
                logger.debug(f"Found {len(urls)} images for '{query}'")
                return urls
            else:
                logger.warning(f"Pexels API error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return []

    def get_category_images(self, category: str) -> list:
        """获取某个类别的图片"""
        if category in self.image_cache:
            return self.image_cache[category]

        search_terms = CONFIG["category_search_terms"].get(category, [category])
        all_urls = []

        for term in search_terms:
            urls = self.search_images(term, CONFIG["per_page"])
            all_urls.extend(urls)

            if len(all_urls) >= CONFIG["images_per_category"]:
                break

            time.sleep(0.5)  # 避免请求太快

        # 去重
        unique_urls = list(dict.fromkeys(all_urls))
        self.image_cache[category] = unique_urls[:CONFIG["images_per_category"]]

        return self.image_cache[category]


def load_products(filepath: str) -> list:
    """加载商品数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('products', [])


def assign_images_to_products(products: list, fetcher: PexelsImageFetcher) -> list:
    """为每个商品分配图片URL"""
    # 按类别预获取图片
    categories = set(p.get('category', 'top') for p in products)
    logger.info(f"Fetching images for categories: {categories}")

    category_image_pools = {}
    for cat in categories:
        logger.info(f"Fetching images for category: {cat}")
        images = fetcher.get_category_images(cat)
        category_image_pools[cat] = images
        logger.info(f"  Got {len(images)} images for {cat}")
        time.sleep(1)  # 避免触发限流

    # 为每个商品分配图片（轮询分配）
    import random
    result = []
    category_indices = {cat: 0 for cat in categories}

    for product in tqdm(products, desc="Assigning images"):
        category = product.get('category', 'top')
        image_pool = category_image_pools.get(category, [])

        if image_pool:
            idx = category_indices[category] % len(image_pool)
            product['image_url'] = image_pool[idx]
            category_indices[category] += 1
        else:
            product['image_url'] = None

        result.append(product)

    return result


def save_results(products: list, output_file: str, failed_file: str):
    """保存结果"""
    # 统计
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

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Success: {success_count}, Failed: {failed_count}")


def main():
    logger.info("Starting Pexels image fetcher...")

    if CONFIG["pexels_api_key"] == "YOUR_PEXELS_API_KEY":
        logger.error("Please set your Pexels API key in CONFIG!")
        logger.info("Get your free API key at: https://www.pexels.com/api/")
        return

    # 加载商品
    products = load_products(CONFIG["input_file"])
    logger.info(f"Loaded {len(products)} products")

    # 获取图片
    fetcher = PexelsImageFetcher(CONFIG["pexels_api_key"])
    products_with_images = assign_images_to_products(products, fetcher)

    # 保存结果
    save_results(products_with_images, CONFIG["output_file"], CONFIG["failed_log"])

    logger.info("Done!")


if __name__ == "__main__":
    main()
