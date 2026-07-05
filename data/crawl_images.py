"""
Amazon商品图片爬虫
从amazon_fashion_processed.json读取商品，爬取Amazon主图URL
"""
import json
import re
import time
import random
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# 配置
CONFIG = {
    "input_file": "D:/clothing-recommender/data/amazon_fashion_processed.json",
    "output_file": "D:/clothing-recommender/data/amazon_fashion_with_images.json",
    "failed_log": "D:/clothing-recommender/data/failed_crawl.txt",
    "delay_min": 1.5,  # 请求间隔最小秒
    "delay_max": 3.5,  # 请求间隔最大秒
    "max_workers": 3,  # 并发线程数（Amazon建议别太高）
    "timeout": 15,
    "max_retries": 3,
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# User-Agent池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# 图片URL模式
IMAGE_PATTERNS = [
    # Amazon主图URL模式
    re.compile(r'"large":"(https://[^"]*\.jpg[^"]*)"'),
    re.compile(r'"hiRes":"(https://[^"]*\.jpg[^"]*)"'),
    re.compile(r'"mainUrl":"(https://[^"]*)"'),
    # 直接图片URL
    re.compile(r'(https://images-na\.ssl-images-amazon\.com/images/I/[^\s"]+\.jpg[^\s"]*)'),
    re.compile(r'(https://m\.media-amazon\.com/images/I/[^\s"]+\.jpg[^\s"]*)'),
]


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    price: float
    source: str
    image_url: Optional[str] = None
    crawl_status: str = "pending"  # pending, success, failed, not_found


def load_products(filepath: str) -> list:
    """加载商品数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('products', [])


def extract_image_url(html: str) -> Optional[str]:
    """从HTML中提取商品主图URL"""
    # 方法1: 正则匹配
    for pattern in IMAGE_PATTERNS:
        matches = pattern.findall(html)
        if matches:
            # 返回第一张高质量图片
            for match in matches:
                if 'jpg' in match.lower() and '._' not in match:
                    return match

    # 方法2: BeautifulSoup解析
    soup = BeautifulSoup(html, 'html.parser')

    # 尝试多种选择器
    selectors = [
        '#imgTagWrapperId img',
        '#landingImage',
        '#main-image',
        '.a-dynamic-image',
        'img[data-a-dynamic-image]',
    ]

    for selector in selectors:
        img = soup.select_one(selector)
        if img:
            # 尝试获取高分辨率版本
            if img.get('data-a-dynamic-image'):
                try:
                    dynamic_images = json.loads(img['data-a-dynamic-image'])
                    if dynamic_images:
                        # 返回最大图片
                        max_url = max(dynamic_images.keys(), key=lambda x: len(x))
                        return max_url
                except json.JSONDecodeError:
                    pass

            src = img.get('src') or img.get('data-src')
            if src and 'images-amazon' in src:
                return src

    return None


def build_amazon_url(product_id: str) -> str:
    """构建Amazon商品URL"""
    return f"https://www.amazon.com/dp/{product_id}"


def crawl_product_image(product: dict, session: requests.Session) -> dict:
    """爬取单个商品的主图"""
    product_id = product['product_id']
    url = build_amazon_url(product_id)

    result = product.copy()
    result['image_url'] = None
    result['crawl_status'] = 'pending'
    result['crawl_error'] = None

    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
    }

    for retry in range(CONFIG['max_retries']):
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=CONFIG['timeout'],
                allow_redirects=True
            )

            if response.status_code == 200:
                image_url = extract_image_url(response.text)
                if image_url:
                    result['image_url'] = image_url
                    result['crawl_status'] = 'success'
                    logger.debug(f"Found image for {product_id}: {image_url[:50]}...")
                    return result
                else:
                    result['crawl_status'] = 'not_found'
                    result['crawl_error'] = 'No image found in page'
            elif response.status_code == 503:
                # Amazon反爬
                result['crawl_status'] = 'failed'
                result['crawl_error'] = f'HTTP {response.status_code} - Blocked'
                logger.warning(f"Blocked {product_id}, retry {retry + 1}/{CONFIG['max_retries']}")
            else:
                result['crawl_status'] = 'failed'
                result['crawl_error'] = f'HTTP {response.status_code}'

        except requests.exceptions.Timeout:
            result['crawl_status'] = 'failed'
            result['crawl_error'] = 'Timeout'
        except requests.exceptions.RequestException as e:
            result['crawl_status'] = 'failed'
            result['crawl_error'] = str(e)[:100]

        if retry < CONFIG['max_retries'] - 1:
            wait_time = CONFIG['delay_min'] + random.random() * (CONFIG['delay_max'] - CONFIG['delay_min'])
            time.sleep(wait_time)

    return result


def save_progress(products: list, output_file: str):
    """保存进度"""
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(products),
        "success_count": sum(1 for p in products if p.get('crawl_status') == 'success'),
        "failed_count": sum(1 for p in products if p.get('crawl_status') == 'failed'),
        "not_found_count": sum(1 for p in products if p.get('crawl_status') == 'not_found'),
        "products": products
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Progress saved to {output_file}")


def save_failed_products(products: list, failed_file: str):
    """保存失败的商品ID"""
    failed = [p for p in products if p.get('crawl_status') in ['failed', 'not_found']]
    with open(failed_file, 'w', encoding='utf-8') as f:
        for p in failed:
            f.write(f"{p['product_id']}: {p.get('crawl_error', 'unknown error')}\n")
    logger.info(f"Failed products saved to {failed_file}")


def main():
    logger.info("Starting Amazon image crawler...")

    # 加载商品
    products = load_products(CONFIG['input_file'])
    logger.info(f"Loaded {len(products)} products")

    # 检查已有进度（支持断点续传）
    output_path = Path(CONFIG['output_file'])
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            existing_products = existing.get('products', [])
            existing_dict = {p['product_id']: p for p in existing_products}

            # 合并：使用已有的成功结果，跳过失败的
            for i, product in enumerate(products):
                pid = product['product_id']
                if pid in existing_dict and existing_dict[pid].get('crawl_status') == 'success':
                    products[i] = existing_dict[pid]
                    products[i]['crawl_status'] = 'success'
                    products[i]['image_url'] = existing_dict[pid].get('image_url')

            logger.info(f"Loaded {len(existing_products)} existing results, continuing...")

    # 统计
    pending = [p for p in products if p.get('crawl_status') != 'success']
    success = [p for p in products if p.get('crawl_status') == 'success']
    logger.info(f"Already have {len(success)} images, {len(pending)} to crawl")

    if not pending:
        logger.info("All products already crawled!")
        save_progress(products, CONFIG['output_file'])
        return

    # 创建session
    session = requests.Session()

    # 带进度条爬取
    logger.info(f"Starting to crawl {len(pending)} products...")

    for i, product in enumerate(pending):
        # 爬取
        result = crawl_product_image(product, session)
        products[[p['product_id'] for p in products].index(result['product_id'])] = result

        # 进度输出
        current_success = sum(1 for p in products if p.get('crawl_status') == 'success')
        current_failed = sum(1 for p in products if p.get('crawl_status') == 'failed')

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(pending)} | Success: {current_success} | Failed: {current_failed}")
            save_progress(products, CONFIG['output_file'])

        # 延迟
        if i < len(pending) - 1:
            delay = CONFIG['delay_min'] + random.random() * (CONFIG['delay_max'] - CONFIG['delay_min'])
            time.sleep(delay)

    # 最终保存
    save_progress(products, CONFIG['output_file'])
    save_failed_products(products, CONFIG['failed_log'])

    # 统计
    success_count = sum(1 for p in products if p.get('crawl_status') == 'success')
    failed_count = sum(1 for p in products if p.get('crawl_status') == 'failed')
    not_found_count = sum(1 for p in products if p.get('crawl_status') == 'not_found')

    logger.info(f"\n{'='*50}")
    logger.info(f"Crawl completed!")
    logger.info(f"Total: {len(products)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Not Found: {not_found_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {success_count/len(products)*100:.1f}%")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
