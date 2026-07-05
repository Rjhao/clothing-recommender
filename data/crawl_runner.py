"""Main crawler script to run data collection.

Usage:
    python data/crawl_runner.py              # Run with defaults
    python data/crawl_runner.py --jd         # Only JD
    python data/crawl_runner.py --tb         # Only Taobao
    python data/crawl_runner.py --pages 5    # 5 pages per category
"""
import argparse
import json
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.crawler import JDScraper, TaobaoScraper, DataWasher, DataExporter
from api.database import get_neo4j_driver, get_mysql_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# JD Categories for clothing
JD_CATEGORIES = [
    "T恤男", "T恤女", "衬衫男", "衬衫女",
    "牛仔裤男", "牛仔裤女", "连衣裙", "半身裙",
    "男士外套", "女士外套", "羽绒服", "棉服",
    "运动裤", "休闲裤", "西装", "卫衣",
    "毛衣", "针织衫", "polo衫", "背心",
    "男鞋", "女鞋", "运动鞋", "休闲鞋",
    "皮带", "围巾", "帽子", "手套",
]

# Taobao Categories
TAOBAO_CATEGORIES = [
    "男士T恤", "女士T恤", "男士衬衫", "女士衬衫",
    "男士牛仔裤", "女士牛仔裤", "连衣裙", "半身裙",
    "男士外套", "女士外套", "羽绒服", "棉服",
    "运动裤", "休闲裤", "西装外套", "卫衣",
    "毛衣", "针织衫", "polo衫", "吊带背心",
    "男鞋", "女鞋", "运动休闲鞋", "皮鞋",
    "皮带", "围巾", "帽子", "手套",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clothing data crawler")

    parser.add_argument(
        "--jd",
        action="store_true",
        help="Only crawl JD.com"
    )
    parser.add_argument(
        "--tb",
        action="store_true",
        help="Only crawl Taobao/Tmall"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        help="Pages per category (default: 3)"
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=20,
        help="Items per page (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crawled_products.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip data cleaning"
    )
    parser.add_argument(
        "--export-mysql",
        action="store_true",
        help="Export to MySQL"
    )
    parser.add_argument(
        "--export-neo4j",
        action="store_true",
        help="Export to Neo4j"
    )

    return parser.parse_args()


def crawl_jd(pages_per_category: int = 3, per_page: int = 20) -> list:
    """
    Crawl products from JD.com.

    Args:
        pages_per_category: Number of pages to crawl per category
        per_page: Items per page

    Returns:
        List of crawled products
    """
    logger.info("=" * 60)
    logger.info("Starting JD.com crawler")
    logger.info("=" * 60)

    scraper = JDScraper()
    all_products = []

    def progress_callback(current, total, count):
        logger.info(f"Progress: {current}/{total} pages, got {count} items this page")

    for i, category in enumerate(JD_CATEGORIES):
        logger.info(f"[{i+1}/{len(JD_CATEGORIES)}] Crawling category: {category}")

        try:
            for page in range(1, pages_per_category + 1):
                products = scraper.search(category, page, per_page)
                all_products.extend(products)
                logger.info(f"  Page {page}: got {len(products)} products")

                if len(products) == 0:
                    break  # No more results

                # Random delay to avoid being blocked
                import random
                import time
                time.sleep(random.uniform(2, 4))

        except Exception as e:
            logger.error(f"Failed to crawl category '{category}': {e}")

    logger.info(f"JD crawler finished: {len(all_products)} total products")
    return all_products


def crawl_taobao(pages_per_category: int = 3) -> list:
    """
    Crawl products from Taobao/Tmall.

    Args:
        pages_per_category: Number of pages to crawl per category

    Returns:
        List of crawled products
    """
    logger.info("=" * 60)
    logger.info("Starting Taobao/Tmall crawler")
    logger.info("=" * 60)

    scraper = TaobaoScraper()
    all_products = []

    for i, category in enumerate(TAOBAO_CATEGORIES):
        logger.info(f"[{i+1}/{len(TAOBAO_CATEGORIES)}] Crawling category: {category}")

        try:
            for page in range(1, pages_per_category + 1):
                # Try Tmall first (more reliable)
                products = scraper.search_tmall(category, page)
                all_products.extend(products)
                logger.info(f"  Page {page}: got {len(products)} products")

                if len(products) == 0:
                    break

                import random
                import time
                time.sleep(random.uniform(3, 5))

        except Exception as e:
            logger.error(f"Failed to crawl category '{category}': {e}")

    logger.info(f"Taobao crawler finished: {len(all_products)} total products")
    return all_products


def save_to_json(products: list, filepath: str):
    """Save products to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "count": len(products),
            "products": products
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(products)} products to {filepath}")


def load_from_json(filepath: str) -> list:
    """Load products from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("products", [])


def main():
    """Main entry point."""
    args = parse_args()

    all_products = []

    # Determine which sources to crawl
    crawl_jd_flag = args.jd or (not args.jd and not args.tb)
    crawl_tb_flag = args.tb or (not args.jd and not args.tb)

    # Crawl data
    if crawl_jd_flag:
        jd_products = crawl_jd(args.pages, args.per_page)
        all_products.extend(jd_products)

    if crawl_tb_flag:
        tb_products = crawl_taobao(args.pages)
        all_products.extend(tb_products)

    # Deduplicate by product_id
    seen_ids = set()
    unique_products = []
    for p in all_products:
        pid = p.get("product_id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_products.append(p)

    logger.info(f"After deduplication: {len(unique_products)} unique products")

    # Save raw data
    raw_output = args.output.replace(".json", "_raw.json")
    save_to_json(unique_products, raw_output)

    # Clean data
    if args.skip_clean:
        cleaned_products = unique_products
        cleaning_stats = {}
    else:
        logger.info("=" * 60)
        logger.info("Cleaning and standardizing data")
        logger.info("=" * 60)

        washer = DataWasher()
        cleaned_products, cleaning_stats = washer.clean_batch(unique_products)

        logger.info("Cleaning statistics:")
        for k, v in cleaning_stats.items():
            logger.info(f"  {k}: {v}")

    # Save cleaned data
    cleaned_output = args.output
    save_to_json(cleaned_products, cleaned_output)

    # Export to databases
    if args.export_mysql or args.export_neo4j:
        logger.info("=" * 60)
        logger.info("Exporting to databases")
        logger.info("=" * 60)

        exporter = DataExporter()

        if args.export_mysql:
            try:
                import pymysql
                mysql_url = get_mysql_url()
                # Parse URL
                # mysql+pymysql://user:pass@host:port/db
                db_url = mysql_url.replace("mysql+pymysql://", "")
                parts = db_url.split("@")
                auth, host_db = parts[0], parts[1]
                user, password = auth.split(":")
                host_port, dbname = host_db.split("/")

                conn = pymysql.connect(
                    host=host_port.split(":")[0],
                    port=int(host_port.split(":")[1]) if ":" in host_port else 3306,
                    user=user,
                    password=password,
                    database=dbname,
                    charset="utf8mb4"
                )
                exporter.mysql_conn = conn
                result = exporter.export_to_mysql(cleaned_products)
                logger.info(f"MySQL export complete: {result} records")
                conn.close()
            except Exception as e:
                logger.error(f"MySQL export failed: {e}")

        if args.export_neo4j:
            try:
                driver = get_neo4j_driver()
                exporter.neo4j_driver = driver
                result = exporter.export_to_neo4j(cleaned_products)
                logger.info(f"Neo4j export complete: {result}")
                driver.close()
            except Exception as e:
                logger.error(f"Neo4j export failed: {e}")

    logger.info("=" * 60)
    logger.info("Crawling complete!")
    logger.info(f"Total products: {len(cleaned_products)}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
