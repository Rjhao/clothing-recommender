"""
将图片URL更新到MySQL数据库
"""
import json
import logging
from pathlib import Path

import pymysql
from tqdm import tqdm

# 配置
CONFIG = {
    "input_file": "D:/clothing-recommender/data/amazon_fashion_with_images.json",
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password",
    "database": "clothing_recommender",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image_data(filepath: str) -> dict:
    """加载图片数据，返回 product_id -> image_url 的字典"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_map = {}
    for product in data.get('products', []):
        if product.get('image_url'):
            image_map[product['product_id']] = product['image_url']

    return image_map


def update_mysql(products_with_images: dict):
    """更新MySQL数据库"""
    connection = pymysql.connect(
        host=CONFIG['host'],
        port=CONFIG['port'],
        user=CONFIG['user'],
        password=CONFIG['password'],
        database=CONFIG['database'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            # 检查表结构
            cursor.execute("DESCRIBE clothing_products")
            columns = [col['Field'] for col in cursor.fetchall()]
            logger.info(f"Current columns: {columns}")

            # 如果没有image_url列，添加它
            if 'image_url' not in columns:
                logger.info("Adding image_url column...")
                cursor.execute("ALTER TABLE clothing_products ADD COLUMN image_url VARCHAR(500) NULL")
                connection.commit()
                logger.info("Column added successfully")

            # 批量更新
            update_count = 0
            batch_size = 100
            product_ids = list(products_with_images.keys())

            for i in tqdm(range(0, len(product_ids), batch_size)):
                batch = product_ids[i:i + batch_size]

                sql = "UPDATE clothing_products SET image_url = %s WHERE product_id = %s"
                values = [(products_with_images[pid], pid) for pid in batch if products_with_images[pid]]

                cursor.executemany(sql, values)
                update_count += cursor.rowcount

            connection.commit()
            logger.info(f"Updated {update_count} records")

            # 验证
            cursor.execute("SELECT COUNT(*) as cnt FROM clothing_products WHERE image_url IS NOT NULL AND image_url != ''")
            result = cursor.fetchone()
            logger.info(f"Total products with images in DB: {result['cnt']}")

    finally:
        connection.close()


def main():
    logger.info("Starting MySQL image URL update...")

    # 加载图片数据
    image_map = load_image_data(CONFIG['input_file'])
    logger.info(f"Loaded {len(image_map)} products with images")

    if not image_map:
        logger.error("No image data found!")
        return

    # 更新数据库
    update_mysql(image_map)

    logger.info("Done!")


if __name__ == "__main__":
    main()
