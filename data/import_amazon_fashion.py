"""Import Amazon Fashion dataset from Kaggle into MySQL and Neo4j.

Usage:
    python -m data.import_amazon_fashion --file data/2019-Oct.csv

Prerequisites:
    1. Download from: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
    2. Save as: data/2019-Oct.csv (or 2019-Nov.csv)
"""
import argparse
import logging
import sys
import os
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.crawler.washer import DataWasher
from data.crawler.exporter import DataExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Fashion-related category codes
FASHION_CATEGORIES = [
    'apparel', 'clothing', 'fashion', 'shoes', 'footwear',
    'Accessories', 'jewelry', 'watch', 'bag', 'purse',
    'underwear', 'swimwear', 'lingerie', 'sportswear',
]


def is_fashion(category_code: str) -> bool:
    """Check if category is fashion-related."""
    if pd.isna(category_code):
        return False
    category_code = str(category_code).lower()
    return any(cat in category_code for cat in FASHION_CATEGORIES)


def categorize_product(category_code: str) -> str:
    """Map Amazon category_code to our category system."""
    if pd.isna(category_code):
        return 'unknown'

    category_code = str(category_code).lower()

    # Top (shirts, t-shirts, blouses)
    if any(x in category_code for x in ['t-shirt', 'tshirt', 'shirt', 'blouse', 'top', 'sweater', 'hoodie', 'sweatshirt', 'polo', 'jersey']):
        return 'top'

    # Bottom (pants, shorts, skirts)
    if any(x in category_code for x in ['pants', 'jean', 'trouser', 'shorts', 'skirt', 'leggings', 'chino']):
        return 'bottom'

    # Dress
    if any(x in category_code for x in ['dress', 'gown', 'jumpsuit', 'romper']):
        return 'dress'

    # Outerwear
    if any(x in category_code for x in ['jacket', 'coat', 'blazer', 'suit', 'outerwear', 'parka', 'windbreaker']):
        return 'outerwear'

    # Shoes
    if any(x in category_code for x in ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'flat', 'loafer', 'slipper', 'footwear']):
        return 'shoes'

    # Accessories
    if any(x in category_code for x in ['bag', 'purse', 'wallet', 'watch', 'jewelry', 'belt', 'hat', 'cap', 'scarf', 'glove', 'sunglasses', 'accessory']):
        return 'accessories'

    return 'unknown'


def extract_style_from_text(name: str, brand: str = '') -> str:
    """Extract style from product name and brand."""
    if pd.isna(name):
        name = ''
    if pd.isna(brand):
        brand = ''

    text = f"{name} {brand}".lower()

    style_keywords = {
        'minimalist': ['minimal', 'simple', 'basic', 'plain', 'essential'],
        'casual': ['casual', 'daily', 'comfort', 'relaxed', 'everyday'],
        'bohemian': ['boho', 'bohemian', 'ethnic', 'tribal', 'folk'],
        'vintage': ['vintage', 'retro', 'classic', 'oldschool', 'nostalgic'],
        'streetwear': ['street', 'urban', 'hip-hop', 'skate', 'graffiti', 'y2k'],
        'business': ['business', 'professional', 'formal', 'office', 'career', 'suit'],
        'sporty': ['sport', 'athletic', 'fitness', 'gym', 'running', 'workout'],
        'romantic': ['romantic', 'elegant', 'graceful', 'feminine', 'soft'],
    }

    for style, keywords in style_keywords.items():
        if any(kw in text for kw in keywords):
            return style

    return 'casual'  # default


def extract_season(name: str) -> str:
    """Extract season from product name."""
    if pd.isna(name):
        return 'all_season'

    text = str(name).lower()

    if any(x in text for x in ['spring', '春', '早春']):
        return 'spring'
    if any(x in text for x in ['summer', '夏', '清凉', '薄款']):
        return 'summer'
    if any(x in text for x in ['fall', 'autumn', '秋', '秋冬']):
        return 'fall'
    if any(x in text for x in ['winter', '冬', '保暖', '加绒', '加厚']):
        return 'winter'
    if any(x in text for x in ['four season', '四季', '全年', 'all season', 'year round']):
        return 'all_season'

    return 'all_season'


def load_and_filter(filepath: str) -> pd.DataFrame:
    """Load Amazon dataset and filter for fashion items."""
    logger.info(f"Loading data from {filepath}...")

    # Read CSV (large file, read in chunks)
    df = pd.read_csv(filepath, parse_dates=['event_time'])

    logger.info(f"Total records: {len(df):,}")

    # Filter for fashion items
    df['is_fashion'] = df['category_code'].apply(is_fashion)
    fashion_df = df[df['is_fashion']].copy()

    logger.info(f"Fashion records: {len(fashion_df):,}")

    return fashion_df


def process_interactions(df: pd.DataFrame) -> list:
    """Process user-item interactions from Amazon data."""
    logger.info("Processing user-item interactions...")

    # Group by user and item
    interactions = []

    # Filter for purchase and cart events (more valuable than views)
    purchase_df = df[df['event_type'].isin(['purchase', 'cart'])]

    # Aggregate interactions
    grouped = purchase_df.groupby(['user_id', 'product_id']).agg({
        'event_type': 'count',  # number of interactions
        'price': 'mean',  # average price
        'event_time': 'max',  # last interaction time
    }).reset_index()

    grouped.columns = ['user_id', 'product_id', 'interaction_count', 'price', 'last_time']

    # Convert Timestamp to string for JSON serialization
    grouped['last_time'] = grouped['last_time'].astype(str)

    logger.info(f"Unique user-product interactions: {len(grouped):,}")
    logger.info(f"Unique users: {grouped['user_id'].nunique():,}")
    logger.info(f"Unique products: {grouped['product_id'].nunique():,}")

    return grouped.to_dict('records')


def process_products(df: pd.DataFrame) -> list:
    """Process product catalog from Amazon data."""
    logger.info("Processing product catalog...")

    # Get unique products with their attributes
    products = df.groupby('product_id').agg({
        'category_code': 'first',
        'brand': 'first',
        'price': 'mean',
        'user_id': 'count',  # interaction count as popularity
    }).reset_index()

    products.columns = ['product_id', 'category_code', 'brand', 'price', 'popularity']

    # Filter out products with very few interactions (noise)
    products = products[products['popularity'] >= 2]

    logger.info(f"Products with 2+ interactions: {len(products):,}")

    # Convert to list of dicts
    # Note: Don't use washer.clean_product() because it expects Chinese text
    # Our Amazon data uses category_code like "apparel.shoes.slipons"
    result = []

    for _, row in products.iterrows():
        category = categorize_product(row['category_code'])
        style = extract_style_from_text(row['category_code'], row.get('brand', ''))
        season = extract_season(row['category_code'])

        product = {
            'product_id': str(row['product_id']),
            'name': str(row.get('category_code', row['product_id'])).replace('.', ' ').title(),
            'category': category if category != 'unknown' else None,
            'category_code': str(row['category_code']),
            'brand': str(row['brand']) if pd.notna(row['brand']) else '',
            'color': None,  # Amazon data doesn't have color
            'size': None,
            'material': None,
            'price': float(row['price']) if pd.notna(row['price']) else 0,
            'platform': 'AMAZON',
            'source': f"https://www.amazon.com/dp/{row['product_id']}",
            'style': style,
            'season': season,
            'styles': [style] if style else [],
            'popularity': int(row['popularity']),
            '_original_name': str(row['category_code']),
            '_original_category': category,
        }
        result.append(product)

    return result


def process_users(df: pd.DataFrame, min_interactions: int = 3) -> list:
    """Extract user profiles from interaction data."""
    logger.info("Processing user profiles...")

    # Get users with sufficient interactions
    user_interactions = df.groupby('user_id').size()
    active_users = user_interactions[user_interactions >= min_interactions].index

    # Get category preferences for active users
    active_df = df[df['user_id'].isin(active_users)]

    user_prefs = active_df.groupby(['user_id', 'category_code']).size().reset_index(name='count')

    # Get top category per user
    user_top_cat = user_prefs.loc[user_prefs.groupby('user_id')['count'].idxmax()]

    users = []
    for _, row in user_top_cat.iterrows():
        style = extract_style_from_text(row['category_code'])
        users.append({
            'user_id': str(row['user_id']),
            'preferred_style': style,
            'interaction_count': int(user_interactions[row['user_id']]),
        })

    logger.info(f"Extracted {len(users)} user profiles")

    return users


def get_category_stats(df: pd.DataFrame) -> dict:
    """Get statistics by category."""
    stats = defaultdict(lambda: {
        'count': 0,
        'avg_price': 0,
        'total_interactions': 0,
        'unique_users': 0,
        'unique_products': 0,
    })

    for cat, group in df.groupby('category_code'):
        if pd.isna(cat):
            continue
        category = categorize_product(cat)
        stats[category]['count'] += 1
        stats[category]['avg_price'] += group['price'].sum()
        stats[category]['total_interactions'] += len(group)
        stats[category]['unique_users'] += group['user_id'].nunique()
        stats[category]['unique_products'] += group['product_id'].nunique()

    for cat in stats:
        if stats[cat]['count'] > 0:
            stats[cat]['avg_price'] /= stats[cat]['count']

    return dict(stats)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Import Amazon Fashion data')
    parser.add_argument('--file', type=str, default='data/2019-Oct.csv',
                        help='Path to Kaggle CSV file')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Limit rows for testing')
    parser.add_argument('--export-mysql', action='store_true',
                        help='Export to MySQL')
    parser.add_argument('--export-neo4j', action='store_true',
                        help='Export to Neo4j')
    parser.add_argument('--save-json', type=str, default='data/amazon_fashion_processed.json',
                        help='Save processed data to JSON')

    args = parser.parse_args()

    # Load and filter data
    df = load_and_filter(args.file)

    if args.max_rows:
        df = df.head(args.max_rows)
        logger.info(f"Limited to {args.max_rows} rows")

    # Show category distribution
    logger.info("\n=== Category Distribution ===")
    category_counts = df['category_code'].value_counts().head(20)
    for cat, count in category_counts.items():
        our_cat = categorize_product(cat)
        logger.info(f"  {cat} -> {our_cat}: {count:,}")

    # Process data
    products = process_products(df)
    users = process_users(df)
    interactions = process_interactions(df)

    # Save processed data
    import json
    from datetime import datetime

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'total_products': len(products),
            'total_users': len(users),
            'total_interactions': len(interactions),
        },
        'products': products,
        'users': users,
        'interactions': interactions,
    }

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved processed data to {args.save_json}")

    # Print summary
    logger.info("\n=== Summary ===")
    logger.info(f"Products: {len(products):,}")
    logger.info(f"Users: {len(users):,}")
    logger.info(f"Interactions: {len(interactions):,}")

    # Export to databases
    if args.export_mysql or args.export_neo4j:
        logger.info("\n=== Exporting to Databases ===")

        exporter = DataExporter()

        if args.export_mysql:
            conn = None
            try:
                from api.database import get_mysql_url
                import pymysql

                mysql_url = get_mysql_url()
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

                result = exporter.export_to_mysql(products)
                logger.info(f"MySQL export: {result} products")

                # Export users
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(64) PRIMARY KEY,
                        preferred_style VARCHAR(64),
                        interaction_count INT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for user in users:
                    cursor.execute("""
                        INSERT IGNORE INTO users (user_id, preferred_style, interaction_count)
                        VALUES (%s, %s, %s)
                    """, (user['user_id'], user['preferred_style'], user['interaction_count']))

                # Export interactions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64),
                        product_id VARCHAR(64),
                        interaction_count INT,
                        price DECIMAL(10,2),
                        last_time DATETIME,
                        INDEX idx_user (user_id),
                        INDEX idx_product (product_id)
                    )
                """)
                for interaction in interactions:
                    cursor.execute("""
                        INSERT IGNORE INTO user_interactions
                        (user_id, product_id, interaction_count, price, last_time)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        interaction['user_id'],
                        interaction['product_id'],
                        interaction['interaction_count'],
                        interaction['price'],
                        interaction['last_time']
                    ))

                conn.commit()
                logger.info("MySQL export complete")

            except Exception as e:
                logger.error(f"MySQL export failed: {e}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        if args.export_neo4j:
            try:
                from api.database import get_neo4j_driver

                driver = get_neo4j_driver()
                exporter.neo4j_driver = driver

                # Export products to Neo4j
                result = exporter.export_to_neo4j(products)
                logger.info(f"Neo4j export: {result}")

                # Export users and interactions
                with driver.session() as session:
                    # Create user nodes
                    for user in users:
                        session.run("""
                            MERGE (u:User {user_id: $user_id})
                            SET u.preferred_style = $style,
                                u.interaction_count = $count
                        """, user_id=user['user_id'], style=user['preferred_style'], count=user['interaction_count'])

                    # Create interaction relationships
                    for interaction in interactions:
                        session.run("""
                            MATCH (u:User {user_id: $user_id})
                            MATCH (p:Clothing {product_id: $product_id})
                            MERGE (u)-[r:INTERACTED {
                                count: $count,
                                price: $price,
                                last_time: $last_time
                            }]->(p)
                        """,
                            user_id=interaction['user_id'],
                            product_id=interaction['product_id'],
                            count=interaction['interaction_count'],
                            price=interaction['price'],
                            last_time=str(interaction['last_time'])
                        )

                driver.close()
                logger.info("Neo4j export complete")

            except Exception as e:
                logger.error(f"Neo4j export failed: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
