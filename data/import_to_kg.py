"""Import Amazon Fashion data into Neo4j knowledge graph.

Usage:
    python -m data.import_to_kg

This script imports clothing products from amazon_fashion_with_images.json
into Neo4j, creating nodes and relationships for the knowledge graph.
"""
import json
import logging
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import get_neo4j_driver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Valid categories in our system
VALID_CATEGORIES = {'top', 'bottom', 'dress', 'outerwear', 'shoes', 'accessories'}

# Valid styles
VALID_STYLES = {'minimalist', 'casual', 'bohemian', 'vintage', 'streetwear', 'business', 'sporty', 'romantic'}

# Valid seasons
VALID_SEASONS = {'spring', 'summer', 'fall', 'winter', 'all_season'}

# Valid occasions
VALID_OCCASIONS = {'casual', 'work', 'formal', 'sports', 'party', 'daily'}

# Outfit matching rules - which categories match with which
MATCHING_RULES = {
    'top': ['bottom', 'dress', 'outerwear', 'shoes', 'accessories'],
    'bottom': ['top', 'outerwear', 'shoes', 'accessories'],
    'dress': ['outerwear', 'shoes', 'accessories'],
    'outerwear': ['top', 'bottom', 'shoes', 'accessories'],
    'shoes': ['top', 'bottom', 'dress', 'accessories'],
    'accessories': ['top', 'bottom', 'dress', 'outerwear'],
}


def clean_category(cat: str) -> str:
    """Normalize category to valid value."""
    if not cat:
        return 'accessories'
    cat = cat.lower().strip()
    if cat not in VALID_CATEGORIES:
        return 'accessories'
    return cat


def clean_style(styles: list) -> str:
    """Get the best style from a list of styles."""
    if not styles:
        return 'casual'
    # Prefer valid style, fallback to first or 'casual'
    for s in styles:
        if s in VALID_STYLES:
            return s
    return 'casual'


def clean_season(product: dict) -> str:
    """Determine season from product."""
    return product.get('season', 'all_season') or 'all_season'


class DataImporter:
    """Import Amazon Fashion data into Neo4j."""

    def __init__(self):
        self.driver = get_neo4j_driver()
        self.stats = {
            'clothing': 0,
            'categories': 0,
            'styles': 0,
            'brands': 0,
            'relationships': 0,
            'skipped': 0
        }

    def clear_graph(self):
        """Clear all existing data in the graph."""
        logger.info("Clearing existing graph data...")
        with self.driver.session() as session:
            # Delete all relationships first, then nodes
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared.")

    def create_constraints(self):
        """Create Neo4j constraints for unique IDs."""
        logger.info("Creating constraints...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clothing) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Style) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    logger.warning(f"Constraint creation: {e}")
        logger.info("Constraints created.")

    def import_categories_and_styles(self):
        """Create Category and Style nodes."""
        logger.info("Creating category and style nodes...")

        categories_query = """
        UNWIND $categories AS cat
        MERGE (c:Category {name: cat})
        """
        styles_query = """
        UNWIND $styles AS style
        MERGE (s:Style {name: style})
        """

        with self.driver.session() as session:
            session.run(categories_query, categories=list(VALID_CATEGORIES))
            session.run(styles_query, styles=list(VALID_STYLES))

        self.stats['categories'] = len(VALID_CATEGORIES)
        self.stats['styles'] = len(VALID_STYLES)
        logger.info(f"Created {self.stats['categories']} categories and {self.stats['styles']} styles.")

    def import_clothing_items(self, products: list):
        """Import clothing items as nodes."""
        logger.info(f"Importing {len(products)} clothing items...")

        clothing_query = """
        MERGE (c:Clothing {id: toInteger($product_id)})
        SET c.name = $name,
            c.category = $category,
            c.brand = $brand,
            c.color = $color,
            c.price = $price,
            c.style = $style,
            c.season = $season,
            c.image_url = $image_url,
            c.platform = $platform,
            c.popularity = $popularity
        """

        category_rel_query = """
        MATCH (c:Clothing {id: toInteger($product_id)})
        MATCH (cat:Category {name: $category})
        MERGE (c)-[:IN_CATEGORY]->(cat)
        """

        style_rel_query = """
        MATCH (c:Clothing {id: toInteger($product_id)})
        MATCH (s:Style {name: $style})
        MERGE (c)-[:HAS_STYLE]->(s)
        """

        brand_query = """
        MATCH (c:Clothing {id: toInteger($product_id)})
        MERGE (b:Brand {name: $brand})
        MERGE (c)-[:FROM_BRAND]->(b)
        """

        count = 0
        with self.driver.session() as session:
            for product in products:
                product_id = product.get('product_id')
                if not product_id:
                    self.stats['skipped'] += 1
                    continue

                try:
                    session.run(clothing_query, {
                        'product_id': product_id,
                        'name': product.get('name', 'Unknown'),
                        'category': clean_category(product.get('category')),
                        'brand': product.get('brand', 'Unknown'),
                        'color': product.get('color'),
                        'price': product.get('price', 0),
                        'style': clean_style(product.get('styles', [])),
                        'season': clean_season(product),
                        'image_url': product.get('image_url'),
                        'platform': product.get('platform', 'AMAZON'),
                        'popularity': product.get('popularity', 0)
                    })

                    # Create relationships
                    category = clean_category(product.get('category'))
                    style = clean_style(product.get('styles', []))

                    session.run(category_rel_query, {
                        'product_id': product_id,
                        'category': category
                    })
                    session.run(style_rel_query, {
                        'product_id': product_id,
                        'style': style
                    })
                    if product.get('brand'):
                        session.run(brand_query, {
                            'product_id': product_id,
                            'brand': product.get('brand')
                        })

                    count += 1
                    if count % 500 == 0:
                        logger.info(f"Imported {count}/{len(products)} items...")

                except Exception as e:
                    logger.warning(f"Failed to import product {product_id}: {e}")
                    self.stats['skipped'] += 1

        self.stats['clothing'] = count
        logger.info(f"Imported {count} clothing items (skipped: {self.stats['skipped']})")

    def create_matching_relationships(self, limit_per_category: int = 50):
        """Create MATCHES_WITH relationships between complementary items."""
        logger.info("Creating matching relationships between clothing items...")

        # For efficiency, limit relationships per category
        for cat1, matching_cats in MATCHING_RULES.items():
            for cat2 in matching_cats:
                query = f"""
                MATCH (c1:Clothing)
                WHERE c1.category = $cat1
                WITH c1 LIMIT $limit
                MATCH (c2:Clothing)
                WHERE c2.category = $cat2 AND c1.id <> c2.id
                WITH c1, c2 LIMIT $limit
                MERGE (c1)-[r:MATCHES_WITH]->(c2)
                SET r.score = 0.6
                """
                try:
                    with self.driver.session() as session:
                        result = session.run(query, {'cat1': cat1, 'cat2': cat2, 'limit': limit_per_category})
                        # Count relationships created
                        summary = result.consume()
                        self.stats['relationships'] += summary.counters.relationships_created
                except Exception as e:
                    logger.warning(f"Failed to create matches for {cat1}-{cat2}: {e}")

        logger.info(f"Created {self.stats['relationships']} matching relationships")

    def create_similarity_relationships(self, limit_per_style: int = 20):
        """Create SIMILAR_TO relationships between items of the same style."""
        logger.info("Creating similarity relationships...")

        for style in VALID_STYLES:
            query = """
            MATCH (c1:Clothing)
            WHERE c1.style = $style
            WITH c1 LIMIT $limit
            MATCH (c2:Clothing)
            WHERE c2.style = $style AND c1.id <> c2.id
            WITH c1, c2 LIMIT $limit
            MERGE (c1)-[r:SIMILAR_TO]->(c2)
            SET r.score = 0.7
            """
            try:
                with self.driver.session() as session:
                    result = session.run(query, {'style': style, 'limit': limit_per_style})
                    summary = result.consume()
                    self.stats['relationships'] += summary.counters.relationships_created
            except Exception as e:
                logger.warning(f"Failed to create similarities for style {style}: {e}")

        logger.info(f"Total relationships now: {self.stats['relationships']}")

    def run(self, json_path: str = None):
        """Run the full import process."""
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'amazon_fashion_real.json'
            )

        logger.info(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        products = data.get('products', [])
        logger.info(f"Loaded {len(products)} products from JSON")

        # ---- 去重：按 (name, category, brand) 合并重复商品 ----
        seen = {}  # key -> first product
        unique_key_products = []
        for p in products:
            key = (p.get('name'), p.get('category'), p.get('brand'))
            if key not in seen:
                seen[key] = True
                unique_key_products.append(p)

        dup_count = len(products) - len(unique_key_products)
        if dup_count > 0:
            logger.info(f"去重: {len(products)} -> {len(unique_key_products)} (移除 {dup_count} 个重复)")
        products = unique_key_products

        # Clear and setup
        self.clear_graph()
        self.create_constraints()
        self.import_categories_and_styles()

        # Import data
        self.import_clothing_items(products)

        # Create relationships
        self.create_matching_relationships(limit_per_category=30)
        self.create_similarity_relationships(limit_per_style=15)

        # Print stats
        logger.info("=" * 50)
        logger.info("Import complete!")
        logger.info(f"  Clothing nodes: {self.stats['clothing']}")
        logger.info(f"  Category nodes: {self.stats['categories']}")
        logger.info(f"  Style nodes: {self.stats['styles']}")
        logger.info(f"  Relationships: {self.stats['relationships']}")
        logger.info(f"  Skipped: {self.stats['skipped']}")
        logger.info("=" * 50)

        # Verify
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            logger.info("Node counts by type:")
            for record in result:
                logger.info(f"  {record['type']}: {record['count']}")


if __name__ == "__main__":
    importer = DataImporter()
    importer.run()
