"""Export cleaned clothing data to MySQL and Neo4j."""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataExporter:
    """Export cleaned product data to databases."""

    def __init__(self, mysql_conn=None, neo4j_driver=None):
        """
        Initialize data exporter.

        Args:
            mysql_conn: MySQL database connection (pymysql connection)
            neo4j_driver: Neo4j driver instance
        """
        self.mysql_conn = mysql_conn
        self.neo4j_driver = neo4j_driver

    def export_to_mysql(
        self,
        products: List[Dict],
        table_name: str = "clothing_products"
    ) -> int:
        """
        Export products to MySQL.

        Args:
            products: List of cleaned product dictionaries
            table_name: Target table name

        Returns:
            Number of records exported
        """
        if not self.mysql_conn:
            logger.warning("MySQL connection not provided, skipping MySQL export")
            return 0

        cursor = self.mysql_conn.cursor()

        # Create table if not exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            product_id VARCHAR(64) UNIQUE NOT NULL,
            name VARCHAR(512),
            category VARCHAR(64),
            sub_category VARCHAR(64),
            brand VARCHAR(128),
            color VARCHAR(64),
            size VARCHAR(32),
            material VARCHAR(64),
            style VARCHAR(64),
            season VARCHAR(32),
            price DECIMAL(10, 2),
            platform VARCHAR(32),
            source_url TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_category (category),
            INDEX idx_style (style),
            INDEX idx_brand (brand),
            INDEX idx_platform (platform)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            cursor.execute(create_table_sql)
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return 0

        # Insert products
        insert_sql = f"""
        INSERT IGNORE INTO {table_name}
        (product_id, name, category, brand, color, size, material, style, season, price, platform, source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        count = 0
        for product in products:
            try:
                cursor.execute(insert_sql, (
                    product.get("product_id", ""),
                    product.get("name", ""),
                    product.get("category", ""),
                    product.get("brand", ""),
                    product.get("color", ""),
                    product.get("size", ""),
                    product.get("material", ""),
                    product.get("style", ""),
                    product.get("season", ""),
                    product.get("price", 0),
                    product.get("platform", ""),
                    product.get("source", ""),
                ))
                count += 1
            except Exception as e:
                logger.debug(f"Failed to insert product {product.get('product_id')}: {e}")

        self.mysql_conn.commit()
        cursor.close()

        logger.info(f"Exported {count} products to MySQL table '{table_name}'")
        return count

    def export_to_neo4j(self, products: List[Dict]) -> Dict[str, int]:
        """
        Export products to Neo4j knowledge graph.

        Args:
            products: List of cleaned product dictionaries

        Returns:
            Statistics dictionary with counts
        """
        # DEPRECATED: this method builds Clothing nodes with a schema
        # (product_id string + BELONGS_TO) that does NOT match the query layer
        # in kg/graph/query.py (integer id + IN_CATEGORY). Graph building must
        # go through data.import_to_kg instead, which produces the canonical
        # schema the recommendation engine expects.
        logger.warning(
            "DataExporter.export_to_neo4j is deprecated; "
            "use data.import_to_kg for graph building. Skipping."
        )
        return {"nodes": 0, "relationships": 0, "deprecated": True}

        with self.neo4j_driver.session() as session:
            # Create indexes
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (c:Clothing) ON (c.product_id)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (cat:Category) ON (cat.name)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (s:Style) ON (s.name)
            """)

            # Create style nodes
            session.run("""
                MERGE (s:Style {name: 'minimalist'})
                SET s.display_name = '简约风'
                MERGE (s2:Style {name: 'casual'})
                SET s2.display_name = '休闲风'
                MERGE (s3:Style {name: 'bohemian'})
                SET s3.display_name = '波西米亚风'
                MERGE (s4:Style {name: 'vintage'})
                SET s4.display_name = '复古风'
                MERGE (s5:Style {name: 'streetwear'})
                SET s5.display_name = '街头风'
                MERGE (s6:Style {name: 'business'})
                SET s6.display_name = '商务风'
                MERGE (s7:Style {name: 'sporty'})
                SET s7.display_name = '运动风'
                MERGE (s8:Style {name: 'romantic'})
                SET s8.display_name = '浪漫风'
            """)

            # Create category nodes
            for cat in ["top", "bottom", "dress", "outerwear", "shoes", "accessories"]:
                session.run(f"""
                    MERGE (c:Category {{name: '{cat}'}})
                """)

            # Create product nodes and relationships
            for product in products:
                pid = product.get("product_id", "")
                category = product.get("category", "unknown")
                style = product.get("style", "")
                brand = product.get("brand", "")

                # Create product node
                session.run("""
                    MERGE (p:Clothing {product_id: $product_id})
                    SET p.name = $name,
                        p.price = $price,
                        p.platform = $platform,
                        p.color = $color,
                        p.material = $material,
                        p.season = $season,
                        p.source = $source
                """, {
                    "product_id": pid,
                    "name": product.get("name", ""),
                    "price": product.get("price", 0),
                    "platform": product.get("platform", ""),
                    "color": product.get("color", ""),
                    "material": product.get("material", ""),
                    "season": product.get("season", ""),
                    "source": product.get("source", ""),
                })
                stats["nodes"] += 1

                # Link to category
                if category:
                    session.run("""
                        MATCH (p:Clothing {product_id: $pid})
                        MATCH (c:Category {name: $cat})
                        MERGE (p)-[:BELONGS_TO]->(c)
                    """, pid=pid, cat=category)
                    stats["relationships"] += 1

                # Link to style
                if style:
                    session.run("""
                        MATCH (p:Clothing {product_id: $pid})
                        MATCH (s:Style {name: $style})
                        MERGE (p)-[:HAS_STYLE]->(s)
                    """, pid=pid, style=style)
                    stats["relationships"] += 1

                # Link to brand
                if brand:
                    session.run("""
                        MATCH (p:Clothing {product_id: $pid})
                        MERGE (b:Brand {name: $brand})
                        MERGE (p)-[:FROM_BRAND]->(b)
                    """, pid=pid, brand=brand)
                    stats["relationships"] += 1

            # Create outfit rules (COORINATES_WITH relationships)
            # tops match with bottoms, dresses, outerwear
            category_rules = {
                "top": ["bottom", "dress", "outerwear"],
                "bottom": ["top", "outerwear", "shoes"],
                "dress": ["outerwear", "shoes", "accessories"],
                "outerwear": ["top", "bottom", "shoes"],
                "shoes": ["top", "bottom", "dress"],
                "accessories": ["top", "dress"],
            }

            for cat1, matching_cats in category_rules.items():
                for cat2 in matching_cats:
                    session.run("""
                        MATCH (c1:Clothing)-[:BELONGS_TO]->(:Category {name: $cat1})
                        MATCH (c2:Clothing)-[:BELONGS_TO]->(:Category {name: $cat2})
                        WHERE c1 <> c2
                        MERGE (c1)-[:MATCHES_WITH {confidence: 0.7}]->(c2)
                    """, cat1=cat1, cat2=cat2)
                    stats["relationships"] += 1

            # Create SIMILAR_TO within same category and style
            session.run("""
                MATCH (c1:Clothing)-[:HAS_STYLE]->(s:Style)
                MATCH (c2:Clothing)-[:HAS_STYLE]->(s)
                MATCH (c1)-[:BELONGS_TO]->(cat:Category)<-[:BELONGS_TO]-(c2)
                WHERE c1 <> c2 AND c1.product_id < c2.product_id
                WITH c1, c2, cat, s, abs(c1.price - c2.price) as price_dist
                WHERE price_dist < 50
                MERGE (c1)-[:SIMILAR_TO {style: s.name, category: cat.name}]->(c2)
            """)
            stats["relationships"] += 1

        logger.info(
            f"Exported {stats['nodes']} nodes and "
            f"{stats['relationships']} relationships to Neo4j"
        )
        return stats

    def export_all(
        self,
        products: List[Dict],
        mysql_conn=None,
        neo4j_driver=None
    ) -> Dict:
        """
        Export products to both MySQL and Neo4j.

        Args:
            products: List of cleaned products
            mysql_conn: MySQL connection
            neo4j_driver: Neo4j driver

        Returns:
            Combined statistics
        """
        if mysql_conn:
            self.mysql_conn = mysql_conn
        if neo4j_driver:
            self.neo4j_driver = neo4j_driver

        results = {}

        # Export to MySQL
        mysql_count = self.export_to_mysql(products)
        results["mysql"] = {"exported": mysql_count}

        # Export to Neo4j
        neo4j_stats = self.export_to_neo4j(products)
        results["neo4j"] = neo4j_stats

        return results


def get_export_stats(products: List[Dict]) -> Dict:
    """
    Get statistics about products for export preview.

    Args:
        products: List of products

    Returns:
        Statistics dictionary
    """
    stats = {
        "total": len(products),
        "by_category": defaultdict(int),
        "by_platform": defaultdict(int),
        "by_style": defaultdict(int),
        "price_range": {"min": float("inf"), "max": 0, "sum": 0},
        "with_category": 0,
        "with_color": 0,
        "with_material": 0,
    }

    for p in products:
        if p.get("category"):
            stats["by_category"][p["category"]] += 1
            stats["with_category"] += 1
        if p.get("platform"):
            stats["by_platform"][p["platform"]] += 1
        if p.get("style"):
            stats["by_style"][p["style"]] += 1
        if p.get("color"):
            stats["with_color"] += 1
        if p.get("material"):
            stats["with_material"] += 1

        price = p.get("price", 0)
        if price > 0:
            stats["price_range"]["min"] = min(stats["price_range"]["min"], price)
            stats["price_range"]["max"] = max(stats["price_range"]["max"], price)
            stats["price_range"]["sum"] += price

    if stats["price_range"]["min"] == float("inf"):
        stats["price_range"]["min"] = 0

    if stats["total"] > 0:
        stats["price_range"]["avg"] = stats["price_range"]["sum"] / stats["total"]

    return dict(stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    sample_products = [
        {
            "product_id": "JD001",
            "name": "简约纯棉T恤 男士 白色",
            "category": "top",
            "brand": "Nike",
            "color": "白色",
            "material": "棉",
            "style": "minimalist",
            "season": "summer",
            "price": 129.0,
            "platform": "JD",
            "source": "https://item.jd.com/12345.html",
        },
        {
            "product_id": "JD002",
            "name": "波西米亚连衣裙 女 碎花",
            "category": "dress",
            "brand": "ZARA",
            "color": "红色",
            "material": "雪纺",
            "style": "bohemian",
            "season": "spring",
            "price": 399.0,
            "platform": "JD",
            "source": "https://item.jd.com/12346.html",
        },
    ]

    stats = get_export_stats(sample_products)
    print("Export preview statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
