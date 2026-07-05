"""Sync image URLs from MySQL into the Neo4j Clothing nodes.

A Clothing node may have been created by either import path:
  - data/import_to_kg.py        -> identity is integer ``id``
  - data/crawler/exporter.py    -> identity is string ``product_id``

We match by EITHER so the backfill works regardless of which path was used,
and we always write the canonical lowercase ``image_url`` property (the one the
API reads). This is idempotent and safe to re-run.
"""
import logging

import pymysql
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _config():
    """Read DB config from the shared settings (single source of truth)."""
    from api.database import get_settings
    s = get_settings()
    return {
        "mysql": {
            "host": s.mysql_host,
            "port": s.mysql_port,
            "user": s.mysql_user,
            "password": s.mysql_password,
            "database": s.mysql_database,
        },
        "neo4j": {
            "uri": s.neo4j_uri,
            "user": s.neo4j_user,
            "password": s.neo4j_password,
        },
    }


def sync_images():
    """Sync image URLs from MySQL clothing_products to Neo4j Clothing nodes."""
    cfg = _config()

    mysql_conn = pymysql.connect(**cfg["mysql"], charset="utf8mb4")
    products_with_images = []
    try:
        with mysql_conn.cursor() as cur:
            cur.execute(
                "SELECT product_id, image_url FROM clothing_products "
                "WHERE image_url IS NOT NULL AND image_url != ''"
            )
            products_with_images = cur.fetchall()
    finally:
        mysql_conn.close()

    logger.info("Found %d products with images in MySQL", len(products_with_images))
    if not products_with_images:
        return

    driver = GraphDatabase.driver(
        cfg["neo4j"]["uri"], auth=(cfg["neo4j"]["user"], cfg["neo4j"]["password"])
    )

    updated = 0
    batch_size = 100

    update_query = """
    MATCH (c:Clothing)
    WHERE c.id = toInteger($product_id) OR c.product_id = $product_id
    SET c.image_url = $image_url
    RETURN c.id AS id, c.product_id AS product_id
    """

    try:
        for i in range(0, len(products_with_images), batch_size):
            batch = products_with_images[i:i + batch_size]
            with driver.session() as session:
                for product_id, image_url in batch:
                    try:
                        result = session.run(
                            update_query,
                            product_id=str(product_id),
                            image_url=image_url,
                        )
                        if result.single() is not None:
                            updated += 1
                    except Exception as e:
                        logger.warning("Error updating %s: %s", product_id, e)
            logger.info("Updated %d/%d so far", updated, len(products_with_images))
    finally:
        driver.close()

    logger.info("Sync complete! Updated %d nodes in Neo4j", updated)


if __name__ == "__main__":
    sync_images()
