"""One-time, idempotent migration to normalize Clothing nodes to the canonical
schema expected by kg/graph/query.py and api/services/clothing.py:

  - identity:  integer ``id``  (backfill from product_id where missing)
  - image:     lowercase ``image_url``  (backfill from Image_url, then remove it)
  - category:  ``IN_CATEGORY``  (rename BELONGS_TO if the exporter path was used)

Safe to re-run: every statement is conditional (only touches rows that need it).

Usage:
    python -m data.migrate_clothing_schema

Run this once after bringing Neo4j up, before relying on recommendations /
image display. It does NOT delete any nodes or relationships.
"""
import logging

from api.database import get_neo4j_driver, close_neo4j_driver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MIGRATIONS = [
    # 1. Backfill integer id from product_id where missing
    (
        "Backfill id from product_id",
        """
        MATCH (c:Clothing)
        WHERE c.id IS NULL AND c.product_id IS NOT NULL
        SET c.id = toInteger(c.product_id)
        RETURN count(c) AS affected
        """,
    ),
    # 2. Normalize image property name: image_url (canonical, lowercase)
    (
        "Normalize image_url (from Image_url)",
        """
        MATCH (c:Clothing)
        WHERE c.image_url IS NULL AND c.Image_url IS NOT NULL
        SET c.image_url = c.Image_url
        REMOVE c.Image_url
        RETURN count(c) AS affected
        """,
    ),
    # 3. Rename BELONGS_TO -> IN_CATEGORY if the exporter path was used
    (
        "Rename BELONGS_TO -> IN_CATEGORY",
        """
        MATCH (c:Clothing)-[r:BELONGS_TO]->(cat:Category)
        MERGE (c)-[:IN_CATEGORY]->(cat)
        DELETE r
        RETURN count(r) AS affected
        """,
    ),
]


def ensure_constraints(session):
    """Ensure the canonical unique constraint on Clothing.id exists."""
    try:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clothing) REQUIRE c.id IS UNIQUE"
        )
        logger.info("Constraint on Clothing.id ensured.")
    except Exception as e:
        # Older Neo4j syntax fallback
        logger.warning("Could not create constraint (may already exist): %s", e)


def main():
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            ensure_constraints(session)
            for label, cypher in MIGRATIONS:
                result = session.run(cypher)
                rec = result.single()
                affected = rec["affected"] if rec is not None else 0
                logger.info("%s: %s rows", label, affected)

            # Report final state
            row = session.run("""
                MATCH (c:Clothing)
                RETURN count(c) AS total,
                       count(c.id) AS has_id,
                       count(c.product_id) AS has_product_id,
                       count(c.image_url) AS has_image_url,
                       count(c.Image_url) AS has_Image_url_upper
            """).single()
            logger.info("Post-migration Clothing state: %s", dict(row))
    finally:
        close_neo4j_driver()


if __name__ == "__main__":
    main()
