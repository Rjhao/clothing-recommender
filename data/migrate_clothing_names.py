"""One-time, idempotent migration to make Clothing node names unique.

Problem
-------
The Amazon dataset has no product title, so ``import_amazon_fashion.py`` fell
back to ``category_code`` (e.g. ``apparel.shoes.slipons``) as the name → 820
items named "Apparel Shoes", 531 "Apparel Shoes Sandals", etc. The
recommendation UI therefore shows rows of identical cards.

Fix
---
Rewrite each name to ``{brand} {base_name} #{id_last4}`` so every name is
unique while staying human-readable. Collision groups (same brand + base +
last-4 id) are widened to 8 / full id until unique.

Scope
-----
- Neo4j: 4872 Clothing nodes live here → always updated.
- MySQL: updated only if a ``clothing_products`` table exists (the current
  deployment keeps the catalogue in Neo4j only, so MySQL is usually skipped).

Safe to re-run: names already ending in ``#{id_last4}`` are skipped.

Usage:
    python -m data.migrate_clothing_names
"""
import logging
from collections import defaultdict

from api.database import (
    get_neo4j_driver,
    close_neo4j_driver,
    get_settings,
)
from data.name_utils import build_unique_name, is_name_migrated, extend_suffix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_collisions(planned):
    """Widen the id suffix for any new_name shared by more than one item.

    ``planned``: list of ``(id, new_name)``. Returns ``{id: unique_name}``.
    """
    by_name = defaultdict(list)
    for cid, nm in planned:
        by_name[nm].append(cid)

    final = {}
    used = set()
    collisions = 0
    for nm, ids in by_name.items():
        if len(ids) == 1:
            final[ids[0]] = nm
            used.add(nm)
            continue
        collisions += 1
        for cid in ids:
            chosen = None
            for width in (8, 12, len(str(cid)) or 4):
                cand = extend_suffix(nm, cid, width)
                if cand not in used:
                    chosen = cand
                    break
            if chosen is None:
                # ultimate fallback: append the whole id (guaranteed unique)
                chosen = f"{extend_suffix(nm, cid, len(str(cid)))} {cid}"
                while chosen in used:
                    chosen += "_"
            final[cid] = chosen
            used.add(chosen)
    if collisions:
        logger.info("Resolved %d collision groups by widening the id suffix", collisions)
    return final


def migrate_neo4j():
    """Rewrite Clothing.name in Neo4j to the unique format."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Clothing)
                WHERE c.id IS NOT NULL
                RETURN c.id AS id, c.name AS name, c.brand AS brand
                """
            )
            records = [(r["id"], r["name"], r["brand"] or "") for r in result]
        logger.info("Loaded %d Clothing nodes from Neo4j", len(records))

        original = {cid: name for cid, name, _ in records}
        planned = []
        skipped = 0
        for cid, name, brand in records:
            if is_name_migrated(name, cid):
                skipped += 1
                planned.append((cid, name))  # keep as-is
            else:
                planned.append((cid, build_unique_name(name, brand, cid)))

        final = _resolve_collisions(planned)
        # Only write back names that actually changed (true idempotency).
        items = [
            {"id": cid, "name": nm}
            for cid, nm in final.items()
            if nm != original.get(cid)
        ]

        BATCH = 500
        if items:
            with driver.session() as session:
                for i in range(0, len(items), BATCH):
                    batch = items[i:i + BATCH]
                    session.run(
                        """
                        UNWIND $batch AS row
                        MATCH (c:Clothing {id: row.id})
                        SET c.name = row.name
                        """,
                        batch=batch,
                    )

        logger.info(
            "Updated %d names (%d already-migrated, skipped)", len(items), skipped
        )
        _report_uniqueness(driver)
    finally:
        close_neo4j_driver()


def migrate_mysql():
    """Rewrite name in MySQL ``clothing_products`` if that table exists."""
    try:
        import pymysql
    except ImportError:
        logger.info("pymysql not available; skipping MySQL")
        return

    settings = get_settings()
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'clothing_products'")
            if not cur.fetchone():
                logger.info(
                    "MySQL has no clothing_products table; skipping "
                    "(catalogue lives in Neo4j only)"
                )
                return
            cur.execute("SELECT product_id, name, brand FROM clothing_products")
            rows = cur.fetchall()
            logger.info("Loaded %d rows from MySQL clothing_products", len(rows))

            original = {}
            planned = []
            skipped = 0
            for pid, name, brand in rows:
                original[pid] = name
                if is_name_migrated(name, pid):
                    skipped += 1
                    planned.append((pid, name))
                else:
                    planned.append((pid, build_unique_name(name, brand or "", pid)))

            # Resolve collisions: widen the id suffix where two rows would clash.
            by_name = defaultdict(list)
            for pid, nm in planned:
                by_name[nm].append(pid)
            final = {}
            used = set()
            for nm, pids in by_name.items():
                if len(pids) == 1:
                    final[pids[0]] = nm
                    used.add(nm)
                    continue
                for pid in pids:
                    cand = extend_suffix(nm, pid, 8)
                    if cand in used:
                        cand = extend_suffix(nm, pid, len(str(pid)) or 4)
                    final[pid] = cand
                    used.add(cand)

            updates = [(nm, pid) for pid, nm in final.items() if nm != original.get(pid)]
            if updates:
                cur.executemany(
                    "UPDATE clothing_products SET name = %s WHERE product_id = %s",
                    updates,
                )
                conn.commit()
            logger.info(
                "MySQL: updated %d names (%d already-migrated, skipped)",
                len(updates), skipped,
            )
    finally:
        conn.close()


def _report_uniqueness(driver):
    """Log the post-migration duplicate count (should be 0)."""
    with driver.session() as session:
        r = session.run(
            """
            MATCH (c:Clothing)
            WHERE c.name IS NOT NULL
            RETURN count(c) AS rows, count(DISTINCT c.name) AS distinct_names
            """
        ).single()
        rows, distinct = r["rows"], r["distinct_names"]
        logger.info(
            "Post-migration: %d rows, %d distinct names, %d duplicates remaining",
            rows, distinct, rows - distinct,
        )
        if rows - distinct > 0:
            logger.warning("Still-duplicated names (showing up to 10):")
            for rec in session.run(
                """
                MATCH (c:Clothing) WHERE c.name IS NOT NULL
                WITH c.name AS name, collect(c.id)[..5] AS ids, count(*) AS cnt
                WHERE cnt > 1
                RETURN name, ids, cnt ORDER BY cnt DESC LIMIT 10
                """
            ):
                logger.warning("  %d x %r -> ids %r", rec["cnt"], rec["name"], rec["ids"])


def main():
    logger.info("=== Migrating Clothing names (Neo4j) ===")
    migrate_neo4j()
    logger.info("=== Migrating Clothing names (MySQL, if applicable) ===")
    migrate_mysql()
    logger.info("Done.")


if __name__ == "__main__":
    main()
