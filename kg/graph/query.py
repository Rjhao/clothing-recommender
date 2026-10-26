"""Knowledge Graph queries for recommendations."""
from neo4j import Driver
from typing import List, Optional


class ClothingGraphQuery:
    """Query class for clothing knowledge graph."""

    # Find clothing similar to user's preferences
    FIND_BY_PREFERENCE = """
    MATCH (u:User {id: $user_id})-[:PREFERS]->(s:Style)<-[:HAS_STYLE]-(c:Clothing)
    WHERE c.category = $category
    RETURN c, s.name as matched_style
    ORDER BY c.price DESC
    LIMIT $limit
    """

    # Find user's purchase history
    GET_USER_HISTORY = """
    MATCH (u:User {id: $user_id})-[r:HAS_STYLE|PREFERS|BOUGHT|VIEWED]->(c:Clothing)
    RETURN c, type(r) as relation_type, r.score as score, r.timestamp as timestamp
    ORDER BY r.timestamp DESC
    LIMIT $limit
    """

    # Find similar clothing items
    GET_SIMILAR_CLOTHING = """
    MATCH (c:Clothing {id: $clothing_id})-[r:SIMILAR_TO]-(similar:Clothing)
    WHERE r.score > $threshold
    RETURN similar, r.score as similarity_score
    ORDER BY similarity_score DESC
    LIMIT $limit
    """

    # Find matching outfits
    GET_MATCHING_OUTFITS = """
    MATCH (c:Clothing {id: $clothing_id})-[r:MATCHES_WITH]-(match:Clothing)
    WHERE r.score > $threshold
    RETURN match, r.score as match_score
    ORDER BY match_score DESC
    LIMIT $limit
    """

    # Find clothing by category and style
    FIND_BY_CATEGORY_STYLE = """
    MATCH (c:Clothing)
    WHERE ($category IS NULL OR c.category = $category)
      AND ($style IS NULL OR c.style = $style)
      AND ($season IS NULL OR c.season = $season)
      AND ($occasion IS NULL OR c.occasion = $occasion)
      AND c.price >= $min_price
      AND c.price <= $max_price
    RETURN c, c.id as id
    ORDER BY c.price DESC
    SKIP $offset
    LIMIT $limit
    """

    # Get user's style profile from graph
    GET_USER_STYLE_PROFILE = """
    MATCH (u:User {id: $user_id})-[r:PREFERS|HAS_STYLE]->(s:Style)
    RETURN s, avg(r.score) as preference_score
    ORDER BY preference_score DESC
    """

    # Collaborative filtering via graph
    FIND_SIMILAR_USERS = """
    MATCH (u1:User {id: $user_id})-[r1:PREFERS]->(s:Style)<-[r2:PREFERS]-(u2:User)
    WHERE u1 <> u2
    WITH u2, collect(s.name) as common_styles, count(s) as style_overlap
    ORDER BY style_overlap DESC
    LIMIT $limit
    MATCH (u2)-[:BOUGHT|VIEWED]->(c:Clothing)
    RETURN u2.id as similar_user_id, common_styles, style_overlap, c
    ORDER BY style_overlap DESC
    LIMIT $limit
    """

    # Get clothing by IDs
    GET_CLOTHING_BY_IDS = """
    MATCH (c:Clothing)
    WHERE c.id IN $clothing_ids
    RETURN c
    """

    # Cold-start fallback: popular clothing (by popularity, then price).
    # Used when a user has no preferences / interactions in the graph yet.
    # coalesce() so items missing popularity don't sort first on NULL.
    FIND_POPULAR_CLOTHING = """
    MATCH (c:Clothing)
    WHERE ($category IS NULL OR c.category = $category)
      AND NOT c.id IN $exclude_ids
    RETURN c
    ORDER BY coalesce(c.popularity, 0) DESC, c.price DESC
    SKIP $offset
    LIMIT $limit
    """

    # === KG visualization endpoints (added for KnowledgeGraph page) ===

    # Count nodes by label
    KG_STATS_NODES = """
    MATCH (n)
    WHERE size(labels(n)) > 0
    RETURN labels(n)[0] AS label, count(n) AS cnt
    """

    # Count relationships by type
    KG_STATS_RELATIONSHIPS = """
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(r) AS cnt
    """

    # Sample N nodes (random) for force-directed layout
    KG_GRAPH_NODES = """
    MATCH (n)
    WHERE size(labels(n)) > 0
    WITH n, rand() AS r
    ORDER BY r
    LIMIT $limit
    RETURN id(n) AS neo4j_id,
           labels(n)[0] AS label,
           coalesce(n.id, toString(id(n))) AS biz_id,
           properties(n) AS props
    """

    # Type-specific sampling so small-label nodes (Style=8, Category=6)
    # are always included even if their population is dwarfed by Clothing.
    KG_GRAPH_NODES_BY_LABEL = """
    MATCH (n:`%s`)
    WITH n, rand() AS r
    ORDER BY r
    LIMIT $limit
    RETURN id(n) AS neo4j_id,
           labels(n)[0] AS label,
           coalesce(n.id, toString(id(n))) AS biz_id,
           properties(n) AS props
    """

    # Edges whose both endpoints are inside the sampled id list.
    KG_GRAPH_EDGES = """
    UNWIND $ids AS nid
    MATCH (n) WHERE id(n) = nid
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE id(m) IN $ids
    RETURN id(n) AS source,
           id(m) AS target,
           type(r) AS rel_type,
           properties(r) AS props
    """

    # Graph-based recommendation score
    # Uses correlated subqueries so each score is aggregated independently,
    # avoiding the cross-product that previously inflated sum()/count() per c.
    COMPUTE_RECOMMEND_SCORE = """
    MATCH (u:User {id: $user_id})
    CALL {
      WITH u
      MATCH (u)-[up:PREFERS]->(s:Style)<-[:HAS_STYLE]-(c:Clothing)
      RETURN c, sum(up.score) AS style_match_score
    }
    CALL {
      WITH c
      MATCH (c)-[:MATCHES_WITH]-(m:Clothing)
      RETURN count(m) AS outfit_match_count
    }
    RETURN c,
           style_match_score,
           outfit_match_count
    ORDER BY style_match_score DESC, outfit_match_count DESC
    LIMIT $limit
    """

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver

    def find_by_preference(
        self,
        user_id: int,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[dict]:
        """Find clothing based on user preferences."""
        with self.driver.session() as session:
            result = session.run(
                self.FIND_BY_PREFERENCE,
                user_id=user_id,
                category=category,
                limit=limit
            )
            return [dict(record) for record in result]

    def get_user_history(self, user_id: int, limit: int = 50) -> List[dict]:
        """Get user's clothing interaction history."""
        with self.driver.session() as session:
            result = session.run(
                self.GET_USER_HISTORY,
                user_id=user_id,
                limit=limit
            )
            return [dict(record) for record in result]

    def get_similar_clothing(
        self,
        clothing_id: int,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[dict]:
        """Get similar clothing items."""
        with self.driver.session() as session:
            result = session.run(
                self.GET_SIMILAR_CLOTHING,
                clothing_id=clothing_id,
                threshold=threshold,
                limit=limit
            )
            return [dict(record) for record in result]

    def get_matching_outfits(
        self,
        clothing_id: int,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[dict]:
        """Get matching outfit items."""
        with self.driver.session() as session:
            result = session.run(
                self.GET_MATCHING_OUTFITS,
                clothing_id=clothing_id,
                threshold=threshold,
                limit=limit
            )
            return [dict(record) for record in result]

    def find_by_category_style(
        self,
        category: str,
        style: Optional[str] = None,
        season: Optional[str] = None,
        occasion: Optional[str] = None,
        min_price: float = 0,
        max_price: float = 10000,
        offset: int = 0,
        limit: int = 20
    ) -> List[dict]:
        """Find clothing by category and style filters."""
        with self.driver.session() as session:
            result = session.run(
                self.FIND_BY_CATEGORY_STYLE,
                category=category,
                style=style,
                season=season,
                occasion=occasion,
                min_price=min_price,
                max_price=max_price,
                offset=offset,
                limit=limit
            )
            result_list = []
            for record in result:
                item = dict(record['c'])
                item['id'] = record['id']
                result_list.append(item)
            result.consume()
            return result_list
    def get_user_style_profile(self, user_id: int) -> List[dict]:
        """Get user's style preference profile."""
        with self.driver.session() as session:
            result = session.run(
                self.GET_USER_STYLE_PROFILE,
                user_id=user_id
            )
            return [dict(record) for record in result]

    def find_similar_users(self, user_id: int, limit: int = 10) -> List[dict]:
        """Find similar users for collaborative filtering."""
        with self.driver.session() as session:
            result = session.run(
                self.FIND_SIMILAR_USERS,
                user_id=user_id,
                limit=limit
            )
            return [dict(record) for record in result]

    def compute_recommendation_scores(
        self,
        user_id: int,
        limit: int = 20
    ) -> List[dict]:
        """Compute recommendation scores using graph algorithms."""
        with self.driver.session() as session:
            result = session.run(
                self.COMPUTE_RECOMMEND_SCORE,
                user_id=user_id,
                limit=limit
            )
            return [dict(record) for record in result]

    def get_clothing_by_ids(self, clothing_ids: List[int]) -> List[dict]:
        """Get clothing items by their IDs."""
        if not clothing_ids:
            return []
        with self.driver.session() as session:
            result = session.run(
                self.GET_CLOTHING_BY_IDS,
                clothing_ids=clothing_ids
            )
            items = []
            for record in result:
                item = dict(record['c'])
                # Neo4j may store id as integer
                item['id'] = item.get('id', 0)
                items.append(item)
            result.consume()
            return items

    def get_graph_stats(self) -> dict:
        """Aggregate node / relationship counts for the KnowledgeGraph page."""
        with self.driver.session() as session:
            node_counts = {
                r["label"]: r["cnt"]
                for r in session.run(self.KG_STATS_NODES)
            }
            rel_counts = {
                r["type"]: r["cnt"]
                for r in session.run(self.KG_STATS_RELATIONSHIPS)
            }
        return {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }

    def get_subgraph(self, limit: int = 100) -> dict:
        """Return a sampled subgraph (nodes + edges) for force-directed viz.

        Sampling is type-aware so small-cardinality labels (Style=8,
        Category=6) are always represented. Edges are restricted to those
        whose endpoints both fall inside the sampled id window, which
        avoids dangling pointers in the frontend.

        ``limit`` caps the *Clothing* sample; small labels (Style/Category)
        are returned in full, Brand is sampled at ``min(limit//3, 80)``.
        """
        clothing_limit = max(10, min(limit, 200))
        brand_limit = max(5, min(limit // 3, 80))
        labels_to_sample_full = ("Style", "Category")

        with self.driver.session() as session:
            nodes = []
            id_to_biz = {}

            # Full inclusion for tiny labels
            for lbl in labels_to_sample_full:
                q = self.KG_GRAPH_NODES_BY_LABEL % lbl
                for r in session.run(q, limit=1000):
                    neo4j_id = r["neo4j_id"]
                    biz_id = str(r["biz_id"])
                    id_to_biz[neo4j_id] = biz_id
                    nodes.append({
                        "id": biz_id,
                        "neo4j_id": neo4j_id,
                        "type": r["label"],
                        "props": dict(r["props"] or {}),
                    })

            # Sampled inclusion for high-cardinality labels
            for lbl, lim in (("Brand", brand_limit), ("Clothing", clothing_limit)):
                q = self.KG_GRAPH_NODES_BY_LABEL % lbl
                for r in session.run(q, limit=lim):
                    neo4j_id = r["neo4j_id"]
                    biz_id = str(r["biz_id"])
                    id_to_biz[neo4j_id] = biz_id
                    nodes.append({
                        "id": biz_id,
                        "neo4j_id": neo4j_id,
                        "type": r["label"],
                        "props": dict(r["props"] or {}),
                    })

            ids = list(id_to_biz.keys())
            edges = []
            for r in session.run(self.KG_GRAPH_EDGES, ids=ids):
                tgt = r["target"]
                if tgt is None:
                    continue
                src = r["source"]
                rel_props = dict(r["props"] or {})
                edges.append({
                    "source": id_to_biz.get(src, str(src)),
                    "target": id_to_biz.get(tgt, str(tgt)),
                    "type": r["rel_type"],
                    "weight": float(rel_props.get("score", 1.0)) or 1.0,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "sample_limit": limit,
            "sampled_labels": list(labels_to_sample_full) + ["Brand", "Clothing"],
        }

    def find_popular_clothing(
        self,
        limit: int = 20,
        exclude_ids: Optional[set] = None,
        category: Optional[str] = None,
        offset: int = 0,
    ) -> List[dict]:
        """Find popular clothing (cold-start fallback when a user has no graph
        profile). Returns clothing dicts ordered by popularity then price."""
        with self.driver.session() as session:
            result = session.run(
                self.FIND_POPULAR_CLOTHING,
                category=category,
                exclude_ids=list(exclude_ids or []),
                offset=offset,
                limit=limit,
            )
            items = []
            for record in result:
                item = dict(record['c'])
                if 'id' not in item:
                    item['id'] = item.get('product_id', 0)
                items.append(item)
            result.consume()
            return items