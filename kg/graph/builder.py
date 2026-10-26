"""Knowledge Graph builder for clothing recommendations."""
from neo4j import Driver
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ClothingGraphBuilder:
    """Builder for constructing clothing knowledge graph in Neo4j."""

    # Cypher queries for creating nodes
    CREATE_USER_QUERY = """
    MERGE (u:User {id: $user_id})
    SET u.username = $username,
        u.gender = $gender,
        u.age = $age,
        u.height = $height,
        u.weight = $weight,
        u.body_shape = $body_shape,
        u.skin_tone = $skin_tone
    RETURN u
    """

    CREATE_CLOTHING_QUERY = """
    MERGE (c:Clothing {id: $clothing_id})
    SET c.name = $name,
        c.category = $category,
        c.brand = $brand,
        c.color = $color,
        c.material = $material,
        c.style = $style,
        c.season = $season,
        c.occasion = $occasion,
        c.price = $price
    RETURN c
    """

    CREATE_STYLE_QUERY = """
    MERGE (s:Style {id: $style_id})
    SET s.name = $name,
        s.description = $description
    RETURN s
    """

    CREATE_CATEGORY_QUERY = """
    MERGE (cat:Category {id: $category_id})
    SET cat.name = $name
    RETURN cat
    """

    # Cypher queries for creating relationships
    USER_PREFERS_STYLE = """
    MATCH (u:User {id: $user_id}), (s:Style {id: $style_id})
    MERGE (u)-[r:PREFERS]->(s)
    SET r.score = $score
    RETURN r
    """

    USER_BOUGHT = """
    MATCH (u:User {id: $user_id}), (c:Clothing {id: $clothing_id})
    MERGE (u)-[:BOUGHT {timestamp: $timestamp}]->(c)
    RETURN u
    """

    USER_VIEWED = """
    MATCH (u:User {id: $user_id}), (c:Clothing {id: $clothing_id})
    MERGE (u)-[:VIEWED {timestamp: $timestamp, duration: $duration}]->(c)
    RETURN u
    """

    USER_RATED = """
    MATCH (u:User {id: $user_id}), (c:Clothing {id: $clothing_id})
    MERGE (u)-[r:RATED]->(c)
    SET r.score = $score, r.timestamp = $timestamp
    RETURN r
    """

    CLOTHING_IN_CATEGORY = """
    MATCH (c:Clothing {id: $clothing_id}), (cat:Category {id: $category_id})
    MERGE (c)-[:IN_CATEGORY]->(cat)
    RETURN c
    """

    CLOTHING_HAS_COLOR = """
    MATCH (c:Clothing {id: $clothing_id}), (col:Color {id: $color_id})
    MERGE (c)-[:HAS_COLOR]->(col)
    RETURN c
    """

    CLOTHING_MATCHES_WITH = """
    MATCH (c1:Clothing {id: $clothing_id_1}), (c2:Clothing {id: $clothing_id_2})
    MERGE (c1)-[r:MATCHES_WITH]->(c2)
    SET r.score = $score
    RETURN r
    """

    CLOTHING_SIMILAR_TO = """
    MATCH (c1:Clothing {id: $clothing_id_1}), (c2:Clothing {id: $clothing_id_2})
    MERGE (c1)-[r:SIMILAR_TO]->(c2)
    SET r.score = $score
    RETURN r
    """

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver

    def create_user(self, user_data: dict) -> dict:
        """Create or update a user node."""
        with self.driver.session() as session:
            result = session.run(self.CREATE_USER_QUERY, **user_data)
            return result.single()[0]

    def create_clothing(self, clothing_data: dict) -> dict:
        """Create or update a clothing node."""
        with self.driver.session() as session:
            result = session.run(self.CREATE_CLOTHING_QUERY, **clothing_data)
            return result.single()[0]

    def create_style(self, style_data: dict) -> dict:
        """Create or update a style node."""
        with self.driver.session() as session:
            result = session.run(self.CREATE_STYLE_QUERY, **style_data)
            return result.single()[0]

    def create_category(self, category_data: dict) -> dict:
        """Create or update a category node."""
        with self.driver.session() as session:
            result = session.run(self.CREATE_CATEGORY_QUERY, **category_data)
            return result.single()[0]

    def add_user_preference(self, user_id: int, style_id: str, score: float = 1.0):
        """Add user preference relationship."""
        with self.driver.session() as session:
            session.run(
                self.USER_PREFERS_STYLE,
                user_id=user_id,
                style_id=style_id,
                score=score
            )

    def add_user_bought(self, user_id: int, clothing_id: int, timestamp=None):
        """Add user bought relationship."""
        with self.driver.session() as session:
            session.run(
                self.USER_BOUGHT,
                user_id=user_id,
                clothing_id=clothing_id,
                timestamp=timestamp
            )

    def add_user_viewed(
        self,
        user_id: int,
        clothing_id: int,
        timestamp=None,
        duration: int = 0
    ):
        """Add user viewed relationship."""
        with self.driver.session() as session:
            session.run(
                self.USER_VIEWED,
                user_id=user_id,
                clothing_id=clothing_id,
                timestamp=timestamp,
                duration=duration
            )

    def add_user_rated(
        self,
        user_id: int,
        clothing_id: int,
        score: float,
        timestamp=None
    ):
        """Add user rated relationship."""
        with self.driver.session() as session:
            session.run(
                self.USER_RATED,
                user_id=user_id,
                clothing_id=clothing_id,
                score=score,
                timestamp=timestamp
            )

    def add_clothing_match(
        self,
        clothing_id_1: int,
        clothing_id_2: int,
        score: float = 1.0
    ):
        """Add clothing matching relationship."""
        with self.driver.session() as session:
            session.run(
                self.CLOTHING_MATCHES_WITH,
                clothing_id_1=clothing_id_1,
                clothing_id_2=clothing_id_2,
                score=score
            )

    def add_clothing_similar(
        self,
        clothing_id_1: int,
        clothing_id_2: int,
        score: float = 1.0
    ):
        """Add clothing similarity relationship."""
        with self.driver.session() as session:
            session.run(
                self.CLOTHING_SIMILAR_TO,
                clothing_id_1=clothing_id_1,
                clothing_id_2=clothing_id_2,
                score=score
            )