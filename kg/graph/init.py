"""Initialize Neo4j database schema and seed data."""
from neo4j import Driver
import logging

logger = logging.getLogger(__name__)


class GraphInitializer:
    """Initialize knowledge graph with schema and seed data."""

    # Create constraints and indexes
    CREATE_CONSTRAINTS = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT clothing_id IF NOT EXISTS FOR (c:Clothing) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT style_id IF NOT EXISTS FOR (s:Style) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (cat:Category) REQUIRE cat.id IS UNIQUE",
    ]

    # Create indexes
    CREATE_INDEXES = [
        "CREATE INDEX user_username IF NOT EXISTS FOR (u:User) ON (u.username)",
        "CREATE INDEX clothing_category IF NOT EXISTS FOR (c:Clothing) ON (c.category)",
        "CREATE INDEX clothing_style IF NOT EXISTS FOR (c:Clothing) ON (c.style)",
        "CREATE INDEX clothing_price IF NOT EXISTS FOR (c:Clothing) ON (c.price)",
    ]

    # Seed style data
    SEED_STYLES = [
        {"style_id": "minimalist", "name": "Minimalist", "description": "简约风格，注重剪裁和色彩"},
        {"style_id": "casual", "name": "Casual", "description": "休闲风格，舒适自在"},
        {"style_id": "bohemian", "name": "Bohemian", "description": "波西米亚风格，浪漫自由"},
        {"style_id": "vintage", "name": "Vintage", "description": "复古风格，怀旧经典"},
        {"style_id": "streetwear", "name": "Streetwear", "description": "街头风格，时尚潮流"},
        {"style_id": "business", "name": "Business", "description": "商务风格，职业正式"},
        {"style_id": "sporty", "name": "Sporty", "description": "运动风格，动感活力"},
        {"style_id": "romantic", "name": "Romantic", "description": "浪漫风格，柔美优雅"},
    ]

    # Seed category data
    SEED_CATEGORIES = [
        {"category_id": "top", "name": "上衣"},
        {"category_id": "bottom", "name": "下装"},
        {"category_id": "dress", "name": "连衣裙"},
        {"category_id": "outerwear", "name": "外套"},
        {"category_id": "shoes", "name": "鞋子"},
        {"category_id": "accessories", "name": "配饰"},
    ]

    # 说明：体型（BodyShape）与场景（Occasion）两类实体及其关系
    # （HAS_BODY_SHAPE / SUITABLE_FOR）在本项目所用公开数据源中无任何
    # 可绑定实例——原始事件日志不含相关字段（见论文 §3.1.2 可得性说明），
    # 故不纳入图谱初始化，避免产生无实例支撑的孤立节点。

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver

    def initialize_schema(self):
        """Create constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints
            for constraint in self.CREATE_CONSTRAINTS:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")

            # Create indexes
            for index in self.CREATE_INDEXES:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    def seed_styles(self):
        """Seed style data."""
        query = """
        MERGE (s:Style {id: $style_id})
        SET s.name = $name,
            s.description = $description
        """
        with self.driver.session() as session:
            for style in self.SEED_STYLES:
                session.run(query, **style)
        logger.info(f"Seeded {len(self.SEED_STYLES)} styles")

    def seed_categories(self):
        """Seed category data."""
        query = """
        MERGE (cat:Category {id: $category_id})
        SET cat.name = $name
        """
        with self.driver.session() as session:
            for category in self.SEED_CATEGORIES:
                session.run(query, **category)
        logger.info(f"Seeded {len(self.SEED_CATEGORIES)} categories")

    def seed_all(self):
        """Seed all initial data."""
        self.initialize_schema()
        self.seed_styles()
        self.seed_categories()
        logger.info("Knowledge graph initialization complete")


# CLI script for initialization
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from api.database import get_neo4j_driver

    logging.basicConfig(level=logging.INFO)
    driver = get_neo4j_driver()
    initializer = GraphInitializer(driver)
    initializer.seed_all()
    driver.close()