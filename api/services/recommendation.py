"""Recommendation service for generating personalized recommendations."""
from typing import List, Optional
from datetime import datetime
from neo4j import Driver

from kg.graph.builder import ClothingGraphBuilder
from kg.graph.query import ClothingGraphQuery
from recommender.hybrid.engine import HybridRecommender


class RecommendationService:
    """Service for generating personalized recommendations."""

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver
        self.builder = ClothingGraphBuilder(driver)
        self.query = ClothingGraphQuery(driver)
        self.hybrid_recommender = HybridRecommender(self.query)

    def recommend(
        self,
        user_id: int,
        category: Optional[str] = None,
        limit: int = 10,
        exclude_viewed: bool = True
    ) -> List[dict]:
        """Get personalized recommendations for a user."""
        return self.hybrid_recommender.recommend(
            user_id=user_id,
            category=category,
            limit=limit,
            exclude_viewed=exclude_viewed
        )

    def get_similar_items(
        self,
        user_id: int,
        clothing_id: int,
        limit: int = 10
    ) -> dict:
        """Get similar and matching items."""
        return self.hybrid_recommender.get_similar_items(
            clothing_id=clothing_id,
            limit=limit
        )

    def get_outfit_recommendations(
        self,
        user_id: int,
        base_item_id: int,
        limit: int = 5
    ) -> List[dict]:
        """Get complete outfit recommendations."""
        return self.hybrid_recommender.get_outfit_recommendations(
            user_id=user_id,
            base_item_id=base_item_id,
            limit=limit
        )

    def record_view(
        self,
        user_id: int,
        clothing_id: int,
        duration: int = 0
    ):
        """Record a user viewing a clothing item."""
        timestamp = datetime.now().isoformat()
        self.builder.add_user_viewed(
            user_id=user_id,
            clothing_id=clothing_id,
            timestamp=timestamp,
            duration=duration
        )

    def record_rating(
        self,
        user_id: int,
        clothing_id: int,
        score: float
    ):
        """Record a user rating a clothing item."""
        timestamp = datetime.now().isoformat()
        self.builder.add_user_rated(
            user_id=user_id,
            clothing_id=clothing_id,
            score=score,
            timestamp=timestamp
        )

    def record_purchase(self, user_id: int, clothing_id: int):
        """Record a user purchasing a clothing item."""
        timestamp = datetime.now().isoformat()
        self.builder.add_user_bought(
            user_id=user_id,
            clothing_id=clothing_id,
            timestamp=timestamp
        )