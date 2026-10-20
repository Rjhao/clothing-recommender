"""User profile service for extracting user preferences from behavior data."""
from typing import Dict, List, Optional
from collections import defaultdict
from neo4j import Driver
import logging

from kg.graph.query import ClothingGraphQuery

logger = logging.getLogger(__name__)


class UserProfileService:
    """Service for building and managing user profiles."""

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver
        self.graph_query = ClothingGraphQuery(driver)

    def build_user_profile(self, user_id: int) -> Dict:
        """
        Build complete user profile from all data sources.

        Returns:
            Dictionary containing static features, preferences, and behavior patterns
        """
        # Get user static info from graph
        user_info = self._get_user_static_info(user_id)

        # Get user's interaction history
        history = self.graph_query.get_user_history(user_id, limit=100)

        # Get user's style profile
        style_profile = self.graph_query.get_user_style_profile(user_id)

        # Extract preferences from interactions
        preferences = self._extract_preferences_from_history(history)

        # Compute style distribution
        style_distribution = self._compute_style_distribution(style_profile)

        # Extract color preferences
        color_preferences = self._extract_color_preferences(history)

        # Extract price range preference
        price_range = self._extract_price_range(history)

        # Extract brand preferences
        brand_preferences = self._extract_brand_preferences(history)

        return {
            "user_id": user_id,
            "static_info": user_info,
            "style_profile": style_profile,
            "preferences": preferences,
            "style_distribution": style_distribution,
            "color_preferences": color_preferences,
            "price_range": price_range,
            "brand_preferences": brand_preferences,
            "interaction_count": len(history)
        }

    def _get_user_static_info(self, user_id: int) -> Dict:
        """Get user's static demographic information."""
        query = """
        MATCH (u:User {id: $user_id})
        RETURN u.id as user_id,
               u.username as username,
               u.gender as gender,
               u.age as age,
               u.height as height,
               u.weight as weight,
               u.body_shape as body_shape,
               u.skin_tone as skin_tone
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            record = result.single()
            if record:
                return dict(record)
            return {}

    def _extract_preferences_from_history(self, history: List[Dict]) -> Dict:
        """Extract user preferences from interaction history."""
        if not history:
            return {}

        preferences = {
            "preferred_categories": defaultdict(int),
            "preferred_styles": defaultdict(int),
            "preferred_seasons": defaultdict(int),
            "preferred_occasions": defaultdict(int),
        }

        for item in history:
            clothing = item.get("c", {})
            if not clothing:
                continue

            # Weight by interaction type
            relation_type = item.get("relation_type", "")
            weight = 1.0
            if relation_type == "BOUGHT":
                weight = 3.0
            elif relation_type == "RATED":
                weight = item.get("score", 3.0) / 5.0 * 2.0

            category = clothing.get("category")
            style = clothing.get("style")
            season = clothing.get("season")
            occasion = clothing.get("occasion")

            if category:
                preferences["preferred_categories"][category] += weight
            if style:
                preferences["preferred_styles"][style] += weight
            if season:
                preferences["preferred_seasons"][season] += weight
            if occasion:
                preferences["preferred_occasions"][occasion] += weight

        # Convert to sorted list
        for key in preferences:
            preferences[key] = dict(sorted(
                preferences[key].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])  # Top 5

        return preferences

    def _compute_style_distribution(self, style_profile: List[Dict]) -> Dict[str, float]:
        """Compute normalized style preference distribution."""
        if not style_profile:
            return {}

        total_score = sum(item.get("preference_score") or 0 for item in style_profile)
        if total_score == 0:
            return {}

        distribution = {}
        for item in style_profile:
            # Neo4j 节点在 dict(record) 后仍是 Node 对象,不是 dict
            # 先尝试 dict(),再 fallback 到 properties
            s = item.get("s")
            if s is None:
                continue
            if hasattr(s, "_properties"):
                props = dict(s._properties)
            elif isinstance(s, dict):
                props = s
            else:
                # neo4j.graph.Node fallback: 直接访问 .items()
                try:
                    props = dict(s)
                except Exception:
                    continue
            # Style 节点只有 name 没有 id,优先 name
            sid = props.get("name") or props.get("id")
            if sid is None:
                continue
            distribution[sid] = (item.get("preference_score") or 0) / total_score
        return distribution

    def _extract_color_preferences(self, history: List[Dict]) -> Dict[str, int]:
        """Extract user's preferred colors from history."""
        color_counts = defaultdict(int)

        for item in history:
            clothing = item.get("c", {})
            color = clothing.get("color")
            if color:
                color_counts[color] += 1

        return dict(sorted(
            color_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

    def _extract_price_range(self, history: List[Dict]) -> Dict[str, float]:
        """Extract user's price range preference."""
        prices = []

        for item in history:
            clothing = item.get("c", {})
            price = clothing.get("price")
            if price:
                prices.append(price)

        if not prices:
            return {"min": 0, "max": 10000, "avg": 500}

        return {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices)
        }

    def _extract_brand_preferences(self, history: List[Dict]) -> Dict[str, int]:
        """Extract user's preferred brands from history."""
        brand_counts = defaultdict(int)

        for item in history:
            clothing = item.get("c", {})
            brand = clothing.get("brand")
            if brand:
                brand_counts[brand] += 1

        return dict(sorted(
            brand_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

    def get_recommended_styles_for_user(self, user_id: int) -> List[str]:
        """Get recommended styles for a user based on their profile."""
        profile = self.build_user_profile(user_id)

        # Get user's body shape
        body_shape = profile.get("static_info", {}).get("body_shape")

        # Map body shape to recommended styles
        body_shape_styles = {
            "slim": ["minimalist", "streetwear", "business"],
            "athletic": ["sporty", "streetwear", "casual"],
            "average": ["casual", "business", "minimalist"],
            "curvy": ["romantic", "bohemian", "casual"],
            "plus_size": ["minimalist", "casual", "bohemian"]
        }

        base_styles = body_shape_styles.get(body_shape, ["casual", "minimalist"])

        # Boost styles user already prefers
        style_dist = profile.get("style_distribution", {})
        boosted_styles = base_styles.copy()

        for style, score in style_dist.items():
            if style in base_styles:
                continue
            if score > 0.2:
                boosted_styles.append(style)

        return boosted_styles[:5]

    def update_user_preferences(self, user_id: int, new_interaction: Dict):
        """
        Update user preferences based on new interaction.
        This would be called after a user views, buys, or rates an item.
        """
        interaction_type = new_interaction.get("type")
        clothing_id = new_interaction.get("clothing_id")
        score = new_interaction.get("score")

        if interaction_type == "RATED" and score and score >= 4.0:
            # Boost style preference for highly rated items
            query = """
            MATCH (u:User {id: $user_id})-[r:RATED]->(c:Clothing {id: $clothing_id})
            WHERE r.score >= 4.0
            MATCH (c)-[:HAS_STYLE]->(s:Style)
            MERGE (u)-[rp:PREFERS]->(s)
            SET rp.score = COALESCE(rp.score, 0) + 0.1
            """
            with self.driver.session() as session:
                session.run(query, user_id=user_id, clothing_id=clothing_id)
                logger.info(f"Updated preferences for user {user_id} after rating")