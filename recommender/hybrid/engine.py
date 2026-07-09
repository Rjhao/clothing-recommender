"""Hybrid recommendation engine combining knowledge graph and collaborative filtering."""
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

from kg.graph.query import ClothingGraphQuery
from kg.image_url import resolve_image


class HybridRecommender:
    """Hybrid recommender combining multiple recommendation strategies."""

    def __init__(
        self,
        graph_query: ClothingGraphQuery,
        weight_kg: float = 0.5,
        weight_cf: float = 0.3,
        weight_content: float = 0.2
    ):
        """
        Initialize hybrid recommender.

        Args:
            graph_query: Knowledge graph query instance
            weight_kg: Weight for knowledge graph based recommendations
            weight_cf: Weight for collaborative filtering recommendations
            weight_content: Weight for content-based recommendations
        """
        self.graph_query = graph_query
        self.weight_kg = weight_kg
        self.weight_cf = weight_cf
        self.weight_content = weight_content

    def recommend(
        self,
        user_id: int,
        category: Optional[str] = None,
        limit: int = 10,
        exclude_viewed: bool = True
    ) -> List[Dict]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id: User ID
            category: Optional category filter
            limit: Maximum number of recommendations
            exclude_viewed: Whether to exclude already viewed items

        Returns:
            List of recommended clothing items with scores
        """
        # Get recommendations from different sources
        kg_recs = self._get_kg_recommendations(user_id, category, limit * 2)
        cf_recs = self._get_collaborative_recommendations(user_id, limit * 2)
        content_recs = self._get_content_recommendations(user_id, category, limit * 2)

        # Get user history for exclusion
        user_history = []
        if exclude_viewed:
            user_history = self.graph_query.get_user_history(user_id, limit=100)
            user_history = {
                item.get("c", {}).get("id")
                for item in user_history
                if item.get("c", {}).get("id") is not None
            }

        # Normalize and combine scores
        combined_scores = self._combine_recommendations(
            kg_recs,
            cf_recs,
            content_recs,
            user_history
        )

        # Sort by score and return top N
        sorted_recs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Build scores dict for _fetch_clothing_details
        rec_ids = [clothing_id for clothing_id, _ in sorted_recs]
        scores_dict = {cid: score for cid, score in sorted_recs}

        # Cold-start / fill: if not enough personalized recs, top up with popular items.
        # Handles brand-new users (no graph profile) and users with sparse data alike.
        if len(rec_ids) < limit:
            exclude = user_history | set(rec_ids)
            popular = self.graph_query.find_popular_clothing(
                limit=limit * 2, exclude_ids=exclude, category=category
            )
            for item in popular:
                raw = item.get('id')
                if raw is None:
                    raw = item.get('product_id')
                try:
                    cid = int(raw)
                except (TypeError, ValueError):
                    continue
                if cid in scores_dict:
                    continue
                rec_ids.append(cid)
                scores_dict[cid] = 1.0
                if len(rec_ids) >= limit:
                    break

        # Fetch full clothing details
        return self._fetch_clothing_details(rec_ids[:limit], scores=scores_dict)

    def _get_kg_recommendations(
        self,
        user_id: int,
        category: Optional[str],
        limit: int
    ) -> Dict[int, float]:
        """Get knowledge graph based recommendations."""
        recs = self.graph_query.compute_recommendation_scores(user_id, limit=limit)
        scores = {}
        for rec in recs:
            clothing = rec["c"]
            if clothing and clothing.get("id"):
                score = (
                    rec.get("style_match_score", 0) * 0.4 +
                    rec.get("body_match_score", 0) * 0.3 +
                    rec.get("outfit_match_count", 0) * 0.3
                )
                scores[clothing["id"]] = score
        return scores

    def _get_collaborative_recommendations(
        self,
        user_id: int,
        limit: int
    ) -> Dict[int, float]:
        """Get collaborative filtering recommendations."""
        similar_users_recs = self.graph_query.find_similar_users(user_id, limit=limit)
        scores = {}
        for rec in similar_users_recs:
            clothing = rec.get("c")
            if clothing and clothing.get("id"):
                # common_styles is a list; normalize by its length to avoid
                # int / list TypeError and keep the score in a sensible range.
                common = rec.get("common_styles") or []
                score = rec.get("style_overlap", 0) / max(len(common), 1)
                scores[clothing["id"]] = scores.get(clothing["id"], 0) + score
        return scores

    def _get_content_recommendations(
        self,
        user_id: int,
        category: Optional[str],
        limit: int
    ) -> Dict[int, float]:
        """Get content-based recommendations."""
        # Use user's style profile to find similar clothing
        style_profile = self.graph_query.get_user_style_profile(user_id)
        if not style_profile:
            return {}

        # Find clothing matching user's preferred styles
        recs = self.graph_query.find_by_preference(user_id, category, limit=limit)
        scores = {}
        for rec in recs:
            clothing = rec.get("c")
            if clothing and clothing.get("id"):
                # Score based on style match
                score = rec.get("preference_score", 1.0)
                scores[clothing["id"]] = score
        return scores

    def _combine_recommendations(
        self,
        kg_recs: Dict[int, float],
        cf_recs: Dict[int, float],
        content_recs: Dict[int, float],
        exclude_ids: set
    ) -> Dict[int, float]:
        """Combine recommendations from multiple sources."""
        # Normalize scores
        kg_norm = self._normalize_scores(kg_recs)
        cf_norm = self._normalize_scores(cf_recs)
        content_norm = self._normalize_scores(content_recs)

        # Get all clothing IDs
        all_ids = set(kg_norm.keys()) | set(cf_norm.keys()) | set(content_norm.keys())

        # Weighted combination
        combined = {}
        for clothing_id in all_ids:
            if clothing_id in exclude_ids:
                continue
            score = (
                kg_norm.get(clothing_id, 0) * self.weight_kg +
                cf_norm.get(clothing_id, 0) * self.weight_cf +
                content_norm.get(clothing_id, 0) * self.weight_content
            )
            combined[clothing_id] = score

        return combined

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        min_val = min(scores.values())
        max_val = max(scores.values())
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def _fetch_clothing_details(self, clothing_ids: List[int], scores: Dict[int, float] = None) -> List[Dict]:
        """Fetch full details for clothing items from Neo4j."""
        if not clothing_ids:
            return []

        # Fetch actual clothing details from graph
        clothing_details = self.graph_query.get_clothing_by_ids(clothing_ids)

        # Build result with scores and reasons
        result = []
        for item in clothing_details:
            cid = item.get('id') or item.get('product_id')
            if cid is None:
                continue

            # Get score if provided
            score = scores.get(cid, 0.0) if scores else 0.0

            # Generate reason based on attributes
            style = item.get('style', '')
            category = item.get('category', '')
            brand = item.get('brand', '')

            reasons = []
            if style:
                reasons.append(f"风格: {style}")
            if category:
                reasons.append(f"类别: {category}")
            if brand:
                reasons.append(f"品牌: {brand}")

            result.append({
                "clothing_id": cid,
                "id": cid,
                "name": item.get('name', ''),
                "category": category,
                "brand": brand,
                "color": item.get('color'),
                "price": item.get('price', 0),
                "style": style,
                "season": item.get('season', ''),
                "image_url": resolve_image(item),
                "score": score,
                "reason": " | ".join(reasons) if reasons else "个性化推荐"
            })

        return result

    def get_similar_items(
        self,
        clothing_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """Get items similar to a specific clothing item."""
        similar = self.graph_query.get_similar_clothing(
            clothing_id, threshold=0.3, limit=limit
        )
        matching = self.graph_query.get_matching_outfits(
            clothing_id, threshold=0.5, limit=limit
        )
        return {"similar": similar, "matching": matching}

    def get_outfit_recommendations(
        self,
        user_id: int,
        base_item_id: int,
        limit: int = 5
    ) -> List[Dict]:
        """Get complete outfit recommendations based on a base item."""
        # Get matching items
        matching = self.graph_query.get_matching_outfits(
            base_item_id, threshold=0.6, limit=limit * 2
        )

        # Filter by user's style preferences
        user_styles = self.graph_query.get_user_style_profile(user_id)
        preferred_styles = {s["s"]["name"] for s in user_styles}

        filtered = []
        for item in matching:
            item_data = item.get("match", {})
            if item_data.get("style") in preferred_styles:
                filtered.append(item)

        return filtered[:limit]