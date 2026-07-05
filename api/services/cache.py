"""Redis caching service for clothing and recommendations."""
import json
import redis
from typing import Optional, Any, List
from functools import wraps

from api.database import get_settings

settings = get_settings()

_redis_client = None


def get_redis_client() -> redis.Redis:
    """Get Redis client instance (lazy singleton)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
    return _redis_client


class CacheService:
    """Redis caching service."""

    # TTL constants (in seconds)
    TTL_CLOTHING_LIST = 300  # 5 minutes
    TTL_RECOMMENDATIONS = 600  # 10 minutes
    TTL_KG_STATS = 1800  # 30 minutes
    TTL_USER_PROFILE = 600  # 10 minutes

    def __init__(self):
        self.client = get_redis_client()

    def _serialize(self, data: Any) -> str:
        """Serialize data to JSON string."""
        if isinstance(data, (list, dict)):
            return json.dumps(data, ensure_ascii=False, default=str)
        return str(data)

    def _deserialize(self, data: str) -> Any:
        """Deserialize JSON string to Python object."""
        if data is None:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    # === Clothing List Cache ===

    def get_clothing_list(self, category: str = None, style: str = None, limit: int = 20, offset: int = 0) -> Optional[List]:
        """Get cached clothing list."""
        key = f"clothing:list:{category or 'all'}:{style or 'all'}:{limit}:{offset}"
        data = self.client.get(key)
        return self._deserialize(data)

    def set_clothing_list(self, data: List, category: str = None, style: str = None, limit: int = 20, offset: int = 0):
        """Cache clothing list."""
        key = f"clothing:list:{category or 'all'}:{style or 'all'}:{limit}:{offset}"
        self.client.setex(key, self.TTL_CLOTHING_LIST, self._serialize(data))

    def invalidate_clothing_list(self):
        """Invalidate all clothing list cache."""
        keys = list(self.client.scan_iter("clothing:list:*"))
        if keys:
            self.client.delete(*keys)

    # === Recommendation Cache ===

    def get_recommendations(self, user_id: int, category: str = None, limit: int = 10) -> Optional[List]:
        """Get cached recommendations for user."""
        key = f"recommend:{user_id}:{category or 'all'}:{limit}"
        data = self.client.get(key)
        return self._deserialize(data)

    def set_recommendations(self, user_id: int, data: List, category: str = None, limit: int = 10):
        """Cache recommendations for user."""
        key = f"recommend:{user_id}:{category or 'all'}:{limit}"
        self.client.setex(key, self.TTL_RECOMMENDATIONS, self._serialize(data))

    def invalidate_user_recommendations(self, user_id: int):
        """Invalidate all recommendations cache for a user."""
        keys = list(self.client.scan_iter(f"recommend:{user_id}:*"))
        if keys:
            self.client.delete(*keys)

    # === Knowledge Graph Stats Cache ===

    def get_kg_stats(self) -> Optional[dict]:
        """Get cached KG statistics."""
        key = "kg:stats"
        data = self.client.get(key)
        return self._deserialize(data)

    def set_kg_stats(self, data: dict):
        """Cache KG statistics."""
        key = "kg:stats"
        self.client.setex(key, self.TTL_KG_STATS, self._serialize(data))

    def invalidate_kg_stats(self):
        """Invalidate KG statistics cache."""
        self.client.delete("kg:stats")

    # === User Profile Cache ===

    def get_user_profile(self, user_id: int) -> Optional[dict]:
        """Get cached user profile."""
        key = f"user:profile:{user_id}"
        data = self.client.get(key)
        return self._deserialize(data)

    def set_user_profile(self, user_id: int, data: dict):
        """Cache user profile."""
        key = f"user:profile:{user_id}"
        self.client.setex(key, self.TTL_USER_PROFILE, self._serialize(data))

    def invalidate_user_profile(self, user_id: int):
        """Invalidate user profile cache."""
        self.client.delete(f"user:profile:{user_id}")

    # === Interaction Invalidation ===

    def invalidate_on_interaction(self, user_id: int):
        """Invalidate caches when user interacts (view/rate/buy)."""
        self.invalidate_user_recommendations(user_id)
        self.invalidate_user_profile(user_id)


# Singleton instance
_cache_service = None


def get_cache_service() -> CacheService:
    """Get cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
