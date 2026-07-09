"""Recommendation API routes."""
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from api.database import get_neo4j_driver
from api.services.recommendation import RecommendationService
from api.services.user_profile import UserProfileService
from api.routes.clothing import _normalize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommendations"])


def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance."""
    driver = get_neo4j_driver()
    return RecommendationService(driver)


def get_user_profile_service() -> UserProfileService:
    """Get recommendation service instance."""
    driver = get_neo4j_driver()
    return UserProfileService(driver)


@router.get("/{user_id}")
def get_recommendations(
    user_id: int,
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    exclude_viewed: bool = Query(True, description="Exclude already viewed items"),
    exclude_ids: str = Query("", description="逗号分隔的 id 列表,推荐结果排除这些(避免与 hot/new 重复)"),
):
    """
    Get personalized recommendations for a user.

    Returns recommended clothing items based on user's preferences,
    behavior history, and similar users' preferences.
    """
    service = get_recommendation_service()
    try:
        # 解析 exclude_ids
        exclude_set = set()
        if exclude_ids:
            for x in exclude_ids.split(","):
                try:
                    exclude_set.add(int(x))
                except ValueError:
                    pass
        recommendations = service.recommend(
            user_id=user_id,
            category=category,
            limit=limit * 4,  # 多取,过滤后保留 limit(去重后可能不够)
            exclude_viewed=exclude_viewed
        )
        # 排除 exclude_ids
        if exclude_set:
            recommendations = [r for r in recommendations if r.get("id") not in exclude_set and r.get("clothing_id") not in exclude_set]
        # 按品牌去重(每个品牌最多 1 件,让 6 件品牌更多样)
        seen_brands = set()
        deduped = []
        for r in recommendations:
            brand = r.get("brand") or "_none_"
            if brand in seen_brands:
                continue
            seen_brands.add(brand)
            deduped.append(r)
            if len(deduped) >= limit:
                break
        recommendations = deduped
        # 如果按品牌去重后不够,补全(不严格去重)
        if len(recommendations) < limit:
            seen_ids = {r.get("id") for r in recommendations}
            for r in original_recs if 'original_recs' in dir() else recommendations:
                pass  # 简化:不再补,保持多样性
        recommendations = recommendations[:limit]
        # 过 _normalize: 统一图片(本地 fallback)、价格(0 元兜底)、名字(去 "Apparel XXX")
        normalized = []
        for item in recommendations:
            # 构造 _normalize 期待的 dict(字段名 + score + reason 保留)
            n = _normalize(item)
            n["score"] = item.get("score", 0.0)
            n["reason"] = item.get("reason", "")
            normalized.append(n)
        return {
            "user_id": user_id,
            "count": len(normalized),
            "recommendations": normalized
        }
    except Exception:
        logger.exception("Recommendation failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Recommendation error")


@router.get("/{user_id}/similar/{clothing_id}")
def get_similar_items(
    user_id: int,
    clothing_id: int,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get similar items to a specific clothing item.

    Returns both similar items (same style/category) and matching items (can be worn together).
    """
    service = get_recommendation_service()
    try:
        result = service.get_similar_items(
            user_id=user_id,
            clothing_id=clothing_id,
            limit=limit
        )
        return {
            "user_id": user_id,
            "base_item_id": clothing_id,
            "similar": result.get("similar", []),
            "matching": result.get("matching", [])
        }
    except Exception:
        logger.exception("Similar items failed for clothing %s", clothing_id)
        raise HTTPException(status_code=500, detail="Similar items error")


@router.get("/{user_id}/outfit/{base_item_id}")
def get_outfit_recommendations(
    user_id: int,
    base_item_id: int,
    limit: int = Query(5, ge=1, le=20)
):
    """
    Get complete outfit recommendations based on a base item.

    Returns a set of items that form a complete outfit with the base item.
    """
    service = get_recommendation_service()
    try:
        outfits = service.get_outfit_recommendations(
            user_id=user_id,
            base_item_id=base_item_id,
            limit=limit
        )
        return {
            "user_id": user_id,
            "base_item_id": base_item_id,
            "outfit_items": outfits
        }
    except Exception:
        logger.exception("Outfit recommendation failed for base %s", base_item_id)
        raise HTTPException(status_code=500, detail="Outfit recommendation error")


@router.get("/{user_id}/profile")
def get_user_profile(user_id: int):
    """
    Get user's complete profile including preferences.

    Returns user's static info, style distribution, color preferences,
    price range, and interaction statistics.
    """
    service = get_user_profile_service()
    try:
        profile = service.build_user_profile(user_id)
        return profile
    except Exception:
        logger.exception("Profile failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Profile error")


@router.get("/{user_id}/styles")
def get_recommended_styles(user_id: int):
    """
    Get recommended styles for a user based on their profile.

    Returns a list of recommended styles considering user's body shape and preferences.
    """
    service = get_user_profile_service()
    try:
        styles = service.get_recommended_styles_for_user(user_id)
        return {
            "user_id": user_id,
            "recommended_styles": styles
        }
    except Exception:
        logger.exception("Style recommendation failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Style recommendation error")
