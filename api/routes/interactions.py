"""User interaction API routes.

All write endpoints derive ``user_id`` from the authenticated token, never from
the request body, so a client cannot record interactions on another user's
behalf (IDOR).
"""
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from pydantic import BaseModel, Field

from api.database import get_neo4j_driver
from api.services.recommendation import RecommendationService
from api.routes.auth import get_current_user
from api.models.user import User
from kg.graph.query import ClothingGraphQuery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/interactions", tags=["interactions"])


class ViewRequest(BaseModel):
    """Request model for recording a view."""
    clothing_id: int
    duration: Optional[int] = Field(default=0, description="View duration in seconds")


class RatingRequest(BaseModel):
    """Request model for recording a rating."""
    clothing_id: int
    score: float = Field(..., ge=0, le=5, description="Rating score from 0 to 5")


class PurchaseRequest(BaseModel):
    """Request model for recording a purchase."""
    clothing_id: int


def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance."""
    driver = get_neo4j_driver()
    return RecommendationService(driver)


@router.post("/view")
def record_view(request: ViewRequest, current_user: User = Depends(get_current_user)):
    """
    Record a user viewing a clothing item.

    This interaction is used to track user interests and improve recommendations.
    """
    service = get_recommendation_service()
    try:
        service.record_view(
            user_id=current_user.id,
            clothing_id=request.clothing_id,
            duration=request.duration
        )
        return {"status": "recorded", "type": "view"}
    except Exception:
        logger.exception("Failed to record view")
        raise HTTPException(status_code=500, detail="Failed to record view")


@router.post("/rate")
def record_rating(request: RatingRequest, current_user: User = Depends(get_current_user)):
    """
    Record a user rating a clothing item.

    Ratings should be between 0 and 5. Higher ratings help improve recommendations.
    """
    service = get_recommendation_service()
    try:
        service.record_rating(
            user_id=current_user.id,
            clothing_id=request.clothing_id,
            score=request.score
        )
        return {"status": "recorded", "type": "rating", "score": request.score}
    except Exception:
        logger.exception("Failed to record rating")
        raise HTTPException(status_code=500, detail="Failed to record rating")


@router.post("/buy")
def record_purchase(request: PurchaseRequest, current_user: User = Depends(get_current_user)):
    """
    Record a user purchasing a clothing item.

    Purchase is the strongest positive signal for recommendations.
    """
    service = get_recommendation_service()
    try:
        service.record_purchase(
            user_id=current_user.id,
            clothing_id=request.clothing_id
        )
        return {"status": "recorded", "type": "purchase"}
    except Exception:
        logger.exception("Failed to record purchase")
        raise HTTPException(status_code=500, detail="Failed to record purchase")


@router.get("/history/{user_id}")
def get_interaction_history(
    user_id: int,
    current_user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get user's interaction history.

    A user may only view their own history.
    """
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view another user's history")

    driver = get_neo4j_driver()
    query = ClothingGraphQuery(driver)

    try:
        history = query.get_user_history(user_id, limit=limit)
        return {
            "user_id": user_id,
            "count": len(history),
            "history": history
        }
    except Exception:
        logger.exception("Failed to get history")
        raise HTTPException(status_code=500, detail="Failed to get history")
