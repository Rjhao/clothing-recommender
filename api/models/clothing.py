"""Pydantic models for Clothing entities."""
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class CategoryEnum(str, Enum):
    """Clothing category enumeration."""
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    SHOES = "shoes"
    ACCESSORIES = "accessories"


class SeasonEnum(str, Enum):
    """Season enumeration."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class OccasionEnum(str, Enum):
    """Occasion enumeration."""
    CASUAL = "casual"
    WORK = "work"
    FORMAL = "formal"
    SPORTS = "sports"
    PARTY = "party"
    DAILY = "daily"


class StyleEnum(str, Enum):
    """Style enumeration."""
    MINIMALIST = "minimalist"
    CASUAL = "casual"
    BOHEMIAN = "bohemian"
    VINTAGE = "vintage"
    STREETWEAR = "streetwear"
    BUSINESS = "business"
    SPORTY = "sporty"
    ROMANTIC = "romantic"


class ClothingBase(BaseModel):
    """Base clothing model."""
    name: str
    category: CategoryEnum
    brand: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    style: Optional[StyleEnum] = None
    season: Optional[SeasonEnum] = None
    occasion: Optional[OccasionEnum] = None
    price: Optional[float] = None


class ClothingCreate(ClothingBase):
    """Model for creating clothing."""
    pass


class Clothing(ClothingBase):
    """Full clothing model with ID."""
    id: int
    image_url: Optional[str] = None

    class Config:
        from_attributes = True