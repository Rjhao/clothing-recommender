"""Models for User entities - both Pydantic and SQLAlchemy."""
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Float, Enum as SQLEnum
from sqlalchemy.orm import declarative_base
from api.models.clothing import StyleEnum, SeasonEnum, OccasionEnum

Base = declarative_base()


class GenderEnum(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class BodyShapeEnum(str, Enum):
    """Body shape enumeration."""
    SLIM = "slim"
    ATHLETIC = "athletic"
    AVERAGE = "average"
    CURVY = "curvy"
    PLUS_SIZE = "plus_size"


class SkinToneEnum(str, Enum):
    """Skin tone enumeration."""
    FAIR = "fair"
    LIGHT = "light"
    MEDIUM = "medium"
    TAN = "tan"
    DARK = "dark"


# SQLAlchemy Model (for MySQL)
class User(Base):
    """SQLAlchemy User model for MySQL."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    gender = Column(String(20), default="other")
    age = Column(Integer, default=25)
    height = Column(Float, default=170.0)
    weight = Column(Float, default=65.0)
    body_shape = Column(String(20), default="average")
    skin_tone = Column(String(20), default="medium")
    created_at = Column(DateTime, default=datetime.now)


# Pydantic Models (for API)
class UserBase(BaseModel):
    """Base user model."""
    username: str
    email: str
    gender: Optional[GenderEnum] = None
    age: Optional[int] = None
    height: Optional[float] = None  # cm
    weight: Optional[float] = None  # kg
    body_shape: Optional[BodyShapeEnum] = None
    skin_tone: Optional[SkinToneEnum] = None


class UserCreate(UserBase):
    """Model for creating user."""
    password: str


class UserPydantic(UserBase):
    """Full user model with ID (Pydantic)."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class UserPreference(BaseModel):
    """User preference model for recommendations."""
    preferred_styles: List[StyleEnum] = []
    preferred_colors: List[str] = []
    preferred_price_range: tuple[float, float] = (0, 10000)
    preferred_seasons: List[SeasonEnum] = []
    preferred_occasions: List[OccasionEnum] = []