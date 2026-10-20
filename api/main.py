"""FastAPI application entry point."""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional

from api.database import get_settings, get_neo4j_driver, close_neo4j_driver, get_db
from api.models.clothing import Clothing, ClothingCreate
from api.models.user import UserPydantic as User, UserCreate, UserUpdate
from api.services.clothing import ClothingService
from api.services.recommendation import RecommendationService
from api.routes.recommend import router as recommend_router
from api.routes.interactions import router as interactions_router
from api.routes.auth import router as auth_router
from api.routes.auth import get_current_user
from api.routes.kg_router import router as kg_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    yield
    # Cleanup on shutdown
    close_neo4j_driver()


app = FastAPI(
    title="Clothing Knowledge Graph Recommender",
    description="A knowledge graph based personalized clothing recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (credentials=False: auth uses Bearer header, not cookies,
# so wildcard origins remain valid per the CORS spec)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommend_router)
app.include_router(interactions_router)
app.include_router(auth_router)
app.include_router(kg_router)
from api.routes.clothing import router as clothing_router
app.include_router(clothing_router)


# Dependency injection
def get_clothing_service() -> ClothingService:
    """Get clothing service instance."""
    driver = get_neo4j_driver()
    return ClothingService(driver)


def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance."""
    driver = get_neo4j_driver()
    return RecommendationService(driver)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# User endpoints
def _to_user_schema(db_user) -> User:
    """Convert the SQLAlchemy User row into the public Pydantic schema."""
    return User(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        gender=db_user.gender,
        age=db_user.age,
        height=db_user.height,
        weight=db_user.weight,
        body_shape=db_user.body_shape,
        skin_tone=db_user.skin_tone,
        created_at=db_user.created_at
    )


@app.post("/users/", response_model=User)
def create_user(user: UserCreate, session=Depends(get_db)):
    """Create a new user."""
    from api.services.user import UserService
    service = UserService(session)
    db_user = service.create_user(user)
    return _to_user_schema(db_user)


@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int, session=Depends(get_db)):
    """Get user by ID."""
    from api.services.user import UserService
    service = UserService(session)
    db_user = service.get_user(user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_schema(db_user)


@app.put("/users/{user_id}", response_model=User)
def update_user(
    user_id: int,
    updates: UserUpdate,
    current_user=Depends(get_current_user),
    session=Depends(get_db),
):
    """Update the current user's profile. A user may only edit their own."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot edit another user's profile")

    from api.services.user import UserService
    service = UserService(session)
    payload = {
        k: (v.value if hasattr(v, "value") else v)
        for k, v in updates.model_dump(exclude_unset=True).items()
    }
    db_user = service.update_user(user_id, payload)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_schema(db_user)


# Clothing endpoints
@app.post("/clothing/", response_model=Clothing)
def create_clothing(clothing: ClothingCreate):
    """Create a new clothing item."""
    service = get_clothing_service()
    return service.create_clothing(clothing)


@app.get("/clothing/{clothing_id}", response_model=Clothing)
def get_clothing(clothing_id: int):
    """Get clothing by ID."""
    service = get_clothing_service()
    clothing = service.get_clothing(clothing_id)
    if not clothing:
        raise HTTPException(status_code=404, detail="Clothing not found")
    return clothing


@app.get("/clothing/", response_model=List[Clothing])
def list_clothing(
    category: Optional[str] = None,
    style: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List clothing items with optional filters."""
    service = get_clothing_service()
    return service.list_clothing(
        category=category,
        style=style,
        limit=limit,
        offset=offset
    )


@app.get("/clothing/{clothing_id}/similar", response_model=List[Clothing])
def get_similar_clothing(clothing_id: int, limit: int = 10):
    """Get similar clothing items (same style/category)."""
    service = get_clothing_service()
    return service.get_similar(clothing_id, limit=limit)


@app.get("/clothing/{clothing_id}/matching", response_model=List[Clothing])
def get_matching_clothing(clothing_id: int, limit: int = 10):
    """Get matching outfit items (can be worn together)."""
    service = get_clothing_service()
    return service.get_matching(clothing_id, limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)