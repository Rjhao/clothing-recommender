"""User service for managing users."""
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from api.models.user import UserCreate, User

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


class UserService:
    """Service for managing users in MySQL."""

    def __init__(self, session: Session):
        """Initialize with SQLAlchemy session."""
        self.session = session

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        hashed_password = pwd_context.hash(user_data.password)
        db_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            gender=str(user_data.gender.value) if user_data.gender else "other",
            age=user_data.age or 25,
            height=user_data.height or 170.0,
            weight=user_data.weight or 65.0,
            body_shape=str(user_data.body_shape.value) if user_data.body_shape else "average",
            skin_tone=str(user_data.skin_tone.value) if user_data.skin_tone else "medium",
            created_at=datetime.now()
        )
        self.session.add(db_user)
        self.session.commit()
        self.session.refresh(db_user)
        return db_user

    def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return self.session.query(User).filter(
            User.id == user_id
        ).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self.session.query(User).filter(
            User.username == username
        ).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self.session.query(User).filter(
            User.email == email
        ).first()

    def verify_password(
        self,
        plain_password: str,
        hashed_password: str
    ) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)