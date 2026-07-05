"""Authentication routes for login and registration."""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from jose import JWTError, jwt

from api.database import get_settings, get_db_session
from api.services.user import UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

settings = get_settings()
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: int
    username: str


class UserRegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    gender: str = "other"
    age: int = 25
    height: float = 170.0
    weight: float = 65.0
    body_shape: str = "average"
    skin_tone: str = "medium"


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        username: str = payload.get("username")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, username=username)
    except JWTError:
        raise credentials_exception

    session = get_db_session()
    try:
        service = UserService(session)
        user = service.get_user(token_data.user_id)
    finally:
        session.close()
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=Token)
async def register(user_data: UserRegisterRequest):
    """Register a new user."""
    session = get_db_session()
    service = UserService(session)
    try:
        # Check if username exists
        existing_user = service.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")

        # Check if email exists
        existing_email = service.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        from api.models.user import UserCreate
        user_create = UserCreate(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            gender=user_data.gender,
            age=user_data.age,
            height=user_data.height,
            weight=user_data.weight,
            body_shape=user_data.body_shape,
            skin_tone=user_data.skin_tone
        )

        new_user = service.create_user(user_create)

        # Generate token
        access_token = create_access_token(
            data={"user_id": new_user.id, "username": new_user.username}
        )
        return Token(access_token=access_token, token_type="bearer")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Registration failed")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with username and password."""
    session = get_db_session()
    service = UserService(session)
    try:
        user = service.get_user_by_username(form_data.username)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not service.verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    finally:
        session.close()

    access_token = create_access_token(
        data={"user_id": user.id, "username": user.username}
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me")
async def get_me(current_user = Depends(get_current_user)):
    """Get current user info."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "gender": current_user.gender,
        "age": current_user.age,
        "height": current_user.height,
        "weight": current_user.weight,
        "body_shape": current_user.body_shape,
        "skin_tone": current_user.skin_tone
    }
