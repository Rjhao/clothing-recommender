"""Database connection configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MySQL Settings
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "password"
    mysql_database: str = "clothing_recommender"

    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # JWT Settings
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# MySQL Database URL
def get_mysql_url() -> str:
    """Get MySQL connection URL."""
    settings = get_settings()
    return f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"


# Neo4j Driver (lazy initialization)
_neo4j_driver = None


def get_neo4j_driver():
    """Get Neo4j driver instance."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        settings = get_settings()
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _neo4j_driver


def close_neo4j_driver():
    """Close Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None


# SQLAlchemy Engine (shared singleton, lazy initialization)
_mysql_engine = None
_SessionLocal = None


def get_mysql_engine():
    """Get shared MySQL SQLAlchemy engine (lazy singleton)."""
    global _mysql_engine, _SessionLocal
    if _mysql_engine is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        _mysql_engine = create_engine(
            get_mysql_url(),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        _SessionLocal = sessionmaker(bind=_mysql_engine, autoflush=False, expire_on_commit=False)
    return _mysql_engine


def get_db_session():
    """Get a SQLAlchemy session from the shared engine.

    Caller is responsible for closing it (prefer ``get_db`` in FastAPI routes).
    """
    get_mysql_engine()
    return _SessionLocal()


def get_db():
    """FastAPI dependency: yield a session and close it after the request."""
    get_mysql_engine()
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()