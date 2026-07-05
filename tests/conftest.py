"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver for testing."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


@pytest.fixture
def sample_clothing_data():
    """Sample clothing data for testing."""
    return [
        {
            "clothing_id": 1,
            "name": "简约纯棉T恤",
            "category": "top",
            "brand": "UNIQLO",
            "color": "白色",
            "material": "棉",
            "style": "minimalist",
            "season": "summer",
            "occasion": "daily",
            "price": 199.0
        },
        {
            "clothing_id": 2,
            "name": "条纹Polo衫",
            "category": "top",
            "brand": "LACOSTE",
            "color": "蓝色",
            "material": "棉",
            "style": "casual",
            "season": "summer",
            "occasion": "casual",
            "price": 499.0
        },
    ]


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "user_id": 1,
        "username": "test_user",
        "gender": "male",
        "age": 28,
        "height": 175.0,
        "weight": 70.0,
        "body_shape": "athletic",
        "skin_tone": "medium"
    }


@pytest.fixture
def sample_interactions():
    """Sample user-clothing interactions for testing."""
    return [
        {"user_id": 1, "clothing_id": 1, "type": "view", "timestamp": "2024-01-01T10:00:00", "duration": 30},
        {"user_id": 1, "clothing_id": 2, "type": "buy", "timestamp": "2024-01-02T15:00:00"},
        {"user_id": 1, "clothing_id": 1, "type": "rate", "score": 4.5, "timestamp": "2024-01-03"},
    ]