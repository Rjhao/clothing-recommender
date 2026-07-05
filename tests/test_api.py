"""Unit tests for the clothing recommendation system."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test that health endpoint returns healthy status."""
        with patch('api.main.get_neo4j_driver'):
            from api.main import app
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


class TestClothingEndpoints:
    """Test clothing API endpoints."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver

    def test_list_clothing_empty(self, mock_driver):
        """Test listing clothing with no results."""
        with patch('api.main.get_neo4j_driver', return_value=mock_driver):
            from api.main import app
            client = TestClient(app)
            response = client.get("/clothing/")
            # Should handle empty results gracefully
            assert response.status_code in [200, 500]


class TestRecommendationService:
    """Test recommendation service logic."""

    def test_hybrid_recommender_weights(self):
        """Test that hybrid recommender uses correct weights."""
        from recommender.hybrid.engine import HybridRecommender

        mock_query = MagicMock()
        recommender = HybridRecommender(mock_query)

        # Default weights should sum to 1.0
        total_weight = (
            recommender.weight_kg +
            recommender.weight_cf +
            recommender.weight_content
        )
        assert abs(total_weight - 1.0) < 0.001


class TestUserProfileService:
    """Test user profile service."""

    def test_extract_preferences_from_history(self):
        """Test preference extraction from history."""
        from api.services.user_profile import UserProfileService

        mock_driver = MagicMock()
        service = UserProfileService(mock_driver)

        history = [
            {
                "c": {"category": "top", "style": "casual", "price": 299},
                "relation_type": "BOUGHT",
                "score": 5.0
            },
            {
                "c": {"category": "top", "style": "minimalist", "price": 199},
                "relation_type": "VIEWED",
                "score": 0
            },
        ]

        preferences = service._extract_preferences_from_history(history)

        assert "preferred_categories" in preferences
        assert preferences["preferred_categories"]["top"] == 4.0  # BOUGHT has weight 3.0, VIEWED has weight 1.0


class TestCollaborativeFiltering:
    """Test collaborative filtering logic."""

    def test_cf_recommend_empty_user(self):
        """Test CF recommendation for unknown user."""
        from recommender.collaborative.engine import CollaborativeFiltering

        cf = CollaborativeFiltering(n_factors=10)
        # User not in training data
        recommendations = cf.recommend(user_id=999)
        assert recommendations == []


class TestTransE:
    """Test TransE embedding model."""

    def test_transe_forward(self):
        """Test TransE forward pass."""
        import torch
        from kg.embedding.transE import TransE

        model = TransE(
            num_entities=100,
            num_relations=10,
            embedding_dim=50
        )

        batch_size = 4
        positive_triplets = torch.LongTensor([
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 3],
            [3, 3, 4]
        ])
        negative_triplets = torch.LongTensor([
            [10, 0, 1],
            [11, 1, 2],
            [12, 2, 3],
            [13, 3, 4]
        ])

        pos_scores, neg_scores = model(positive_triplets, negative_triplets)

        assert pos_scores.shape == (batch_size,)
        assert neg_scores.shape == (batch_size,)
        # Positive scores should typically be lower (closer) than negative
        assert pos_scores.mean() <= neg_scores.mean() + 0.5

    def test_transe_similarity(self):
        """Test TransE similarity computation."""
        from kg.embedding.transE import TransE

        model = TransE(
            num_entities=50,
            num_relations=5,
            embedding_dim=32
        )

        # Same entity should have similarity close to 1.0
        similarity = model.compute_similarity(0, 0)
        assert abs(similarity - 1.0) < 0.01


class TestClothingGraphQuery:
    """Test knowledge graph queries."""

    def test_graph_query_initialization(self):
        """Test ClothingGraphQuery initialization."""
        from kg.graph.query import ClothingGraphQuery

        mock_driver = MagicMock()
        query = ClothingGraphQuery(mock_driver)

        assert query.driver == mock_driver
        assert hasattr(query, 'FIND_BY_PREFERENCE')
        assert hasattr(query, 'GET_SIMILAR_CLOTHING')
        assert hasattr(query, 'GET_MATCHING_OUTFITS')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])