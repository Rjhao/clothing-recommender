"""Collaborative Filtering for clothing recommendations."""
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class CollaborativeFiltering:
    """User-based and Item-based collaborative filtering."""

    def __init__(
        self,
        n_factors: int = 50,
        n_users: int = 100,
        n_items: int = 100
    ):
        """
        Initialize collaborative filtering.

        Args:
            n_factors: Number of latent factors
            n_users: Number of users
            n_items: Number of items
        """
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items

        # User-Item matrix
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None

        # ID mappings
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}

    def fit(
        self,
        interactions: List[Tuple[int, int, float]],
        user_ids: List[int],
        item_ids: List[int]
    ):
        """
        Fit the collaborative filtering model.

        Args:
            interactions: List of (user_id, item_id, rating) tuples
            user_ids: List of unique user IDs
            item_ids: List of unique item IDs
        """
        # Build mappings
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        # Build user-item matrix
        rows, cols, data = [], [], []
        for user_id, item_id, rating in interactions:
            if user_id in self.user_id_to_idx and item_id in self.item_id_to_idx:
                rows.append(self.user_id_to_idx[user_id])
                cols.append(self.item_id_to_idx[item_id])
                data.append(rating)

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items)
        )

        # Initialize latent factors using SVD
        self._compute_matrix_factorization()

        logger.info(f"CF model fitted with {self.n_users} users and {self.n_items} items")

    def _compute_matrix_factorization(self):
        """Compute user and item latent factors using SVD."""
        from scipy.sparse.linalg import svds

        # Normalize ratings
        matrix = self.user_item_matrix.toarray()
        user_means = np.mean(matrix, axis=1, where=matrix > 0, keepdims=True)
        matrix_normalized = matrix - user_means * (matrix > 0)

        # SVD decomposition
        n_factors = min(self.n_factors, min(self.n_users, self.n_items) - 1)
        U, sigma, Vt = svds(
            csr_matrix(matrix_normalized),
            k=n_factors
        )

        # Convert to latent factors
        sigma = np.diag(sigma)
        self.user_factors = U @ sigma
        self.item_factors = (Vt.T @ sigma).T

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_id_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return []

        user_idx = self.user_id_to_idx[user_id]
        user_vector = self.user_factors[user_idx]

        # Compute similarities
        scores = cosine_similarity(
            [user_vector],
            self.item_factors.T
        )[0]

        # Get item indices sorted by score
        item_indices = np.argsort(scores)[::-1]

        # Get already rated items
        rated_items = set()
        if exclude_rated:
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            rated_items = set(np.where(user_ratings > 0)[0])

        # Build recommendations
        recommendations = []
        for item_idx in item_indices:
            if item_idx in rated_items:
                continue
            item_id = self.idx_to_item_id[item_idx]
            recommendations.append((item_id, float(scores[item_idx])))
            if len(recommendations) >= n_recommendations:
                break

        return recommendations

    def get_similar_users(
        self,
        user_id: int,
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find similar users to a given user.

        Args:
            user_id: Target user ID
            n_similar: Number of similar users to return

        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_id_to_idx:
            return []

        user_idx = self.user_id_to_idx[user_id]
        user_vector = self.user_factors[user_idx]

        # Compute similarities with all users
        similarities = cosine_similarity([user_vector], self.user_factors)[0]

        # Exclude self
        similarities[user_idx] = -1

        # Get top similar users
        similar_indices = np.argsort(similarities)[::-1][:n_similar]

        return [
            (self.idx_to_user_id[idx], float(similarities[idx]))
            for idx in similar_indices if similarities[idx] > 0
        ]

    def get_similar_items(
        self,
        item_id: int,
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find similar items to a given item.

        Args:
            item_id: Target item ID
            n_similar: Number of similar items to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_id_to_idx:
            return []

        item_idx = self.item_id_to_idx[item_id]
        item_vector = self.item_factors[:, item_idx]

        # Compute similarities with all items
        similarities = cosine_similarity([item_vector], self.item_factors.T)[0]

        # Exclude self
        similarities[item_idx] = -1

        # Get top similar items
        similar_indices = np.argsort(similarities)[::-1][:n_similar]

        return [
            (self.idx_to_item_id[idx], float(similarities[idx]))
            for idx in similar_indices if similarities[idx] > 0
        ]


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering (NCF) implementation."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_layers: List[int] = [64, 32, 16]
    ):
        """
        Initialize NCF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            hidden_layers: Hidden layer sizes
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # Build model
        self.model = self._build_model(hidden_layers)

    def _build_model(self, hidden_layers: List[int]) -> "torch.nn.Module":
        """Build PyTorch NCF model."""
        import torch.nn as nn

        class NCFModel(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim, hidden_layers):
                super().__init__()
                # GMF part
                self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
                self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

                # MLP part
                self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
                self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

                mlp_layers = []
                input_dim = embedding_dim * 2
                for hidden_size in hidden_layers:
                    mlp_layers.append(nn.Linear(input_dim, hidden_size))
                    mlp_layers.append(nn.ReLU())
                    mlp_layers.append(nn.Dropout(0.2))
                    input_dim = hidden_size

                self.mlp = nn.Sequential(*mlp_layers)

                # Final prediction layer
                self.output = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, user_ids, item_ids):
                # GMF
                user_gmf = self.user_embedding_gmf(user_ids)
                item_gmf = self.item_embedding_gmf(item_ids)
                gmf_output = user_gmf * item_gmf

                # MLP
                user_mlp = self.user_embedding_mlp(user_ids)
                item_mlp = self.item_embedding_mlp(item_ids)
                mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
                mlp_output = self.mlp(mlp_input)

                # Concatenate GMF and MLP
                output = torch.cat([gmf_output, mlp_output], dim=1)
                output = self.output(output)
                return self.sigmoid(output)

        return NCFModel(
            self.n_users,
            self.n_items,
            self.embedding_dim,
            hidden_layers
        )

    def fit(
        self,
        interactions: List[Tuple[int, int, float]],
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ):
        """Train the NCF model."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Prepare data
        user_ids = torch.LongTensor([i[0] for i in interactions])
        item_ids = torch.LongTensor([i[1] for i in interactions])
        ratings = torch.FloatTensor([i[2] for i in interactions])

        dataset = TensorDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user_batch, item_batch, rating_batch in dataloader:
                optimizer.zero_grad()
                predictions = self.model(user_batch, item_batch)
                loss = criterion(predictions.squeeze(), rating_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"NCF Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def recommend(
        self,
        user_id: int,
        item_candidates: List[int],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user."""
        import torch

        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * len(item_candidates))
            item_tensor = torch.LongTensor(item_candidates)
            # squeeze(-1) keeps a 1-d tensor even for a single-item batch
            # (squeeze() would collapse [1,1] -> 0-d and break argsort)
            scores = self.model(user_tensor, item_tensor).squeeze(-1).numpy()
        if scores.ndim == 0:
            scores = scores.reshape(1)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1][:n_recommendations]
        return [(item_candidates[i], float(scores[i])) for i in sorted_indices]