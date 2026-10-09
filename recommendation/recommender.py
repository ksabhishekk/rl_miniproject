import numpy as np
from typing import Optional, List, Tuple


class UserBasedCF:
    """
    User-Based Collaborative Filtering (UBCF) Recommender System
    supporting dynamic user weighting (for RL defensive interventions).
    
    Prediction Formula for user u and item i:
        r_hat(u, i) = mu_u + (sum_{v in N_k(u)} w_v * sim(u, v) * (r(v, i) - mu_v)) /
                              (sum_{v in N_k(u)} w_v * |sim(u, v)| + eps)
    """

    def __init__(self, k_neighbors: int = 40, min_similarity: float = 0.0, eps: float = 1e-8):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.eps = eps
        
        self.R: Optional[np.ndarray] = None  # User-Item rating matrix (M x N)
        self.num_users: int = 0
        self.num_items: int = 0
        self.user_means: Optional[np.ndarray] = None
        self.user_weights: Optional[np.ndarray] = None  # Defense weights w_v in [0, 1]
        self.similarity_matrix: Optional[np.ndarray] = None

    def fit(self, R: np.ndarray, user_weights: Optional[np.ndarray] = None):
        """
        Fit recommender on rating matrix R (shape: M x N).
        Optionally accept initial user weights (default: all 1.0).
        """
        self.R = R.astype(np.float32)
        self.num_users, self.num_items = R.shape

        # Calculate mean rating for each user (ignoring unrated zeros)
        rated_mask = (self.R > 0)
        sums = np.sum(self.R, axis=1)
        counts = np.sum(rated_mask, axis=1)
        self.user_means = np.zeros(self.num_users, dtype=np.float32)
        nonzero_users = (counts > 0)
        self.user_means[nonzero_users] = sums[nonzero_users] / counts[nonzero_users]

        # Initialize user weights (1.0 = fully trusted)
        if user_weights is None:
            self.user_weights = np.ones(self.num_users, dtype=np.float32)
        else:
            self.user_weights = user_weights.astype(np.float32)

        # Compute pairwise Cosine / Pearson similarity
        self._compute_similarity()

    def set_user_weights(self, weights: np.ndarray):
        """Update defensive weights for users without recomputing similarities."""
        assert len(weights) == self.num_users, f"Expected {self.num_users} weights, got {len(weights)}"
        self.user_weights = weights.astype(np.float32)

    def set_single_user_weight(self, user_idx: int, weight: float):
        """Update defensive weight for a single user."""
        self.user_weights[user_idx] = float(weight)

    def _compute_similarity(self):
        """Compute Pearson Correlation Similarity between all user pairs."""
        # Center ratings around user mean
        rated_mask = (self.R > 0).astype(np.float32)
        centered_R = np.where(rated_mask, self.R - self.user_means[:, None], 0.0)

        # Norms
        norms = np.sqrt(np.sum(centered_R ** 2, axis=1)) + self.eps

        # Cosine of mean-centered vectors is Pearson correlation
        dot_products = centered_R @ centered_R.T
        self.similarity_matrix = dot_products / (norms[:, None] * norms[None, :] + self.eps)

        # Zero out self-similarity and negative correlations if min_similarity >= 0
        np.fill_diagonal(self.similarity_matrix, 0.0)
        if self.min_similarity >= 0:
            self.similarity_matrix = np.clip(self.similarity_matrix, 0.0, 1.0)

    def predict_user_ratings(self, user_idx: int) -> np.ndarray:
        """
        Predict ratings for all items for a given user user_idx.
        Incorporates defensive weights w_v into neighbor aggregation.
        """
        # Effective similarity scaled by user defense weights
        sims = self.similarity_matrix[user_idx] * self.user_weights

        # Find top-k neighbors with positive similarity
        if self.k_neighbors < self.num_users:
            top_k_indices = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
            # Filter out neighbors with zero or negative similarity
            top_k_indices = top_k_indices[sims[top_k_indices] > 0]
        else:
            top_k_indices = np.where(sims > 0)[0]

        if len(top_k_indices) == 0:
            # Fallback to user mean or global mean
            return np.full(self.num_items, self.user_means[user_idx], dtype=np.float32)

        neighbor_sims = sims[top_k_indices]  # shape: (K,)
        neighbor_ratings = self.R[top_k_indices]  # shape: (K, num_items)
        neighbor_means = self.user_means[top_k_indices]  # shape: (K,)

        # Mask of items actually rated by neighbors
        rated_mask = (neighbor_ratings > 0).astype(np.float32)

        # Centered neighbor ratings
        centered_neighbor_ratings = rated_mask * (neighbor_ratings - neighbor_means[:, None])

        # Numerator: sum_{v in top_k} sim(u, v) * (r_{v, i} - mu_v)
        num = neighbor_sims @ centered_neighbor_ratings

        # Denominator: sum_{v in top_k} |sim(u, v)| * rated_mask_{v, i}
        denom = neighbor_sims @ rated_mask + self.eps

        predicted_diff = num / denom
        predictions = self.user_means[user_idx] + predicted_diff
        return np.clip(predictions, 1.0, 5.0)

    def recommend_top_k(
        self,
        user_idx: int,
        k: int = 10,
        filter_already_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommended movie items for user user_idx.
        Returns list of (item_idx, predicted_rating).
        """
        preds = self.predict_user_ratings(user_idx)

        if filter_already_rated:
            already_rated = np.where(self.R[user_idx] > 0)[0]
            preds[already_rated] = -np.inf

        top_indices = np.argsort(preds)[::-1][:k]
        return [(int(idx), float(preds[idx])) for idx in top_indices if preds[idx] > -np.inf]
