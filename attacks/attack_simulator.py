import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class ShillingAttackSimulator:
    """
    Shilling (Profile Injection) Attack Simulator for Recommendation Systems.
    Implements:
      1. Random Attack
      2. Average Attack
      3. Bandwagon Attack
    """

    def __init__(
        self,
        clean_matrix: np.ndarray,
        target_item_idx: int,
        filler_size: float = 0.08,
        random_seed: int = 42
    ):
        """
        clean_matrix: shape (num_users, num_items) with genuine ratings
        target_item_idx: 0-indexed column of the movie to promote (Push attack)
        filler_size: fraction of catalog rated as filler items (e.g. 0.08 = ~134 movies)
        """
        self.clean_matrix = clean_matrix.astype(np.float32)
        self.num_users, self.num_items = clean_matrix.shape
        self.target_item_idx = target_item_idx
        self.filler_size = filler_size
        self.num_filler = max(5, int(self.num_items * filler_size))
        self.rng = np.random.RandomState(random_seed)

        # Precompute item statistics for Average and Bandwagon attacks
        self._compute_item_stats()

    def _compute_item_stats(self):
        """Precompute item means, stds, and identify popular bandwagon items."""
        rated_mask = (self.clean_matrix > 0)
        counts = np.sum(rated_mask, axis=0)
        sums = np.sum(self.clean_matrix, axis=0)

        self.item_counts = counts
        self.item_means = np.full(self.num_items, 3.0, dtype=np.float32)
        nonzero_items = (counts > 0)
        self.item_means[nonzero_items] = sums[nonzero_items] / counts[nonzero_items]

        # Standard deviations
        self.item_stds = np.full(self.num_items, 1.0, dtype=np.float32)
        for i in range(self.num_items):
            if counts[i] > 1:
                ratings = self.clean_matrix[self.clean_matrix[:, i] > 0, i]
                self.item_stds[i] = float(np.std(ratings))

        # Top popular items for Bandwagon attack (most rated items, excluding target item)
        sorted_popular = np.argsort(counts)[::-1]
        self.popular_items = [idx for idx in sorted_popular if idx != self.target_item_idx][:20]

    def generate_random_attack_profile(self) -> np.ndarray:
        """
        Random Attack:
          - Target item: 5.0
          - Filler items: chosen uniformly at random, rated uniformly from [1, 5]
        """
        profile = np.zeros(self.num_items, dtype=np.float32)
        # Target item push
        profile[self.target_item_idx] = 5.0

        # Candidate filler pool (all items except target)
        candidate_pool = [i for i in range(self.num_items) if i != self.target_item_idx]
        filler_indices = self.rng.choice(candidate_pool, size=self.num_filler, replace=False)

        # Random ratings {1, 2, 3, 4, 5}
        filler_ratings = self.rng.randint(1, 6, size=self.num_filler).astype(np.float32)
        profile[filler_indices] = filler_ratings
        return profile

    def generate_average_attack_profile(self) -> np.ndarray:
        """
        Average Attack:
          - Target item: 5.0
          - Filler items: chosen uniformly at random, ratings sampled from N(mu_i, sigma_i^2)
        """
        profile = np.zeros(self.num_items, dtype=np.float32)
        profile[self.target_item_idx] = 5.0

        candidate_pool = [i for i in range(self.num_items) if i != self.target_item_idx]
        filler_indices = self.rng.choice(candidate_pool, size=self.num_filler, replace=False)

        for idx in filler_indices:
            mu = self.item_means[idx]
            sigma = max(0.2, self.item_stds[idx])
            sample_rating = round(float(self.rng.normal(mu, sigma)))
            profile[idx] = float(np.clip(sample_rating, 1.0, 5.0))

        return profile

    def generate_bandwagon_attack_profile(self, num_popular_to_rate: int = 10) -> np.ndarray:
        """
        Bandwagon Attack:
          - Target item: 5.0
          - Selected popular blockbuster items: rated 5.0 (ties profile to mainstream users)
          - Remaining filler items: sampled around average ratings
        """
        profile = np.zeros(self.num_items, dtype=np.float32)
        profile[self.target_item_idx] = 5.0

        # Select top popular items
        popular_selected = self.rng.choice(
            self.popular_items,
            size=min(num_popular_to_rate, len(self.popular_items)),
            replace=False
        )
        profile[popular_selected] = 5.0

        # Additional random filler
        remaining_filler_count = max(0, self.num_filler - len(popular_selected))
        candidate_pool = [
            i for i in range(self.num_items)
            if i != self.target_item_idx and i not in popular_selected
        ]
        filler_indices = self.rng.choice(candidate_pool, size=remaining_filler_count, replace=False)

        for idx in filler_indices:
            mu = self.item_means[idx]
            sigma = max(0.2, self.item_stds[idx])
            sample_rating = round(float(self.rng.normal(mu, sigma)))
            profile[idx] = float(np.clip(sample_rating, 1.0, 5.0))

        return profile

    def simulate_attack(
        self,
        attack_type: str = "random",
        attack_size_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject synthetic attack profiles into clean rating matrix.
        Returns:
          attacked_matrix: shape (num_users + num_attackers, num_items)
          labels: 0 for genuine, 1 for attacker (shape: M_total,)
          attack_profiles: array of fake profiles (shape: num_attackers, num_items)
        """
        attack_type = attack_type.lower()
        num_attackers = max(1, int(self.num_users * attack_size_ratio))

        fake_profiles = []
        for _ in range(num_attackers):
            if attack_type == "random":
                fake_profiles.append(self.generate_random_attack_profile())
            elif attack_type == "average":
                fake_profiles.append(self.generate_average_attack_profile())
            elif attack_type == "bandwagon":
                fake_profiles.append(self.generate_bandwagon_attack_profile())
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

        fake_profiles = np.array(fake_profiles, dtype=np.float32)
        attacked_matrix = np.vstack([self.clean_matrix, fake_profiles])

        labels = np.zeros(len(attacked_matrix), dtype=int)
        labels[self.num_users:] = 1  # 1 indicates malicious profile

        return attacked_matrix, labels, fake_profiles
