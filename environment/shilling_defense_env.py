import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, List, Any

from recommendation.recommender import UserBasedCF
from recommendation.data_preprocessing import load_processed_matrix
from attacks.attack_simulator import ShillingAttackSimulator


class ShillingDefenseEnv(gym.Env):
    """
    Gymnasium Custom Environment: RecShield Shilling Attack Defense
    
    The environment simulates an interactive audit stream of incoming user profiles
    (a mix of genuine users and synthetic shilling attack bots).
    
    State Space:
        Box(5,) in [0.0, 1.0]:
        1. f_RDMA: Rating Deviation from Mean Agreement (community consensus divergence)
        2. f_DegSim: Degree of Similarity with Popular Items (blockbuster agreement)
        3. f_ActRatio: Profile Activity Ratio (volume of rated items)
        4. f_TargetDist: Target Item Affinity (rates monitored target with 5.0)
        5. f_SystemRankShift: Live System Target Predicted Rating Inflation
        
    Action Space:
        Discrete(3):
        0: ACCEPT     (defense weight w = 1.00)
        1: DOWNWEIGHT (defense weight w = 0.25)
        2: BLOCK      (defense weight w = 0.00)
        
    Reward Function:
        Multi-objective balance between attack neutralization, genuine user fidelity,
        and system target rank shift delta.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        train_matrix_path: Optional[str] = None,
        episode_length: int = 100,
        genuine_ratio: float = 0.70,
        target_items: Optional[List[int]] = None,
        k_neighbors: int = 30,
        num_eval_users: int = 30,
        reward_config: Optional[Dict[str, float]] = None,
        target_feature_mode: str = "monitored",
        seed: Optional[int] = 42
    ):
        super().__init__()

        # Action: 0=ACCEPT (1.0), 1=DOWNWEIGHT (0.25), 2=BLOCK (0.0)
        self.action_space = spaces.Discrete(3)
        self.action_to_weight = {0: 1.0, 1: 0.25, 2: 0.0}

        # Observation: 5 normalized features in [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Configurable reward weights (defaults preserve original behavior)
        self.reward_config = reward_config or {
            "block_attacker": 2.0,
            "downweight_attacker": 0.8,
            "accept_attacker": -2.5,
            "accept_genuine": 1.0,
            "downweight_genuine": -0.5,
            "block_genuine": -2.0,
            "rank_scale": 2.0,
            "rank_clip": 1.0,
        }

        # Dataset loading
        if train_matrix_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_matrix_path = os.path.join(root_dir, "data", "processed", "train_matrix.npy")

        self.clean_matrix = load_processed_matrix(train_matrix_path)
        self.num_clean_users, self.num_items = self.clean_matrix.shape

        self.episode_length = episode_length
        self.genuine_ratio = genuine_ratio
        self.num_eval_users = min(num_eval_users, self.num_clean_users)
        self.k_neighbors = k_neighbors

        # Monitored target items (MovieLens indices: 40=Billy Madison, 77=Free Willy, 15=French Twist)
        self.target_items = target_items if target_items is not None else [40, 77, 15]
        self.target_feature_mode = target_feature_mode

        # Compute precomputed dataset statistics (including dynamic normalization constants)
        self._precompute_dataset_stats()

        # Evaluation subset of genuine users for tracking target rank shift
        self.eval_user_indices = np.arange(self.num_eval_users)

        # Episode state variables
        self.current_step = 0
        self.current_target_idx = self.target_items[0]
        self.current_attack_type = "bandwagon"
        self.queue_profile_indices: List[int] = []
        self.queue_is_attacker: List[bool] = []
        self.active_matrix: Optional[np.ndarray] = None
        self.recommender: Optional[UserBasedCF] = None
        self.user_weights: Optional[np.ndarray] = None
        self.clean_target_pred: float = 3.0
        self.current_target_pred: float = 3.0
        self.prev_target_pred_shift: float = 0.0
        self.shift_norm: float = 0.5  # Updated dynamically per episode in reset()
        self.episode_events: List[Dict[str, Any]] = []

        self.np_random = np.random.RandomState(seed)

    def _precompute_dataset_stats(self):
        """Precompute item means, popularity counts, blockbuster items, and dynamic normalization constants."""
        rated_mask = (self.clean_matrix > 0)
        item_rating_sums = np.sum(self.clean_matrix, axis=0)
        self.item_counts = np.sum(rated_mask, axis=0)
        
        self.item_means = np.zeros(self.num_items, dtype=np.float32)
        nonzero = self.item_counts > 0
        self.item_means[nonzero] = item_rating_sums[nonzero] / self.item_counts[nonzero]

        # Top 10% most popular items for DegSim calculation
        pop_threshold = np.percentile(self.item_counts, 90)
        self.popular_item_indices = np.where(self.item_counts >= pop_threshold)[0]

        # Mean user rating count
        user_counts = np.sum(rated_mask, axis=1)
        self.mean_user_activity = float(np.mean(user_counts))

        # Auto-compute RDMA normalization constant from clean user distribution
        # Uses the 95th percentile of RDMA values across all clean users so that
        # genuine users mostly fall in [0, 1] and only extreme deviations saturate.
        rdma_values = []
        for u in range(self.num_clean_users):
            row = self.clean_matrix[u]
            rated_idx = np.where(row > 0)[0]
            if len(rated_idx) > 0:
                dev = np.abs(row[rated_idx] - self.item_means[rated_idx])
                rdma_val = np.mean(dev / np.maximum(1, self.item_counts[rated_idx]))
                rdma_values.append(rdma_val)
        if rdma_values:
            self.rdma_norm = float(np.percentile(rdma_values, 95))
            # Guard against degenerate datasets where 95th percentile is near zero
            self.rdma_norm = max(self.rdma_norm, 1e-6)
        else:
            self.rdma_norm = 0.05  # Fallback for empty datasets

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to a fresh episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        options = options or {}
        self.episode_events = []
        # Select target item for this episode
        self.current_target_idx = options.get(
            "target_item",
            self.np_random.choice(self.target_items)
        )
        # Select attack type for this episode
        self.current_attack_type = options.get(
            "attack_type",
            self.np_random.choice(["random", "average", "bandwagon"])
        )

        num_attackers = int(self.episode_length * (1.0 - self.genuine_ratio))
        num_genuine = self.episode_length - num_attackers

        # 1. Synthesize fake profiles
        simulator = ShillingAttackSimulator(
            self.clean_matrix,
            target_item_idx=self.current_target_idx,
            filler_size=0.08,
            random_seed=int(self.np_random.randint(0, 100000))
        )

        attack_profiles = []
        for _ in range(num_attackers):
            if self.current_attack_type == "random":
                p = simulator.generate_random_attack_profile()
            elif self.current_attack_type == "average":
                p = simulator.generate_average_attack_profile()
            else:
                p = simulator.generate_bandwagon_attack_profile()
            attack_profiles.append(p)
        attack_profiles = np.array(attack_profiles, dtype=np.float32)

        # 2. Combine into active matrix
        self.active_matrix = np.vstack([self.clean_matrix, attack_profiles])
        total_users = len(self.active_matrix)

        # 3. Fit recommender
        self.user_weights = np.ones(total_users, dtype=np.float32)
        self.recommender = UserBasedCF(k_neighbors=self.k_neighbors)
        self.recommender.fit(self.active_matrix, user_weights=self.user_weights)

        # 4. Measure clean vs initial attacked target rating
        # Clean baseline with attackers blocked (weights = 0)
        clean_weights = np.ones(total_users, dtype=np.float32)
        clean_weights[self.num_clean_users:] = 0.0
        self.recommender.set_user_weights(clean_weights)
        self.clean_target_pred = self._compute_avg_target_pred()

        # Reset weights back to 1.0 for the start of the audit episode
        self.recommender.set_user_weights(self.user_weights)
        initial_attacked_pred = self._compute_avg_target_pred()
        self.current_target_pred = initial_attacked_pred
        self.prev_target_pred_shift = max(0.0, initial_attacked_pred - self.clean_target_pred)

        # Dynamically set shift normalization to the episode's initial attack impact.
        # This means f_SystemRankShift measures "fraction of initial attack damage remaining"
        # rather than being compared to an arbitrary constant.
        self.shift_norm = max(self.prev_target_pred_shift, 0.1)

        # 5. Form the audit stream queue (shuffled genuine + attackers)
        sampled_genuine_indices = self.np_random.choice(
            self.num_clean_users, size=num_genuine, replace=False
        ).tolist()
        attacker_indices = list(range(self.num_clean_users, total_users))

        combined = [(idx, False) for idx in sampled_genuine_indices] + [(idx, True) for idx in attacker_indices]
        self.np_random.shuffle(combined)

        self.queue_profile_indices = [item[0] for item in combined]
        self.queue_is_attacker = [item[1] for item in combined]

        self.current_step = 0
        obs = self._extract_state_features(self.queue_profile_indices[0])
        info = {
            "target_item": int(self.current_target_idx),
            "attack_type": self.current_attack_type,
            "is_attacker": self.queue_is_attacker[0],
            "initial_target_pred_shift": float(self.prev_target_pred_shift)
        }

        return obs, info

    def _extract_state_features(self, profile_idx: int) -> np.ndarray:
        """
        Extract the 5-dimensional normalized feature vector s_t in [0.0, 1.0]^5:
        1. RDMA
        2. DegSim
        3. ActRatio
        4. TargetDist
        5. SystemRankShift
        """
        row = self.active_matrix[profile_idx]
        rated_idx = np.where(row > 0)[0]
        n_rated = len(rated_idx)

        # 1. RDMA (Rating Deviation from Mean Agreement)
        if n_rated > 0:
            dev = np.abs(row[rated_idx] - self.item_means[rated_idx])
            rdma_val = np.mean(dev / np.maximum(1, self.item_counts[rated_idx]))
            # Dynamic normalization: uses 95th percentile of clean user RDMA distribution
            f_rdma = float(np.clip(rdma_val / self.rdma_norm, 0.0, 1.0))
        else:
            f_rdma = 0.0

        # 2. DegSim (Degree of Similarity with Popular Items)
        rated_pop = np.intersect1d(rated_idx, self.popular_item_indices)
        if len(rated_pop) > 0:
            dev_pop = np.abs(row[rated_pop] - self.item_means[rated_pop])
            degsim_val = np.mean(dev_pop) / 4.0  # max rating range 5-1 = 4
            f_degsim = float(np.clip(1.0 - degsim_val, 0.0, 1.0))
        else:
            f_degsim = 0.0

        # 3. Activity Ratio
        f_act = float(np.clip(n_rated / (self.mean_user_activity * 2.0 + 1e-6), 0.0, 1.0))

        # 4. Target Item Affinity
        if self.target_feature_mode == "target_agnostic":
            # Target-agnostic: highest positive deviation above item mean on ANY rated item
            if n_rated > 0:
                pos_devs = np.maximum(0.0, row[rated_idx] - self.item_means[rated_idx])
                max_dev = float(np.max(pos_devs))
                f_target = float(np.clip(max_dev / 4.0, 0.0, 1.0))
            else:
                f_target = 0.0
        elif self.target_feature_mode == "unmonitored":
            # Defender does not monitor the attacked item -> zero target signal
            f_target = 0.0
        else:
            # Monitored target item (oracle knowledge of target item identity)
            target_rating = row[self.current_target_idx]
            if target_rating > 0:
                deviation = target_rating - self.item_means[self.current_target_idx]
                # Normalize by max possible deviation in a 1-5 rating scale (= 4.0)
                f_target = float(np.clip(deviation / 4.0, 0.0, 1.0))
            else:
                f_target = 0.0

        # 5. Live System Target Predicted Rating Shift
        # Dynamic normalization: uses this episode's initial attack impact as scale
        f_sys_shift = float(np.clip(self.prev_target_pred_shift / self.shift_norm, 0.0, 1.0))

        return np.array([f_rdma, f_degsim, f_act, f_target, f_sys_shift], dtype=np.float32)

    def _compute_avg_target_pred(self) -> float:
        """Compute the average predicted rating for current target item across eval users."""
        preds = []
        for u in self.eval_user_indices:
            user_preds = self.recommender.predict_user_ratings(u)
            preds.append(user_preds[self.current_target_idx])
        return float(np.mean(preds))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one defense action:
        0: ACCEPT     (w = 1.0)
        1: DOWNWEIGHT (w = 0.25)
        2: BLOCK      (w = 0.0)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_profile_idx = self.queue_profile_indices[self.current_step]
        is_attacker = self.queue_is_attacker[self.current_step]
        profile_features = self._extract_state_features(current_profile_idx)
        target_prediction_before = self.current_target_pred
        target_prediction_shift_before = self.prev_target_pred_shift
        profile_ratings = self.active_matrix[current_profile_idx]
        weight = self.action_to_weight[action]

        # Apply action to recommender
        self.recommender.set_single_user_weight(current_profile_idx, weight)
        self.user_weights[current_profile_idx] = weight

        # Measure new system rank shift
        curr_target_pred = self._compute_avg_target_pred()
        curr_target_pred_shift = max(0.0, curr_target_pred - self.clean_target_pred)
        rank_delta = self.prev_target_pred_shift - curr_target_pred_shift
        self.current_target_pred = curr_target_pred
        self.prev_target_pred_shift = curr_target_pred_shift

        # Compute Multi-Objective Reward (using configurable reward weights)
        rc = self.reward_config
        if is_attacker:
            if action == 2:    # BLOCK
                r_mitigation = rc["block_attacker"]
            elif action == 1:  # DOWNWEIGHT
                r_mitigation = rc["downweight_attacker"]
            else:              # ACCEPT (Penetrated)
                r_mitigation = rc["accept_attacker"]
            r_quality = 0.0
        else:
            if action == 0:    # ACCEPT (Preserved)
                r_quality = rc["accept_genuine"]
            elif action == 1:  # DOWNWEIGHT
                r_quality = rc["downweight_genuine"]
            else:              # BLOCK (False positive)
                r_quality = rc["block_genuine"]
            r_mitigation = 0.0

        # Rank inflation improvement bonus/penalty (configurable scale and clip)
        r_rank = float(np.clip(rank_delta * rc["rank_scale"], -rc["rank_clip"], rc["rank_clip"]))

        reward = float(r_mitigation + r_quality + r_rank)

        event = {
            "step": self.current_step,
            "profile_index": int(current_profile_idx),
            "profile_source": "synthetic_attack" if is_attacker else "movielens_genuine",
            "is_attacker": bool(is_attacker),
            "ratings": [
                {"item_index": int(item_idx), "rating": float(profile_ratings[item_idx])}
                for item_idx in np.flatnonzero(profile_ratings > 0)
            ],
            "features": {
                "rdma": float(profile_features[0]),
                "degsim": float(profile_features[1]),
                "activity_ratio": float(profile_features[2]),
                "target_affinity": float(profile_features[3]),
                "system_prediction_shift": float(profile_features[4]),
            },
            "action": int(action),
            "action_name": ("ACCEPT", "DOWNWEIGHT", "BLOCK")[action],
            "defense_weight": float(weight),
            "reward": reward,
            "reward_components": {
                "mitigation": float(r_mitigation),
                "quality": float(r_quality),
                "rank": float(r_rank),
            },
            "target_prediction_before": float(target_prediction_before),
            "target_prediction_after": float(curr_target_pred),
            "target_prediction_shift_before": float(target_prediction_shift_before),
            "target_prediction_shift_after": float(curr_target_pred_shift),
        }
        self.episode_events.append(event)

        self.current_step += 1
        terminated = (self.current_step >= self.episode_length)
        truncated = False

        if not terminated:
            next_profile_idx = self.queue_profile_indices[self.current_step]
            next_obs = self._extract_state_features(next_profile_idx)
            next_is_attacker = self.queue_is_attacker[self.current_step]
        else:
            next_obs = np.zeros(5, dtype=np.float32)
            next_is_attacker = False

        info = {
            "step": self.current_step,
            "action": int(action),
            "defense_weight": float(weight),
            "is_attacker": is_attacker,
            "target_pred_shift": float(curr_target_pred_shift),
            "target_prediction_before": float(target_prediction_before),
            "target_prediction_after": float(curr_target_pred),
            "reward_mitigation": float(r_mitigation),
            "reward_quality": float(r_quality),
            "reward_rank": float(r_rank),
            "audit_event": event,
        }

        return next_obs, reward, terminated, truncated, info

    @staticmethod
    def discretize_state(continuous_state: np.ndarray, num_bins: int = 3) -> int:
        """
        Discretize 5-dim continuous state into 3^5 = 243 discrete states for Tabular Q-Learning.
        Each feature in [0.0, 1.0] is mapped to {0, 1, 2}.
        """
        bins = np.clip(np.floor(continuous_state * num_bins).astype(int), 0, num_bins - 1)
        state_idx = (
            bins[0] * (num_bins ** 4) +
            bins[1] * (num_bins ** 3) +
            bins[2] * (num_bins ** 2) +
            bins[3] * (num_bins ** 1) +
            bins[4]
        )
        return int(state_idx)
