import os
import json
import numpy as np
from typing import Optional, Tuple, Dict, Any
from environment.shilling_defense_env import ShillingDefenseEnv


class QLearningAgent:
    """
    Tabular Q-Learning Agent for Shilling Defense.
    
    States: 243 discrete states (3 quantile bins across 5 features: 3^5 = 243).
    Actions: 3 discrete actions (0: ACCEPT w=1.0, 1: DOWNWEIGHT w=0.25, 2: BLOCK w=0.0).
    """

    def __init__(
        self,
        num_states: int = 243,
        num_actions: int = 3,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: Optional[int] = 42
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.rng = np.random.RandomState(seed)
        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def select_action(self, continuous_state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy exploration strategy."""
        state_idx = ShillingDefenseEnv.discretize_state(continuous_state)

        if training and self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, self.num_actions))
        else:
            # Exploit best Q-value (break ties randomly)
            q_values = self.q_table[state_idx]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(self.rng.choice(best_actions))

    def update(
        self,
        continuous_state: np.ndarray,
        action: int,
        reward: float,
        next_continuous_state: np.ndarray,
        done: bool
    ) -> float:
        """
        Bellman equation update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        s_idx = ShillingDefenseEnv.discretize_state(continuous_state)
        next_s_idx = ShillingDefenseEnv.discretize_state(next_continuous_state)

        target = reward if done else reward + self.gamma * np.max(self.q_table[next_s_idx])
        td_error = target - self.q_table[s_idx, action]
        self.q_table[s_idx, action] += self.lr * td_error

        return float(td_error)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save Q-table and hyperparameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.q_table)
        meta_path = filepath.replace(".npy", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "num_states": self.num_states,
                "num_actions": self.num_actions,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": float(self.epsilon)
            }, f, indent=2)

    def load(self, filepath: str):
        """Load Q-table from disk."""
        self.q_table = np.load(filepath).astype(np.float32)
        meta_path = filepath.replace(".npy", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.epsilon = meta.get("epsilon", self.epsilon_min)
