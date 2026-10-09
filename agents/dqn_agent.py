import os
import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment.shilling_defense_env import ShillingDefenseEnv


def create_dqn_agent(
    env: ShillingDefenseEnv,
    learning_rate: float = 1e-3,
    buffer_size: int = 10000,
    learning_starts: int = 100,
    batch_size: int = 64,
    gamma: float = 0.95,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    seed: Optional[int] = 42,
    verbose: int = 0
) -> DQN:
    """
    Construct a Deep Q-Network (DQN) agent operating on continuous Box(5,) observations.
    """
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=200,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=verbose
    )
    return model


def train_dqn(
    model: DQN,
    total_timesteps: int = 5000,
    save_path: Optional[str] = None
) -> DQN:
    """Train DQN agent and optionally save to disk."""
    model.learn(total_timesteps=total_timesteps)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
    return model


def load_dqn_agent(model_path: str, env: Optional[ShillingDefenseEnv] = None) -> DQN:
    """Load trained DQN model from disk."""
    return DQN.load(model_path, env=env)
