import os
import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from environment.shilling_defense_env import ShillingDefenseEnv


def create_ppo_agent(
    env: ShillingDefenseEnv,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 32,
    n_epochs: int = 5,
    gamma: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    seed: Optional[int] = 42,
    verbose: int = 0
) -> PPO:
    """
    Construct a Proximal Policy Optimization (PPO) Actor-Critic agent
    operating on continuous Box(5,) observations.
    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        seed=seed,
        verbose=verbose
    )
    return model


def train_ppo(
    model: PPO,
    total_timesteps: int = 5000,
    save_path: Optional[str] = None
) -> PPO:
    """Train PPO agent and optionally save to disk."""
    model.learn(total_timesteps=total_timesteps)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
    return model


def load_ppo_agent(model_path: str, env: Optional[ShillingDefenseEnv] = None) -> PPO:
    """Load trained PPO model from disk."""
    return PPO.load(model_path, env=env)
