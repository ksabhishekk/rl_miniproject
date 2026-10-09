import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.shilling_defense_env import ShillingDefenseEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import create_dqn_agent, train_dqn
from agents.ppo_agent import create_ppo_agent, train_ppo


def test_q_learning(env):
    print("\n--- Testing Tabular Q-Learning Agent ---")
    agent = QLearningAgent(seed=42)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0

    for step in range(25):
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, _, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)
        total_reward += reward
        obs = next_obs
        if terminated:
            break

    agent.decay_epsilon()
    save_path = "results/checkpoints/q_table_test.npy"
    agent.save(save_path)
    assert os.path.exists(save_path), "Q-table failed to save"
    print(f"[PASS] Q-Learning executed 25 steps, Total Reward: {total_reward:.2f}, Saved to {save_path}")


def test_dqn(env):
    print("\n--- Testing Deep Q-Network (DQN) Agent ---")
    model = create_dqn_agent(env, learning_starts=10, buffer_size=500, verbose=0)
    train_dqn(model, total_timesteps=30, save_path="results/checkpoints/dqn_test.zip")
    assert os.path.exists("results/checkpoints/dqn_test.zip"), "DQN model failed to save"
    print("[PASS] DQN initialized, trained 30 timesteps, and saved checkpoint successfully.")


def test_ppo(env):
    print("\n--- Testing Proximal Policy Optimization (PPO) Agent ---")
    model = create_ppo_agent(env, n_steps=32, batch_size=16, verbose=0)
    train_ppo(model, total_timesteps=32, save_path="results/checkpoints/ppo_test.zip")
    assert os.path.exists("results/checkpoints/ppo_test.zip"), "PPO model failed to save"
    print("[PASS] PPO initialized, trained 32 timesteps, and saved checkpoint successfully.")


if __name__ == "__main__":
    env = ShillingDefenseEnv(episode_length=30, num_eval_users=15, seed=42)
    test_q_learning(env)
    test_dqn(env)
    test_ppo(env)
    print("\n[ALL TESTS PASSED] Phase 1 implementation verified successfully!")
