import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.shilling_defense_env import ShillingDefenseEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import create_dqn_agent, train_dqn
from agents.ppo_agent import create_ppo_agent, train_ppo
from recommendation.evaluation import evaluate_target_item_impact, evaluate_test_rmse
from recommendation.recommender import UserBasedCF
from attacks.attack_simulator import ShillingAttackSimulator


def train_q_learning(env: ShillingDefenseEnv, num_episodes: int = 150) -> QLearningAgent:
    """Train Tabular Q-Learning agent on ShillingDefenseEnv."""
    print(f"\n[1/3] Training Tabular Q-Learning ({num_episodes} episodes)...")
    agent = QLearningAgent(learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.05, seed=42)
    start_time = time.time()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  Q-Learning Ep {ep+1:03d}/{num_episodes} | Total Reward: {ep_reward:+6.2f} | Epsilon: {agent.epsilon:.3f}")

    os.makedirs("results/checkpoints", exist_ok=True)
    agent.save("results/checkpoints/q_table_final.npy")
    print(f"  Tabular Q-Learning training completed in {time.time()-start_time:.2f}s")
    return agent


def train_dqn_policy(env: ShillingDefenseEnv, total_timesteps: int = 5000):
    """Train Deep Q-Network agent on ShillingDefenseEnv."""
    print(f"\n[2/3] Training Deep Q-Network DQN ({total_timesteps} timesteps)...")
    start_time = time.time()
    dqn_model = create_dqn_agent(env, learning_rate=1e-3, buffer_size=10000, learning_starts=100, batch_size=64, seed=42, verbose=0)
    train_dqn(dqn_model, total_timesteps=total_timesteps, save_path="results/checkpoints/dqn_final.zip")
    print(f"  DQN training completed in {time.time()-start_time:.2f}s")
    return dqn_model


def train_ppo_policy(env: ShillingDefenseEnv, total_timesteps: int = 5000):
    """Train Proximal Policy Optimization Actor-Critic agent on ShillingDefenseEnv."""
    print(f"\n[3/3] Training PPO Actor-Critic ({total_timesteps} timesteps)...")
    start_time = time.time()
    ppo_model = create_ppo_agent(env, learning_rate=3e-4, n_steps=128, batch_size=32, n_epochs=5, seed=42, verbose=0)
    train_ppo(ppo_model, total_timesteps=total_timesteps, save_path="results/checkpoints/ppo_final.zip")
    print(f"  PPO training completed in {time.time()-start_time:.2f}s")
    return ppo_model


def evaluate_agent_on_attack(
    agent_name: str,
    policy_fn,
    clean_matrix: np.ndarray,
    test_matrix: np.ndarray,
    target_idx: int,
    attack_type: str,
    attack_ratio: float = 0.08,
    k_neighbors: int = 30
) -> Dict[str, Any]:
    """
    Evaluate a policy against an injected attack matrix.
    Measures: Prediction Shift, HitRatio@10, Genuine Test RMSE, F1 Classification Score.
    """
    num_clean = clean_matrix.shape[0]
    num_attackers = max(1, int(num_clean * attack_ratio))

    # 1. Generate attack profiles
    simulator = ShillingAttackSimulator(clean_matrix, target_item_idx=target_idx, filler_size=0.08, random_seed=42)
    fake_profiles = []
    for _ in range(num_attackers):
        if attack_type == "random":
            fake_profiles.append(simulator.generate_random_attack_profile())
        elif attack_type == "average":
            fake_profiles.append(simulator.generate_average_attack_profile())
        else:
            fake_profiles.append(simulator.generate_bandwagon_attack_profile())
    fake_profiles = np.array(fake_profiles, dtype=np.float32)

    attacked_matrix = np.vstack([clean_matrix, fake_profiles])
    total_users = len(attacked_matrix)
    labels = np.zeros(total_users, dtype=int)
    labels[num_clean:] = 1  # 1 = attacker, 0 = genuine

    # Evaluation set of genuine users who haven't rated target item
    eval_users = [u for u in range(num_clean) if clean_matrix[u, target_idx] == 0][:100]

    # Precalculate Clean Baseline
    cf_clean = UserBasedCF(k_neighbors=k_neighbors)
    cf_clean.fit(clean_matrix)
    clean_impact = evaluate_target_item_impact(cf_clean, target_idx, eval_users, k=10)
    clean_target_pred = clean_impact["avg_target_predicted_rating"]
    clean_hr10 = clean_impact["target_hit_ratio_at_k"]
    clean_rmse = evaluate_test_rmse(cf_clean, test_matrix, max_eval_users=80)

    # 2. Extract features for all profiles
    env_helper = ShillingDefenseEnv(train_matrix_path="data/processed/train_matrix.npy", target_items=[target_idx])
    env_helper.active_matrix = attacked_matrix
    env_helper.current_target_idx = target_idx
    env_helper.prev_target_pred_shift = 0.15  # baseline assumption for feature calculation

    assigned_weights = np.ones(total_users, dtype=np.float32)
    actions = np.zeros(total_users, dtype=int)

    for i in range(total_users):
        feat = env_helper._extract_state_features(i)
        action = policy_fn(feat, i, attacked_matrix)
        actions[i] = action
        assigned_weights[i] = 1.0 if action == 0 else (0.25 if action == 1 else 0.0)

    # 3. Fit recommender with assigned defense weights
    cf_defended = UserBasedCF(k_neighbors=k_neighbors)
    cf_defended.fit(attacked_matrix, user_weights=assigned_weights)

    defended_impact = evaluate_target_item_impact(cf_defended, target_idx, eval_users, k=10)
    defended_target_pred = defended_impact["avg_target_predicted_rating"]
    defended_hr10 = defended_impact["target_hit_ratio_at_k"]
    defended_rmse = evaluate_test_rmse(cf_defended, test_matrix, max_eval_users=80)

    pred_shift = defended_target_pred - clean_target_pred
    hr_shift = defended_hr10 - clean_hr10

    # 4. Classification Metrics (treating Block or Downweight as Positive Detection)
    pred_labels = (actions > 0).astype(int)  # 1 if intervention applied
    tp = np.sum((pred_labels == 1) & (labels == 1))
    fp = np.sum((pred_labels == 1) & (labels == 0))
    fn = np.sum((pred_labels == 0) & (labels == 1))
    tn = np.sum((pred_labels == 0) & (labels == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Action distribution
    genuine_actions = actions[:num_clean]
    attack_actions = actions[num_clean:]

    return {
        "agent": agent_name,
        "attack_type": attack_type,
        "attack_ratio": attack_ratio,
        "target_pred_rating": float(defended_target_pred),
        "clean_target_pred_rating": float(clean_target_pred),
        "prediction_shift": float(pred_shift),
        "target_hit_ratio_at_10": float(defended_hr10),
        "clean_hit_ratio_at_10": float(clean_hr10),
        "hit_ratio_shift": float(hr_shift),
        "test_rmse": float(defended_rmse),
        "clean_rmse": float(clean_rmse),
        "rmse_delta": float(defended_rmse - clean_rmse),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "action_distribution": {
            "genuine_accept_pct": float(np.mean(genuine_actions == 0) * 100),
            "genuine_downweight_pct": float(np.mean(genuine_actions == 1) * 100),
            "genuine_block_pct": float(np.mean(genuine_actions == 2) * 100),
            "attack_accept_pct": float(np.mean(attack_actions == 0) * 100),
            "attack_downweight_pct": float(np.mean(attack_actions == 1) * 100),
            "attack_block_pct": float(np.mean(attack_actions == 2) * 100)
        }
    }


def run_full_comparative_benchmark():
    print("=================================================================")
    print("RecShield-RL: Comparative Benchmark Across RL Defense Algorithms")
    print("=================================================================")

    # 1. Initialize training environment
    env = ShillingDefenseEnv(episode_length=100, num_eval_users=25, seed=42)

    # 2. Train agents
    q_agent = train_q_learning(env, num_episodes=150)
    dqn_model = train_dqn_policy(env, total_timesteps=6000)
    ppo_model = train_ppo_policy(env, total_timesteps=6000)

    # 3. Define policies
    def policy_no_defense(feat, idx, mat):
        return 0  # always ACCEPT

    def policy_heuristic_threshold(feat, idx, mat):
        # Heuristic based on RDMA and Target item 5.0 rating
        f_rdma, f_degsim, f_act, f_target, _ = feat
        if f_target > 0.5 and (f_rdma > 0.4 or f_degsim > 0.7):
            return 2  # BLOCK
        elif f_target > 0.5:
            return 1  # DOWNWEIGHT
        return 0      # ACCEPT

    def policy_q_learning(feat, idx, mat):
        return q_agent.select_action(feat, training=False)

    def policy_dqn(feat, idx, mat):
        action, _ = dqn_model.predict(feat, deterministic=True)
        return int(action)

    def policy_ppo(feat, idx, mat):
        action, _ = ppo_model.predict(feat, deterministic=True)
        return int(action)

    policies = {
        "No Defense (Attacked)": policy_no_defense,
        "Static Heuristic Rule": policy_heuristic_threshold,
        "Tabular Q-Learning": policy_q_learning,
        "Deep Q-Network (DQN)": policy_dqn,
        "PPO (Actor-Critic)": policy_ppo
    }

    # 4. Load datasets for evaluation
    clean_matrix = np.load("data/processed/train_matrix.npy")
    test_matrix = np.load("data/processed/test_matrix.npy")
    target_idx = 40  # Billy Madison (1995)

    benchmark_results = {}

    print("\n-----------------------------------------------------------------")
    print("Evaluating Defense Policies across Random, Average & Bandwagon Attacks")
    print("-----------------------------------------------------------------")

    for attack_type in ["random", "average", "bandwagon"]:
        benchmark_results[attack_type] = {}
        print(f"\n>>> Attack Scenario: {attack_type.upper()} PUSH ATTACK (@ 8% attack size)")
        print(f"{'Model Name':24s} | {'Pred Shift':10s} | {'HR@10':7s} | {'RMSE Delta':10s} | {'F1-Score':8s} | {'Fake Blocked':12s}")
        print("-" * 80)

        for name, fn in policies.items():
            res = evaluate_agent_on_attack(
                agent_name=name,
                policy_fn=fn,
                clean_matrix=clean_matrix,
                test_matrix=test_matrix,
                target_idx=target_idx,
                attack_type=attack_type,
                attack_ratio=0.08
            )
            benchmark_results[attack_type][name] = res

            fake_blocked_pct = res["action_distribution"]["attack_block_pct"] + res["action_distribution"]["attack_downweight_pct"]
            print(f"{name:24s} | {res['prediction_shift']:+10.4f} | {res['target_hit_ratio_at_10']:7.4f} | "
                  f"{res['rmse_delta']:+10.4f} | {res['f1_score']:8.3f} | {fake_blocked_pct:10.1f}%")

    # 5. Save results to JSON
    output_path = "results/comparative_benchmark_results.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n=================================================================")
    print(f"[SUCCESS] Comparative benchmark completed! Saved to {output_path}")
    print("=================================================================")


if __name__ == "__main__":
    run_full_comparative_benchmark()
