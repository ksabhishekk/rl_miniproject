import os
import sys
import json
import time
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Callable
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.shilling_defense_env import ShillingDefenseEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import create_dqn_agent, train_dqn
from agents.ppo_agent import create_ppo_agent, train_ppo
from recommendation.evaluation import (
    evaluate_target_item_impact,
    evaluate_test_rmse,
    evaluate_catalog_coverage,
    evaluate_genuine_top_k_metrics,
)
from recommendation.recommender import UserBasedCF
from recommendation.data_preprocessing import load_processed_matrix
from attacks.attack_simulator import ShillingAttackSimulator

# ===== EXPERIMENT CONFIGURATION =====
TRAINING_SEEDS = [42, 123, 456]
Q_LEARNING_EPISODES = 150              # 150 episodes x 100 steps = 15,000 timesteps
DQN_TIMESTEPS = 15000                  # Equal budget: ~150 episodes
PPO_TIMESTEPS = 15000                  # Equal budget: ~150 episodes
ATTACK_TYPES = ["random", "average", "bandwagon"]
TARGET_ITEMS = [40, 77, 15]
TARGET_NAMES = {40: "Billy Madison (40)", 77: "Free Willy (77)", 15: "French Twist (15)"}
EVAL_ATTACK_RATIO = 0.08               # 8% profile injection ratio during EVALUATION (75 attackers / 943 genuine)
K_NEIGHBORS = 30
# Training episodes use a 70/30 genuine/attacker stream ratio (configured in env)
# =====================================


def train_q_learning(env: ShillingDefenseEnv, num_episodes: int = Q_LEARNING_EPISODES,
                     seed: int = 42) -> QLearningAgent:
    """Train Tabular Q-Learning agent."""
    agent = QLearningAgent(
        learning_rate=0.1, gamma=0.95, epsilon=1.0,
        epsilon_decay=0.98, epsilon_min=0.05, seed=seed
    )
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        agent.decay_epsilon()
    return agent


def train_dqn_policy(env: ShillingDefenseEnv, total_timesteps: int = DQN_TIMESTEPS,
                     seed: int = 42):
    """Train Deep Q-Network agent (equal budget to Q-Learning)."""
    dqn_model = create_dqn_agent(
        env, learning_rate=1e-3, buffer_size=10000, learning_starts=100,
        batch_size=64, seed=seed, verbose=0
    )
    train_dqn(dqn_model, total_timesteps=total_timesteps)
    return dqn_model


def train_ppo_policy(env: ShillingDefenseEnv, total_timesteps: int = PPO_TIMESTEPS,
                     seed: int = 42):
    """Train PPO Actor-Critic agent (equal budget to Q-Learning)."""
    ppo_model = create_ppo_agent(
        env, learning_rate=3e-4, n_steps=128, batch_size=32,
        n_epochs=5, seed=seed, verbose=0
    )
    train_ppo(ppo_model, total_timesteps=total_timesteps)
    return ppo_model


def compute_clean_baseline(
    clean_matrix: np.ndarray,
    test_matrix: np.ndarray,
    target_idx: int,
    k_neighbors: int = K_NEIGHBORS
) -> Dict[str, Any]:
    """Compute and cache clean CF performance once per target item."""
    num_clean = clean_matrix.shape[0]
    total_items = clean_matrix.shape[1]
    eval_users = [u for u in range(num_clean) if clean_matrix[u, target_idx] == 0][:100]

    cf_clean = UserBasedCF(k_neighbors=k_neighbors)
    cf_clean.fit(clean_matrix)
    clean_impact = evaluate_target_item_impact(cf_clean, target_idx, eval_users, k=10)
    clean_rmse = evaluate_test_rmse(cf_clean, test_matrix, max_eval_users=50)
    clean_coverage = evaluate_catalog_coverage(cf_clean, total_items, eval_users[:50], k=10)
    clean_genuine_topk = evaluate_genuine_top_k_metrics(cf_clean, test_matrix, rating_threshold=3.5, k=10, max_eval_users=50)

    return {
        "cf_clean": cf_clean,
        "clean_target_pred": clean_impact["avg_target_predicted_rating"],
        "clean_hr10": clean_impact["target_hit_ratio_at_k"],
        "clean_rmse": clean_rmse,
        "clean_coverage": clean_coverage,
        "clean_genuine_hit_rate": clean_genuine_topk["hit_rate@10"],
        "clean_genuine_precision": clean_genuine_topk["precision@10"],
        "eval_users": eval_users,
    }


def evaluate_agent_on_attack(
    agent_name: str,
    policy_fn,
    clean_matrix: np.ndarray,
    test_matrix: np.ndarray,
    target_idx: int,
    attack_type: str,
    clean_baseline: Optional[Dict[str, Any]] = None,
    attack_ratio: float = EVAL_ATTACK_RATIO,
    k_neighbors: int = K_NEIGHBORS,
    eval_seed: int = 42,
    target_feature_mode: str = "monitored"
) -> Dict[str, Any]:
    """
    Evaluate a defense policy against an injected attack.

    Evaluation Mix:
        - Clean users: N_clean (943 users, ~92.6% of evaluation stream)
        - Attackers: N_attackers (int(943 * 0.08) = 75 users, ~7.4% of stream)
        - eval_seed varies per seed so attack fillers & injection differ across evaluations.
    """
    num_clean = clean_matrix.shape[0]
    total_items = clean_matrix.shape[1]
    num_attackers = max(1, int(num_clean * attack_ratio))

    if clean_baseline is None:
        clean_baseline = compute_clean_baseline(clean_matrix, test_matrix, target_idx, k_neighbors)

    eval_users = clean_baseline["eval_users"]
    clean_target_pred = clean_baseline["clean_target_pred"]
    clean_hr10 = clean_baseline["clean_hr10"]
    clean_rmse = clean_baseline["clean_rmse"]
    clean_coverage = clean_baseline["clean_coverage"]

    # 1. Generate attack profiles with eval_seed for proper statistical variance
    simulator = ShillingAttackSimulator(
        clean_matrix, target_item_idx=target_idx, filler_size=0.08, random_seed=eval_seed
    )
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
    labels[num_clean:] = 1  # 1 = attacker

    # Feature extraction helper
    env_helper = ShillingDefenseEnv(
        train_matrix_path="data/processed/train_matrix.npy",
        target_items=[target_idx],
        target_feature_mode=target_feature_mode
    )
    env_helper.active_matrix = attacked_matrix
    env_helper.current_target_idx = target_idx

    cf_attacked_initial = UserBasedCF(k_neighbors=k_neighbors)
    cf_attacked_initial.fit(attacked_matrix)
    attacked_preds = [
        cf_attacked_initial.predict_user_ratings(u)[target_idx] for u in eval_users[:30]
    ]
    initial_attacked_pred = float(np.mean(attacked_preds))
    actual_initial_shift = max(0.0, initial_attacked_pred - clean_target_pred)
    env_helper.prev_target_pred_shift = actual_initial_shift
    env_helper.shift_norm = max(actual_initial_shift, 0.1)

    # 2. Run policy on every profile
    assigned_weights = np.ones(total_users, dtype=np.float32)
    actions = np.zeros(total_users, dtype=int)

    for i in range(total_users):
        feat = env_helper._extract_state_features(i)
        action = policy_fn(feat, i, attacked_matrix)
        actions[i] = action
        assigned_weights[i] = 1.0 if action == 0 else (0.25 if action == 1 else 0.0)

    # 3. Fit defended recommender
    cf_defended = UserBasedCF(k_neighbors=k_neighbors)
    cf_defended.fit(attacked_matrix, user_weights=assigned_weights)

    defended_impact = evaluate_target_item_impact(cf_defended, target_idx, eval_users, k=10)
    defended_target_pred = defended_impact["avg_target_predicted_rating"]
    defended_hr10 = defended_impact["target_hit_ratio_at_k"]
    defended_rmse = evaluate_test_rmse(cf_defended, test_matrix, max_eval_users=50)
    defended_coverage = evaluate_catalog_coverage(cf_defended, total_items, eval_users[:50], k=10)
    defended_genuine_topk = evaluate_genuine_top_k_metrics(cf_defended, test_matrix, rating_threshold=3.5, k=10, max_eval_users=50)

    pred_shift = defended_target_pred - clean_target_pred
    hr_shift = defended_hr10 - clean_hr10

    # 4. Classification metrics
    # Dual evaluation: Any Intervention (actions > 0) vs Strict Block (actions == 2)
    # A) Intervention metrics (Downweight or Block treated as positive detection)
    pred_int = (actions > 0).astype(int)
    tp_int = int(np.sum((pred_int == 1) & (labels == 1)))
    fp_int = int(np.sum((pred_int == 1) & (labels == 0)))
    fn_int = int(np.sum((pred_int == 0) & (labels == 1)))
    prec_int = float(tp_int / (tp_int + fp_int)) if (tp_int + fp_int) > 0 else 0.0
    rec_int = float(tp_int / (tp_int + fn_int)) if (tp_int + fn_int) > 0 else 0.0
    f1_int = float(2 * prec_int * rec_int / (prec_int + rec_int)) if (prec_int + rec_int) > 0 else 0.0

    # B) Strict Block metrics (only Action 2 = BLOCK treated as positive)
    pred_blk = (actions == 2).astype(int)
    tp_blk = int(np.sum((pred_blk == 1) & (labels == 1)))
    fp_blk = int(np.sum((pred_blk == 1) & (labels == 0)))
    fn_blk = int(np.sum((pred_blk == 0) & (labels == 1)))
    prec_blk = float(tp_blk / (tp_blk + fp_blk)) if (tp_blk + fp_blk) > 0 else 0.0
    rec_blk = float(tp_blk / (tp_blk + fn_blk)) if (tp_blk + fn_blk) > 0 else 0.0
    f1_blk = float(2 * prec_blk * rec_blk / (prec_blk + rec_blk)) if (prec_blk + rec_blk) > 0 else 0.0

    genuine_actions = actions[:num_clean]
    attack_actions = actions[num_clean:]

    return {
        "agent": agent_name,
        "attack_type": attack_type,
        "attack_ratio": attack_ratio,
        "eval_seed": eval_seed,
        "target_feature_mode": target_feature_mode,
        "evaluation_mix": {
            "num_genuine": num_clean,
            "num_attackers": num_attackers,
            "attacker_pct": float(num_attackers / total_users * 100),
        },
        "target_pred_rating": float(defended_target_pred),
        "clean_target_pred_rating": float(clean_target_pred),
        "prediction_shift": float(pred_shift),
        "abs_prediction_shift": float(abs(pred_shift)),
        "target_hit_ratio_at_10": float(defended_hr10),
        "clean_hit_ratio_at_10": float(clean_hr10),
        "hit_ratio_shift": float(hr_shift),
        "test_rmse": float(defended_rmse),
        "clean_rmse": float(clean_rmse),
        "rmse_delta": float(defended_rmse - clean_rmse),
        "catalog_coverage": float(defended_coverage),
        "clean_catalog_coverage": float(clean_coverage),
        "coverage_delta": float(defended_coverage - clean_coverage),
        "genuine_hit_rate_at_10": float(defended_genuine_topk["hit_rate@10"]),
        "clean_genuine_hit_rate": float(clean_baseline["clean_genuine_hit_rate"]),
        # Classification (Intervention: Actions >= 1)
        "precision": prec_int,
        "recall": rec_int,
        "f1_score": f1_int,
        # Classification (Strict Block: Action == 2)
        "precision_block": prec_blk,
        "recall_block": rec_blk,
        "f1_block": f1_blk,
        "action_distribution": {
            "genuine_accept_pct": float(np.mean(genuine_actions == 0) * 100),
            "genuine_downweight_pct": float(np.mean(genuine_actions == 1) * 100),
            "genuine_block_pct": float(np.mean(genuine_actions == 2) * 100),
            "attack_accept_pct": float(np.mean(attack_actions == 0) * 100),
            "attack_downweight_pct": float(np.mean(attack_actions == 1) * 100),
            "attack_block_pct": float(np.mean(attack_actions == 2) * 100),
        },
    }


def _predict_agent_action(agent: Any, observation: np.ndarray) -> int:
    """Get a deterministic discrete action from any supported project agent."""
    if hasattr(agent, "select_action"):
        action = agent.select_action(observation, training=False)
    elif hasattr(agent, "predict"):
        action, _ = agent.predict(observation, deterministic=True)
    else:
        raise TypeError("Agents must provide select_action() or predict().")

    action = int(np.asarray(action).item())
    if action not in (0, 1, 2):
        raise ValueError(f"Agent returned invalid defense action: {action}")
    return action


def _binary_classification_metrics(
    predicted_positive: np.ndarray,
    actual_positive: np.ndarray
) -> Dict[str, float]:
    true_positive = int(np.sum(predicted_positive & actual_positive))
    false_positive = int(np.sum(predicted_positive & ~actual_positive))
    false_negative = int(np.sum(~predicted_positive & actual_positive))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def run_shared_scenario_comparison(
    agents: Dict[str, Any],
    seed: int = 10042,
    target_idx: int = 40,
    attack_type: str = "bandwagon",
    episode_length: int = 100,
    genuine_ratio: float = 0.70,
    target_feature_mode: str = "monitored",
    k_neighbors: int = K_NEIGHBORS,
    progress_callback: Optional[Callable[[int, Dict[str, Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Run policies in isolated environments over one reproducible scenario.

    Each environment is reset with identical inputs, so the attack profiles and
    audit ordering match. The policy-specific actions and system feedback evolve
    independently after the initial state.
    """
    if not agents:
        raise ValueError("At least one trained agent is required.")

    clean_matrix = load_processed_matrix("data/processed/train_matrix.npy")
    test_matrix = load_processed_matrix("data/processed/test_matrix.npy")
    clean_baseline = compute_clean_baseline(clean_matrix, test_matrix, target_idx, k_neighbors)

    scenario_reference = None
    environments: Dict[str, ShillingDefenseEnv] = {}
    observations: Dict[str, np.ndarray] = {}
    total_rewards = {name: 0.0 for name in agents}

    # Reset every independent environment with the same seed and options.
    for agent_name, agent in agents.items():
        env = ShillingDefenseEnv(
            episode_length=episode_length,
            genuine_ratio=genuine_ratio,
            target_items=[target_idx],
            target_feature_mode=target_feature_mode,
            k_neighbors=k_neighbors,
            seed=seed,
        )
        observation, _ = env.reset(
            seed=seed,
            options={"target_item": target_idx, "attack_type": attack_type},
        )

        scenario = (
            env.active_matrix.copy(),
            tuple(env.queue_profile_indices),
            tuple(env.queue_is_attacker),
        )
        if scenario_reference is None:
            scenario_reference = scenario
        elif (
            not np.array_equal(scenario[0], scenario_reference[0])
            or scenario[1:] != scenario_reference[1:]
        ):
            raise RuntimeError("Agent environments did not receive the same attack scenario and profile stream.")

        environments[agent_name] = env
        observations[agent_name] = observation

    agent_names = list(agents)
    for step_index in range(episode_length):
        step_events: Dict[str, Dict[str, Any]] = {}
        for agent_name in agent_names:
            env = environments[agent_name]
            action = _predict_agent_action(agents[agent_name], observations[agent_name])
            observation, reward, terminated, truncated, info = env.step(action)
            observations[agent_name] = observation
            total_rewards[agent_name] += reward
            step_events[agent_name] = info["audit_event"]
            if truncated:
                raise RuntimeError("The shared-scenario episode was unexpectedly truncated.")
            if terminated != (step_index == episode_length - 1):
                raise RuntimeError(f"Agent {agent_name} terminated at an unexpected step.")
        if progress_callback is not None:
            progress_callback(step_index + 1, step_events)

    agent_results: Dict[str, Any] = {}
    for agent_name in agent_names:
        env = environments[agent_name]
        events = env.episode_events
        actions = np.array([event["action"] for event in events], dtype=int)
        labels = np.array([event["is_attacker"] for event in events], dtype=bool)
        intervention_metrics = _binary_classification_metrics(actions > 0, labels)
        block_metrics = _binary_classification_metrics(actions == 2, labels)

        target_impact = evaluate_target_item_impact(
            env.recommender, target_idx, clean_baseline["eval_users"], k=10
        )
        test_rmse = evaluate_test_rmse(env.recommender, test_matrix, max_eval_users=50)
        coverage = evaluate_catalog_coverage(
            env.recommender, clean_matrix.shape[1], clean_baseline["eval_users"][:50], k=10
        )
        genuine_topk = evaluate_genuine_top_k_metrics(
            env.recommender, test_matrix, rating_threshold=3.5, k=10, max_eval_users=50
        )
        genuine_actions = actions[~labels]
        attack_actions = actions[labels]

        agent_results[agent_name] = {
            "total_reward": float(total_rewards[agent_name]),
            "steps": len(events),
            "target_prediction": target_impact["avg_target_predicted_rating"],
            "clean_target_prediction": clean_baseline["clean_target_pred"],
            "target_prediction_shift": float(
                target_impact["avg_target_predicted_rating"] - clean_baseline["clean_target_pred"]
            ),
            "target_hit_ratio_at_10": target_impact["target_hit_ratio_at_k"],
            "clean_target_hit_ratio_at_10": clean_baseline["clean_hr10"],
            "test_rmse": float(test_rmse),
            "clean_test_rmse": clean_baseline["clean_rmse"],
            "catalog_coverage": float(coverage),
            "clean_catalog_coverage": clean_baseline["clean_coverage"],
            "genuine_hit_rate_at_10": genuine_topk["hit_rate@10"],
            "clean_genuine_hit_rate_at_10": clean_baseline["clean_genuine_hit_rate"],
            "intervention_metrics": intervention_metrics,
            "strict_block_metrics": block_metrics,
            "action_distribution": {
                "genuine_accept_pct": float(np.mean(genuine_actions == 0) * 100) if len(genuine_actions) else 0.0,
                "genuine_downweight_pct": float(np.mean(genuine_actions == 1) * 100) if len(genuine_actions) else 0.0,
                "genuine_block_pct": float(np.mean(genuine_actions == 2) * 100) if len(genuine_actions) else 0.0,
                "attack_accept_pct": float(np.mean(attack_actions == 0) * 100) if len(attack_actions) else 0.0,
                "attack_downweight_pct": float(np.mean(attack_actions == 1) * 100) if len(attack_actions) else 0.0,
                "attack_block_pct": float(np.mean(attack_actions == 2) * 100) if len(attack_actions) else 0.0,
            },
            "events": events,
        }

    return {
        "scenario": {
            "seed": int(seed),
            "target_index": int(target_idx),
            "attack_type": attack_type,
            "episode_length": int(episode_length),
            "genuine_ratio": float(genuine_ratio),
            "target_feature_mode": target_feature_mode,
            "num_genuine_in_stream": int(sum(not label for label in scenario_reference[2])),
            "num_attackers_in_stream": int(sum(scenario_reference[2])),
            "profile_stream": [
                {"profile_index": int(index), "is_attacker": bool(label)}
                for index, label in zip(scenario_reference[1], scenario_reference[2])
            ],
        },
        "agents": agent_results,
    }


def _shared_demo_checkpoint_paths(checkpoint_dir: Optional[str] = None) -> Dict[str, str]:
    checkpoint_dir = checkpoint_dir or os.path.join(
        PROJECT_ROOT, "results", "checkpoints", "shared_demo"
    )
    return {
        "directory": checkpoint_dir,
        "manifest": os.path.join(checkpoint_dir, "manifest.json"),
        "q_learning": os.path.join(checkpoint_dir, "q_learning.npy"),
        "q_learning_meta": os.path.join(checkpoint_dir, "q_learning_meta.json"),
        "dqn": os.path.join(checkpoint_dir, "dqn_policy.pth"),
        "ppo": os.path.join(checkpoint_dir, "ppo_policy.pth"),
    }


def _load_shared_demo_agents(
    paths: Dict[str, str],
    target_idx: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a complete trained-agent checkpoint set and its manifest."""
    with open(paths["manifest"], "r") as f:
        manifest = json.load(f)
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported shared-demo checkpoint format: {manifest.get('format_version')}")

    q_agent = QLearningAgent(
        learning_rate=0.1, gamma=0.95, epsilon=1.0,
        epsilon_decay=0.98, epsilon_min=0.05, seed=seed,
    )
    q_agent.load(paths["q_learning"])

    dqn_env = ShillingDefenseEnv(target_items=[target_idx], seed=seed)
    ppo_env = ShillingDefenseEnv(target_items=[target_idx], seed=seed)
    dqn_model = create_dqn_agent(
        dqn_env, learning_rate=1e-3, buffer_size=10000, learning_starts=100,
        batch_size=64, seed=seed, verbose=0,
    )
    ppo_model = create_ppo_agent(
        ppo_env, learning_rate=3e-4, n_steps=128, batch_size=32,
        n_epochs=5, seed=seed, verbose=0,
    )
    dqn_model.policy.load_state_dict(torch.load(paths["dqn"], map_location="cpu", weights_only=True))
    ppo_model.policy.load_state_dict(torch.load(paths["ppo"], map_location="cpu", weights_only=True))
    agents = {
        "Q-Learning": q_agent,
        "DQN": dqn_model,
        "PPO": ppo_model,
    }
    return agents, manifest


def _train_and_save_shared_demo_agents(
    paths: Dict[str, str],
    training_seed: int,
    target_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train all three policies once and persist a complete checkpoint set."""
    os.makedirs(paths["directory"], exist_ok=True)
    if os.path.exists(paths["manifest"]):
        os.remove(paths["manifest"])

    q_env = ShillingDefenseEnv(target_items=[target_idx], seed=training_seed)
    dqn_env = ShillingDefenseEnv(target_items=[target_idx], seed=training_seed)
    ppo_env = ShillingDefenseEnv(target_items=[target_idx], seed=training_seed)

    print(f"Training Q-Learning ({Q_LEARNING_EPISODES} episodes)...")
    q_agent = train_q_learning(q_env, seed=training_seed)
    q_agent.save(paths["q_learning"])

    print(f"Training DQN ({DQN_TIMESTEPS} timesteps)...")
    dqn_model = train_dqn_policy(dqn_env, seed=training_seed)
    torch.save(dqn_model.policy.state_dict(), paths["dqn"])

    print(f"Training PPO ({PPO_TIMESTEPS} timesteps)...")
    ppo_model = train_ppo_policy(ppo_env, seed=training_seed)
    torch.save(ppo_model.policy.state_dict(), paths["ppo"])

    manifest = {
        "format_version": 1,
        "training_seed": int(training_seed),
        "training_budget": {
            "q_learning_episodes": Q_LEARNING_EPISODES,
            "episode_length": 100,
            "dqn_timesteps": DQN_TIMESTEPS,
            "ppo_timesteps": PPO_TIMESTEPS,
        },
        "target_index_used_for_training": int(target_idx),
    }
    with open(paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)
    return {"Q-Learning": q_agent, "DQN": dqn_model, "PPO": ppo_model}, manifest


def load_or_train_shared_demo_agents(
    training_seed: int = 42,
    target_idx: int = 40,
    retrain: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reuse saved agents by default; train and save only if needed or requested."""
    paths = _shared_demo_checkpoint_paths(checkpoint_dir)
    required_files = [
        paths["manifest"], paths["q_learning"], paths["q_learning_meta"],
        paths["dqn"], paths["ppo"],
    ]
    checkpoint_complete = all(os.path.isfile(path) for path in required_files)

    if checkpoint_complete and not retrain:
        print(f"Loading trained agents from {paths['directory']}...")
        return _load_shared_demo_agents(paths, target_idx, training_seed)

    if not retrain and any(os.path.exists(path) for path in required_files):
        print("A partial shared-demo checkpoint was found; training a complete replacement set.")
    elif retrain:
        print("Retraining shared-demo agents as requested.")
    else:
        print("No trained shared-demo checkpoints found; training the agents now.")

    agents, manifest = _train_and_save_shared_demo_agents(paths, training_seed, target_idx)
    print(f"Trained agents saved to {paths['directory']}")
    return agents, manifest


def run_shared_scenario_demo(
    training_seed: int = 42,
    scenario_seed: Optional[int] = None,
    target_idx: int = 40,
    attack_type: str = "bandwagon",
    episode_length: int = 100,
    retrain_agents: bool = False,
) -> Dict[str, Any]:
    """Load or train the three RL agents, then compare on one shared scenario."""
    scenario_seed = training_seed + 10000 if scenario_seed is None else scenario_seed
    agents, training_manifest = load_or_train_shared_demo_agents(
        training_seed=training_seed,
        target_idx=target_idx,
        retrain=retrain_agents,
    )

    result = run_shared_scenario_comparison(
        agents,
        seed=scenario_seed,
        target_idx=target_idx,
        attack_type=attack_type,
        episode_length=episode_length,
    )
    result["training"] = training_manifest
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "shared_scenario_comparison.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Shared-scenario results saved to {output_path}")
    for agent_name, metrics in result["agents"].items():
        print(
            f"  {agent_name:12s} | target shift={metrics['target_prediction_shift']:+.4f} "
            f"| HR@10={metrics['target_hit_ratio_at_10']:.3f} "
            f"| intervention F1={metrics['intervention_metrics']['f1']:.3f}"
        )
    return result


# ===== FEATURE ABLATION SUPPORT =====

class FeatureMaskedEnv(gym.Wrapper):
    """Wrapper that zeros out specific feature indices for ablation studies."""

    def __init__(self, env, mask_indices: List[int]):
        super().__init__(env)
        self.mask_indices = mask_indices

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.copy()
        obs[self.mask_indices] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.copy()
        obs[self.mask_indices] = 0.0
        return obs, reward, terminated, truncated, info


# ===== AGGREGATION UTILITIES =====

def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    """Aggregate evaluation results across multiple seeds -> mean +/- std."""
    if len(seed_results) == 1:
        r = seed_results[0].copy()
        for key in [
            "abs_prediction_shift", "prediction_shift", "f1_score", "precision",
            "recall", "f1_block", "precision_block", "recall_block", "rmse_delta",
            "target_hit_ratio_at_10", "hit_ratio_shift", "catalog_coverage", "coverage_delta",
            "genuine_hit_rate_at_10"
        ]:
            r[f"{key}_std"] = 0.0
        dist = r.get("action_distribution", {})
        for dk in list(dist.keys()):
            dist[f"{dk}_std"] = 0.0
        return r

    agg = {}
    scalar_keys = [
        "prediction_shift", "abs_prediction_shift", "target_hit_ratio_at_10",
        "hit_ratio_shift", "test_rmse", "rmse_delta", "precision", "recall", "f1_score",
        "precision_block", "recall_block", "f1_block", "catalog_coverage", "coverage_delta",
        "genuine_hit_rate_at_10",
    ]
    for key in scalar_keys:
        vals = [r[key] for r in seed_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    dist_keys = [
        "genuine_accept_pct", "genuine_downweight_pct", "genuine_block_pct",
        "attack_accept_pct", "attack_downweight_pct", "attack_block_pct",
    ]
    agg["action_distribution"] = {}
    for dk in dist_keys:
        vals = [r["action_distribution"][dk] for r in seed_results]
        agg["action_distribution"][dk] = float(np.mean(vals))
        agg["action_distribution"][f"{dk}_std"] = float(np.std(vals))

    for key in [
        "agent", "attack_type", "attack_ratio", "clean_target_pred_rating",
        "clean_hit_ratio_at_10", "clean_rmse", "clean_catalog_coverage",
        "clean_genuine_hit_rate", "evaluation_mix", "target_feature_mode"
    ]:
        if key in seed_results[0]:
            agg[key] = seed_results[0][key]

    agg["num_seeds"] = len(seed_results)
    return agg


# ===== MAIN BENCHMARK =====

def run_full_comparative_benchmark(num_seeds: int = 3):
    """
    Rigorously audited comparative benchmark.

    Fixes applied:
    1. Multi-seed training AND multi-seed evaluation (eval_seed=seed so attack injection has true variance).
    2. Comprehensive metrics: Genuine Test RMSE delta, Catalog Coverage delta, Genuine HitRate@10.
    3. Dual Precision: Intervention (Actions >= 1) vs Strict Block (Action == 2).
    4. Explicit evaluation mix: 943 genuine (92.6%) vs 75 attackers (7.4%).
    """
    seeds = TRAINING_SEEDS[:num_seeds]

    print("=" * 120)
    print("RecShield-RL: Rigorous Comparative Benchmark")
    print("=" * 120)
    print(f"  Training & Evaluation seeds: {seeds}")
    print(f"  Q-Learning budget:           {Q_LEARNING_EPISODES} episodes x 100 steps = {Q_LEARNING_EPISODES * 100} timesteps")
    print(f"  DQN budget:                  {DQN_TIMESTEPS} timesteps")
    print(f"  PPO budget:                  {PPO_TIMESTEPS} timesteps")
    print(f"  Eval attack ratio:           {EVAL_ATTACK_RATIO * 100:.0f}% (75 attackers injected into 943 genuine users)")
    print(f"  Training stream:             70% genuine / 30% attacker per episode")
    print(f"  Targets:                     {[TARGET_NAMES[t] for t in TARGET_ITEMS]}")
    print(f"  Attacks:                     {ATTACK_TYPES}")
    print("=" * 120)

    clean_matrix = load_processed_matrix("data/processed/train_matrix.npy")
    test_matrix = load_processed_matrix("data/processed/test_matrix.npy")

    # Precompute clean baseline per target once
    print("\n  Precomputing clean CF baselines for all target items...")
    clean_baselines = {}
    for target_idx in TARGET_ITEMS:
        clean_baselines[target_idx] = compute_clean_baseline(clean_matrix, test_matrix, target_idx, K_NEIGHBORS)
    print("  Clean baselines ready.")

    all_results: Dict[str, Dict[str, Dict[str, List]]] = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 70}")
        print(f"  SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'=' * 70}")

        env = ShillingDefenseEnv(episode_length=100, num_eval_users=25, seed=seed)

        # Train agents with equal budget
        print(f"  [1/3] Training Q-Learning ({Q_LEARNING_EPISODES} episodes)...", end=" ")
        t0 = time.time()
        q_agent = train_q_learning(env, num_episodes=Q_LEARNING_EPISODES, seed=seed)
        print(f"Done in {time.time() - t0:.1f}s")

        print(f"  [2/3] Training DQN ({DQN_TIMESTEPS} timesteps)...", end=" ")
        t0 = time.time()
        dqn_model = train_dqn_policy(env, total_timesteps=DQN_TIMESTEPS, seed=seed)
        print(f"Done in {time.time() - t0:.1f}s")

        print(f"  [3/3] Training PPO ({PPO_TIMESTEPS} timesteps)...", end=" ")
        t0 = time.time()
        ppo_model = train_ppo_policy(env, total_timesteps=PPO_TIMESTEPS, seed=seed)
        print(f"Done in {time.time() - t0:.1f}s")

        rng = np.random.RandomState(seed)

        def policy_no_defense(feat, idx, mat):
            return 0

        def policy_random(feat, idx, mat):
            return int(rng.randint(0, 3))

        def policy_heuristic(feat, idx, mat):
            f_rdma, f_degsim, f_act, f_target, _ = feat
            if f_target > 0.3 and (f_rdma > 0.4 or f_degsim > 0.7):
                return 2  # BLOCK
            elif f_target > 0.3:
                return 1  # DOWNWEIGHT
            return 0

        def policy_q(feat, idx, mat):
            return q_agent.select_action(feat, training=False)

        def policy_dqn(feat, idx, mat):
            action, _ = dqn_model.predict(feat, deterministic=True)
            return int(action)

        def policy_ppo(feat, idx, mat):
            action, _ = ppo_model.predict(feat, deterministic=True)
            return int(action)

        policies = {
            "No Defense": policy_no_defense,
            "Random Agent": policy_random,
            "Static Heuristic": policy_heuristic,
            "Q-Learning": policy_q,
            "DQN": policy_dqn,
            "PPO": policy_ppo,
        }

        print(f"\n  Evaluating {len(policies)} policies x {len(ATTACK_TYPES)} attacks x {len(TARGET_ITEMS)} targets...")

        for attack_type in ATTACK_TYPES:
            if attack_type not in all_results:
                all_results[attack_type] = {}
            for target_idx in TARGET_ITEMS:
                tkey = f"target_{target_idx}"
                if tkey not in all_results[attack_type]:
                    all_results[attack_type][tkey] = {}
                for pname, pfn in policies.items():
                    if pname not in all_results[attack_type][tkey]:
                        all_results[attack_type][tkey][pname] = []

                    res = evaluate_agent_on_attack(
                        agent_name=pname, policy_fn=pfn,
                        clean_matrix=clean_matrix, test_matrix=test_matrix,
                        target_idx=target_idx, attack_type=attack_type,
                        clean_baseline=clean_baselines[target_idx],
                        attack_ratio=EVAL_ATTACK_RATIO,
                        eval_seed=seed,  # Evaluates with distinct attack injection per seed!
                    )
                    all_results[attack_type][tkey][pname].append(res)

    policy_order = ["No Defense", "Random Agent", "Static Heuristic",
                    "Q-Learning", "DQN", "PPO"]

    print("\n\n" + "=" * 140)
    print(f"  AGGREGATED RESULTS -- mean +/- std across {len(seeds)} seed(s)")
    print(f"  Eval: 8% injection (75 attackers, 943 genuine users)  |  Training budget: 15,000 steps each")
    print("=" * 140)

    aggregated_results: Dict = {}
    show_std = len(seeds) > 1

    for attack_type in ATTACK_TYPES:
        aggregated_results[attack_type] = {}
        print(f"\n  >>> {attack_type.upper()} PUSH ATTACK")

        for target_idx in TARGET_ITEMS:
            tkey = f"target_{target_idx}"
            aggregated_results[attack_type][tkey] = {}
            print(f"\n    Target: {TARGET_NAMES[target_idx]}")

            hdr = (f"    {'Policy':20s} | {'|dPred| v':14s} | {'HR@10':10s} | "
                   f"{'RMSE d':12s} | {'Cov d':10s} | {'F1 (Int)':10s} | {'Prec (Int)':10s} | "
                   f"{'F1 (Blk)':10s} | {'GenBlk%':8s} | {'GenDW%':8s}")
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            for pname in policy_order:
                seed_res = all_results[attack_type][tkey][pname]
                agg = aggregate_seed_results(seed_res)
                aggregated_results[attack_type][tkey][pname] = agg

                def fmt(key):
                    v = agg[key]
                    if show_std:
                        s = agg.get(f"{key}_std", 0)
                        return f"{v:.4f}+/-{s:.4f}"
                    return f"{v:.4f}"

                def fmt3(key):
                    v = agg[key]
                    if show_std:
                        s = agg.get(f"{key}_std", 0)
                        return f"{v:.3f}+/-{s:.3f}"
                    return f"{v:.3f}"

                gb = agg["action_distribution"]["genuine_block_pct"]
                gd = agg["action_distribution"]["genuine_downweight_pct"]

                print(f"    {pname:20s} | {fmt('abs_prediction_shift'):>14s} | "
                      f"{fmt('target_hit_ratio_at_10'):>10s} | "
                      f"{fmt('rmse_delta'):>12s} | "
                      f"{fmt('coverage_delta'):>10s} | "
                      f"{fmt3('f1_score'):>10s} | {fmt3('precision'):>10s} | "
                      f"{fmt3('f1_block'):>10s} | {gb:6.1f}% | {gd:6.1f}%")

    output_path = "results/comparative_benchmark_results.json"
    os.makedirs("results", exist_ok=True)

    json_out = {}
    for at in aggregated_results:
        json_out[at] = {}
        for tk in aggregated_results[at]:
            json_out[at][tk] = {}
            for pn in aggregated_results[at][tk]:
                r = aggregated_results[at][tk][pn]
                json_out[at][tk][pn] = {
                    k: v for k, v in r.items() if isinstance(v, (int, float, str, dict, list))
                }

    with open(output_path, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\n{'=' * 140}")
    print(f"  [DONE] Results saved to {output_path}")
    print(f"{'=' * 140}")
    return aggregated_results


# ===== MULTI-SEED ABLATION & GENERALIZATION STUDY =====

def run_ablation_study(num_seeds: int = 3, num_episodes: int = Q_LEARNING_EPISODES):
    """
    Multi-seed feature ablation study with Q-Learning and DQN on Bandwagon Attack.

    Evaluates:
      1. Full (all 5)         - standard oracle monitoring
      2. Target-Agnostic (f4) - max positive deviation across all items (no target identity needed)
      3. Unmonitored Target   - f4 = 0 (attack targeting unmonitored item)
      4. No SysShift (-f5)    - mask feature index 4 (tests impact of 243 -> 81 state compression)
      5. No f4 + No f5        - mask indices 3 and 4
    """
    seeds = TRAINING_SEEDS[:num_seeds]
    print("\n" + "=" * 120)
    print(f"  MULTI-SEED ABLATION STUDY: Feature Importance & Generalization across {len(seeds)} Seeds")
    print("=" * 120)

    clean_matrix = load_processed_matrix("data/processed/train_matrix.npy")
    test_matrix = load_processed_matrix("data/processed/test_matrix.npy")

    clean_baselines = {}
    for target_idx in TARGET_ITEMS:
        clean_baselines[target_idx] = compute_clean_baseline(clean_matrix, test_matrix, target_idx, K_NEIGHBORS)

    configs = {
        "Full (all 5)":           {"mask": [],     "mode": "monitored"},
        "Target-Agnostic (f4)":   {"mask": [],     "mode": "target_agnostic"},
        "Unmonitored Target":     {"mask": [],     "mode": "unmonitored"},
        "No SysShift (-f5)":      {"mask": [4],    "mode": "monitored"},
        "No f4 + No f5":          {"mask": [3, 4], "mode": "monitored"},
    }

    # Structure: {cfg_name: {target_key: [seed_results]}}
    raw_ablation: Dict[str, Dict[str, List]] = {c: {f"target_{t}": [] for t in TARGET_ITEMS} for c in configs}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  [Seed {seed_idx + 1}/{len(seeds)}: {seed}]")
        for cfg_name, cfg in configs.items():
            mask_idx = cfg["mask"]
            mode = cfg["mode"]

            base_env = ShillingDefenseEnv(
                episode_length=100, num_eval_users=25,
                target_feature_mode=mode, seed=seed
            )
            train_env = FeatureMaskedEnv(base_env, mask_idx) if mask_idx else base_env

            agent = QLearningAgent(
                learning_rate=0.1, gamma=0.95, epsilon=1.0,
                epsilon_decay=0.98, epsilon_min=0.05, seed=seed
            )
            t0 = time.time()
            for ep in range(num_episodes):
                obs, _ = train_env.reset(seed=seed + ep)
                done = False
                while not done:
                    action = agent.select_action(obs, training=True)
                    next_obs, reward, terminated, truncated, _ = train_env.step(action)
                    done = terminated or truncated
                    agent.update(obs, action, reward, next_obs, done)
                    obs = next_obs
                agent.decay_epsilon()

            def _make_policy(ag, mi):
                def _policy(feat, idx, mat):
                    f = feat.copy()
                    if mi:
                        f[mi] = 0.0
                    return ag.select_action(f, training=False)
                return _policy

            policy_fn = _make_policy(agent, mask_idx)

            for target_idx in TARGET_ITEMS:
                res = evaluate_agent_on_attack(
                    agent_name=cfg_name, policy_fn=policy_fn,
                    clean_matrix=clean_matrix, test_matrix=test_matrix,
                    target_idx=target_idx, attack_type="bandwagon",
                    clean_baseline=clean_baselines[target_idx],
                    attack_ratio=EVAL_ATTACK_RATIO,
                    eval_seed=seed,
                    target_feature_mode=mode
                )
                raw_ablation[cfg_name][f"target_{target_idx}"].append(res)

    # Aggregate
    aggregated_ablation: Dict = {}
    print(f"\n{'=' * 120}")
    print("  ABLATION RESULTS (Bandwagon Attack, mean +/- std)")
    print(f"{'=' * 120}")

    hdr = (f"    {'Config':25s} | {'Target':20s} | {'|dPred| v':14s} | "
           f"{'F1 (Int)':12s} | {'Prec (Int)':12s} | {'F1 (Blk)':12s} | {'GenBlk%':8s} | {'GenDW%':8s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for cfg_name in configs:
        aggregated_ablation[cfg_name] = {}
        for target_idx in TARGET_ITEMS:
            tkey = f"target_{target_idx}"
            agg = aggregate_seed_results(raw_ablation[cfg_name][tkey])
            aggregated_ablation[cfg_name][tkey] = agg

            s_shift = f"{agg['abs_prediction_shift']:.4f}+/-{agg.get('abs_prediction_shift_std', 0):.4f}"
            s_f1 = f"{agg['f1_score']:.3f}+/-{agg.get('f1_score_std', 0):.3f}"
            s_prec = f"{agg['precision']:.3f}+/-{agg.get('precision_std', 0):.3f}"
            s_f1_blk = f"{agg['f1_block']:.3f}+/-{agg.get('f1_block_std', 0):.3f}"
            gb = agg["action_distribution"]["genuine_block_pct"]
            gd = agg["action_distribution"]["genuine_downweight_pct"]

            print(f"    {cfg_name:25s} | {TARGET_NAMES[target_idx]:20s} | "
                  f"{s_shift:>14s} | {s_f1:>12s} | {s_prec:>12s} | {s_f1_blk:>12s} | "
                  f"{gb:6.1f}% | {gd:6.1f}%")

    output_path = "results/ablation_results.json"
    serializable = {}
    for cn in aggregated_ablation:
        serializable[cn] = {}
        for tk in aggregated_ablation[cn]:
            r = aggregated_ablation[cn][tk]
            serializable[cn][tk] = {
                k: v for k, v in r.items() if isinstance(v, (int, float, str, dict, list))
            }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  [DONE] Multi-seed ablation results saved to {output_path}")
    return aggregated_ablation


# ===== UNSEEN TARGET GENERALIZATION EXPERIMENT =====

def run_unseen_target_generalization(num_seeds: int = 3):
    """
    Test generalization to an UNSEEN target movie.
    Train on targets [40, 77] (Billy Madison, Free Willy).
    Evaluate on target [15] (French Twist) - completely held out from training!
    """
    seeds = TRAINING_SEEDS[:num_seeds]
    print("\n" + "=" * 120)
    print("  UNSEEN TARGET GENERALIZATION EXPERIMENT")
    print("  Trained on Targets: [40, 77]  -->  Evaluated on UNSEEN Target: 15 (French Twist)")
    print("=" * 120)

    clean_matrix = load_processed_matrix("data/processed/train_matrix.npy")
    test_matrix = load_processed_matrix("data/processed/test_matrix.npy")
    clean_baseline = compute_clean_baseline(clean_matrix, test_matrix, 15, K_NEIGHBORS)

    eval_modes = {
        "Oracle Target (Monitored)": "monitored",
        "Target-Agnostic (Max Dev)": "target_agnostic",
        "Unmonitored Target (f4=0)": "unmonitored"
    }

    gen_results = {}

    for mode_name, feat_mode in eval_modes.items():
        seed_res = []
        for seed in seeds:
            # Train only on targets [40, 77]
            train_env = ShillingDefenseEnv(
                episode_length=100, num_eval_users=25,
                target_items=[40, 77],
                target_feature_mode=feat_mode,
                seed=seed
            )
            q_ag = train_q_learning(train_env, num_episodes=Q_LEARNING_EPISODES, seed=seed)

            res = evaluate_agent_on_attack(
                agent_name=f"Q-Learning ({mode_name})",
                policy_fn=lambda feat, idx, mat: q_ag.select_action(feat, training=False),
                clean_matrix=clean_matrix, test_matrix=test_matrix,
                target_idx=15,  # Unseen target!
                attack_type="bandwagon",
                clean_baseline=clean_baseline,
                attack_ratio=EVAL_ATTACK_RATIO,
                eval_seed=seed,
                target_feature_mode=feat_mode
            )
            seed_res.append(res)

        agg = aggregate_seed_results(seed_res)
        gen_results[mode_name] = agg

        s_shift = f"{agg['abs_prediction_shift']:.4f}+/-{agg.get('abs_prediction_shift_std', 0):.4f}"
        s_f1 = f"{agg['f1_score']:.3f}+/-{agg.get('f1_score_std', 0):.3f}"
        s_prec = f"{agg['precision']:.3f}+/-{agg.get('precision_std', 0):.3f}"
        gb = agg["action_distribution"]["genuine_block_pct"]
        gd = agg["action_distribution"]["genuine_downweight_pct"]

        print(f"    {mode_name:30s} | |dPred|: {s_shift:>14s} | F1: {s_f1:>10s} | "
              f"Prec: {s_prec:>10s} | GenBlk: {gb:5.1f}% | GenDW: {gd:5.1f}%")

    output_path = "results/unseen_target_generalization.json"
    with open(output_path, "w") as f:
        json.dump({k: {k2: v2 for k2, v2 in v.items() if isinstance(v2, (int, float, str, dict, list))}
                   for k, v in gen_results.items()}, f, indent=2)
    print(f"\n  [DONE] Unseen target generalization saved to {output_path}")
    return gen_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecShield-RL Rigorous Benchmark")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--ablation", action="store_true", help="Include multi-seed ablation")
    parser.add_argument("--generalization", action="store_true", help="Include unseen target generalization test")
    parser.add_argument("--all", action="store_true", help="Run benchmark + ablation + generalization")
    parser.add_argument(
        "--shared-scenario",
        action="store_true",
        help="Train Q-Learning, DQN, and PPO, then compare them on one identical attack episode",
    )
    parser.add_argument("--training-seed", type=int, default=42, help="Seed for training the shared-scenario agents")
    parser.add_argument("--scenario-seed", type=int, help="Scenario seed (defaults to training seed + 10000)")
    parser.add_argument("--target-index", type=int, choices=TARGET_ITEMS, default=40, help="Zero-based target item index")
    parser.add_argument("--attack-type", choices=ATTACK_TYPES, default="bandwagon")
    parser.add_argument(
        "--retrain-agents",
        action="store_true",
        help="Force retraining and overwrite the saved shared-demo agent checkpoints",
    )
    args = parser.parse_args()

    if args.shared_scenario:
        run_shared_scenario_demo(
            training_seed=args.training_seed,
            scenario_seed=args.scenario_seed,
            target_idx=args.target_index,
            attack_type=args.attack_type,
            retrain_agents=args.retrain_agents,
        )
    else:
        if args.retrain_agents:
            parser.error("--retrain-agents can only be used with --shared-scenario")
        run_full_comparative_benchmark(num_seeds=args.seeds)

        if args.ablation or args.all:
            run_ablation_study(num_seeds=args.seeds)

        if args.generalization or args.all:
            run_unseen_target_generalization(num_seeds=args.seeds)
