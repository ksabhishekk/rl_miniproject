import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.shilling_defense_env import ShillingDefenseEnv
from recommendation.data_preprocessing import load_processed_matrix


def test_processed_matrix_is_cached():
    first = load_processed_matrix("data/processed/train_matrix.npy")
    second = load_processed_matrix("data/processed/train_matrix.npy")

    assert first is second
    assert not first.flags.writeable


def test_environment_loop():
    print("Initializing ShillingDefenseEnv...")
    env = ShillingDefenseEnv(episode_length=20, num_eval_users=15, seed=42)

    obs, info = env.reset(seed=42)
    print(f"[RESET] Initial observation: {obs}")
    print(f"[RESET] Info: {info}")

    assert obs.shape == (5,), f"Expected shape (5,), got {obs.shape}"
    assert np.all((obs >= 0.0) & (obs <= 1.0)), f"Observation values out of [0, 1] range: {obs}"
    assert env.episode_events == [], "A reset should start with an empty audit event log"

    discrete_state = ShillingDefenseEnv.discretize_state(obs)
    print(f"[DISCRETIZE] Discrete state index (0..242): {discrete_state}")
    assert 0 <= discrete_state < 243, f"Discrete state {discrete_state} out of bounds"

    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()  # random action in {0, 1, 2}
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        event = step_info["audit_event"]
        assert event == env.episode_events[-1], "Step info and episode event log should match"
        json.dumps(event)
        assert event["action"] == action
        assert event["defense_weight"] == env.action_to_weight[action]
        assert event["profile_source"] in {"movielens_genuine", "synthetic_attack"}
        assert event["ratings"], "The recorded profile should include its rated items"
        assert len(event["features"]) == 5
        assert event["target_prediction_after"] == step_info["target_prediction_after"]

        print(f"Step {step+1:02d}: Action={action} ({['ACCEPT', 'DOWNWEIGHT', 'BLOCK'][action]:10s}) | "
              f"Attacker={str(step_info['is_attacker']):5s} | "
              f"Reward={reward:+5.2f} | "
              f"Shift={step_info['target_pred_shift']:.4f}")

        if terminated:
            print(f"[EPISODE COMPLETED] Total Reward: {total_reward:.2f}")
            break

    print("\n[SUCCESS] Environment validation passed cleanly!")


if __name__ == "__main__":
    test_environment_loop()
