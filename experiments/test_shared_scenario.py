import json
import os
import sys
import tempfile
from unittest.mock import patch
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_agent import create_dqn_agent
from agents.ppo_agent import create_ppo_agent
from agents.q_learning_agent import QLearningAgent
from environment.shilling_defense_env import ShillingDefenseEnv
from experiments.run_comparative_benchmark import (
    _shared_demo_checkpoint_paths,
    load_or_train_shared_demo_agents,
    run_shared_scenario_comparison,
)


class FixedActionAgent:
    def __init__(self, action):
        self.action = action

    def select_action(self, observation, training=False):
        return self.action


class FixedPredictionAgent:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic=True):
        return self.action, None


def test_agents_share_profile_stream_and_keep_independent_actions():
    result = run_shared_scenario_comparison(
        {
            "accept": FixedActionAgent(0),
            "downweight": FixedPredictionAgent(1),
            "block": FixedPredictionAgent(2),
        },
        seed=142,
        target_idx=40,
        attack_type="bandwagon",
        episode_length=10,
    )

    stream = result["scenario"]["profile_stream"]
    assert len(stream) == 10
    for name, expected_action in (("accept", 0), ("downweight", 1), ("block", 2)):
        events = result["agents"][name]["events"]
        assert len(events) == len(stream)
        assert [event["profile_index"] for event in events] == [
            entry["profile_index"] for entry in stream
        ]
        assert [event["is_attacker"] for event in events] == [
            entry["is_attacker"] for entry in stream
        ]
        assert {event["action"] for event in events} == {expected_action}
    json.dumps(result)


def test_saved_demo_agents_load_without_retraining():
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        paths = _shared_demo_checkpoint_paths(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        env = ShillingDefenseEnv(episode_length=10, target_items=[40], seed=19)

        q_agent = QLearningAgent(seed=19)
        q_agent.save(paths["q_learning"])
        dqn_model = create_dqn_agent(
            env, learning_starts=1, buffer_size=32, batch_size=8, seed=19
        )
        ppo_model = create_ppo_agent(
            env, n_steps=8, batch_size=8, n_epochs=1, seed=19
        )
        expected_dqn_state = {
            name: tensor.clone() for name, tensor in dqn_model.policy.state_dict().items()
        }
        expected_ppo_state = {
            name: tensor.clone() for name, tensor in ppo_model.policy.state_dict().items()
        }
        torch.save(dqn_model.policy.state_dict(), paths["dqn"])
        torch.save(ppo_model.policy.state_dict(), paths["ppo"])
        manifest = {
            "format_version": 1,
            "training_seed": 19,
            "training_budget": {},
            "target_index_used_for_training": 40,
        }
        with open(paths["manifest"], "w") as manifest_file:
            json.dump(manifest, manifest_file)

        with patch("experiments.run_comparative_benchmark.train_q_learning") as train_q, \
                patch("experiments.run_comparative_benchmark.train_dqn_policy") as train_dqn, \
                patch("experiments.run_comparative_benchmark.train_ppo_policy") as train_ppo:
            agents, loaded_manifest = load_or_train_shared_demo_agents(
                training_seed=19,
                target_idx=40,
                checkpoint_dir=checkpoint_dir,
            )

        assert set(agents) == {"Q-Learning", "DQN", "PPO"}
        assert loaded_manifest == manifest
        train_q.assert_not_called()
        train_dqn.assert_not_called()
        train_ppo.assert_not_called()
        assert agents["Q-Learning"].select_action([0.0] * 5, training=False) in (0, 1, 2)
        for agent_name, expected_state in (
            ("DQN", expected_dqn_state),
            ("PPO", expected_ppo_state),
        ):
            loaded_state = agents[agent_name].policy.state_dict()
            assert all(torch.equal(expected_state[name], loaded_state[name]) for name in expected_state)
