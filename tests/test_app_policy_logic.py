import unittest

import numpy as np

from app import (
    append_run_history,
    calculate_stream_counts,
    get_comparison_mode_details,
    should_auto_refresh,
)
from experiments.run_comparative_benchmark import _binary_classification_metrics


class ShouldAutoRefreshTests(unittest.TestCase):
    def test_should_auto_refresh_when_scenario_changes(self):
        last_signature = {
            "target_idx": 40,
            "attack_type": "bandwagon",
            "attack_share_pct": 30,
            "episode_length": 100,
            "scenario_seed": 10042,
        }
        next_signature = {
            "target_idx": 40,
            "attack_type": "random",
            "attack_share_pct": 30,
            "episode_length": 100,
            "scenario_seed": 10042,
        }

        self.assertTrue(
            should_auto_refresh(
                last_signature=last_signature,
                current_signature=next_signature,
                agents={"Q-Learning": object()},
                loaded_target_idx=40,
                target_idx=40,
            )
        )

    def test_should_not_auto_refresh_when_signature_is_unchanged(self):
        signature = {
            "target_idx": 77,
            "attack_type": "average",
            "attack_share_pct": 20,
            "episode_length": 50,
            "scenario_seed": 42,
            "comparison_mode": "shared_scenario",
        }

        self.assertFalse(
            should_auto_refresh(
                last_signature=signature,
                current_signature=signature,
                agents={"Q-Learning": object()},
                loaded_target_idx=77,
                target_idx=77,
            )
        )

    def test_should_auto_refresh_when_comparison_mode_changes(self):
        last_signature = {
            "target_idx": 40,
            "attack_type": "bandwagon",
            "attack_share_pct": 30,
            "episode_length": 100,
            "scenario_seed": 10042,
            "comparison_mode": "shared_scenario",
        }
        next_signature = {
            "target_idx": 40,
            "attack_type": "bandwagon",
            "attack_share_pct": 30,
            "episode_length": 100,
            "scenario_seed": 10042,
            "comparison_mode": "independent_runs",
        }

        self.assertTrue(
            should_auto_refresh(
                last_signature=last_signature,
                current_signature=next_signature,
                agents={"Q-Learning": object()},
                loaded_target_idx=40,
                target_idx=40,
            )
        )

    def test_comparison_mode_details_explain_shared_scenario_safely(self):
        details = get_comparison_mode_details("shared_scenario")

        self.assertEqual(details["label"], "Shared-scenario replay")
        self.assertIn("same generated stream", details["description"].lower())
        self.assertIn("same stream", details["interpretation"].lower())
        self.assertTrue(details["shared_scenario"])

    def test_calculate_stream_counts_respects_custom_episode_length(self):
        self.assertEqual(calculate_stream_counts(150, 30), (45, 105))
        self.assertEqual(calculate_stream_counts(200, 10), (20, 180))

    def test_binary_classification_metrics_exposes_false_positive_rate(self):
        predicted = np.array([1, 1, 0, 0], dtype=bool)
        actual = np.array([1, 0, 1, 0], dtype=bool)
        metrics = _binary_classification_metrics(predicted, actual)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)

    def test_append_run_history_increments_run_id(self):
        import streamlit as st

        st.session_state.clear()
        result = {
            "scenario": {
                "num_attackers_in_stream": 5,
                "num_genuine_in_stream": 15,
            },
            "agents": {
                "Q-Learning": {
                    "total_reward": 10.0,
                    "target_prediction_shift": 0.5,
                    "target_hit_ratio_at_10": 0.2,
                    "clean_target_hit_ratio_at_10": 0.1,
                    "test_rmse": 0.8,
                    "clean_test_rmse": 0.7,
                    "catalog_coverage": 0.4,
                    "clean_catalog_coverage": 0.3,
                    "genuine_hit_rate_at_10": 0.5,
                    "clean_genuine_hit_rate_at_10": 0.4,
                    "strict_block_metrics": {"recall": 1.0, "false_positive_rate": 0.1},
                },
                "DQN": {
                    "total_reward": 11.0,
                    "target_prediction_shift": 0.4,
                    "target_hit_ratio_at_10": 0.2,
                    "clean_target_hit_ratio_at_10": 0.1,
                    "test_rmse": 0.8,
                    "clean_test_rmse": 0.7,
                    "catalog_coverage": 0.4,
                    "clean_catalog_coverage": 0.3,
                    "genuine_hit_rate_at_10": 0.5,
                    "clean_genuine_hit_rate_at_10": 0.4,
                    "strict_block_metrics": {"recall": 1.0, "false_positive_rate": 0.1},
                },
                "PPO": {
                    "total_reward": 12.0,
                    "target_prediction_shift": 0.3,
                    "target_hit_ratio_at_10": 0.2,
                    "clean_target_hit_ratio_at_10": 0.1,
                    "test_rmse": 0.8,
                    "clean_test_rmse": 0.7,
                    "catalog_coverage": 0.4,
                    "clean_catalog_coverage": 0.3,
                    "genuine_hit_rate_at_10": 0.5,
                    "clean_genuine_hit_rate_at_10": 0.4,
                    "strict_block_metrics": {"recall": 1.0, "false_positive_rate": 0.1},
                },
            },
        }
        scenario = {
            "target_idx": 40,
            "attack_type": "bandwagon",
            "attack_share_pct": 30,
            "episode_length": 90,
            "scenario_seed": 10042,
            "comparison_mode": "shared_scenario",
        }

        first = append_run_history(result, scenario)
        second = append_run_history(result, scenario)

        self.assertEqual(first["run_id"], 1)
        self.assertEqual(second["run_id"], 2)
        self.assertEqual(len(st.session_state["run_history"]), 2)


if __name__ == "__main__":
    unittest.main()
