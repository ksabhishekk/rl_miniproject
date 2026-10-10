import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recommendation.data_preprocessing import load_movies, load_processed_matrix
from recommendation.recommender import UserBasedCF
from attacks.attack_simulator import ShillingAttackSimulator
from recommendation.evaluation import evaluate_target_item_impact

def run_attack_experiments():
    train_matrix = load_processed_matrix("data/processed/train_matrix.npy")
    movies_df = load_movies()

    # We evaluate across 3 representative target movies:
    # 1. Item 41: "Billy Madison (1995)" (modest comedy, ratings count ~30)
    # 2. Item 78: "Free Willy (1993)" (family film, ratings count ~25)
    # 3. Item 16: "French Twist (1995)" (foreign comedy, ratings count ~36)
    target_items = [40, 77, 15]
    
    results = {}
    eval_users = list(range(120))  # genuine user sample for fast, precise evaluation

    for target_idx in target_items:
        item_id = target_idx + 1
        title = movies_df[movies_df["item_id"] == item_id]["title"].values[0]
        print(f"\n==========================================")
        print(f"Target Item: {title} (ID {item_id}, index {target_idx})")
        print(f"==========================================")

        unrated_eval_users = [u for u in eval_users if train_matrix[u, target_idx] == 0]

        # 1. Baseline Clean CF
        cf_clean = UserBasedCF(k_neighbors=30)
        cf_clean.fit(train_matrix)
        clean_impact = evaluate_target_item_impact(cf_clean, target_idx, unrated_eval_users, k=10)
        print(f"Baseline Clean: Predicted Rating = {clean_impact['avg_target_predicted_rating']:.4f}, "
              f"HitRatio@10 = {clean_impact['target_hit_ratio_at_k']:.4f}")

        sim = ShillingAttackSimulator(
            train_matrix,
            target_item_idx=target_idx,
            filler_size=0.08,
            random_seed=42
        )

        item_results = {
            "title": title,
            "item_id": int(item_id),
            "clean_baseline": clean_impact,
            "attacks": {}
        }

        for attack_type in ["random", "average", "bandwagon"]:
            item_results["attacks"][attack_type] = {}
            for rate in [0.03, 0.05, 0.08]:
                attacked_matrix, labels, fake_profiles = sim.simulate_attack(
                    attack_type=attack_type,
                    attack_size_ratio=rate
                )
                cf_att = UserBasedCF(k_neighbors=30)
                cf_att.fit(attacked_matrix)
                att_impact = evaluate_target_item_impact(cf_att, target_idx, unrated_eval_users, k=10)

                pred_shift = att_impact["avg_target_predicted_rating"] - clean_impact["avg_target_predicted_rating"]
                hr_shift = att_impact["target_hit_ratio_at_k"] - clean_impact["target_hit_ratio_at_k"]

                item_results["attacks"][attack_type][f"{int(rate*100)}%"] = {
                    "num_fake_profiles": int(len(fake_profiles)),
                    "attacked_predicted_rating": att_impact["avg_target_predicted_rating"],
                    "prediction_shift": pred_shift,
                    "attacked_hit_ratio_at_10": att_impact["target_hit_ratio_at_k"],
                    "hit_ratio_shift": hr_shift
                }
                print(f"  [{attack_type.capitalize():9s} @ {int(rate*100)}%]: "
                      f"Pred = {att_impact['avg_target_predicted_rating']:.4f} "
                      f"(Shift: +{pred_shift:+.4f}), "
                      f"HR@10 = {att_impact['target_hit_ratio_at_k']:.4f} "
                      f"(Shift: +{hr_shift:+.4f})")

        results[f"item_{item_id}"] = item_results

    os.makedirs("results", exist_ok=True)
    with open("results/shilling_attack_impact_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[SUCCESS] All attack experiments completed. Results saved to results/shilling_attack_impact_results.json")

if __name__ == "__main__":
    run_attack_experiments()
