import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from recommendation.recommender import UserBasedCF


def evaluate_rating_prediction(recommender: UserBasedCF, test_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate RMSE and MAE on held-out test ratings.
    """
    errors = []
    abs_errors = []

    # Cache predicted ratings per user to avoid recomputation
    cached_preds = {}

    for row in test_df.itertuples():
        u_idx = int(row.user_id) - 1
        i_idx = int(row.item_id) - 1
        actual = float(row.rating)

        if u_idx not in cached_preds:
            cached_preds[u_idx] = recommender.predict_user_ratings(u_idx)

        predicted = float(cached_preds[u_idx][i_idx])
        diff = actual - predicted
        errors.append(diff ** 2)
        abs_errors.append(abs(diff))

    rmse = float(np.sqrt(np.mean(errors)))
    mae = float(np.mean(abs_errors))
    return {"rmse": rmse, "mae": mae}


def evaluate_top_k_recommendations(
    recommender: UserBasedCF,
    test_df: pd.DataFrame,
    k: int = 10,
    rating_threshold: float = 3.5,
    max_eval_users: int = 200
) -> Dict[str, float]:
    """
    Evaluate ranking metrics: Precision@K, Recall@K, and Hit Rate@K.
    An item in the test set is considered relevant if its rating >= rating_threshold.
    """
    precisions = []
    recalls = []
    hits = []

    user_groups = test_df.groupby("user_id")
    eval_count = 0

    for user_id, group in user_groups:
        u_idx = int(user_id) - 1
        # Ground-truth relevant items in test set
        relevant_items = set(group[group["rating"] >= rating_threshold]["item_id"] - 1)
        if len(relevant_items) == 0:
            continue

        recs = recommender.recommend_top_k(u_idx, k=k, filter_already_rated=True)
        recommended_items = set([item_idx for item_idx, _ in recs])

        n_rel_and_rec = len(relevant_items.intersection(recommended_items))

        precisions.append(n_rel_and_rec / k)
        recalls.append(n_rel_and_rec / len(relevant_items))
        hits.append(1.0 if n_rel_and_rec > 0 else 0.0)

        eval_count += 1
        if max_eval_users is not None and eval_count >= max_eval_users:
            break

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0
    }


def evaluate_target_item_impact(
    recommender: UserBasedCF,
    target_item_idx: int,
    eval_user_indices: List[int],
    k: int = 10
) -> Dict[str, float]:
    """
    Measure the target item's visibility across genuine users:
    - Average predicted rating for the target item.
    - Hit Ratio @ K (fraction of users whose top-k recommendations include target item).
    - Average rank of target item when recommended.
    """
    predicted_ratings = []
    hits_in_top_k = 0
    ranks = []

    for u_idx in eval_user_indices:
        preds = recommender.predict_user_ratings(u_idx)
        predicted_ratings.append(preds[target_item_idx])

        # Check top-k recommendations (ignoring items already rated in train set)
        recs = recommender.recommend_top_k(u_idx, k=k, filter_already_rated=True)
        rec_item_ids = [item_idx for item_idx, _ in recs]

        if target_item_idx in rec_item_ids:
            hits_in_top_k += 1
            ranks.append(rec_item_ids.index(target_item_idx) + 1)

    avg_predicted_rating = float(np.mean(predicted_ratings))
    hit_ratio = float(hits_in_top_k / len(eval_user_indices)) if eval_user_indices else 0.0
    avg_rank = float(np.mean(ranks)) if ranks else float(k + 1)

    return {
        "avg_target_predicted_rating": avg_predicted_rating,
        "target_hit_ratio_at_k": hit_ratio,
        "target_avg_rank_in_top_k": avg_rank
    }


def evaluate_test_rmse(
    recommender: UserBasedCF,
    test_matrix: np.ndarray,
    max_eval_users: int = 200
) -> float:
    """
    Compute RMSE on held-out test ratings stored as a numpy matrix.
    Only evaluates on entries where test_matrix > 0 (actual ratings exist).
    """
    squared_errors = []
    num_users = min(max_eval_users, test_matrix.shape[0])

    for u_idx in range(num_users):
        rated_items = np.where(test_matrix[u_idx] > 0)[0]
        if len(rated_items) == 0:
            continue

        preds = recommender.predict_user_ratings(u_idx)
        for i_idx in rated_items:
            actual = float(test_matrix[u_idx, i_idx])
            predicted = float(preds[i_idx])
            squared_errors.append((actual - predicted) ** 2)

    if squared_errors:
        return float(np.sqrt(np.mean(squared_errors)))
    return 0.0


def evaluate_catalog_coverage(
    recommender: UserBasedCF,
    total_items: int,
    eval_user_indices: List[int],
    k: int = 10
) -> float:
    """
    Compute catalog coverage: fraction of total items recommended at least once in top-k.
    """
    recommended_items = set()
    for u_idx in eval_user_indices:
        recs = recommender.recommend_top_k(u_idx, k=k, filter_already_rated=True)
        for i_idx, _ in recs:
            recommended_items.add(i_idx)
    return float(len(recommended_items) / total_items) if total_items > 0 else 0.0


def evaluate_genuine_top_k_metrics(
    recommender: UserBasedCF,
    test_matrix: np.ndarray,
    rating_threshold: float = 3.5,
    k: int = 10,
    max_eval_users: int = 100
) -> Dict[str, float]:
    """
    Evaluate ranking metrics: HitRate@K, Precision@K, Recall@K on genuine users' test ratings.
    """
    precisions = []
    recalls = []
    hits = []
    num_users = min(max_eval_users, test_matrix.shape[0])

    for u_idx in range(num_users):
        relevant_items = set(np.where(test_matrix[u_idx] >= rating_threshold)[0])
        if len(relevant_items) == 0:
            continue
        recs = recommender.recommend_top_k(u_idx, k=k, filter_already_rated=True)
        recommended_items = set([item_idx for item_idx, _ in recs])
        n_rel_rec = len(relevant_items.intersection(recommended_items))

        precisions.append(n_rel_rec / k)
        recalls.append(n_rel_rec / len(relevant_items))
        hits.append(1.0 if n_rel_rec > 0 else 0.0)

    return {
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
    }

