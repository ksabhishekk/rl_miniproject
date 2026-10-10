import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Tuple, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ML100K_DIR = os.path.join(DATA_DIR, "ml-100k")
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@lru_cache(maxsize=4)
def _load_processed_matrix_cached(matrix_path: str) -> np.ndarray:
    """Load one processed matrix once and share it as read-only data."""
    matrix = np.load(matrix_path).astype(np.float32, copy=False)
    matrix.setflags(write=False)
    return matrix


def load_processed_matrix(matrix_path: str) -> np.ndarray:
    """Load a processed matrix, reusing the cached array for the same file.

    Relative paths are resolved from the project root rather than the caller's
    current working directory.
    """
    if not os.path.isabs(matrix_path):
        matrix_path = os.path.join(PROJECT_ROOT, matrix_path)
    normalized_path = os.path.normcase(os.path.realpath(matrix_path))
    return _load_processed_matrix_cached(normalized_path)


def download_and_extract_ml100k(data_dir: str = DATA_DIR) -> str:
    """Download MovieLens 100K dataset if not present, and extract it."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extracted_folder = os.path.join(data_dir, "ml-100k")

    if not os.path.exists(os.path.join(extracted_folder, "u.data")):
        print(f"Downloading MovieLens 100K from {ML100K_URL}...")
        try:
            import requests
            resp = requests.get(ML100K_URL, timeout=30, verify=False)
            with open(zip_path, "wb") as f:
                f.write(resp.content)
        except Exception:
            import ssl
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(ML100K_URL, context=ctx) as u, open(zip_path, 'wb') as f:
                f.write(u.read())
        print("Extracting ml-100k.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Dataset extracted successfully to", extracted_folder)
    else:
        print("MovieLens 100K dataset already present at", extracted_folder)

    return extracted_folder


def load_ratings(dataset_dir: str = ML100K_DIR) -> pd.DataFrame:
    """Load ratings dataset (u.data)."""
    ratings_path = os.path.join(dataset_dir, "u.data")
    columns = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(ratings_path, sep="\t", names=columns, engine="python")
    return df


def load_movies(dataset_dir: str = ML100K_DIR) -> pd.DataFrame:
    """Load movie metadata (u.item)."""
    movies_path = os.path.join(dataset_dir, "u.item")
    columns = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    df = pd.read_csv(movies_path, sep="|", names=columns, encoding="latin-1", engine="python")
    return df[["item_id", "title", "release_date"]]


def create_user_item_matrix(ratings_df: pd.DataFrame, num_users: int = None, num_items: int = None) -> np.ndarray:
    """
    Construct user-item matrix where rows are users (0-indexed)
    and columns are items (0-indexed). Unrated entries are 0.0.
    Dimensions are auto-detected from the data if not specified.
    """
    if num_users is None:
        num_users = int(ratings_df["user_id"].max())
    if num_items is None:
        num_items = int(ratings_df["item_id"].max())
    matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for row in ratings_df.itertuples():
        u_idx = int(row.user_id) - 1
        i_idx = int(row.item_id) - 1
        matrix[u_idx, i_idx] = float(row.rating)
    return matrix


def split_train_test(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a clean train/test split.
    Ensures genuine evaluation by holding out test ratings per user or stratified randomly.
    """
    rng = np.random.RandomState(random_seed)
    shuffled = ratings_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    # We do a masked split per user to ensure every user has test items if possible
    train_list = []
    test_list = []

    for user_id, group in shuffled.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        test_group = group.iloc[:n_test]
        train_group = group.iloc[n_test:]
        train_list.append(train_group)
        test_list.append(test_group)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


def inspect_dataset_summary(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict:
    """Compute and display summary statistics of MovieLens 100K."""
    n_users = ratings_df["user_id"].nunique()
    n_items = ratings_df["item_id"].nunique()
    n_ratings = len(ratings_df)
    sparsity = 1.0 - (n_ratings / (n_users * n_items))
    mean_rating = ratings_df["rating"].mean()
    rating_counts = ratings_df["rating"].value_counts().sort_index().to_dict()

    summary = {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": float(sparsity),
        "mean_rating": float(mean_rating),
        "rating_distribution": rating_counts
    }
    return summary


if __name__ == "__main__":
    download_and_extract_ml100k()
    ratings_df = load_ratings()
    movies_df = load_movies()
    summary = inspect_dataset_summary(ratings_df, movies_df)
    
    print("\n--- MovieLens 100K Dataset Inspection Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    train_df, test_df = split_train_test(ratings_df)
    print(f"\nTrain ratings count: {len(train_df)}")
    print(f"Test ratings count: {len(test_df)}")
    
    train_matrix = create_user_item_matrix(train_df)
    test_matrix = create_user_item_matrix(test_df)
    
    # Save processed matrices for fast re-use
    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "train_matrix.npy"), train_matrix)
    np.save(os.path.join(processed_dir, "test_matrix.npy"), test_matrix)
    train_df.to_csv(os.path.join(processed_dir, "train_df.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_df.csv"), index=False)
    print(f"\nProcessed matrices and CSVs saved to {processed_dir}")
