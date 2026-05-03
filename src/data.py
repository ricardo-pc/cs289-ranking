"""
data.py — MovieLens 1M data pipeline

Responsibilities:
  1. Load raw .dat files
  2. Remap user/movie IDs to contiguous 0-indexed integers
  3. Leave-one-out split (last item by timestamp = test, second-to-last = val)
  4. Sparsity subsampling of the training set
  5. Pre-sample 99 negatives per user for evaluation
  6. PyTorch Dataset classes for training and evaluation

Usage:
  from data import load_ml1m, build_eval_negatives, TrainDataset, EvalDataset
  data = load_ml1m('data/raw/ml-1m')
  negatives = build_eval_negatives(data, seed=42)
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ML1MData:
    """Holds all splits and metadata after preprocessing."""
    train: pd.DataFrame          # columns: user, item  (subsampled at density d)
    val:   pd.DataFrame          # one row per user — the second-to-last interaction
    test:  pd.DataFrame          # one row per user — the last interaction
    n_users: int
    n_items: int
    user2idx: Dict[int, int]     # original ID -> 0-indexed
    item2idx: Dict[int, int]
    all_items: np.ndarray        # all valid item indices (for negative sampling)


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_ml1m(data_dir: str, density: float = 1.0, seed: int = 42) -> ML1MData:
    """
    Load MovieLens 1M, remap IDs, and return train/val/test splits.

    density: fraction of each user's training interactions to keep (1.0 = full).
             Val and test items are never subsampled.
    """
    assert 0.0 < density <= 1.0, "density must be in (0, 1]"
    data_dir = Path(data_dir)

    ratings = pd.read_csv(
        data_dir / "ratings.dat", sep="::", engine="python", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "movie_id": np.int32,
               "rating": np.float32, "timestamp": np.int64},
    )

    # Remap IDs to contiguous 0-indexed integers
    user_ids  = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    item2idx  = {m: i for i, m in enumerate(movie_ids)}

    ratings["user"] = ratings["user_id"].map(user2idx).astype(np.int32)
    ratings["item"] = ratings["movie_id"].map(item2idx).astype(np.int32)

    n_users = len(user_ids)
    n_items = len(movie_ids)

    # Sort by user then timestamp so .nth(-1) is the last interaction
    ratings = ratings.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)

    # Leave-one-out: last interaction -> test, second-to-last -> val
    last        = ratings.groupby("user").nth(-1)[["user", "item"]].reset_index(drop=True)
    second_last = ratings.groupby("user").nth(-2)[["user", "item"]].reset_index(drop=True)

    # Training pool: everything except the last two interactions per user
    test_idx  = ratings.groupby("user").tail(1).index
    val_idx   = ratings.groupby("user").tail(2).index.difference(test_idx)
    train_all = ratings.drop(index=test_idx.union(val_idx))[["user", "item"]].reset_index(drop=True)

    train = _subsample(train_all, density, seed)

    return ML1MData(
        train=train,
        val=second_last,
        test=last,
        n_users=n_users,
        n_items=n_items,
        user2idx=user2idx,
        item2idx=item2idx,
        all_items=np.arange(n_items, dtype=np.int32),
    )


def _subsample(train: pd.DataFrame, density: float, seed: int) -> pd.DataFrame:
    """Keep `density` fraction of each user's training interactions (min 1)."""
    if density == 1.0:
        return train.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for user, group in train.groupby("user"):
        n = max(1, int(len(group) * density))
        parts.append(group.sample(n=n, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Negative sampling for evaluation
# ---------------------------------------------------------------------------

def build_eval_negatives(
    data: ML1MData,
    n_neg: int = 99,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    For each user, sample `n_neg` items they have NOT interacted with.
    Returns a dict: user_idx -> list of n_neg negative item indices.

    Negatives are fixed for the entire experiment so results are
    comparable across models and density levels.
    """
    rng = random.Random(seed)

    # Full set of items each user has seen (train + val + test)
    seen: Dict[int, set] = {u: set() for u in range(data.n_users)}
    for df in (data.train, data.val, data.test):
        for row in df.itertuples(index=False):
            seen[row.user].add(row.item)

    all_items = data.all_items.tolist()
    negatives: Dict[int, List[int]] = {}
    for user in range(data.n_users):
        user_seen  = seen[user]
        candidates = [it for it in all_items if it not in user_seen]
        negatives[user] = rng.sample(candidates, min(n_neg, len(candidates)))

    return negatives


# ---------------------------------------------------------------------------
# PyTorch Dataset classes
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """
    Implicit feedback training dataset with online negative sampling.

    Each __getitem__ returns (user, pos_item, neg_item).
    Used for BPR loss (ranker) or binary cross-entropy with negatives (NCF/MF).
    """

    def __init__(self, data: ML1MData, n_neg_per_pos: int = 4, seed: int = 42):
        self.n_items = data.n_items
        self.n_neg   = n_neg_per_pos
        self.rng     = np.random.default_rng(seed)

        # Per-user positive sets for fast negative rejection
        self.user_pos: Dict[int, set] = {}
        for row in data.train.itertuples(index=False):
            self.user_pos.setdefault(row.user, set()).add(row.item)

        # Expand: one row per (user, pos_item) x n_neg_per_pos
        users = data.train["user"].to_numpy(dtype=np.int64)
        items = data.train["item"].to_numpy(dtype=np.int64)
        self.users = np.repeat(users, n_neg_per_pos)
        self.items = np.repeat(items, n_neg_per_pos)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = int(self.users[idx])
        pos  = int(self.items[idx])
        seen = self.user_pos[user]
        while True:
            neg = int(self.rng.integers(self.n_items))
            if neg not in seen:
                break
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            torch.tensor(neg,  dtype=torch.long),
        )


class EvalDataset(Dataset):
    """
    Evaluation dataset: for each user, return (user, pos_item, neg_items).

    pos_item is from data.test (or data.val).
    neg_items are the pre-sampled 99 negatives from build_eval_negatives().
    """

    def __init__(
        self,
        data: ML1MData,
        negatives: Dict[int, List[int]],
        split: str = "test",
    ):
        assert split in ("test", "val")
        pos_df = data.test if split == "test" else data.val

        self.users     = pos_df["user"].to_numpy(dtype=np.int64)
        self.pos_items = pos_df["item"].to_numpy(dtype=np.int64)
        self.neg_items = [negatives[u] for u in self.users]

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = torch.tensor(self.users[idx], dtype=torch.long)
        pos  = torch.tensor(self.pos_items[idx], dtype=torch.long)
        negs = torch.tensor(self.neg_items[idx], dtype=torch.long)
        return user, pos, negs


# ---------------------------------------------------------------------------
# Collate: build 100-item candidate list (pos + 99 negs) for ranking metrics
# ---------------------------------------------------------------------------

def eval_collate(batch):
    """
    candidates: (B, 100) — col 0 is the positive, cols 1-99 are negatives.
    labels    : (B, 100) — 1 at col 0, 0 elsewhere.
    """
    users, pos_items, neg_items = zip(*batch)
    users      = torch.stack(users)
    pos_items  = torch.stack(pos_items).unsqueeze(1)
    neg_items  = torch.stack(neg_items)
    candidates = torch.cat([pos_items, neg_items], dim=1)
    labels     = torch.zeros_like(candidates, dtype=torch.float)
    labels[:, 0] = 1.0
    return users, candidates, labels


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ml-1m"

    print("Loading ml-1m at density=1.0 ...")
    data = load_ml1m(data_dir, density=1.0)
    print(f"  n_users : {data.n_users:,}")
    print(f"  n_items : {data.n_items:,}")
    print(f"  train   : {len(data.train):,} interactions")
    print(f"  val     : {len(data.val):,} rows")
    print(f"  test    : {len(data.test):,} rows")

    assert len(data.val)  == data.n_users, "val must have one row per user"
    assert len(data.test) == data.n_users, "test must have one row per user"

    # No val/test items should appear in training
    train_set  = set(zip(data.train["user"], data.train["item"]))
    val_leaks  = sum(1 for r in data.val.itertuples()  if (r.user, r.item) in train_set)
    test_leaks = sum(1 for r in data.test.itertuples() if (r.user, r.item) in train_set)
    print(f"  val leaks into train  : {val_leaks}  (expect 0)")
    print(f"  test leaks into train : {test_leaks} (expect 0)")
    assert val_leaks == 0 and test_leaks == 0

    print("\nBuilding eval negatives ...")
    negatives = build_eval_negatives(data, n_neg=99, seed=42)
    assert all(len(v) == 99 for v in negatives.values())
    print("  99 negatives per user — OK")

    print("\nSparsity subsampling check:")
    for d in [0.8, 0.6, 0.4, 0.2]:
        d_data = load_ml1m(data_dir, density=d)
        print(f"  density={d:.0%}  train={len(d_data.train):,}")

    print("\nAll checks passed.")
