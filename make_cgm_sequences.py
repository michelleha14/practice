"""
Chunk 5-minute CGM data into midnight-to-midnight, 288-step daily sequences
for transformer pretraining.

Expected input columns (CSV or DataFrame):
    subject_id, timestamp, hour_sin, hour_cos, norm_gl, norm_speed, norm_acceleration

Output:
    X        : (N, 288, 5) float32  - features (missing values filled with 0)
    obs_mask : (N, 288)    bool     - True where a real measurement exists
    meta     : DataFrame with subject_id, date, coverage per sequence
"""

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288          # 24h * 60 / 5
FREQ = "5min"
TIME_FEATURES = ["hour_sin", "hour_cos"]               # deterministic, always known
SIGNAL_FEATURES = ["norm_gl", "norm_speed", "norm_acc"]  # can be missing
FEATURES = TIME_FEATURES + SIGNAL_FEATURES


def make_daily_sequences(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    time_col: str = "timestamp",
    min_coverage: float = 0.7,     # keep days with >= 70% of glucose points present
    max_gap_steps: int | None = 36,  # drop days with a gap longer than 3h (None = ignore)
):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Time-of-day encoding for a full day, computed once (so it is never missing)
    tod = np.arange(STEPS_PER_DAY) * 5 / (24 * 60)          # fraction of day
    hour_sin = np.sin(2 * np.pi * tod).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * tod).astype(np.float32)

    X_list, mask_list, meta_rows = [], [], []

    for subj, g in df.groupby(subject_col):
        g = g.sort_values(time_col)

        # Snap to the 5-min grid and remove duplicates (keep first reading)
        g[time_col] = g[time_col].dt.round(FREQ)
        g = g.drop_duplicates(subset=time_col).set_index(time_col)

        # Walk through each calendar day, midnight -> midnight
        for day, day_df in g.groupby(g.index.normalize()):
            grid = pd.date_range(day, periods=STEPS_PER_DAY, freq=FREQ)
            day_df = day_df[SIGNAL_FEATURES].reindex(grid)    # NaN where missing

            # Observed = glucose present (speed/accel are derived from glucose)
            observed = day_df["norm_gl"].notna().to_numpy()
            coverage = observed.mean()
            if coverage < min_coverage:
                continue

            if max_gap_steps is not None and _longest_gap(observed) > max_gap_steps:
                continue

            signals = day_df.to_numpy(dtype=np.float32)
            signals = np.nan_to_num(signals, nan=0.0)         # fill; use mask to ignore

            x = np.column_stack([hour_sin, hour_cos, signals]).astype(np.float32)

            X_list.append(x)
            mask_list.append(observed)
            meta_rows.append(
                {"subject_id": subj, "date": day.date(), "coverage": float(coverage)}
            )

    X = np.stack(X_list)                    # (N, 288, 5)
    obs_mask = np.stack(mask_list)          # (N, 288)
    meta = pd.DataFrame(meta_rows)
    return X, obs_mask, meta


def _longest_gap(observed: np.ndarray) -> int:
    """Longest run of consecutive missing steps."""
    longest = current = 0
    for ok in observed:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return longest


def split_by_subject(meta: pd.DataFrame, val_frac=0.1, test_frac=0.1, seed=42):
    """Split by SUBJECT (not by day) to avoid leakage between train/val/test."""
    rng = np.random.default_rng(seed)
    subjects = meta["subject_id"].unique()
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_frac))
    n_test = max(1, int(len(subjects) * test_frac))
    val_s = set(subjects[:n_val])
    test_s = set(subjects[n_val:n_val + n_test])

    split = np.where(
        meta["subject_id"].isin(test_s), "test",
        np.where(meta["subject_id"].isin(val_s), "val", "train"),
    )
    return split


# ---------------------------------------------------------------- PyTorch
try:
    import torch
    from torch.utils.data import Dataset

    class CGMDayDataset(Dataset):
        def __init__(self, X, obs_mask):
            self.X = torch.from_numpy(X)                 # (N, 288, 5)
            self.mask = torch.from_numpy(obs_mask)       # (N, 288) bool

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return {"x": self.X[i], "obs_mask": self.mask[i]}
except ImportError:
    pass


if __name__ == "__main__":
    df = pd.read_csv("pseudo_cgm_5min_data_500_subjects.csv")   # <- change to your file

    X, obs_mask, meta = make_daily_sequences(df)
    print("X:", X.shape, "| mask:", obs_mask.shape)
    print(meta.groupby("subject_id").size().describe())

    meta["split"] = split_by_subject(meta)
    np.savez_compressed("cgm_daily_sequences.npz", X=X, obs_mask=obs_mask)
    meta.to_csv("cgm_daily_meta.csv", index=False)
