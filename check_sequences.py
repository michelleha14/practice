import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # draw to file, no window needed
import matplotlib.pyplot as plt

# ---- load everything --------------------------------------------------
df = pd.read_csv("cgm_data.csv")                       # original data
df["timestamp"] = pd.to_datetime(df["timestamp"])
d = np.load("cgm_daily_sequences.npz")
X, obs_mask = d["X"], d["obs_mask"]
meta = pd.read_csv("cgm_daily_meta.csv")

# ---- 1. shapes --------------------------------------------------------
print("X shape:", X.shape)                 # expect (n_days, 288, 5)
print("mask shape:", obs_mask.shape)       # expect (n_days, 288)
assert X.shape[1:] == (288, 5)
assert len(meta) == X.shape[0] == obs_mask.shape[0]

# ---- 2. no NaNs left --------------------------------------------------
print("Any NaN in X?", np.isnan(X).any())  # expect False

# ---- 3. missing values are filled with 0 and flagged in the mask ------
missing_vals = X[..., 2][~obs_mask]        # glucose where mask is False
print("Filled glucose all zero?", np.all(missing_vals == 0))

# ---- 4. time features: step 0 = midnight, step 72 = 06:00 -------------
print("Step 0  (sin, cos):", X[0, 0, 0].round(3), X[0, 0, 1].round(3))    # 0, 1
print("Step 72 (sin, cos):", X[0, 72, 0].round(3), X[0, 72, 1].round(3))  # 1, 0

# ---- 5. summaries -----------------------------------------------------
print("\nDays per subject:")
print(meta.groupby("subject_id").size().describe())
print("\nCoverage per day:")
print(meta["coverage"].describe())

# ---- 6. round trip: compare sequences to the raw data -----------------
rng = np.random.default_rng(0)
for i in rng.choice(len(meta), size=min(5, len(meta)), replace=False):
    subj = meta.loc[i, "subject_id"]
    day = pd.Timestamp(meta.loc[i, "date"])

    s = df[df["subject_id"] == subj].sort_values("timestamp").copy()
    s["ts"] = s["timestamp"].dt.round("5min")
    s = s.drop_duplicates("ts").dropna(subset=["norm_gl"])
    s = s[s["ts"].dt.normalize() == day]

    step = ((s["ts"] - day) / pd.Timedelta("5min")).astype(int).to_numpy()
    values_match = np.allclose(X[i, step, 2], s["norm_gl"].to_numpy(), atol=1e-5)
    count_match = obs_mask[i].sum() == len(s)
    print(f"seq {i} (subject {subj}, {day.date()}): "
          f"values match = {values_match}, count match = {count_match}")

# ---- 7. plot a few days -----------------------------------------------
idx = rng.choice(len(meta), size=min(3, len(meta)), replace=False)
fig, axes = plt.subplots(len(idx), 1, figsize=(10, 3 * len(idx)), sharex=True)
axes = np.atleast_1d(axes)
for ax, i in zip(axes, idx):
    gl = np.where(obs_mask[i], X[i, :, 2], np.nan)   # NaN -> gaps in the line
    ax.plot(np.arange(288) / 12, gl)                 # x axis in hours
    ax.set_title(f"subject {meta.loc[i, 'subject_id']}  {meta.loc[i, 'date']}")
    ax.set_ylabel("norm_gl")
axes[-1].set_xlabel("hour of day")
plt.tight_layout()
plt.savefig("check_days.png", dpi=120)
print("\nSaved check_days.png")
