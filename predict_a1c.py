"""
predict_a1c.py  -  Step 2: predict A1c from CGM using the pretrained transformer
===============================================================================

WHAT THIS SCRIPT DOES (the big picture)
---------------------------------------
Pretraining taught the transformer ("encoder") to read CGM days. Now we use it as
a feature extractor and ask: do its summaries of a person's CGM days predict that
person's A1c better than simple alternatives?

    1. Load the daily CGM sequences and the A1c value of each subject.
    2. Run every day through the pretrained encoder  -> one vector ("embedding")
       per day. (The encoder is FROZEN: its weights are not changed.)
    3. Average each subject's day-vectors             -> one vector per subject.
    4. Fit a ridge regression (embedding -> A1c) and test it with
       cross-validation, where every subject is predicted by a model that never
       saw that subject.
    5. Compare against baselines so we know whether the transformer adds value:
         - "Mean A1c"                   predicts the average A1c for everyone
         - "Mean glucose"               A1c ~ average glucose (the classic GMI idea)
         - "Mean glucose + SD"          adds glucose variability
         - "Random encoder + ridge"     same pipeline but WITHOUT pretraining
                                        (tells us whether pretraining helped)
         - "Pretrained encoder + ridge" the model we care about

WHY FROZEN EMBEDDINGS + RIDGE (and not fine-tuning the whole transformer)?
    A1c is one number per SUBJECT. With a few dozen subjects, a transformer with
    hundreds of thousands of weights would memorise them. A ridge regression
    (a linear model with a penalty that keeps weights small) is much safer.

FILES YOU NEED (all in the same folder as this script)
    pretrain_cgm.py            the pretraining script (we re-use its model class)
    cgm_pretrained.pt          the checkpoint saved by pretraining
    cgm_daily_sequences.npz    made by make_cgm_sequences.py
    cgm_daily_meta.csv         made by make_cgm_sequences.py
    a1c_labels.csv             YOU provide: one row per subject, e.g.
                                   subject_id,a1c
                                   S001,6.8
                                   S002,7.4
                               (subject_id must match the ids in your CGM data)

ONE-TIME INSTALL (in a terminal):
    pip install scikit-learn scipy

RUN:
    python predict_a1c.py

OUTPUT (in the folder  a1c_results/ ):
    metrics.csv          error / correlation of each model, with 95% intervals
    predictions.csv      each subject's true A1c and every model's prediction
    a1c_predictions.png  true vs predicted A1c for each model + an error summary

GLOSSARY
    embedding        a list of numbers (here 128) that summarises something
    frozen           weights are used but not updated
    cross-validation repeatedly hide some subjects, fit on the rest, predict the hidden
    out-of-fold      a prediction made for a subject by a model that did NOT see them
    ridge regression linear regression + a penalty on large weights (less overfitting)
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                  # draw straight to files; no pop-up windows
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Re-use the model definition and helper from the pretraining script, so the
# encoder here is built exactly as it was during pretraining.
from pretrain_cgm import GL, MaskedGlucoseModel, build_inputs


# ==========================================================================
# SETTINGS  (edit these; everything else can stay as is)
# ==========================================================================
# A Python "dict" is a set of name -> value pairs, like a named list in R.
SETTINGS = dict(
    # ---- input / output files ------------------------------------------
    ckpt_file="cgm_pretrained.pt",
    seq_file="cgm_daily_sequences.npz",
    meta_file="cgm_daily_meta.csv",
    a1c_file="pseudo_cgm_data_1200_subjects_r.csv",
    label_subject_col="subject_id",     # name of the subject column in a1c_labels.csv
    label_a1c_col="a1c",                # name of the A1c column in a1c_labels.csv
    out_dir="a1c_results_1200",

    # ---- data selection -------------------------------------------------
    min_days_per_subject=3,             # subjects with fewer valid days are dropped

    # ---- cross-validation -----------------------------------------------
    # "loo"   = leave-one-subject-out: best when you have < ~60 subjects.
    # "kfold" = repeated k-fold: use when you have many subjects.
    cv="loo",
    n_splits=5,                         # only used if cv == "kfold"
    n_repeats=20,                       # only used if cv == "kfold"

    # ---- ridge regression ------------------------------------------------
    # Candidate penalty strengths; RidgeCV picks the best one automatically
    # (using only the training subjects of each fold).
    ridge_alphas=np.logspace(-2, 4, 25),

    # ---- statistics ------------------------------------------------------
    n_boot=2000,                        # bootstrap repeats for the 95% intervals
    seed=42,                            # fixes randomness so results repeat exactly
)

# Names of the models. Defined once so the table/plot labels always match.
NULL = "Mean A1c (null)"
GLU = "Mean glucose"
GLU_SD = "Mean glucose + SD"
RAND = "Random encoder + ridge"
PRE = "Pretrained encoder + ridge"


# ==========================================================================
# STEP 1: load the data
# ==========================================================================
def load_labels(s):
    """Read the A1c file -> DataFrame with columns subject_id (text) and a1c (number)."""
    lab = pd.read_csv(s["a1c_file"])
    lab = lab.rename(columns={s["label_subject_col"]: "subject_id", s["label_a1c_col"]: "a1c"})
    lab = lab[["subject_id", "a1c"]]
    # Make ids text so "1" in one file matches "1" in the other (not 1 vs "1").
    lab["subject_id"] = lab["subject_id"].astype(str).str.strip()
    lab["a1c"] = pd.to_numeric(lab["a1c"], errors="coerce")   # bad entries -> NaN
    n_missing = int(lab["a1c"].isna().sum())
    if n_missing:
        print(f"  Dropping {n_missing} label rows with a missing/non-numeric A1c")
        lab = lab.dropna(subset=["a1c"])
    lab = lab.drop_duplicates("subject_id")                    # one row per subject
    if not lab["a1c"].between(3, 20).all():
        print("  WARNING: some A1c values are outside 3-20. Are they in % (not mmol/mol)?")
    return lab


def load_sequences(s):
    """
    Read the daily CGM sequences.
        X     : (n_days, 288, 5)  features per 5-minute step
        obs   : (n_days, 288)     True where a real glucose reading exists
        meta  : one row per day (subject_id, date, ...)
    Drops subjects with fewer than `min_days_per_subject` valid days.
    """
    d = np.load(s["seq_file"])
    X = d["X"].astype(np.float32)
    obs = d["obs_mask"].astype(bool)
    meta = pd.read_csv(s["meta_file"])
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()

    days_per_subject = meta.groupby("subject_id")["subject_id"].transform("size")
    keep = (days_per_subject >= s["min_days_per_subject"]).to_numpy()
    n_dropped = meta.loc[~keep, "subject_id"].nunique()
    if n_dropped:
        print(f"  Dropping {n_dropped} subjects with < {s['min_days_per_subject']} days")
    return X[keep], obs[keep], meta[keep].reset_index(drop=True)


# ==========================================================================
# STEP 2: turn each day into an embedding with the encoder
# ==========================================================================
def load_pretrained(s, device):
    """
    Load the pretrained model from the checkpoint file.
    Returns (model, cfg) where cfg holds the architecture settings used in pretraining.
    """
    ckpt = torch.load(s["ckpt_file"], map_location=device)
    cfg = ckpt["cfg"]
    model = MaskedGlucoseModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def compute_day_embeddings(model, X, obs, device, batch_size=128):
    """
    Run every day through the encoder and return one vector per day.

    - We do NOT hide any glucose here (we only flag truly missing steps), because
      at this stage we want the model to see the whole day.
    - The encoder gives one vector per 5-minute step: (288, d_model). To get ONE
      vector per day we average over the steps that have real data ("mean pooling").
    - model.eval() turns off dropout so the output is deterministic.
    - torch.no_grad() skips gradient bookkeeping: faster, less memory.
    """
    model.eval()
    X_t = torch.from_numpy(X)
    obs_t = torch.from_numpy(obs)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):            # process in small batches
            x = X_t[i:i + batch_size].to(device)
            o = obs_t[i:i + batch_size].to(device)
            no_masking = torch.zeros_like(o)                 # nothing hidden for the task
            h = model.encoder(build_inputs(x, o, no_masking), ~o)   # (B, 288, d_model)
            w = o.unsqueeze(-1).float()                      # 1 for real steps, 0 for missing
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)   # (B, d_model)
            out.append(pooled.cpu())
    return torch.cat(out).numpy()                            # (n_days, d_model)


# ==========================================================================
# STEP 3: one row per subject
# ==========================================================================
def pool_by_subject(day_values, subject_ids):
    """
    Average the day-level rows within each subject.
    R analogy:  df %>% group_by(subject_id) %>% summarise(across(everything(), mean))
    Returns a DataFrame whose row names are subject ids.
    """
    df = pd.DataFrame(day_values, index=subject_ids.to_numpy())
    return df.groupby(level=0).mean()


def glucose_summary_by_subject(X, obs, subject_ids):
    """
    Simple hand-made features for the baselines: each subject's mean and standard
    deviation of normalised glucose, over all of their valid 5-minute readings.
    (Because norm_gl is a linear rescaling of glucose, "mean of norm_gl" carries the
    same information as mean glucose in mg/dL - which is what GMI is built from.)
    """
    rows = {}
    ids = subject_ids.to_numpy()
    for subj in np.unique(ids):
        idx = np.where(ids == subj)[0]
        vals = X[idx][:, :, GL][obs[idx]]        # all real glucose readings of this subject
        rows[subj] = {"mean_gl": vals.mean(), "sd_gl": vals.std()}
    return pd.DataFrame.from_dict(rows, orient="index")


# ==========================================================================
# STEP 4: models and cross-validation
# ==========================================================================
def build_experiments(s):
    """
    Return the list of models to compare. Each item is:
        (display name, sklearn model, which feature table to give it)

    sklearn basics: every model has .fit(X, y) to learn and .predict(X) to predict.
    A Pipeline chains steps: here "standardise each column, then ridge". Chaining
    inside the pipeline matters: during cross-validation the standardisation is
    learned from the training subjects only, so nothing leaks from the held-out ones.
    """
    def ridge():
        return Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=s["ridge_alphas"])),
        ])

    return [
        (NULL, DummyRegressor(strategy="mean"), "glucose"),     # ignores its inputs
        (GLU, LinearRegression(), "mean_only"),
        (GLU_SD, LinearRegression(), "mean_sd"),
        (RAND, ridge(), "random_emb"),
        (PRE, ridge(), "pretrained_emb"),
    ]


def out_of_fold_predictions(model, X, y, s):
    """
    Predict every subject using models that never saw that subject.

    "loo":   fit on all subjects but one, predict the one left out, repeat for everyone.
    "kfold": split subjects into k groups, hold out one group at a time; repeat with
             different random splits and average the predictions for stability.
    Each row of X is ONE SUBJECT, so holding out rows = holding out subjects.
    """
    if s["cv"] == "loo":
        return cross_val_predict(model, X, y, cv=LeaveOneOut())
    all_preds = []
    for rep in range(s["n_repeats"]):
        splitter = KFold(n_splits=s["n_splits"], shuffle=True, random_state=s["seed"] + rep)
        all_preds.append(cross_val_predict(model, X, y, cv=splitter))
    return np.mean(all_preds, axis=0)


# ==========================================================================
# STEP 5: metrics
# ==========================================================================
def compute_metrics(y, p, with_correlation=True):
    """
    Summaries of how close predictions p are to the truth y.
      MAE  mean absolute error, in A1c % points (easy to interpret)
      RMSE root mean squared error (penalises big misses more)
      R2   1 = perfect, 0 = no better than always guessing the mean,
           negative = worse than guessing the mean
      r    Pearson correlation;  rho  Spearman (rank) correlation
    """
    out = {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "r": np.nan, "rho": np.nan,
    }
    if with_correlation:
        out["r"] = float(pearsonr(y, p)[0])
        out["rho"] = float(spearmanr(y, p)[0])
    return out


def bootstrap_ci(y, p, func, n_boot, seed):
    """
    95% interval by bootstrapping SUBJECTS: re-draw subjects with replacement many
    times, recompute the metric each time, and take the 2.5th and 97.5th percentiles.
    With few subjects these intervals are wide - that is honest, not a bug.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals.append(func(y[idx], p[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mae_fn(y, p):
    return np.mean(np.abs(y - p))


def r_fn(y, p):
    return np.corrcoef(y, p)[0, 1]


def evaluate_models(features, y, subject_ids, s):
    """
    Run every experiment.
    features: dict of DataFrames (rows = subjects, same order as y)
    Returns (metrics DataFrame, predictions DataFrame).
    """
    preds = pd.DataFrame({"subject_id": subject_ids, "a1c_true": y})
    rows = []
    for name, model, feat_key in build_experiments(s):
        X = features[feat_key].to_numpy()
        p = out_of_fold_predictions(model, X, y, s)
        preds[name] = p

        # Correlation is meaningless for the "mean A1c" model: when you leave a
        # subject out, the mean of the others goes DOWN when that subject's A1c is
        # high, producing an artificial correlation of exactly -1. We skip it.
        #
        # The SAME artifact appears (as a strongly NEGATIVE r under cv="loo") for any
        # model that learns nothing useful and falls back to predicting the training
        # mean - e.g. the random-encoder control when ridge shrinks all its weights
        # to ~0. If you see r close to -1 for a weak model, read it as "no signal",
        # and judge models by MAE and R2 instead.
        is_null = name == NULL
        m = compute_metrics(y, p, with_correlation=not is_null)
        m["MAE_ci_low"], m["MAE_ci_high"] = bootstrap_ci(y, p, mae_fn, s["n_boot"], s["seed"])
        if is_null:
            m["r_ci_low"] = m["r_ci_high"] = np.nan
        else:
            m["r_ci_low"], m["r_ci_high"] = bootstrap_ci(y, p, r_fn, s["n_boot"], s["seed"])
        m["model"] = name
        rows.append(m)

    cols = ["model", "MAE", "MAE_ci_low", "MAE_ci_high", "RMSE", "R2",
            "r", "r_ci_low", "r_ci_high", "rho"]
    return pd.DataFrame(rows)[cols], preds


# ==========================================================================
# STEP 6: plots
# ==========================================================================
def plot_results(path, metrics, preds):
    """One scatter panel (true vs out-of-fold predicted A1c) per model + an MAE bar chart."""
    names = list(metrics["model"])
    y = preds["a1c_true"].to_numpy()
    all_vals = np.concatenate([y] + [preds[n].to_numpy() for n in names])
    lo, hi = all_vals.min() - 0.2, all_vals.max() + 0.2

    ncols = 3
    nrows = int(np.ceil((len(names) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        row = metrics.set_index("model").loc[name]
        color = "tab:red" if name == PRE else "tab:blue"
        ax.scatter(y, preds[name], s=32, color=color, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)             # perfect-prediction line
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        r_txt = "n/a" if np.isnan(row["r"]) else f"{row['r']:.2f}"
        ax.set_title(f"{name}\nMAE {row['MAE']:.2f}   r {r_txt}", fontsize=10)
        ax.set_xlabel("true A1c (%)")
        ax.set_ylabel("predicted A1c (%) - out-of-fold")
        ax.grid(alpha=0.25)

    # MAE bar chart with bootstrap intervals (lower = better)
    ax = axes[len(names)]
    m = metrics.set_index("model")
    maes = m["MAE"].to_numpy()
    err = np.vstack([maes - m["MAE_ci_low"].to_numpy(), m["MAE_ci_high"].to_numpy() - maes])
    colors = ["tab:red" if n == PRE else "0.6" for n in names]
    ax.bar(range(len(names)), maes, yerr=err, color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("MAE (A1c % points), lower = better")
    ax.set_title("Error summary (95% bootstrap interval)", fontsize=10)

    for ax in axes[len(names) + 1:]:
        ax.set_visible(False)                                # hide unused panels
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# MAIN: run all the steps in order
# ==========================================================================
def main(s=SETTINGS):
    os.makedirs(s["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(s["seed"])

    print("[1/6] Loading data ...")
    labels = load_labels(s)
    X, obs, meta = load_sequences(s)
    print(f"  {len(meta)} days from {meta['subject_id'].nunique()} subjects; "
          f"{len(labels)} subjects with an A1c value")

    print("[2/6] Computing day embeddings (pretrained and random-weight control) ...")
    model, cfg = load_pretrained(s, device)
    day_emb_pre = compute_day_embeddings(model, X, obs, device)

    # CONTROL: an encoder with the same architecture but random, untrained weights.
    # If it predicts A1c as well as the pretrained one, pretraining added nothing.
    torch.manual_seed(s["seed"])
    random_model = MaskedGlucoseModel(cfg).to(device)
    day_emb_rand = compute_day_embeddings(random_model, X, obs, device)

    print("[3/6] Pooling to one row per subject ...")
    subj = meta["subject_id"]
    glucose = glucose_summary_by_subject(X, obs, subj)
    features = {
        "pretrained_emb": pool_by_subject(day_emb_pre, subj),
        "random_emb": pool_by_subject(day_emb_rand, subj),
        "glucose": glucose[["mean_gl"]],           # only a placeholder for the null model
        "mean_only": glucose[["mean_gl"]],
        "mean_sd": glucose[["mean_gl", "sd_gl"]],
    }

    # Keep only subjects that have BOTH CGM data and an A1c value, in a fixed order.
    both = sorted(set(glucose.index) & set(labels["subject_id"]))
    print(f"  {len(both)} subjects have both CGM data and A1c")
    if len(both) < 8:
        raise SystemExit("Fewer than 8 subjects matched. Check that subject ids in "
                         "a1c_labels.csv match the ids in cgm_daily_meta.csv.")
    features = {k: v.loc[both] for k, v in features.items()}
    y = labels.set_index("subject_id").loc[both, "a1c"].to_numpy(dtype=float)

    print(f"[4/6] Cross-validating (cv = {s['cv']}) ...")
    metrics, preds = evaluate_models(features, y, both, s)
    preds.insert(1, "n_days", meta.groupby("subject_id").size().loc[both].to_numpy())

    print("[5/6] Saving tables ...")
    metrics.to_csv(os.path.join(s["out_dir"], "metrics.csv"), index=False)
    preds.to_csv(os.path.join(s["out_dir"], "predictions.csv"), index=False)

    print("[6/6] Plotting ...")
    plot_results(os.path.join(s["out_dir"], "a1c_predictions.png"), metrics, preds)

    print("\nResults (out-of-fold, A1c in % points):")
    show = metrics[["model", "MAE", "RMSE", "R2", "r", "rho"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = metrics.loc[metrics["MAE"].idxmin(), "model"]
    print(f"\nLowest MAE: {best}")
    print("Tip: judge models mainly by MAE and R2. A strongly negative r for a weak "
          "model under cv='loo' is a known artifact, not a bug (see evaluate_models).")
    print(f"Saved results to ./{s['out_dir']}/")


if __name__ == "__main__":
    main()
