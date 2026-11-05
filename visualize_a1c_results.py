"""
visualize_a1c_results.py  -  Plots for judging the A1c models made by predict_a1c.py
====================================================================================

Run AFTER predict_a1c.py:     python visualize_a1c_results.py

It reads  a1c_results/predictions.csv  (one row per subject: the true A1c and every
model's OUT-OF-FOLD prediction, i.e. each subject was predicted by a model that
never saw them) and writes, into the same folder:

    a1c_scatter_by_model.png   predicted vs actual A1c, one panel per model
    a1c_diagnostics.png        six panels that help answer "does the model work?"
    a1c_model_summary.csv      MAE, RMSE, R2, r, calibration slope, SD ratio, AUC
    a1c_largest_errors.csv     the subjects the main model gets most wrong

THE SIX DIAGNOSTIC PANELS (and what you want to see)
 1. Bias by A1c level    mean (predicted - true) for groups of subjects with similar
                         true A1c. Flat at 0 is ideal. A downward slope means the
                         model "shrinks toward the mean" (high A1c under-predicted,
                         low over-predicted) - common, and the first sign of a weak model.
 2. Distribution         histogram of true A1c vs each model's predictions. If a model's
                         predictions are much narrower than the truth (small SD), it is
                         close to predicting the average for everyone.
 3. ROC curves           how well each model separates subjects above/below an A1c
                         threshold (default 6.5%). AUC 0.5 = coin flip, 1.0 = perfect.
                         This ignores shrinkage, so it is a fairer test of "ranking".
 4. Paired MAE change    for the main model minus each other model, with 95% intervals.
                         Entirely LEFT of 0 = the main model is reliably better.
 5. Added value          Does the transformer know something that mean glucose doesn't?
                         x = (transformer prediction - mean-glucose prediction),
                         y = (true A1c - mean-glucose prediction). A clearly positive
                         slope means the transformer's disagreements with mean glucose
                         move toward the truth. A flat cloud = no added information.
 6. Error vs days        absolute error against how many CGM days a subject has.
                         Error should shrink as days increase.

KEY NUMBERS IN a1c_model_summary.csv
    MAE, RMSE, R2    accuracy (A1c % points; R2 = 0 means no better than the average)
    r                correlation between predicted and true A1c
    cal_slope        slope of TRUE on PREDICTED. 1 = well calibrated; < 1 = predictions
                     too spread out (overconfident); > 1 = too shrunk toward the mean.
                     Left blank when predictions are almost constant (unstable)
    sd_ratio         SD(predictions) / SD(true). Near 0 = collapsed to the mean
    AUC              ability to flag A1c above the threshold

Note on the "Mean A1c (null)" model: it predicts the training average for everyone, so
correlation-based numbers are meaningless for it (leave-one-out makes them an artifact).
They are left blank, and only its MAE / RMSE / R2 are shown.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # draw straight to files; no pop-up windows
import matplotlib.pyplot as plt

# ==========================================================================
# SETTINGS  (edit these)
# ==========================================================================
SETTINGS = dict(
    pred_file="a1c_results_1200/predictions.csv",   # made by predict_a1c.py
    out_dir="a1c_results_1200",                     # where the pictures/tables are written
    # Names must match the column names in predictions.csv.
    main_model="Pretrained encoder + ridge",   # the model you care about
    reference_model="Mean glucose",            # the simple model it must beat
    control_model="Random encoder + ridge",    # same pipeline WITHOUT pretraining
    null_model="Mean A1c (null)",              # predicts the average for everyone
    a1c_threshold=6.5,                         # for the ROC panel (6.5 = diabetes cut-off; 5.7 = prediabetes)
    n_boot=2000,                               # bootstrap repeats for 95% intervals
    seed=42,                                   # fixes the random numbers
)


# ==========================================================================
# Loading
# ==========================================================================
def load_predictions(s):
    """
    Read predictions.csv. Every column except subject_id / n_days / a1c_true is a model.
    Returns (data frame, list of model names, dict of roles -> model name or None).
    """
    df = pd.read_csv(s["pred_file"])
    if "a1c_true" not in df.columns:
        raise SystemExit("predictions.csv has no 'a1c_true' column. Run predict_a1c.py first.")
    models = [c for c in df.columns if c not in ("subject_id", "n_days", "a1c_true")]
    roles = {}
    for role in ["main", "reference", "control", "null"]:
        name = s[f"{role}_model"]
        roles[role] = name if name in models else None      # None if that column doesn't exist
        if roles[role] is None:
            print(f"  NOTE: model '{name}' not found in predictions.csv; panels using it are skipped")
    if roles["main"] is None:                               # fall back to the last model column
        roles["main"] = models[-1]
        print(f"  Using '{roles['main']}' as the main model instead")
    return df, models, roles


def assign_colors(models, roles):
    """One fixed colour per model, so the same model has the same colour in every panel."""
    fixed = {roles["main"]: "tab:red", roles["reference"]: "tab:blue",
             roles["control"]: "tab:orange", roles["null"]: "0.55"}
    spare = iter(["tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive", "tab:pink"])
    return {m: fixed.get(m) or next(spare) for m in models}


# ==========================================================================
# Statistics helpers
# ==========================================================================
def basic_metrics(y, p):
    """MAE, RMSE, R2 of predictions p against truth y."""
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def spread_stats(y, p):
    """r, calibration slope (true ~ predicted) and SD ratio. NaN if predictions are constant."""
    if np.std(p) < 1e-9:
        return {"r": np.nan, "cal_slope": np.nan, "sd_ratio": 0.0}
    sd_ratio = float(np.std(p) / np.std(y))
    # With almost-constant predictions the slope is dominated by noise (it can be huge or
    # negative), so we leave it blank when the predictions have < 5% of the truth's spread.
    cal_slope = float(np.polyfit(p, y, 1)[0]) if sd_ratio >= 0.05 else np.nan   # slope of a line with predicted on x, true on y
    return {
        "r": float(np.corrcoef(y, p)[0, 1]),
        "cal_slope": cal_slope,
        "sd_ratio": sd_ratio,
    }


def auc_score(y_bin, score):
    """
    Area under the ROC curve = chance that a randomly chosen 'high A1c' subject gets a
    higher score than a randomly chosen 'not high' subject (ties count half).
    Computed from ranks (the Mann-Whitney formula), so no extra libraries are needed.
    """
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = int(y_bin.sum()), int((~y_bin).sum())
    return float((ranks[y_bin].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_points(y_bin, score):
    """False- and true-positive rates as the decision threshold sweeps from high to low."""
    order = np.argsort(-score, kind="mergesort")            # highest scores first
    hits = y_bin[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / hits.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / (~hits).sum()])
    return fpr, tpr


def has_both_classes(y_bin, minimum=5):
    """The ROC panel needs at least a few subjects on each side of the threshold."""
    return y_bin.sum() >= minimum and (~y_bin).sum() >= minimum


def bias_by_level(y, p, n_bins):
    """Group subjects by true A1c; return per-group mean true A1c and mean (pred - true)."""
    bins = pd.qcut(pd.Series(y), n_bins, duplicates="drop")  # equal-sized groups
    g = pd.DataFrame({"y": y, "res": p - y, "bin": bins}).groupby("bin", observed=True)
    return g["y"].mean().to_numpy(), g["res"].mean().to_numpy(), \
        (g["res"].std() / np.sqrt(g["res"].size())).to_numpy()


def fmt2(v):
    """Format a number with 2 decimals, or 'n/a' if it is missing (NaN)."""
    return "n/a" if v != v else f"{v:.2f}"                  # NaN is the only value not equal to itself


def unavailable(ax, message):
    """Fill a panel with a note when it can't be drawn."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_axis_off()


# ==========================================================================
# Figure 1: predicted vs actual, one panel per model
# ==========================================================================
def plot_scatter_grid(df, models, roles, colors, path):
    y = df["a1c_true"].to_numpy()
    lo = min(y.min(), *(df[m].min() for m in models)) - 0.2     # shared axis limits so panels are comparable
    hi = max(y.max(), *(df[m].max() for m in models)) + 0.2
    ncols = 3
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, m in zip(axes, models):
        p = df[m].to_numpy()
        if len(y) > 500:                                        # many subjects: density plot instead of dots
            ax.hexbin(y, p, gridsize=35, bins="log", mincnt=1, cmap="viridis",
                      extent=(lo, hi, lo, hi))
        else:
            ax.scatter(y, p, s=22, alpha=0.65, color=colors[m])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)                # perfect prediction
        is_null = m == roles["null"]
        if not is_null and np.std(p) > 1e-9:                    # dotted line: where the predictions actually sit
            slope, intercept = np.polyfit(y, p, 1)
            ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
                    ":", color="k", lw=1.6)
        bm, sm = basic_metrics(y, p), spread_stats(y, p)
        if is_null:                                             # correlation is an artifact for the null model: show n/a
            sm = dict(sm, r=np.nan, cal_slope=np.nan)
        txt = (f"MAE {bm['MAE']:.2f}   R2 {bm['R2']:.2f}\n"
               f"r {fmt2(sm['r'])}   cal.slope {fmt2(sm['cal_slope'])}\n"
               f"SD ratio {sm['sd_ratio']:.2f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(m, fontsize=10, fontweight="bold" if m == roles["main"] else "normal")
        ax.set_xlabel("true A1c (%)")
        ax.set_ylabel("predicted A1c (%) - out-of-fold")
        ax.grid(alpha=0.25)
    for ax in axes[len(models):]:
        ax.set_visible(False)                                   # hide unused panels
    fig.suptitle("Predicted vs actual A1c   (dashed = perfect;  dotted = fitted trend of predictions)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Figure 2: diagnostic panels
# ==========================================================================
def panel_bias(ax, y, df, models, roles, colors):
    n_bins = int(min(10, max(3, len(y) // 15)))
    for m in models:
        xs, ys, es = bias_by_level(y, df[m].to_numpy(), n_bins)
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3.5, capsize=2, color=colors[m],
                    ls="--" if m == roles["null"] else "-", lw=2 if m == roles["main"] else 1.2,
                    label=m)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("true A1c, group means (%)")
    ax.set_ylabel("mean (predicted - true)")
    ax.set_title("1. Bias by A1c level\n(downward slope = shrinks toward the mean)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def panel_distribution(ax, y, df, models, roles, colors):
    shown = [m for m in models if m != roles["null"]]           # a constant model is just one spike
    allv = np.concatenate([y] + [df[m].to_numpy() for m in shown])
    bins = np.linspace(allv.min(), allv.max(), 26)
    ax.hist(y, bins=bins, color="0.85", edgecolor="k", label=f"TRUE (SD {np.std(y):.2f})")
    for m in shown:
        p = df[m].to_numpy()
        ax.hist(p, bins=bins, histtype="step", lw=2.0 if m == roles["main"] else 1.3,
                color=colors[m], label=f"{m} (SD {np.std(p):.2f})")
    ax.set_xlabel("A1c (%)")
    ax.set_ylabel("number of subjects")
    ax.set_title("2. Distribution of predictions vs truth\n(much narrower = close to guessing the average)")
    ax.legend(fontsize=7)


def panel_roc(ax, y, df, models, roles, colors, threshold):
    y_bin = y >= threshold
    if not has_both_classes(y_bin):
        unavailable(ax, f"3. ROC\nfewer than 5 subjects on one side of A1c {threshold}")
        return
    for m in models:
        if m == roles["null"]:
            continue                                            # constant predictions: no ranking ability
        p = df[m].to_numpy()
        fpr, tpr = roc_points(y_bin, p)
        ax.plot(fpr, tpr, color=colors[m], lw=2.2 if m == roles["main"] else 1.3,
                label=f"{m} (AUC {auc_score(y_bin, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"3. Flagging A1c >= {threshold}\n({int(y_bin.sum())} high, {int((~y_bin).sum())} not high)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)


def panel_forest(ax, y, df, models, roles, s):
    main = roles["main"]
    others = [m for m in models if m != main]
    if not others:
        unavailable(ax, "4. Paired MAE change\nno other models to compare with")
        return
    rng = np.random.default_rng(s["seed"])
    n = len(y)
    idx = rng.integers(0, n, (s["n_boot"], n))                  # resampled subjects, shared by both models (= "paired")
    ae_main = np.abs(y - df[main].to_numpy())
    for k, m in enumerate(others):
        ae_other = np.abs(y - df[m].to_numpy())
        diff = ae_main.mean() - ae_other.mean()
        boot = ae_main[idx].mean(axis=1) - ae_other[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        color = "tab:green" if hi < 0 else ("tab:red" if lo > 0 else "0.4")   # green = reliably better, red = reliably worse
        ax.errorbar(diff, k, xerr=[[diff - lo], [hi - diff]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels(others, fontsize=8)
    ax.set_xlabel(f"MAE({main}) - MAE(other), A1c % points")
    ax.set_title("4. Paired MAE change with 95% interval\n(left of 0 = main model better; green = reliably)")
    ax.grid(alpha=0.25)


def panel_added_value(ax, y, df, roles, s):
    main, ref = roles["main"], roles["reference"]
    if ref is None or ref == main:
        unavailable(ax, "5. Added value\nneeds a reference model (e.g. 'Mean glucose')")
        return
    x_add = df[main].to_numpy() - df[ref].to_numpy()            # what the transformer adds to the reference
    y_add = y - df[ref].to_numpy()                              # what the reference misses
    if np.std(x_add) < 1e-9:
        unavailable(ax, "5. Added value\nmain model's predictions equal the reference's")
        return
    slope, intercept = np.polyfit(x_add, y_add, 1)
    r = np.corrcoef(x_add, y_add)[0, 1]
    rng = np.random.default_rng(s["seed"])                      # bootstrap interval for the slope
    idx = rng.integers(0, len(y), (s["n_boot"], len(y)))
    xs, ys = x_add[idx], y_add[idx]
    xc, yc = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
    boot_slope = (xc * yc).sum(axis=1) / (xc ** 2).sum(axis=1)
    lo, hi = np.percentile(boot_slope, [2.5, 97.5])
    ax.scatter(x_add, y_add, s=18, alpha=0.55, color="tab:red")
    xx = np.array([x_add.min(), x_add.max()])
    ax.plot(xx, intercept + slope * xx, "k-", lw=1.6)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"{main} minus {ref}\n(what the transformer adds)")
    ax.set_ylabel(f"true A1c minus {ref}\n(what '{ref}' misses)")
    ax.set_title(f"5. Added value over '{ref}'\nslope {slope:.2f} (95% CI {lo:.2f} to {hi:.2f}), r {r:.2f}",
                 fontsize=10)
    ax.grid(alpha=0.25)


def panel_error_vs_days(ax, y, df, roles, colors):
    if "n_days" not in df.columns:
        unavailable(ax, "6. Error vs days\nno n_days column")
        return
    main, ref = roles["main"], roles["reference"]
    nd = df["n_days"]
    group = nd if nd.nunique() <= 8 else pd.qcut(nd, 6, duplicates="drop")
    curves = [(main, colors[main])] + ([(ref, colors[ref])] if ref and ref != main else [])
    for m, col in curves:
        ae = np.abs(y - df[m].to_numpy())
        g = pd.DataFrame({"grp": group, "nd": nd, "ae": ae}).groupby("grp", observed=True)
        ax.errorbar(g["nd"].mean(), g["ae"].mean(), yerr=g["ae"].std() / np.sqrt(g["ae"].size()),
                    fmt="-o", capsize=2, color=col, label=f"{m} (group mean +/- SE)")
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(nd))   # nudge dots sideways so they don't overlap
    ax.scatter(nd + jitter, np.abs(y - df[main].to_numpy()), s=10, alpha=0.25, color=colors[main])
    ax.set_xlabel("number of CGM days for the subject")
    ax.set_ylabel("absolute error (A1c % points)")
    ax.set_title("6. Error vs amount of CGM data\n(should fall as days increase)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def plot_diagnostics(df, models, roles, colors, s, path):
    y = df["a1c_true"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    panel_bias(axes[0, 0], y, df, models, roles, colors)
    panel_distribution(axes[0, 1], y, df, models, roles, colors)
    panel_roc(axes[0, 2], y, df, models, roles, colors, s["a1c_threshold"])
    panel_forest(axes[1, 0], y, df, models, roles, s)
    panel_added_value(axes[1, 1], y, df, roles, s)
    panel_error_vs_days(axes[1, 2], y, df, roles, colors)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Tables
# ==========================================================================
def summarize(df, models, roles, s):
    y = df["a1c_true"].to_numpy()
    y_bin = y >= s["a1c_threshold"]
    rows = []
    for m in models:
        p = df[m].to_numpy()
        row = {"model": m, **basic_metrics(y, p)}
        if m == roles["null"]:
            row.update(r=np.nan, cal_slope=np.nan, sd_ratio=0.0, AUC=np.nan)
        else:
            row.update(spread_stats(y, p))
            row["AUC"] = auc_score(y_bin, p) if has_both_classes(y_bin) and np.std(p) > 1e-9 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)[["model", "MAE", "RMSE", "R2", "r", "cal_slope", "sd_ratio", "AUC"]]


# ==========================================================================
# Main
# ==========================================================================
def main(s=SETTINGS):
    os.makedirs(s["out_dir"], exist_ok=True)
    print("Loading predictions ...")
    df, models, roles = load_predictions(s)
    colors = assign_colors(models, roles)
    print(f"  {len(df)} subjects, {len(models)} models: {models}")

    summary = summarize(df, models, roles, s)
    summary.to_csv(os.path.join(s["out_dir"], "a1c_model_summary.csv"), index=False)
    print("\nModel summary (out-of-fold, A1c in % points):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  cal_slope: 1 = well calibrated (<1 overconfident, >1 too shrunk)."
          "\n  sd_ratio : near 0 = predictions collapsed toward the average.")

    print("\nDrawing figures ...")
    plot_scatter_grid(df, models, roles, colors, os.path.join(s["out_dir"], "a1c_scatter_by_model.png"))
    plot_diagnostics(df, models, roles, colors, s, os.path.join(s["out_dir"], "a1c_diagnostics.png"))

    main_model = roles["main"]
    worst = df.assign(pred=df[main_model], error=df[main_model] - df["a1c_true"])
    worst = worst.reindex(worst["error"].abs().sort_values(ascending=False).index).head(10)
    cols = [c for c in ["subject_id", "n_days", "a1c_true", "pred", "error"] if c in worst.columns]
    worst[cols].to_csv(os.path.join(s["out_dir"], "a1c_largest_errors.csv"), index=False)
    print(f"\nLargest errors for '{main_model}' (check these subjects' data and labels):")
    print(worst[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved figures and tables to ./{s['out_dir']}/")


if __name__ == "__main__":
    main()
