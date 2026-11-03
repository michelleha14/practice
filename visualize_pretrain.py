"""
Diagnostics for the masked-glucose pretraining (pretrain_cgm.py).

Run AFTER pretraining:   python visualize_pretrain.py

Creates in ./viz :
    per_subject_predictions.pdf  one page per subject; one panel per day showing
                                 original data, what the model saw, and predictions
    example_subject.png          first page of that PDF as an image
    diagnostics.png              training curves, predicted-vs-true, baselines,
                                 error by hour, bias by glucose level, per-subject
    embeddings_pca.png           2-D PCA of day embeddings
    per_subject_metrics.csv      numbers behind the per-subject panel
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pretrain_cgm import GL, MaskedGlucoseModel, make_span_mask, build_inputs

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
VIZ = dict(
    ckpt_file="cgm_pretrained.pt",
    seq_file="cgm_daily_sequences.npz",
    meta_file="cgm_daily_meta.csv",
    log_file="training_log.csv",            # written by the updated pretrain_cgm.py
    out_dir="viz",
    diag_splits=("val", "test"),            # days used for the diagnostic plots
    pdf_splits=("val", "test"),             # subjects shown in the per-subject PDF
                                            # add "train" to compare with training subjects
    max_subjects_pdf=None,                  # e.g. 10 to limit PDF size
    gl_mean=None, gl_std=None,              # set both to plot in mg/dL instead of normalized units
    mask_seed=123,                          # fixed seed -> same masks every run
)

T = 288


# --------------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------------
def predict(model, X, obs, masked, device, bs=128):
    """Reconstruct glucose for every step, given the masked input."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, o, m = (t[i:i + bs].to(device) for t in (X, obs, masked))
            out.append(model(build_inputs(x, o, m), ~o).cpu())
    return torch.cat(out).numpy()


def embed(model, X, obs, device, bs=128):
    """One vector per day: mean of encoder outputs over observed steps (nothing masked)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, o = (t[i:i + bs].to(device) for t in (X, obs))
            h = model.encoder(build_inputs(x, o, torch.zeros_like(o)), ~o)
            w = o.unsqueeze(-1).float()
            out.append(((h * w).sum(1) / w.sum(1).clamp(min=1.0)).cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------
# Baselines and metrics
# --------------------------------------------------------------------------
def interp_baseline(gl, obs, masked):
    """Linear interpolation across the hidden steps, using only visible points."""
    out = np.zeros_like(gl)
    t = np.arange(gl.shape[1])
    for i in range(len(gl)):
        vis = obs[i] & ~masked[i]
        out[i] = np.interp(t, t[vis], gl[i, vis]) if vis.any() else 0.0
    return out


def daymean_baseline(gl, obs, masked):
    """Predict the mean of that day's visible glucose everywhere."""
    vis = obs & ~masked
    mean = (gl * vis).sum(1) / np.maximum(vis.sum(1), 1)
    return np.repeat(mean[:, None], gl.shape[1], axis=1)


def rmse(pred, true, m):
    return float(np.sqrt(((pred - true)[m] ** 2).mean())) if m.any() else float("nan")


def rmse_by_hour(pred, true, m):
    out = np.full(24, np.nan)
    for h in range(24):
        sl = slice(h * 12, (h + 1) * 12)
        mm = m[:, sl]
        if mm.any():
            out[h] = np.sqrt(((pred[:, sl] - true[:, sl])[mm] ** 2).mean())
    return out


def per_subject_table(meta, true, pred, interp, dmean, m):
    rows = []
    for subj, idx in meta.groupby("subject_id").groups.items():
        idx = np.asarray(idx)
        mm = m[idx]
        rows.append(dict(
            subject_id=subj, split=meta.loc[idx[0], "split"], n_days=len(idx),
            n_masked_steps=int(mm.sum()),
            rmse_model=rmse(pred[idx], true[idx], mm),
            rmse_interp=rmse(interp[idx], true[idx], mm),
            rmse_daymean=rmse(dmean[idx], true[idx], mm),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plot 1: per-subject pages (original / what the model saw / predictions)
# --------------------------------------------------------------------------
def plot_subject_pages(meta, gl, pred, obs, masked, units, pdf_path, png_path, max_subjects):
    hours = np.arange(T) / 12
    handles = [
        Line2D([0], [0], color="0.65", lw=3, label="original data"),
        Line2D([0], [0], color="tab:blue", lw=1.2, label="visible to model"),
        Patch(color="tab:orange", alpha=0.25, label="masked span"),
        Line2D([0], [0], color="tab:red", lw=1.4, marker=".", label="model prediction"),
    ]
    subjects = meta.loc[meta["pdf_row"], "subject_id"].unique()
    if max_subjects:
        subjects = subjects[:max_subjects]

    with PdfPages(pdf_path) as pdf:
        for k, subj in enumerate(subjects):
            sel = meta[(meta["subject_id"] == subj) & meta["pdf_row"]].sort_values("date")
            rows = sel.index.to_numpy()
            n, ncol = len(rows), 2
            nrow = int(np.ceil(n / ncol))
            fig_h = 2.3 * nrow + 1.0
            fig, axes = plt.subplots(nrow, ncol, figsize=(14, fig_h), sharex=True, squeeze=False)
            axes = axes.ravel()

            for ax, i in zip(axes, rows):
                o, m = obs[i], masked[i]
                ax.fill_between(hours, 0, 1, where=m, transform=ax.get_xaxis_transform(),
                                color="tab:orange", alpha=0.18, lw=0)
                ax.plot(hours, np.where(o, gl[i], np.nan), color="0.65", lw=3,
                        solid_capstyle="round")
                ax.plot(hours, np.where(o & ~m, gl[i], np.nan), color="tab:blue", lw=1.2)
                ax.plot(hours, np.where(m, pred[i], np.nan), color="tab:red", lw=1.4,
                        marker=".", ms=3)
                day_rmse = rmse(pred[i], gl[i], m)
                ax.set_title(f"{meta.loc[i, 'date']}   |   masked-step RMSE {day_rmse:.3f}",
                             fontsize=9, loc="left")
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 6, 12, 18, 24])
                ax.grid(alpha=0.25)
            for ax in axes[n:]:
                ax.set_visible(False)
            for ax in axes[: n]:
                ax.set_ylabel(f"glucose ({units})", fontsize=8)
            for ax in axes[max(0, n - ncol): n]:
                ax.set_xlabel("hour of day", fontsize=8)

            split = sel["split"].iloc[0]
            fig.suptitle(f"Subject {subj}  ({split}, {n} days)", x=0.01, ha="left",
                         y=0.995, fontsize=12, fontweight="bold")
            fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.995),
                       ncol=4, fontsize=8, frameon=False)
            fig.tight_layout(rect=[0, 0, 1, 1 - 0.40 / fig_h])
            pdf.savefig(fig)
            if k == 0:
                fig.savefig(png_path, dpi=130)
            plt.close(fig)


# --------------------------------------------------------------------------
# Plot 2: diagnostics grid
# --------------------------------------------------------------------------
def plot_diagnostics(path, log_file, best_epoch, true, pred, interp, dmean, m, subj_df, units):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # (1) training curves ---------------------------------------------------
    ax = axes[0, 0]
    if os.path.exists(log_file):
        log = pd.read_csv(log_file)
        ax.plot(log["epoch"], log["train_mse"], label="train (random masks)")
        ax.plot(log["epoch"], log["val_mse"], label="val (fixed masks)")
        ax.plot(log["epoch"], log["baseline_mse"], "k--", lw=1, label="day-mean baseline (val)")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="tab:red", ls=":", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (normalized units)")
        ax.set_title("Training curves")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, f"{log_file} not found.\nRe-run the updated pretrain_cgm.py\nto record training curves.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # (2) predicted vs true -------------------------------------------------
    ax = axes[0, 1]
    t, p = true[m], pred[m]
    hb = ax.hexbin(t, p, gridsize=50, bins="log", mincnt=1, cmap="viridis")
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel(f"true glucose ({units})")
    ax.set_ylabel(f"predicted glucose ({units})")
    ax.set_title(f"Predicted vs true at masked steps\nRMSE {rmse(pred, true, m):.3f},  r = {np.corrcoef(t, p)[0, 1]:.3f}")
    fig.colorbar(hb, ax=ax, label="count (log scale)")

    # (3) model vs baselines -----------------------------------------------
    ax = axes[0, 2]
    names = ["Transformer", "Linear\ninterpolation", "Day mean"]
    vals = [rmse(pred, true, m), rmse(interp, true, m), rmse(dmean, true, m)]
    bars = ax.bar(names, vals, color=["tab:red", "tab:green", "0.6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel(f"RMSE at masked steps ({units})")
    ax.set_title("Model vs simple baselines (lower is better)")

    # (4) error by hour of day ---------------------------------------------
    ax = axes[1, 0]
    ax.plot(range(24), rmse_by_hour(pred, true, m), "-o", color="tab:red", ms=3, label="Transformer")
    ax.plot(range(24), rmse_by_hour(interp, true, m), "-o", color="tab:green", ms=3, label="Linear interpolation")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("RMSE")
    ax.set_title("Error by time of day")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (5) bias by true glucose level ---------------------------------------
    ax = axes[1, 1]
    res = (pred - true)[m]
    edges = np.quantile(t, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(t, edges[1:-1]), 0, 9)
    xs, ys, es = [], [], []
    for k in range(10):
        sel = b == k
        if sel.sum() > 1:
            xs.append(t[sel].mean())
            ys.append(res[sel].mean())
            es.append(res[sel].std() / np.sqrt(sel.sum()))
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color="tab:purple", ms=4, capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(f"true glucose, decile means ({units})")
    ax.set_ylabel("mean (predicted - true)")
    ax.set_title("Bias by glucose level\n(slope down = shrinks toward the mean)")
    ax.grid(alpha=0.25)

    # (6) per-subject -------------------------------------------------------
    ax = axes[1, 2]
    if len(subj_df):
        ax.scatter(subj_df["rmse_interp"], subj_df["rmse_model"], s=28, color="tab:red", alpha=0.8)
        top = max(subj_df["rmse_interp"].max(), subj_df["rmse_model"].max()) * 1.05
        ax.plot([0, top], [0, top], "k--", lw=1)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        better = int((subj_df["rmse_model"] < subj_df["rmse_interp"]).sum())
        ax.set_title(f"Per-subject RMSE (each dot = subject)\nTransformer better for {better}/{len(subj_df)}")
    ax.set_xlabel("linear interpolation RMSE")
    ax.set_ylabel("Transformer RMSE")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 3: embeddings (PCA)
# --------------------------------------------------------------------------
def plot_embeddings(path, Z, day_mean, day_sd, subject_codes, units):
    if len(Z) < 3:
        print("Too few days for the embedding plot, skipping.")
        return
    Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    pcs = Zc @ Vt[:2].T
    var = S ** 2 / np.sum(S ** 2)

    for name, v in [("mean glucose", day_mean), ("glucose SD", day_sd)]:
        r1, r2 = np.corrcoef(pcs[:, 0], v)[0, 1], np.corrcoef(pcs[:, 1], v)[0, 1]
        print(f"Correlation of PC1 / PC2 with day {name}: {r1:+.2f} / {r2:+.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    panels = [
        (day_mean, "viridis", f"day mean glucose ({units})"),
        (day_sd, "magma", f"day glucose SD ({units})"),
        (subject_codes % 20, "tab20", "subject (colors repeat)"),
    ]
    for ax, (c, cmap, label) in zip(axes, panels):
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, cmap=cmap, s=18)
        if "subject" not in label:
            fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"Colored by {label}", fontsize=10)
        ax.set_xlabel(f"PC1 ({var[0]:.0%})")
        ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    fig.suptitle("Day embeddings (mean-pooled encoder output, nothing masked)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(v=VIZ):
    os.makedirs(v["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model ----------------------------------------------------------
    ckpt = torch.load(v["ckpt_file"], map_location=device)
    cfg = ckpt["cfg"]
    model = MaskedGlucoseModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val MSE {ckpt['val_mse']:.4f})")

    # ---- data -----------------------------------------------------------
    d = np.load(v["seq_file"])
    meta = pd.read_csv(v["meta_file"])
    keep = meta["split"].isin(set(v["diag_splits"]) | set(v["pdf_splits"])).to_numpy()
    if keep.sum() == 0:
        raise SystemExit("No days found for the requested splits. Check diag_splits / pdf_splits.")
    keep_t = torch.from_numpy(keep)
    X = torch.from_numpy(d["X"]).float()[keep_t]
    obs = torch.from_numpy(d["obs_mask"]).bool()[keep_t]
    meta = meta[keep].reset_index(drop=True)
    meta["diag_row"] = meta["split"].isin(v["diag_splits"])
    meta["pdf_row"] = meta["split"].isin(v["pdf_splits"])
    print(f"Days used: {len(meta)} from {meta['subject_id'].nunique()} subjects")

    # ---- masks + predictions -------------------------------------------
    g = torch.Generator().manual_seed(v["mask_seed"])
    masked = make_span_mask(obs, cfg["mask_ratio"], cfg["span_len"], g)
    pred = predict(model, X, obs, masked, device)

    gl, obs_np, masked_np = X[..., GL].numpy(), obs.numpy(), masked.numpy()
    interp = interp_baseline(gl, obs_np, masked_np)
    dmean = daymean_baseline(gl, obs_np, masked_np)

    # optional conversion back to mg/dL (all quantities are affine in glucose)
    if v["gl_mean"] is not None and v["gl_std"] is not None:
        conv = lambda a: a * v["gl_std"] + v["gl_mean"]
        gl, pred, interp, dmean = conv(gl), conv(pred), conv(interp), conv(dmean)
        units = "mg/dL"
    else:
        units = "normalized units"

    # ---- plots ----------------------------------------------------------
    out = v["out_dir"]
    plot_subject_pages(
        meta, gl, pred, obs_np, masked_np, units,
        os.path.join(out, "per_subject_predictions.pdf"),
        os.path.join(out, "example_subject.png"),
        v["max_subjects_pdf"],
    )

    rows = np.where(meta["diag_row"].to_numpy())[0]
    mdiag = meta.iloc[rows].reset_index(drop=True)
    subj_df = per_subject_table(mdiag, gl[rows], pred[rows], interp[rows], dmean[rows], masked_np[rows])
    subj_df.to_csv(os.path.join(out, "per_subject_metrics.csv"), index=False)

    plot_diagnostics(
        os.path.join(out, "diagnostics.png"), v["log_file"], ckpt.get("epoch"),
        gl[rows], pred[rows], interp[rows], dmean[rows], masked_np[rows], subj_df, units,
    )

    Z = embed(model, X[torch.from_numpy(rows)], obs[torch.from_numpy(rows)], device)
    day_vals = np.where(obs_np[rows], gl[rows], np.nan)
    plot_embeddings(
        os.path.join(out, "embeddings_pca.png"), Z,
        np.nanmean(day_vals, axis=1), np.nanstd(day_vals, axis=1),
        pd.factorize(mdiag["subject_id"])[0], units,
    )

    m = masked_np[rows]
    print(f"\nHeld-out RMSE at masked steps ({units}):")
    print(f"  Transformer        {rmse(pred[rows], gl[rows], m):.4f}")
    print(f"  Linear interpolat. {rmse(interp[rows], gl[rows], m):.4f}")
    print(f"  Day mean           {rmse(dmean[rows], gl[rows], m):.4f}")
    print(f"\nSaved plots to ./{out}/")


if __name__ == "__main__":
    main()
