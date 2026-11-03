"""
finetune_a1c.py  -  Fine-tune the pretrained CGM transformer to predict A1c
===========================================================================

WHAT "FINE-TUNING" MEANS
------------------------
Pretraining taught the transformer to understand CGM days (by reconstructing hidden
glucose). Fine-tuning takes those pretrained weights as a starting point and keeps
training them on the task we actually care about: predicting A1c.

    pretrained encoder  ->  average over the day's time steps  ->  small "head"  ->  A1c

HOW THE DATA ARE USED
---------------------
* A1c is one number per SUBJECT, but the model reads one DAY at a time. So every
  day is a training example labelled with its subject's A1c. At evaluation time we
  average the predictions of a subject's days to get ONE prediction per subject.
* We re-use the train / val / test split of subjects made earlier (the "split"
  column in cgm_daily_meta.csv):
      train  subjects: the model learns from these
      val    subjects: used to decide when to stop training ("early stopping")
      test   subjects: looked at ONCE, at the very end, for the honest final score
  Pretraining only ever saw train-split subjects' CGM (never any A1c), so val and
  test subjects are completely new to the model.

TRAINING RECIPE (and why)
-------------------------
1. Standardise A1c (subtract the training mean, divide by the training SD) so the
   loss has a sensible scale. Predictions are converted back to A1c % for reporting.
2. "Head warm-up": for the first few epochs only the new head is trained and the
   pretrained encoder is frozen. A brand-new head makes large random errors at
   first; this stops those errors from damaging the pretrained weights.
3. Then everything is trained, but the encoder gets a much SMALLER learning rate
   than the head (we want to nudge it, not rewrite it).
4. Light augmentation: each epoch a few random glucose stretches are hidden, like
   in pretraining. This acts as regularisation.
5. Early stopping: keep the weights from the epoch with the best validation error.
6. Every subject counts equally: a day's loss is weighted by 1 / (number of days
   that subject has), so people with more days don't dominate.

COMPARISONS (so you know whether it is worth it)
------------------------------------------------
   Mean A1c            predict the training-set average for everyone
   Mean glucose        A1c ~ average glucose (fitted on train subjects)
   Mean glucose + SD   adds glucose variability
   Fine-tuned model    the transformer
Run the script a second time with  --scratch  to train the SAME architecture
from random weights. If the pretrained version is not clearly better than the
scratch version, pretraining did not help.

FILES YOU NEED (same folder as this script)
    pretrain_cgm.py, cgm_pretrained.pt, cgm_daily_sequences.npz, cgm_daily_meta.csv
    a1c_labels.csv   one row per subject:   subject_id,a1c

RUN
    python finetune_a1c.py             # start from the pretrained weights
    python finetune_a1c.py --scratch   # control: random starting weights

OUTPUT (folder a1c_finetune/ ; file names end in _pretrained or _scratch)
    a1c_finetuned_*.pt      best model weights
    finetune_log_*.csv      per-epoch training/validation numbers
    metrics_*.csv           validation and test metrics for all models
    test_predictions_*.csv  each test subject's true and predicted A1c
    finetune_results_*.png  learning curves, true-vs-predicted, error summary
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")                  # draw straight to files; no pop-up windows
import matplotlib.pyplot as plt

# Re-use building blocks from the pretraining script so the encoder is identical.
from pretrain_cgm import CGMEncoder, GL, IN_DIM, build_inputs, make_span_mask


# ==========================================================================
# SETTINGS  (edit these; everything else can stay as is)
# ==========================================================================
# A Python "dict" is a set of name -> value pairs, like a named list in R.
SETTINGS = dict(
    # ---- files ------------------------------------------------------------
    ckpt_file="cgm_pretrained.pt",
    seq_file="cgm_daily_sequences.npz",
    meta_file="cgm_daily_meta.csv",
    a1c_file="pseudo_cgm_5min_data_500_subjects.csv",
    label_subject_col="subject_id",    # subject column name in a1c_labels.csv
    label_a1c_col="a1c",               # A1c column name in a1c_labels.csv
    out_dir="a1c_finetune",

    # ---- starting point ---------------------------------------------------
    use_pretrained=True,               # False = random weights (or run with --scratch)
    min_days_per_subject=3,            # subjects with fewer valid days are dropped

    # ---- training ---------------------------------------------------------
    batch_size=64,
    epochs=30,
    patience=6,                        # stop if val error hasn't improved for 6 epochs
    head_warmup_epochs=2,              # epochs with the encoder frozen (pretrained only)
    lr_encoder=5e-5,                   # small: gently adjust pretrained weights
    lr_encoder_scratch=3e-4,           # larger: used when training from random weights
    lr_head=1e-3,                      # the new head learns faster
    weight_decay=0.01,
    warmup_frac=0.05,                  # fraction of steps with a rising learning rate
    grad_clip=1.0,
    head_dropout=0.1,
    aug_mask_ratio=0.05,               # fraction of glucose hidden at random each epoch
    weight_by_subject=True,            # each subject counts equally

    # ---- statistics -------------------------------------------------------
    n_boot=2000,                       # bootstrap repeats for 95% intervals
    seed=42,                           # fixes randomness so runs repeat
)

# Display names for the models (defined once so tables and plots match).
NULL = "Mean A1c"
GLU = "Mean glucose"
GLU_SD = "Mean glucose + SD"


# ==========================================================================
# STEP 1: load and prepare the data
# ==========================================================================
def load_labels(s):
    """Read the A1c file -> DataFrame with columns subject_id (text) and a1c (number)."""
    lab = pd.read_csv(s["a1c_file"])
    lab = lab.rename(columns={s["label_subject_col"]: "subject_id", s["label_a1c_col"]: "a1c"})
    lab = lab[["subject_id", "a1c"]]
    lab["subject_id"] = lab["subject_id"].astype(str).str.strip()   # ids as text
    lab["a1c"] = pd.to_numeric(lab["a1c"], errors="coerce")         # bad entries -> NaN
    lab = lab.dropna(subset=["a1c"]).drop_duplicates("subject_id")
    if not lab["a1c"].between(3, 20).all():
        print("  WARNING: some A1c values are outside 3-20. Are they in % (not mmol/mol)?")
    return lab


def prepare_data(s):
    """
    Load the CGM days and attach each subject's A1c.

    Returns
        X     (n_days, 288, 5)  features per 5-minute step
        obs   (n_days, 288)     True where a real glucose reading exists
        meta  one row per day: subject_id, date, split, a1c, n_days
    Days are dropped if the subject has too few days or no A1c value.
    """
    labels = load_labels(s)
    d = np.load(s["seq_file"])
    X = d["X"].astype(np.float32)
    obs = d["obs_mask"].astype(bool)
    meta = pd.read_csv(s["meta_file"])
    if "split" not in meta.columns:
        raise SystemExit("cgm_daily_meta.csv has no 'split' column. Re-run make_cgm_sequences.py.")
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()

    # Drop subjects with too few days.
    n_days = meta.groupby("subject_id")["subject_id"].transform("size")
    keep = (n_days >= s["min_days_per_subject"]).to_numpy()
    X, obs, meta = X[keep], obs[keep], meta[keep].reset_index(drop=True)

    # A subject must belong to exactly one split, otherwise test subjects would leak.
    if (meta.groupby("subject_id")["split"].nunique() > 1).any():
        raise SystemExit("Some subjects appear in more than one split. Fix the split first.")

    # Attach A1c (a "left join": keeps every day, adds the matching label).
    meta = meta.merge(labels, on="subject_id", how="left")
    has_label = meta["a1c"].notna().to_numpy()
    n_unlabelled = meta.loc[~has_label, "subject_id"].nunique()
    if n_unlabelled:
        print(f"  Dropping {n_unlabelled} subjects without an A1c value")
    X, obs, meta = X[has_label], obs[has_label], meta[has_label].reset_index(drop=True)
    meta["n_days"] = meta.groupby("subject_id")["subject_id"].transform("size")
    return X, obs, meta


def glucose_summary_by_subject(X, obs, subject_ids):
    """
    Each subject's mean and SD of normalised glucose over all valid readings.
    Used only for the simple baselines. (norm_gl is a linear rescaling of glucose,
    so the mean of norm_gl carries the same information as mean glucose in mg/dL.)
    """
    rows = {}
    ids = np.asarray(subject_ids)
    for subj in np.unique(ids):
        idx = np.where(ids == subj)[0]
        vals = X[idx][:, :, GL][obs[idx]]
        rows[subj] = {"mean_gl": vals.mean(), "sd_gl": vals.std()}
    return pd.DataFrame.from_dict(rows, orient="index")


# ==========================================================================
# STEP 2: the model = pretrained encoder + a small regression head
# ==========================================================================
class A1cModel(nn.Module):
    """
    A "class" is a blueprint for an object. This one holds two parts:
        encoder : the pretrained transformer (one vector per 5-minute step)
        head    : turns the day's average vector into a single A1c number
    """

    def __init__(self, encoder, d_model, head_dropout):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, 1))

    def forward(self, x_in, pad_mask):
        # x_in: (batch, 288, 6) model input; pad_mask: True where data are missing
        h = self.encoder(x_in, pad_mask)                    # (batch, 288, d_model)
        w = (~pad_mask).unsqueeze(-1).float()               # 1 for real steps, 0 for missing
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)   # average over real steps
        return self.head(pooled).squeeze(-1)                # (batch,) standardised A1c


def build_model(s, device):
    """Create the model; load pretrained encoder weights unless training from scratch."""
    ckpt = torch.load(s["ckpt_file"], map_location=device)
    cfg = ckpt["cfg"]                                       # architecture used in pretraining
    encoder = CGMEncoder(IN_DIM, cfg["d_model"], cfg["n_heads"], cfg["n_layers"],
                         cfg["d_ff"], cfg["dropout"])
    if s["use_pretrained"]:
        encoder.load_state_dict(ckpt["encoder_state"])
        print("  Loaded pretrained encoder weights")
    else:
        print("  Using RANDOM encoder weights (scratch control)")
    return A1cModel(encoder, cfg["d_model"], s["head_dropout"]).to(device), cfg


def set_encoder_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) the encoder's weights."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable


# ==========================================================================
# STEP 3: training and prediction
# ==========================================================================
def train_one_epoch(model, loader, opt, sched, device, s, span_len):
    """One pass over the training days. Returns the average (weighted) loss."""
    model.train()                                           # turns dropout ON
    total, count = 0.0, 0
    for x, o, y, w in loader:                               # x: days, o: real-data mask, y: A1c, w: weight
        # Augmentation: hide a few random glucose stretches (on CPU, then move).
        masked = make_span_mask(o, s["aug_mask_ratio"], span_len)
        x, o, masked, y, w = (t.to(device) for t in (x, o, masked, y, w))

        pred = model(build_inputs(x, o, masked), ~o)
        loss = (w * (pred - y) ** 2).sum() / w.sum()        # weighted mean squared error

        opt.zero_grad()                                     # clear old gradients
        loss.backward()                                     # compute new gradients
        nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"])   # avoid huge steps
        opt.step()                                          # update weights
        sched.step()                                        # update learning rate
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def predict_days(model, X, obs, device, batch_size=256):
    """Predict (standardised) A1c for every day, with nothing hidden."""
    model.eval()                                            # turns dropout OFF
    out = []
    with torch.no_grad():                                   # no gradient bookkeeping
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(device)
            o = obs[i:i + batch_size].to(device)
            out.append(model(build_inputs(x, o, torch.zeros_like(o)), ~o).cpu())
    return torch.cat(out).numpy()


def predict_subjects(model, X, obs, subject_ids, y_mean, y_std, device):
    """
    Day-level predictions -> ONE prediction per subject (the average over their days),
    converted back to A1c %. Returns a pandas Series indexed by subject_id.
    """
    day_pred = predict_days(model, X, obs, device) * y_std + y_mean
    df = pd.DataFrame({"subject_id": np.asarray(subject_ids), "pred": day_pred})
    return df.groupby("subject_id")["pred"].mean()


# ==========================================================================
# STEP 4: metrics, baselines, bootstrap
# ==========================================================================
def compute_metrics(y, p):
    """
    How close predictions p are to truth y (both in A1c % points).
      MAE  mean absolute error (easy to interpret)
      RMSE root mean squared error (punishes big misses)
      R2   1 = perfect, 0 = no better than guessing the mean, negative = worse
      r    Pearson correlation;  rho  Spearman (rank) correlation
    """
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan
    rho = pd.Series(y).rank().corr(pd.Series(p).rank()) if np.std(p) > 0 else np.nan
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "r": float(r), "rho": float(rho), "n_subjects": len(y),
    }


def baseline_predictions(glucose, truth, train_ids, eval_ids):
    """
    Simple comparison models, fitted on TRAIN subjects only, applied to eval_ids.
    Returns {model name: predictions array}.
    """
    y_train = truth.loc[train_ids].to_numpy()
    out = {NULL: np.full(len(eval_ids), y_train.mean())}
    for name, cols in [(GLU, ["mean_gl"]), (GLU_SD, ["mean_gl", "sd_gl"])]:
        A = np.column_stack([np.ones(len(train_ids)), glucose.loc[train_ids, cols].to_numpy()])
        coef = np.linalg.lstsq(A, y_train, rcond=None)[0]     # ordinary least squares
        B = np.column_stack([np.ones(len(eval_ids)), glucose.loc[eval_ids, cols].to_numpy()])
        out[name] = B @ coef
    return out


def bootstrap_mae_ci(y, p, n_boot, seed):
    """95% interval for MAE: re-draw subjects with replacement many times."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    boot = np.abs(y - p)[idx].mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def bootstrap_paired_diff_ci(y, p_model, p_base, n_boot, seed):
    """
    95% interval for (MAE of model - MAE of baseline) using the SAME resampled
    subjects for both. Entirely below 0 => the model is reliably better.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    diff = np.abs(y - p_model)[idx].mean(axis=1) - np.abs(y - p_base)[idx].mean(axis=1)
    return tuple(np.percentile(diff, [2.5, 97.5]))


# ==========================================================================
# STEP 5: plots
# ==========================================================================
def plot_results(path, log, y_test, preds_test, ft_name, best_epoch, metrics_test):
    """2x2 figure: learning curves, test scatter, MAE comparison, bias by A1c level."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) learning curves -----------------------------------------------------
    ax = axes[0, 0]
    ax.plot(log["epoch"], log["train_loss"], color="tab:blue", label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (standardised MSE)", color="tab:blue")
    ax2 = ax.twinx()                                        # second y axis on the right
    ax2.plot(log["epoch"], log["val_RMSE"], color="tab:red", label="val RMSE")
    ax2.set_ylabel("validation RMSE (A1c % points)", color="tab:red")
    ax.axvline(best_epoch, color="k", ls=":", lw=1)
    ax.set_title(f"Learning curves (dotted = best epoch {best_epoch})")
    ax.grid(alpha=0.25)

    # (2) test scatter --------------------------------------------------------
    ax = axes[0, 1]
    p = preds_test[ft_name]
    ax.scatter(y_test, p, s=18, alpha=0.5, color="tab:red")
    lo, hi = min(y_test.min(), p.min()) - 0.2, max(y_test.max(), p.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)                # perfect-prediction line
    m = metrics_test.set_index("model").loc[ft_name]
    ax.set_title(f"Test subjects: true vs predicted A1c\nMAE {m['MAE']:.2f}   r {m['r']:.2f}   R2 {m['R2']:.2f}")
    ax.set_xlabel("true A1c (%)")
    ax.set_ylabel("predicted A1c (%)")
    ax.grid(alpha=0.25)

    # (3) MAE comparison ------------------------------------------------------
    ax = axes[1, 0]
    mt = metrics_test.set_index("model")
    names = list(mt.index)
    maes = mt["MAE"].to_numpy()
    err = np.vstack([maes - mt["MAE_ci_low"].to_numpy(), mt["MAE_ci_high"].to_numpy() - maes])
    ax.bar(range(len(names)), maes, yerr=err, capsize=3,
           color=["tab:red" if n == ft_name else "0.6" for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (A1c % points), lower = better")
    ax.set_title("Test error with 95% bootstrap intervals")

    # (4) bias by true A1c level ---------------------------------------------
    # Models tend to shrink toward the middle: high A1c under-predicted, low over-predicted.
    ax = axes[1, 1]
    resid = p - y_test
    n_bins = int(min(10, max(2, len(y_test) // 15)))
    bins = pd.qcut(pd.Series(y_test), n_bins, duplicates="drop")
    g = pd.DataFrame({"y": y_test, "res": resid, "bin": bins}).groupby("bin", observed=True)
    xs, ys = g["y"].mean(), g["res"].mean()
    es = g["res"].std() / np.sqrt(g["res"].size())
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color="tab:purple", capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("true A1c, bin means (%)")
    ax.set_ylabel("mean (predicted - true)")
    ax.set_title("Bias by A1c level\n(downward slope = shrinks toward the mean)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# MAIN: run everything in order
# ==========================================================================
def main(s=SETTINGS):
    if "--scratch" in sys.argv:                             # command-line switch
        s = dict(s, use_pretrained=False)
    tag = "pretrained" if s["use_pretrained"] else "scratch"
    ft_name = f"Fine-tuned transformer ({tag})"
    os.makedirs(s["out_dir"], exist_ok=True)
    torch.manual_seed(s["seed"])
    np.random.seed(s["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | starting point: {tag}")

    # ---- load data ------------------------------------------------------------
    print("[1/5] Loading data ...")
    X, obs, meta = prepare_data(s)
    sub_info = meta.drop_duplicates("subject_id").set_index("subject_id")
    truth = sub_info["a1c"]                                 # true A1c, indexed by subject
    rows = {sp: np.where(meta["split"] == sp)[0] for sp in ["train", "val", "test"]}
    for sp, idx in rows.items():
        n_subj = meta.loc[idx, "subject_id"].nunique()
        print(f"  {sp:5s}: {len(idx):6d} days, {n_subj:5d} subjects")
        if n_subj == 0:
            raise SystemExit(f"No labelled subjects in the '{sp}' split.")

    # Standardise A1c using TRAIN subjects only.
    train_ids = sub_info.index[sub_info["split"] == "train"]
    y_mean, y_std = float(truth.loc[train_ids].mean()), float(truth.loc[train_ids].std())
    print(f"  train A1c: mean {y_mean:.2f}, SD {y_std:.2f}")

    # Tensors (PyTorch's version of arrays) for the three splits.
    def to_tensors(idx):
        return torch.from_numpy(X[idx]), torch.from_numpy(obs[idx])

    y_day = ((meta["a1c"].to_numpy() - y_mean) / y_std).astype(np.float32)
    w_day = (1.0 / meta["n_days"].to_numpy() if s["weight_by_subject"]
             else np.ones(len(meta))).astype(np.float32)
    Xtr, otr = to_tensors(rows["train"])
    train_loader = DataLoader(
        TensorDataset(Xtr, otr, torch.from_numpy(y_day[rows["train"]]),
                      torch.from_numpy(w_day[rows["train"]])),
        batch_size=s["batch_size"], shuffle=True, drop_last=len(Xtr) > s["batch_size"],
    )
    eval_data = {sp: (*to_tensors(rows[sp]), meta.loc[rows[sp], "subject_id"].to_numpy())
                 for sp in ["val", "test"]}

    # ---- model and optimiser -----------------------------------------------------
    print("[2/5] Building model ...")
    model, cfg = build_model(s, device)
    enc_lr = s["lr_encoder"] if s["use_pretrained"] else s["lr_encoder_scratch"]
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": enc_lr},       # gentle for the encoder
         {"params": model.head.parameters(), "lr": s["lr_head"]}],   # faster for the new head
        weight_decay=s["weight_decay"],
    )
    total_steps = s["epochs"] * len(train_loader)
    warm = max(1, int(s["warmup_frac"] * total_steps))

    def lr_lambda(step):                                    # warm-up, then cosine decay
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- training loop -------------------------------------------------------------
    print("[3/5] Training ...")
    head_warmup = s["head_warmup_epochs"] if s["use_pretrained"] else 0
    best_rmse, best_epoch, best_state, bad_epochs = np.inf, 0, None, 0
    history = []
    for epoch in range(1, s["epochs"] + 1):
        stage = "head-only" if epoch <= head_warmup else "full"
        set_encoder_trainable(model, stage == "full")
        train_loss = train_one_epoch(model, train_loader, opt, sched, device, s, cfg["span_len"])

        Xv, ov, sv = eval_data["val"]
        val_pred = predict_subjects(model, Xv, ov, sv, y_mean, y_std, device)
        vm = compute_metrics(truth.loc[val_pred.index].to_numpy(), val_pred.to_numpy())
        history.append(dict(epoch=epoch, stage=stage, train_loss=train_loss,
                            **{f"val_{k}": vm[k] for k in ["MAE", "RMSE", "R2", "r"]}))

        flag = ""
        if vm["RMSE"] < best_rmse - 1e-6:                   # improved on validation
            best_rmse, best_epoch, bad_epochs = vm["RMSE"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            flag = "  <- best"
        else:
            bad_epochs += 1
        print(f"  epoch {epoch:2d} [{stage:9s}] train loss {train_loss:.3f} | "
              f"val MAE {vm['MAE']:.3f} RMSE {vm['RMSE']:.3f} R2 {vm['R2']:.3f}{flag}")
        if bad_epochs >= s["patience"] and epoch > head_warmup:
            print(f"  Early stopping: no improvement for {s['patience']} epochs")
            break

    log = pd.DataFrame(history)
    log.to_csv(os.path.join(s["out_dir"], f"finetune_log_{tag}.csv"), index=False)
    model.load_state_dict(best_state)                       # go back to the best epoch
    torch.save({"model_state": best_state, "cfg": cfg, "y_mean": y_mean, "y_std": y_std,
                "best_epoch": best_epoch, "settings": s},
               os.path.join(s["out_dir"], f"a1c_finetuned_{tag}.pt"))

    # ---- final evaluation (test set is used only here) ---------------------------------
    print("[4/5] Evaluating best model on val and test ...")
    glucose = glucose_summary_by_subject(X, obs, meta["subject_id"])
    metric_rows, preds_test, y_test = [], {}, None
    for sp in ["val", "test"]:
        Xe, oe, se = eval_data[sp]
        pred = predict_subjects(model, Xe, oe, se, y_mean, y_std, device)
        ids = list(pred.index)
        y = truth.loc[ids].to_numpy()
        all_preds = {ft_name: pred.to_numpy()}
        all_preds.update(baseline_predictions(glucose, truth, list(train_ids), ids))
        for name in [NULL, GLU, GLU_SD, ft_name]:           # fixed display order
            m = compute_metrics(y, all_preds[name])
            m.update(split=sp, model=name)
            lo, hi = bootstrap_mae_ci(y, all_preds[name], s["n_boot"], s["seed"])
            m.update(MAE_ci_low=lo, MAE_ci_high=hi)
            if name == ft_name:                             # paired comparison vs mean glucose
                dlo, dhi = bootstrap_paired_diff_ci(y, all_preds[ft_name], all_preds[GLU],
                                                    s["n_boot"], s["seed"])
                m.update(dMAE_vs_meanglucose=m["MAE"] - compute_metrics(y, all_preds[GLU])["MAE"],
                         dMAE_ci_low=dlo, dMAE_ci_high=dhi)
            metric_rows.append(m)
        if sp == "test":
            y_test, preds_test = y, all_preds
            pd.DataFrame({"subject_id": ids, "a1c_true": y, "a1c_pred": all_preds[ft_name],
                          "n_days": sub_info.loc[ids, "n_days"].to_numpy()}
                         ).to_csv(os.path.join(s["out_dir"], f"test_predictions_{tag}.csv"), index=False)

    metrics = pd.DataFrame(metric_rows)
    first = ["split", "model", "n_subjects", "MAE", "MAE_ci_low", "MAE_ci_high",
             "RMSE", "R2", "r", "rho", "dMAE_vs_meanglucose", "dMAE_ci_low", "dMAE_ci_high"]
    metrics = metrics[first]
    metrics.to_csv(os.path.join(s["out_dir"], f"metrics_{tag}.csv"), index=False)

    print("[5/5] Plotting ...")
    plot_results(os.path.join(s["out_dir"], f"finetune_results_{tag}.png"), log, y_test,
                 preds_test, ft_name, best_epoch, metrics[metrics["split"] == "test"])

    print(f"\nTEST results (A1c in % points; best epoch {best_epoch}):")
    show = metrics[metrics["split"] == "test"][["model", "MAE", "RMSE", "R2", "r"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    row = metrics[(metrics["split"] == "test") & (metrics["model"] == ft_name)].iloc[0]
    print(f"\nMAE change vs 'Mean glucose': {row['dMAE_vs_meanglucose']:+.3f} "
          f"(95% CI {row['dMAE_ci_low']:+.3f} to {row['dMAE_ci_high']:+.3f}); "
          "an interval entirely below 0 means the transformer is reliably better.")
    print(f"Saved results to ./{s['out_dir']}/")


if __name__ == "__main__":
    main()
