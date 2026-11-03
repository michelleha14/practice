"""
Masked-glucose pretraining of a transformer encoder on daily CGM sequences.

Inputs (made by make_cgm_sequences.py):
    cgm_daily_sequences.npz  -> X (N, 288, 5), obs_mask (N, 288)
    cgm_daily_meta.csv       -> subject_id, date, coverage, split

Feature order in X: hour_sin, hour_cos, norm_gl, norm_speed, norm_acceleration

Task: hide some glucose stretches (and the speed/acceleration at the same
steps, because those are derived from glucose and would leak the answer),
then train the model to reconstruct the hidden norm_gl values.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------
CFG = dict(
    seq_file="cgm_daily_sequences.npz",
    meta_file="cgm_daily_meta.csv",
    out_file="cgm_pretrained.pt",
    # masking
    mask_ratio=0.10,     # fraction of steps hidden per day (approx.)
    span_len=6,          # hide contiguous chunks of 6 steps (30 min). 1 = random single points
    # model
    d_model=128, # size of embedding table
    n_heads=4, # slice embedding into 4 sections
    n_layers=4, # attention > layer norm > feedforward 4 time
    d_ff=256, # dimension of feedforward, number of nodes (size of dense layer)
    dropout=0.1, # randomly turn off the nodes to prevent overfitting; a form of regularization
    # training
    batch_size=32, # looking at 64 random sequences out of 193 at the same time, 
    # when we calculate the loss )to calc the gradient) to update the model, we want it to reflect the entire data
    epochs=500, # itiration through entire dataset
    lr=3e-4, # learning rate
    weight_decay=0.01, # making lr smaller at each training step
    warmup_frac=0.05, 
    grad_clip=1.0, # capping gradient to avoid inf, often not needed
    seed=42,
)

GL = 2                      # index of norm_gl in the feature axis
SIGNAL_IDX = [2, 3, 4]      # norm_gl, norm_speed, norm_acceleration
N_FEATURES = 5
IN_DIM = N_FEATURES + 1     # + 1 flag channel ("this step is hidden/missing")


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def make_span_mask(obs_mask, ratio, span_len, generator=None):
    """
    Returns a bool (B, T) tensor: True = step is hidden for the pretext task.
    Hides contiguous spans of `span_len` steps so that about `ratio` of each
    day is hidden. Only steps with real data (obs_mask) can be hidden.
    """
    B, T = obs_mask.shape
    # per-step probability of starting a span so expected coverage ~= ratio
    p = 1 - (1 - ratio) ** (1 / span_len)
    starts = (torch.rand(B, T, generator=generator) < p).float().unsqueeze(1)
    # left-pad then max-pool: step t is hidden if a span started in [t-L+1, t]
    spans = F.max_pool1d(F.pad(starts, (span_len - 1, 0)), kernel_size=span_len, stride=1)
    return (spans.squeeze(1) > 0) & obs_mask


def build_inputs(x, obs_mask, masked):
    """
    x: (B,T,5)  obs_mask: (B,T) bool  masked: (B,T) bool
    -> (B,T,6): glucose/speed/accel zeroed where hidden or missing,
       plus a 0/1 flag channel telling the model where they are hidden.
    """
    hidden = masked | ~obs_mask
    x_in = x.clone()
    x_in[..., SIGNAL_IDX] = x_in[..., SIGNAL_IDX].masked_fill(hidden.unsqueeze(-1), 0.0)
    return torch.cat([x_in, hidden.unsqueeze(-1).float()], dim=-1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class CGMEncoder(nn.Module):
    """Transformer encoder. Re-use this class (and its weights) for A1c later."""

    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ff, dropout, max_len=288):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model) # matrix multiplication
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

    def forward(self, x_in, pad_mask):
        # pad_mask: (B,T) bool, True = ignore this step as an attention key
        h = self.in_proj(x_in) + self.pos_emb[:, : x_in.size(1)]
        return self.encoder(h, src_key_padding_mask=pad_mask)   # (B,T,d_model)


class MaskedGlucoseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CGMEncoder(
            IN_DIM, cfg["d_model"], cfg["n_heads"], cfg["n_layers"],
            cfg["d_ff"], cfg["dropout"],
        )
        self.head = nn.Linear(cfg["d_model"], 1)

    def forward(self, x_in, pad_mask):
        return self.head(self.encoder(x_in, pad_mask)).squeeze(-1)   # (B,T)


# --------------------------------------------------------------------------
# Loss / evaluation
# --------------------------------------------------------------------------
def masked_mse(pred, target, masked):
    m = masked.float()
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_model = tot_base = tot_n = 0.0
    for x, obs, masked in loader:
        x, obs, masked = x.to(device), obs.to(device), masked.to(device)
        pred = model(build_inputs(x, obs, masked), ~obs)
        target = x[..., GL]
        n = masked.sum().item()
        tot_model += masked_mse(pred, target, masked).item() * n

        # naive baseline: predict the mean of the visible glucose in that day
        visible = (obs & ~masked).float()
        day_mean = (target * visible).sum(1, keepdim=True) / visible.sum(1, keepdim=True).clamp(min=1)
        tot_base += masked_mse(day_mean.expand_as(target), target, masked).item() * n
        tot_n += n
    return tot_model / tot_n, tot_base / tot_n


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(cfg=CFG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- data -----------------------------------------------------------
    d = np.load(cfg["seq_file"])
    X = torch.from_numpy(d["X"]).float()
    obs = torch.from_numpy(d["obs_mask"]).bool()
    meta = pd.read_csv(cfg["meta_file"])

    tr = torch.from_numpy((meta["split"] == "train").to_numpy())
    va = torch.from_numpy((meta["split"] == "val").to_numpy())
    print(f"train days: {int(tr.sum())} | val days: {int(va.sum())}")

    train_loader = DataLoader(
        TensorDataset(X[tr], obs[tr]), batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True,
    )
    # validation masks are generated ONCE so val loss is comparable across epochs
    g = torch.Generator().manual_seed(123)
    val_masked = make_span_mask(obs[va], cfg["mask_ratio"], cfg["span_len"], g)
    val_loader = DataLoader(
        TensorDataset(X[va], obs[va], val_masked), batch_size=cfg["batch_size"]
    )

    # ---- model / optimiser ---------------------------------------------
    model = MaskedGlucoseModel(cfg).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = cfg["epochs"] * len(train_loader)
    warm = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- training loop --------------------------------------------------
    best = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        run, n_batches = 0.0, 0
        for x, o in train_loader:
            masked = make_span_mask(o, cfg["mask_ratio"], cfg["span_len"])  # new mask every batch
            x, o, masked = x.to(device), o.to(device), masked.to(device)

            pred = model(build_inputs(x, o, masked), ~o)
            loss = masked_mse(pred, x[..., GL], masked)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            run += loss.item()
            n_batches += 1

        val_mse, base_mse = evaluate(model, val_loader, device)
        flag = ""
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder_state": model.encoder.state_dict(),
                    "cfg": cfg, "epoch": epoch, "val_mse": val_mse,
                },
                cfg["out_file"],
            )
            flag = "  <- saved"
        print(
            f"epoch {epoch:3d} | train mse {run / n_batches:.4f} | "
            f"val mse {val_mse:.4f} | day-mean baseline {base_mse:.4f}{flag}"
        )

    print(f"Best val MSE: {best:.4f}  (checkpoint: {cfg['out_file']})")


if __name__ == "__main__":
    main()
    
