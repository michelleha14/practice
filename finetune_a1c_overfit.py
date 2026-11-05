"""
finetune_a1c.py  -  Fine-tune the pretrained CGM transformer to predict A1c
===========================================================================
(This version has a comment on every line, written for someone coming from R.)

----------------------------------------------------------------------------
PYTHON FOR R USERS - READ THIS FIRST
----------------------------------------------------------------------------
    Python                         R equivalent / what it means
    -----------------------------  ----------------------------------------------
    x = 5                          x <- 5
    # comment                      # comment (same)
    None                           NULL
    np.nan                         NA (a missing number)
    True / False                   TRUE / FALSE (capitalised in Python)
    and, or, not                   &&, ||, !   (for single TRUE/FALSE values)
    &  |  ~                        &  |  !     (element-wise, on whole columns/arrays)
    [1, 2, 3]                      a "list" (like c(1,2,3) or list(1,2,3))
    {"a": 1}                       a "dict": like a named list, list(a = 1)
    (1, 2)                         a "tuple": a fixed-length list you can't modify
    x[0]                           x[1]   <-- Python counts from 0, not 1 !
    x[0:3]                         x[1:3] <-- the END is excluded: gives items 0,1,2
    x[-1]                          the LAST item (R: x[length(x)])
    df["col"]                      df$col
    df.loc[rows, "col"]            df[rows, "col"]   (select by label / TRUE-FALSE)
    df.iloc[0]                     df[1, ]           (select by position)
    df.groupby("g")["x"].mean()    df %>% group_by(g) %>% summarise(mean(x))
    a.merge(b, on="k", how="left") left_join(a, b, by = "k")
    len(x)                         length(x) or nrow(df)
    def f(a, b=2):                 f <- function(a, b = 2)
    return z                       Python needs an explicit "return" (R returns the
                                   last value automatically)
    for i in range(3):             for (i in 0:2)   (range(3) is 0, 1, 2)
    f"n = {n}"                     glue("n = {n}") / paste0("n = ", n)
    obj.method(x)                  method(obj, x): functions "belong" to objects
    a.b().c()                      chaining, like  a %>% b() %>% c()
    f(**d)                         do.call(f, d): spread a dict into named arguments
    [g(v) for v in vs]             sapply(vs, g)  ("list comprehension")
    lambda x: x + 1                \\(x) x + 1
    A if cond else B               ifelse(cond, A, B)
    a @ b                          a %*% b   (matrix multiplication)

TWO MORE THINGS THAT SURPRISE R USERS
  * INDENTATION IS SYNTAX. The lines indented under "def", "for", "if", "with" are
    the body (R uses { } for this).
  * Assignment does NOT copy. If  b = a  and you change b in place, a changes too
    (R copies automatically). That is why the script uses copy.deepcopy() when it
    needs an independent snapshot of the model weights.

----------------------------------------------------------------------------
WHAT "FINE-TUNING" MEANS
----------------------------------------------------------------------------
Pretraining taught the transformer to understand CGM days (by reconstructing hidden
glucose). Fine-tuning takes those pretrained weights as a starting point and keeps
training them on the task we actually care about: predicting A1c.

    pretrained encoder  ->  average over the day's time steps  ->  small "head"  ->  A1c

HOW THE DATA ARE USED
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
1. Standardise A1c (subtract the training mean, divide by the training SD) so the
   loss has a sensible scale. Predictions are converted back to A1c % for reporting.
2. "Head warm-up": for the first few epochs only the new head is trained and the
   pretrained encoder is frozen, so the head's early random errors can't damage the
   pretrained weights.
3. Then everything is trained, but the encoder gets a much SMALLER learning rate
   than the head (we want to nudge it, not rewrite it).
4. Light augmentation: each epoch a few random glucose stretches are hidden, like
   in pretraining. This acts as regularisation.
5. Early stopping is OPTIONAL and OFF by default (early_stopping=False in SETTINGS).
   Off: train for all epochs and use the final weights. On: stop when validation
   error stops improving and use the best epoch's weights.
6. Every subject counts equally: a day's loss is weighted by 1 / (number of days
   that subject has), so people with more days don't dominate.

COMPARISONS (so you know whether it is worth it)
   Mean A1c            predict the training-set average for everyone
   Mean glucose        A1c ~ average glucose (fitted on train subjects)
   Mean glucose + SD   adds glucose variability
   Fine-tuned model    the transformer
Run the script a second time with  --scratch  to train the SAME architecture
from random weights. If the pretrained version is not clearly better than the
scratch version, pretraining did not help.

OVERFIT CHECK (a first sanity test: can the model fit the training data at all?)
   Before worrying about generalisation, confirm the model CAN learn. Train with no
   early stopping and no regularisation, for many epochs, and watch the TRAIN error:
       early_stopping = False, aug_mask_ratio = 0.0, head_dropout = 0.0,
       weight_decay = 0.0, epochs = 100 (or more)
   What you should see in the plot (top-left panel, blue line):
       * train RMSE falls well BELOW the dashed "always predict the mean" line,
         and the TRAIN scatter hugs the diagonal  -> the model works
       * validation RMSE staying high or rising is expected here: that is
         overfitting, and a later job for regularisation
   If train RMSE never leaves the dashed line, the model is not learning at all
   (look at the learning rates, the labels, or the data pipeline, not at overfitting).

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
    finetune_results_*.png  error per epoch (train and val), true-vs-predicted for train
                            and test, error summaries
"""

# ==========================================================================
# IMPORTS: load the libraries (packages) this script uses.
# In R:  library(pkg).   In Python:  import pkg   (then call pkg.function())
# ==========================================================================
import copy                            # tools for making independent duplicates of objects
import os                              # operating-system helpers: folders, file paths
import sys                             # gives access to command-line arguments (sys.argv)

import numpy as np                     # "as np" = nickname, so we write np.mean() instead of numpy.mean(). Arrays + maths.
import pandas as pd                    # data frames (like R's data.frame / tibble); nickname pd
import torch                           # PyTorch: the deep-learning library (tensors, autograd)
import torch.nn as nn                  # neural-network building blocks (layers, losses); nickname nn
from torch.utils.data import DataLoader, TensorDataset   # "from X import A, B": bring in just these two names (no prefix needed)
import matplotlib                      # plotting library (like base R graphics)
matplotlib.use("Agg")                  # draw straight into image files; never open a pop-up window
import matplotlib.pyplot as plt        # the plotting interface (like plot(), hist()...); nickname plt

# Re-use building blocks from our own pretraining script (pretrain_cgm.py in the same folder),
# so the encoder here is built exactly the way it was during pretraining.
from pretrain_cgm import CGMEncoder, GL, IN_DIM, build_inputs, make_span_mask


# ==========================================================================
# SETTINGS  (edit these; everything else can stay as is)
# ==========================================================================
# A dict {name: value, ...} is Python's version of an R named list. We look values up as SETTINGS["epochs"].
SETTINGS = dict(                       # dict(...) builds a dict from name=value pairs
    # ---- files ------------------------------------------------------------
    ckpt_file="cgm_pretrained_50.pt",     # the pretraining checkpoint (saved model weights)
    seq_file="cgm_daily_sequences_50.npz",    # the daily CGM arrays made earlier
    meta_file="cgm_daily_meta_50.csv",        # one row per day: subject, date, split ...
    a1c_file="pseudo_cgm_data_50_subjects_r.csv",             # YOUR file: one A1c per subject
    label_subject_col="subject_id",    # name of the subject column inside a1c_labels.csv
    label_a1c_col="a1c",               # name of the A1c column inside a1c_labels.csv
    out_dir="a1c_finetune_50",            # folder where results are written

    # ---- starting point ---------------------------------------------------
    use_pretrained=True,               # True = start from pretrained weights; False = random weights
    min_days_per_subject=3,            # subjects with fewer valid days are dropped

    # ---- training ---------------------------------------------------------
    batch_size=64,                     # how many days are processed together in one update
    epochs=300,                         # maximum passes over the whole training set
    early_stopping=False,              # False = train ALL epochs and use the final weights (overfit check); True = stop early and use the best-validation epoch
    patience=6,                        # only used if early_stopping is True: stop after 6 epochs without validation improvement
    head_warmup_epochs=2,              # epochs with the encoder frozen (pretrained runs only)
    lr_encoder=5e-5,                   # learning rate for the encoder: small = gentle changes (5e-5 = 0.00005)
    lr_encoder_scratch=3e-4,           # larger rate used when the encoder starts from random weights
    lr_head=1e-3,                      # the new head learns faster
    weight_decay=0.01,                 # penalty that keeps weights small (regularisation)
    warmup_frac=0.05,                  # fraction of steps during which the learning rate ramps up
    grad_clip=1.0,                     # cap on how big one update step may be
    head_dropout=0.1,                  # chance of randomly zeroing a feature in the head during training
    aug_mask_ratio=0.05,               # fraction of glucose hidden at random each epoch (augmentation)
    weight_by_subject=True,            # True = each subject counts equally

    # ---- statistics -------------------------------------------------------
    n_boot=2000,                       # bootstrap repeats for 95% intervals
    seed=42,                           # fixes the random numbers so runs repeat exactly
)   # end of the SETTINGS dict (the closing bracket matches "dict(" above)

# Display names for the models. Defined once ("constants", written in CAPITALS by convention)
# so tables and plots always use the same labels.
NULL = "Mean A1c"                      # baseline: everyone gets the training-average A1c
GLU = "Mean glucose"                   # baseline: linear model on average glucose
GLU_SD = "Mean glucose + SD"           # baseline: linear model on average glucose and its SD


# ==========================================================================
# STEP 1: load and prepare the data
# ==========================================================================
def load_labels(s):                    # "def" defines a function (R: load_labels <- function(s) {...}); s is the settings dict
    """Read the A1c file -> DataFrame with columns subject_id (text) and a1c (number)."""   # text right after "def" is the function's "docstring" (its help text)
    lab = pd.read_csv(s["a1c_file"])   # read the CSV into a data frame (R: read.csv). s["a1c_file"] looks up that name in the dict
    # rename columns using a dict {old name: new name}, so later code can rely on "subject_id" and "a1c"
    lab = lab.rename(columns={s["label_subject_col"]: "subject_id", s["label_a1c_col"]: "a1c"})
    lab = lab[["subject_id", "a1c"]]   # keep only these two columns (the inner [...] is a list of column names; R: lab[, c("subject_id","a1c")])
    lab["subject_id"] = lab["subject_id"].astype(str).str.strip()   # convert ids to text, then trim spaces. Chained calls read left to right, like %>%
    lab["a1c"] = pd.to_numeric(lab["a1c"], errors="coerce")         # convert to numbers; anything unparseable becomes NaN (R: as.numeric gives NA)
    lab = lab.dropna(subset=["a1c"]).drop_duplicates("subject_id")  # drop rows with missing A1c, then keep one row per subject
    if not lab["a1c"].between(3, 20).all():   # .between(3,20) is TRUE/FALSE per row; .all() = every row TRUE; "not" flips it
        print("  WARNING: some A1c values are outside 3-20. Are they in % (not mmol/mol)?")   # print a message to the console
    return lab                         # hand the cleaned table back to whoever called this function


def prepare_data(s):   # function: load CGM days and attach each subject's A1c (details in the docstring below)
    """
    Load the CGM days and attach each subject's A1c.

    Returns
        X     (n_days, 288, 5)  features per 5-minute step
        obs   (n_days, 288)     True where a real glucose reading exists
        meta  one row per day: subject_id, date, split, a1c, n_days
    Days are dropped if the subject has too few days or no A1c value.
    """
    labels = load_labels(s)            # call the function defined above; result stored in "labels"
    d = np.load(s["seq_file"])         # open the .npz file: a bundle of named arrays (like a list of arrays in R)
    X = d["X"].astype(np.float32)      # pull out the feature array; store numbers as 32-bit decimals (half the memory of the default)
    obs = d["obs_mask"].astype(bool)   # pull out the real-reading flags as TRUE/FALSE
    meta = pd.read_csv(s["meta_file"]) # read the per-day table (one row per day)
    if "split" not in meta.columns:    # "x in y" / "x not in y" tests membership; meta.columns = the column names
        raise SystemExit("cgm_daily_meta.csv has no 'split' column. Re-run make_cgm_sequences.py.")   # stop the whole program with this message (R: stop())
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()   # ids as clean text, so they match the label file

    # Drop subjects with too few days.
    n_days = meta.groupby("subject_id")["subject_id"].transform("size")   # for each ROW, how many rows its subject has (R: ave(..., FUN = length) / add_count())
    keep = (n_days >= s["min_days_per_subject"]).to_numpy()   # TRUE for rows whose subject has enough days; .to_numpy() turns the column into a plain array
    X, obs, meta = X[keep], obs[keep], meta[keep].reset_index(drop=True)   # keep only TRUE rows in all three at once (tuple assignment); reset_index renumbers rows from 0

    # A subject must belong to exactly one split, otherwise test subjects would leak.
    if (meta.groupby("subject_id")["split"].nunique() > 1).any():   # nunique = number of distinct splits per subject; any() = is there at least one subject with >1?
        raise SystemExit("Some subjects appear in more than one split. Fix the split first.")   # stop with an error message

    # Attach A1c (a "left join": keeps every day, adds the matching label).
    meta = meta.merge(labels, on="subject_id", how="left")   # R: left_join(meta, labels, by = "subject_id")
    has_label = meta["a1c"].notna().to_numpy()   # TRUE where A1c is present (R: !is.na())
    n_unlabelled = meta.loc[~has_label, "subject_id"].nunique()   # "~" means NOT on a TRUE/FALSE array; count distinct subjects lacking A1c
    if n_unlabelled:                   # a number counts as TRUE if it is not zero
        print(f"  Dropping {n_unlabelled} subjects without an A1c value")   # f"..." = f-string: {n_unlabelled} is replaced by its value
    X, obs, meta = X[has_label], obs[has_label], meta[has_label].reset_index(drop=True)   # keep only labelled days everywhere
    meta["n_days"] = meta.groupby("subject_id")["subject_id"].transform("size")   # add a column: number of days of each row's subject
    return X, obs, meta                # a function can return several things at once (as a tuple)


def glucose_summary_by_subject(X, obs, subject_ids):   # function: per-subject mean and SD of glucose, for the baselines
    """
    Each subject's mean and SD of normalised glucose over all valid readings.
    Used only for the simple baselines. (norm_gl is a linear rescaling of glucose,
    so the mean of norm_gl carries the same information as mean glucose in mg/dL.)
    """
    rows = {}                          # start with an empty dict (R: list())
    ids = np.asarray(subject_ids)      # make sure ids are a plain array
    for subj in np.unique(ids):        # loop over each distinct subject id (R: for (subj in unique(ids)))
        idx = np.where(ids == subj)[0] # positions of this subject's days (R: which(ids == subj)); [0] extracts the array from the tuple np.where returns
        vals = X[idx][:, :, GL][obs[idx]]   # X[idx] = this subject's days; [:, :, GL] = all days, all steps, glucose feature only (GL = 2, the third); [obs[idx]] keeps only real readings -> a flat 1-D array
        rows[subj] = {"mean_gl": vals.mean(), "sd_gl": vals.std()}   # add an entry: key = subject id, value = a small dict of the two summaries
    return pd.DataFrame.from_dict(rows, orient="index")   # turn the dict of dicts into a data frame with one row per subject


# ==========================================================================
# STEP 2: the model = pretrained encoder + a small regression head
# ==========================================================================
class A1cModel(nn.Module):             # "class" = a blueprint for an object; (nn.Module) means it inherits PyTorch's model machinery
    """
    A "class" is a blueprint for an object. This one holds two parts:
        encoder : the pretrained transformer (one vector per 5-minute step)
        head    : turns the day's average vector into a single A1c number
    """

    def __init__(self, encoder, d_model, head_dropout):   # __init__ runs once when the object is created. "self" = the object itself (first argument of every method)
        super().__init__()             # first let PyTorch's nn.Module set itself up (required boilerplate)
        self.encoder = encoder         # store the encoder on the object, so it is available as self.encoder later
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, 1))   # head = a chain: random dropout, then a linear layer d_model numbers -> 1 number (like a regression)

    def forward(self, x_in, pad_mask):   # "forward" defines what the model computes from an input (called automatically when you do model(...))
        # x_in: (batch, 288, 6) model input; pad_mask: True where data are missing
        h = self.encoder(x_in, pad_mask)                    # run the encoder -> (batch, 288, d_model): one vector per time step
        w = (~pad_mask).unsqueeze(-1).float()               # ~ flips TRUE/FALSE -> 1 for real steps; unsqueeze(-1) adds a length-1 last dimension -> (batch, 288, 1); .float() makes it 1.0/0.0
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)   # weighted mean over the 288 steps: zero out missing steps, sum, divide by the count of real steps (clamp avoids dividing by 0) -> (batch, d_model)
        return self.head(pooled).squeeze(-1)                # head gives (batch, 1); squeeze(-1) drops the last dimension -> (batch,): one standardised A1c per day


def build_model(s, device):   # function: build the model, optionally loading pretrained weights
    """Create the model; load pretrained encoder weights unless training from scratch."""
    ckpt = torch.load(s["ckpt_file"], map_location=device)   # read the saved checkpoint file; map_location puts tensors on the chosen device (cpu/gpu)
    cfg = ckpt["cfg"]                                       # the architecture settings used in pretraining (stored in the checkpoint)
    encoder = CGMEncoder(IN_DIM, cfg["d_model"], cfg["n_heads"], cfg["n_layers"],   # build an encoder with the SAME shape as in pretraining...
                         cfg["d_ff"], cfg["dropout"])                               # ...(arguments continue on the next line)
    if s["use_pretrained"]:            # if we want the pretrained starting point...
        encoder.load_state_dict(ckpt["encoder_state"])      # ...copy the saved weights into the encoder
        print("  Loaded pretrained encoder weights")   # tell the user which starting point is being used
    else:                              # otherwise leave the weights as the random values they were created with
        print("  Using RANDOM encoder weights (scratch control)")   # tell the user which starting point is being used
    return A1cModel(encoder, cfg["d_model"], s["head_dropout"]).to(device), cfg   # build the full model, move it to the device, and return it together with cfg


def set_encoder_trainable(model, trainable):   # function: freeze or unfreeze the encoder
    """Freeze (False) or unfreeze (True) the encoder's weights."""
    for p in model.encoder.parameters():   # loop over every weight tensor in the encoder
        p.requires_grad = trainable        # True = training may change it; False = frozen


# ==========================================================================
# STEP 3: training and prediction
# ==========================================================================
def train_one_epoch(model, loader, opt, sched, device, s, span_len):
    """One pass over the training days. Returns the average (weighted) loss."""
    model.train()                                           # training mode: dropout ON
    total, count = 0.0, 0                                   # two running totals, set in one line (tuple assignment)
    for x, o, y, w in loader:                               # each loop gives one batch: x = days, o = real-data flags, y = standardised A1c, w = weights
        # Augmentation: hide a few random glucose stretches (made on CPU, then moved).
        masked = make_span_mask(o, s["aug_mask_ratio"], span_len)   # TRUE where a stretch is hidden this time
        x, o, masked, y, w = (t.to(device) for t in (x, o, masked, y, w))   # move all five tensors to the device (cpu/gpu); the (... for t in ...) is a one-line loop

        pred = model(build_inputs(x, o, masked), ~o)        # build the model input (hidden values zeroed + flag channel), run the model -> one prediction per day
        loss = (w * (pred - y) ** 2).sum() / w.sum()        # weighted mean squared error; "**" is power (R: ^)

        opt.zero_grad()                                     # wipe the gradients left from the previous batch
        loss.backward()                                     # backpropagation: compute how each weight should change
        nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"])   # shrink the update if it is unusually large (stability)
        opt.step()                                          # actually update the weights
        sched.step()                                        # move the learning-rate schedule forward one step
        total += loss.item() * len(y)                       # "+=" adds in place; .item() turns a 1-number tensor into a normal number; times batch size
        count += len(y)                                     # running count of days seen
    return total / count                                    # average loss per day over the whole epoch


def predict_days(model, X, obs, device, batch_size=256):   # batch_size=256 is a default: used if the caller doesn't supply one
    """Predict (standardised) A1c for every day, with nothing hidden."""
    model.eval()                                            # evaluation mode: dropout OFF, so predictions are deterministic
    out = []                                                # an empty list we will fill with batch results
    with torch.no_grad():                                   # inside this block PyTorch skips gradient bookkeeping (faster, less memory)
        for i in range(0, len(X), batch_size):              # i = 0, 256, 512, ... (start, stop, step)
            x = X[i:i + batch_size].to(device)              # slice: days i up to (not including) i+batch_size; move to device
            o = obs[i:i + batch_size].to(device)            # the matching real-data flags
            out.append(model(build_inputs(x, o, torch.zeros_like(o)), ~o).cpu())   # predict with NOTHING hidden (zeros_like = all FALSE); .cpu() brings the result back; append adds it to the list IN PLACE
    return torch.cat(out).numpy()                           # glue the batch results into one tensor, convert to a NumPy array


def predict_subjects(model, X, obs, subject_ids, y_mean, y_std, device):   # function: one A1c prediction per subject (average of their days)
    """
    Day-level predictions -> ONE prediction per subject (the average over their days),
    converted back to A1c %. Returns a pandas Series indexed by subject_id.
    """
    day_pred = predict_days(model, X, obs, device) * y_std + y_mean   # undo the standardisation: multiply by SD, add mean -> A1c in %
    df = pd.DataFrame({"subject_id": np.asarray(subject_ids), "pred": day_pred})   # small table: each day's subject and prediction
    return df.groupby("subject_id")["pred"].mean()          # average the predictions within each subject (R: group_by %>% summarise(mean))


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
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 1e-9 else np.nan   # "A if cond else B": correlation, unless predictions are all (practically) identical, when it is undefined -> NaN. (1e-9 rather than 0 because rounding can leave a tiny non-zero SD.) corrcoef returns a 2x2 matrix; [0, 1] picks the off-diagonal
    rho = pd.Series(y).rank().corr(pd.Series(p).rank()) if np.std(p) > 1e-9 else np.nan   # Spearman = Pearson correlation of the ranks
    return {                                                # build and return a dict of results
        "MAE": float(np.mean(np.abs(y - p))),               # average of |error|
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),      # square root of average squared error
        "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),   # 1 - (model error) / (error of just guessing the mean)
        "r": float(r), "rho": float(rho), "n_subjects": len(y),   # float() makes sure these are plain numbers
    }   # end of the dict being returned


def baseline_predictions(glucose, truth, train_ids, eval_ids):   # function: the simple comparison models (mean A1c, mean glucose, mean glucose + SD)
    """
    Simple comparison models, fitted on TRAIN subjects only, applied to eval_ids.
    Returns {model name: predictions array}.
    """
    y_train = truth.loc[train_ids].to_numpy()               # true A1c of the train subjects (.loc selects rows by their labels = subject ids)
    out = {NULL: np.full(len(eval_ids), y_train.mean())}    # baseline 1: np.full repeats the training-mean A1c once per evaluation subject
    for name, cols in [(GLU, ["mean_gl"]), (GLU_SD, ["mean_gl", "sd_gl"])]:   # loop over two (name, feature columns) pairs; "name, cols" unpacks each pair
        A = np.column_stack([np.ones(len(train_ids)), glucose.loc[train_ids, cols].to_numpy()])   # design matrix: a column of 1s (intercept) next to the features (R: cbind(1, X))
        coef = np.linalg.lstsq(A, y_train, rcond=None)[0]   # ordinary least squares (R: lm / solve); [0] keeps just the coefficients
        B = np.column_stack([np.ones(len(eval_ids)), glucose.loc[eval_ids, cols].to_numpy()])     # the same design matrix for the evaluation subjects
        out[name] = B @ coef                                # predictions = matrix multiplication (R: B %*% coef); stored under the model's name
    return out   # hand back the dict {model name: predictions}


def bootstrap_mae_ci(y, p, n_boot, seed):   # function: 95% interval for MAE by resampling subjects
    """95% interval for MAE: re-draw subjects with replacement many times."""
    rng = np.random.default_rng(seed)                       # a random-number generator with a fixed seed (R: set.seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))         # a table of random subject positions: n_boot rows (one per repeat), len(y) columns
    boot = np.abs(y - p)[idx].mean(axis=1)                  # |error| per subject, picked out by idx (a resampled table), averaged along each row -> one MAE per repeat
    return tuple(np.percentile(boot, [2.5, 97.5]))          # the 2.5th and 97.5th percentiles = 95% interval (R: quantile)


def bootstrap_paired_diff_ci(y, p_model, p_base, n_boot, seed):   # function: 95% interval for the MAE difference between two models
    """
    95% interval for (MAE of model - MAE of baseline) using the SAME resampled
    subjects for both. Entirely below 0 => the model is reliably better.
    """
    rng = np.random.default_rng(seed)                       # fixed-seed random generator
    idx = rng.integers(0, len(y), (n_boot, len(y)))         # the resampled subject positions, shared by both models (that is what makes it "paired")
    diff = np.abs(y - p_model)[idx].mean(axis=1) - np.abs(y - p_base)[idx].mean(axis=1)   # per repeat: model MAE minus baseline MAE
    return tuple(np.percentile(diff, [2.5, 97.5]))          # 95% interval of that difference


# ==========================================================================
# STEP 5: plots
# ==========================================================================
def draw_scatter(ax, y, p, title, color, lo, hi):
    """Draw one true-vs-predicted scatter panel on the panel `ax`, with the y = x line."""
    ax.scatter(y, p, s=18, alpha=0.5, color=color)          # one dot per subject: true A1c (x) vs predicted A1c (y); s = size, alpha = transparency
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)                # dashed y = x line: points on it are perfect predictions
    ax.set_xlim(lo, hi)                                     # same x range on every scatter panel, so they are comparable
    ax.set_ylim(lo, hi)                                     # same y range
    m = compute_metrics(y, p)                               # MAE, RMSE, R2, r for this set of subjects
    ax.set_title(f"{title}\nMAE {m['MAE']:.2f}   r {m['r']:.2f}   R2 {m['R2']:.2f}")   # "\n" = new line; ":.2f" = 2 decimals
    ax.set_xlabel("true A1c (%)")                           # x label
    ax.set_ylabel("predicted A1c (%)")                      # y label
    ax.grid(alpha=0.25)                                     # faint grid


def bias_by_level(y, p):
    """Group subjects by true A1c; return per group: mean true A1c, mean (pred - true), and its standard error."""
    resid = p - y                                           # residual = predicted - true, for every subject
    n_bins = int(min(10, max(2, len(y) // 15)))             # number of groups: 10 normally, fewer if few subjects ("//" = integer division)
    bins = pd.qcut(pd.Series(y), n_bins, duplicates="drop") # cut true A1c into equal-sized groups (quantile bins; R: cut(..., quantile(...)))
    g = pd.DataFrame({"y": y, "res": resid, "bin": bins}).groupby("bin", observed=True)   # table of (true, residual, bin), then group by bin
    return g["y"].mean(), g["res"].mean(), g["res"].std() / np.sqrt(g["res"].size())      # per-bin mean true A1c, mean residual, standard error (SD / sqrt(n))


def plot_results(path, log, y_train, pred_train, y_test, preds_test, ft_name, final_epoch, metrics, y_ref):
    """2x3 figure: error per epoch, train and test scatter, test MAE of all models, MAE by split, bias by A1c level."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))        # create a figure with a 2x3 grid of panels; axes[row, col] picks one (counting from 0)

    # (1) error per epoch -----------------------------------------------------
    ax = axes[0, 0]                                         # top-left panel
    ax.plot(log["epoch"], log["train_RMSE"], color="tab:blue", label="train subjects")        # TRAIN error each epoch (A1c % points, dropout off, nothing hidden)
    ax.plot(log["epoch"], log["val_RMSE"], color="tab:red", label="validation subjects")      # VALIDATION error each epoch
    ax.axhline(y_ref, color="k", ls="--", lw=1, label="always predict the mean")              # horizontal dashed line: the error of ignoring the CGM entirely
    ax.axvline(final_epoch, color="k", ls=":", lw=1)        # vertical dotted line at the epoch whose weights were used
    ax.set_xlabel("epoch")                                  # x label
    ax.set_ylabel("RMSE (A1c % points)")                    # y label
    ax.set_title(f"Error per epoch (dotted = epoch of the weights used: {final_epoch})")   # title (an f-string)
    ax.legend(fontsize=8)                                   # show the line labels
    ax.grid(alpha=0.25)                                     # faint grid

    # (2) and (3): train and test scatter -------------------------------------
    p_test = preds_test[ft_name]                            # the fine-tuned model's test predictions (looked up by name in a dict)
    everything = np.concatenate([y_train, pred_train, y_test, p_test])   # glue all the values together (R: c(...)) to find shared axis limits
    lo, hi = everything.min() - 0.2, everything.max() + 0.2 # shared axis limits, padded by 0.2
    draw_scatter(axes[0, 1], y_train, pred_train, "TRAIN subjects: true vs predicted A1c", "tab:blue", lo, hi)   # top-middle panel
    draw_scatter(axes[0, 2], y_test, p_test, "TEST subjects: true vs predicted A1c", "tab:red", lo, hi)          # top-right panel

    # (4) test MAE of every model -----------------------------------------------
    ax = axes[1, 0]                                         # bottom-left panel
    mt = metrics[metrics["split"] == "test"].set_index("model")   # keep only the test rows, with the model name as row labels
    names = list(mt.index)                                  # the model names as a plain list
    maes = mt["MAE"].to_numpy()                             # their MAE values as an array
    err = np.vstack([maes - mt["MAE_ci_low"].to_numpy(), mt["MAE_ci_high"].to_numpy() - maes])   # error-bar lengths below and above each bar (R: rbind)
    ax.bar(range(len(names)), maes, yerr=err, capsize=3,    # bar chart with error bars; range(n) gives positions 0..n-1...
           color=["tab:red" if n == ft_name else "0.6" for n in names])   # ...red for our model, grey for baselines (a list comprehension: R sapply/ifelse)
    ax.set_xticks(range(len(names)))                        # one tick under each bar
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)   # tick labels = model names, tilted so they don't overlap
    ax.set_ylabel("test MAE (A1c % points), lower = better")   # y label
    ax.set_title("Test error with 95% bootstrap intervals") # title

    # (5) fine-tuned model vs "always predict the mean", by split ---------------
    ax = axes[1, 1]                                         # bottom-middle panel
    splits = ["train", "val", "test"]                       # the three groups of subjects

    def mae_of(model_name, sp):                             # small helper: look up one MAE in the metrics table
        return metrics[(metrics["split"] == sp) & (metrics["model"] == model_name)]["MAE"].iloc[0]   # rows matching both conditions, the MAE column, first row

    base_mae = [mae_of(NULL, sp) for sp in splits]          # MAE of "always predict the mean" in each split
    ft_mae = [mae_of(ft_name, sp) for sp in splits]         # MAE of the fine-tuned model in each split
    xpos = np.arange(len(splits))                           # bar positions 0, 1, 2
    ax.bar(xpos - 0.2, base_mae, width=0.4, color="0.6", label=NULL)           # grey bars, shifted left
    ax.bar(xpos + 0.2, ft_mae, width=0.4, color="tab:red", label=ft_name)      # red bars, shifted right
    for x, v in zip(xpos - 0.2, base_mae):                  # loop over bars (zip pairs up two lists element by element)
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)        # write the value above each grey bar
    for x, v in zip(xpos + 0.2, ft_mae):                    # same for the red bars
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)        # write the value above each red bar
    ax.set_xticks(xpos)                                     # one tick per split
    ax.set_xticklabels(splits)                              # label the ticks train / val / test
    ax.set_ylabel("MAE (A1c % points), lower = better")     # y label
    ax.set_title("Fine-tuned model vs 'always predict the mean'")   # title
    ax.legend(fontsize=8)                                   # show which colour is which

    # (6) bias by true A1c level ------------------------------------------------
    # Models tend to shrink toward the middle: high A1c under-predicted, low over-predicted.
    ax = axes[1, 2]                                         # bottom-right panel
    for yy, pp, label, col in [(y_train, pred_train, "train", "tab:blue"), (y_test, p_test, "test", "tab:red")]:   # loop over two sets; each item is unpacked into 4 names
        xs, ys, es = bias_by_level(yy, pp)                  # per-group mean true A1c, mean residual, standard error
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color=col, capsize=2, label=label)   # points joined by a line, with error bars
    ax.axhline(0, color="k", lw=1)                          # horizontal line at 0 = no bias
    ax.set_xlabel("true A1c, bin means (%)")                # x label
    ax.set_ylabel("mean (predicted - true)")                # y label
    ax.set_title("Bias by A1c level\n(downward slope = shrinks toward the mean)")   # title
    ax.legend(fontsize=8)                                   # show which colour is which
    ax.grid(alpha=0.25)                                     # faint grid

    fig.tight_layout()                                      # adjust spacing so labels don't overlap
    fig.savefig(path, dpi=130)                              # write the picture to a file
    plt.close(fig)                                          # free the figure's memory


# ==========================================================================
# MAIN: run everything in order
# ==========================================================================
def main(s=SETTINGS):                                       # s defaults to the SETTINGS dict above
    if "--scratch" in sys.argv:                             # sys.argv = the words typed after "python"; did the user add --scratch?
        s = dict(s, use_pretrained=False)                   # make a COPY of the settings with use_pretrained switched off
    tag = "pretrained" if s["use_pretrained"] else "scratch"   # a label used in file names and titles
    ft_name = f"Fine-tuned transformer ({tag})"             # display name of our model
    os.makedirs(s["out_dir"], exist_ok=True)                # create the output folder; exist_ok=True = don't complain if it already exists
    torch.manual_seed(s["seed"])                            # fix PyTorch's random numbers (R: set.seed)
    np.random.seed(s["seed"])                               # fix NumPy's random numbers
    device = "cuda" if torch.cuda.is_available() else "cpu" # use the GPU if one exists, else the CPU
    print(f"Device: {device} | starting point: {tag}")      # report the choices

    # ---- load data ------------------------------------------------------------
    print("[1/5] Loading data ...")                         # progress message
    X, obs, meta = prepare_data(s)                          # unpack the three returned items into three names
    sub_info = meta.drop_duplicates("subject_id").set_index("subject_id")   # one row per subject, with subject_id as the row labels
    truth = sub_info["a1c"]                                 # each subject's true A1c (a Series: a named vector, labelled by subject id)
    rows = {sp: np.where(meta["split"] == sp)[0] for sp in ["train", "val", "test"]}   # dict comprehension: for each split name, the row positions of its days
    for sp, idx in rows.items():                            # .items() loops over a dict's (key, value) pairs
        n_subj = meta.loc[idx, "subject_id"].nunique()      # how many distinct subjects are in this split
        print(f"  {sp:5s}: {len(idx):6d} days, {n_subj:5d} subjects")   # ":5s" / ":6d" / ":5d" pad text/integers to a fixed width so lines align
        if n_subj == 0:                                     # a split with nobody in it cannot be used
            raise SystemExit(f"No labelled subjects in the '{sp}' split.")   # stop with a helpful message

    # Standardise A1c using TRAIN subjects only.
    train_ids = sub_info.index[sub_info["split"] == "train"]   # ids of train subjects (.index = the row labels)
    y_mean, y_std = float(truth.loc[train_ids].mean()), float(truth.loc[train_ids].std())   # mean and SD of train A1c, computed once per SUBJECT
    print(f"  train A1c: mean {y_mean:.2f}, SD {y_std:.2f}")   # ":.2f" = 2 decimals

    # Tensors (PyTorch's version of arrays) for the three splits.
    def to_tensors(idx):                                    # a function defined INSIDE main; it can use main's variables (X, obs)
        return torch.from_numpy(X[idx]), torch.from_numpy(obs[idx])   # pick these rows and convert NumPy arrays -> PyTorch tensors

    y_day = ((meta["a1c"].to_numpy() - y_mean) / y_std).astype(np.float32)   # standardised A1c for every DAY (z-score using the train mean/SD)
    w_day = (1.0 / meta["n_days"].to_numpy() if s["weight_by_subject"]      # per-day weight: 1 / (days of that subject) so each subject counts equally...
             else np.ones(len(meta))).astype(np.float32)                    # ...or all 1s if weighting is turned off
    Xtr, otr = to_tensors(rows["train"])                    # the training days (features, real-data flags) as tensors
    train_loader = DataLoader(                              # DataLoader serves the data in shuffled mini-batches
        TensorDataset(Xtr, otr, torch.from_numpy(y_day[rows["train"]]),   # bundle features, flags, labels...
                      torch.from_numpy(w_day[rows["train"]])),            # ...and weights, so they are shuffled together
        batch_size=s["batch_size"], shuffle=True, drop_last=len(Xtr) > s["batch_size"],   # batch size; reshuffle every epoch; drop an incomplete last batch if there are enough days
    )   # end of the DataLoader(...) call
    eval_data = {sp: (*to_tensors(rows[sp]), meta.loc[rows[sp], "subject_id"].to_numpy())   # for train, val and test: (features, flags, subject ids) as a tuple; "*" spreads the two returned tensors into it
                 for sp in ["train", "val", "test"]}        # ...one entry per split (train is included so we can measure the TRAINING error too)

    # ---- model and optimiser -----------------------------------------------------
    print("[2/5] Building model ...")                       # progress message
    model, cfg = build_model(s, device)                     # build the model; keep the architecture settings too
    enc_lr = s["lr_encoder"] if s["use_pretrained"] else s["lr_encoder_scratch"]   # gentle learning rate for pretrained weights, larger for random ones
    opt = torch.optim.AdamW(                                # the optimiser: the rule that updates weights from gradients
        [{"params": model.encoder.parameters(), "lr": enc_lr},       # group 1: encoder weights with the (small) encoder rate
         {"params": model.head.parameters(), "lr": s["lr_head"]}],   # group 2: head weights with the larger head rate
        weight_decay=s["weight_decay"],                     # shrink weights slightly each step (regularisation)
    )   # end of the AdamW(...) call
    total_steps = s["epochs"] * len(train_loader)           # total number of updates if we train all epochs (len(loader) = batches per epoch)
    warm = max(1, int(s["warmup_frac"] * total_steps))      # how many updates the warm-up lasts (at least 1)

    def lr_lambda(step):                                    # learning-rate multiplier as a function of the step number
        if step < warm:                                     # during warm-up...
            return (step + 1) / warm                        # ...ramp the multiplier from ~0 up to 1
        prog = (step - warm) / max(1, total_steps - warm)   # after warm-up: how far through the remaining steps we are (0 to 1)
        return 0.5 * (1 + np.cos(np.pi * prog))             # cosine decay: smoothly goes from 1 down to 0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)   # the scheduler applies lr_lambda to the learning rates at each step

    # ---- training loop -------------------------------------------------------------
    print("[3/5] Training ...")                             # progress message
    head_warmup = s["head_warmup_epochs"] if s["use_pretrained"] else 0   # only pretrained runs get the frozen-encoder warm-up
    best_rmse, best_epoch, best_state, bad_epochs = np.inf, 0, None, 0    # trackers: best error so far (starts at infinity), its epoch, a snapshot of its weights (None = nothing yet), epochs without improvement
    history = []                                            # will collect one dict of numbers per epoch
    for epoch in range(1, s["epochs"] + 1):                 # epoch = 1, 2, ..., epochs (range stops BEFORE its end, hence +1)
        stage = "head-only" if epoch <= head_warmup else "full"   # which phase are we in?
        set_encoder_trainable(model, stage == "full")       # freeze the encoder in the head-only phase, unfreeze after
        train_loss = train_one_epoch(model, train_loader, opt, sched, device, s, cfg["span_len"])   # one pass over the training data

        Xt, ot, st = eval_data["train"]                     # unpack the TRAIN subjects' tensors (same layout as the validation ones below)
        tr_pred = predict_subjects(model, Xt, ot, st, y_mean, y_std, device)   # one predicted A1c per TRAIN subject (dropout off, nothing hidden)
        tm = compute_metrics(truth.loc[tr_pred.index].to_numpy(), tr_pred.to_numpy())   # the TRAINING A1c error in % points (MAE, RMSE, R2, r)
        Xv, ov, sv = eval_data["val"]                       # unpack the validation tuple into three names
        val_pred = predict_subjects(model, Xv, ov, sv, y_mean, y_std, device)   # one predicted A1c per validation subject
        vm = compute_metrics(truth.loc[val_pred.index].to_numpy(), val_pred.to_numpy())   # compare with the truth (rows matched by subject id)
        history.append(dict(epoch=epoch, stage=stage, train_loss=train_loss,   # record this epoch's numbers; "**{...}" spreads a dict into name=value arguments
                            **{f"train_{k}": tm[k] for k in ["MAE", "RMSE", "R2", "r"]},   # creates train_MAE, train_RMSE, train_R2, train_r
                            **{f"val_{k}": vm[k] for k in ["MAE", "RMSE", "R2", "r"]}))    # creates val_MAE, val_RMSE, val_R2, val_r

        flag = ""                                           # text appended to the printout when this epoch has the best validation error so far
        if vm["RMSE"] < best_rmse - 1e-6:                   # did validation error improve (by more than a tiny amount)?
            best_rmse, best_epoch, bad_epochs = vm["RMSE"], epoch, 0   # yes: remember it and reset the counter
            best_state = copy.deepcopy(model.state_dict())  # snapshot of the weights (deepcopy = independent copy; a plain assignment would just be a second name for the same, still-changing weights)
            flag = "  <- best val"                          # mark it in the printout
        elif stage == "full":                               # no improvement (epochs in the head-only warm-up are not counted)
            bad_epochs += 1                                 # count another epoch without improvement
        print(f"  epoch {epoch:3d} [{stage:9s}] train loss {train_loss:.3f} | "   # two string pieces side by side are joined into one
              f"train RMSE {tm['RMSE']:.3f} | "             # the training A1c error (% points)
              f"val MAE {vm['MAE']:.3f} RMSE {vm['RMSE']:.3f} R2 {vm['R2']:.3f}{flag}")   # print this epoch's summary line
        if s["early_stopping"] and bad_epochs >= s["patience"] and epoch > head_warmup:   # only stop early if the switch is ON; "and" = all conditions must hold
            print(f"  Early stopping: no improvement for {s['patience']} epochs")   # (single quotes inside, because the f-string itself uses double quotes)
            break                                           # leave the epoch loop early

    log = pd.DataFrame(history)                             # list of dicts -> data frame (one row per epoch)
    log.to_csv(os.path.join(s["out_dir"], f"finetune_log_{tag}.csv"), index=False)   # save it; os.path.join glues folder + file name (R: file.path); index=False = no row-number column
    if s["early_stopping"]:                                 # early stopping ON: use the weights of the best validation epoch
        final_state, final_epoch = best_state, best_epoch   # (best_state is the saved snapshot)
        model.load_state_dict(best_state)                   # put those weights back into the model
    else:                                                   # early stopping OFF: use the weights after the LAST epoch
        final_state, final_epoch = copy.deepcopy(model.state_dict()), epoch   # snapshot of the current weights; the loop variable "epoch" keeps its last value after the loop ends
    print(f"  Using the weights from epoch {final_epoch} ({'best validation epoch' if s['early_stopping'] else 'last epoch'})")   # tell the user which weights were used
    torch.save({"model_state": final_state, "cfg": cfg, "y_mean": y_mean, "y_std": y_std,   # save a dict containing the weights and what is needed to use them later...
                "final_epoch": final_epoch, "best_val_epoch": best_epoch, "settings": s},   # ...(dict continues)
               os.path.join(s["out_dir"], f"a1c_finetuned_{tag}.pt"))   # ...into this file

    # ---- final evaluation (test set is used only here) ---------------------------------
    print("[4/5] Evaluating on train, val and test ...")   # progress message
    glucose = glucose_summary_by_subject(X, obs, meta["subject_id"])   # each subject's mean/SD glucose, for the baselines
    metric_rows, preds_test, y_test, y_train_eval, pred_train = [], {}, None, None, None   # an empty list, an empty dict, and None (R: NULL) as placeholders
    for sp in ["train", "val", "test"]:                     # evaluate on train first, then validation, then test
        Xe, oe, se = eval_data[sp]                          # unpack this split's tensors and subject ids
        pred = predict_subjects(model, Xe, oe, se, y_mean, y_std, device)   # one prediction per subject, in A1c %
        ids = list(pred.index)                              # subject ids, in the same order as the predictions
        y = truth.loc[ids].to_numpy()                       # the true A1c of those subjects, same order
        all_preds = {ft_name: pred.to_numpy()}              # dict: the transformer's predictions under its name
        all_preds.update(baseline_predictions(glucose, truth, list(train_ids), ids))   # add the baselines' predictions to the same dict (update modifies it in place)
        for name in [NULL, GLU, GLU_SD, ft_name]:           # fixed display order
            m = compute_metrics(y, all_preds[name])         # MAE, RMSE, R2, r ... for this model
            m.update(split=sp, model=name)                  # add two more entries to the dict (in place)
            lo, hi = bootstrap_mae_ci(y, all_preds[name], s["n_boot"], s["seed"])   # 95% interval for its MAE
            m.update(MAE_ci_low=lo, MAE_ci_high=hi)         # store the interval
            if name == ft_name:                             # only for the transformer: compare with mean glucose
                dlo, dhi = bootstrap_paired_diff_ci(y, all_preds[ft_name], all_preds[GLU],   # 95% interval of the MAE difference...
                                                    s["n_boot"], s["seed"])                  # ...(arguments continue)
                m.update(dMAE_vs_meanglucose=m["MAE"] - compute_metrics(y, all_preds[GLU])["MAE"],   # MAE change vs the mean-glucose baseline (negative = better)
                         dMAE_ci_low=dlo, dMAE_ci_high=dhi)   # and its interval
            metric_rows.append(m)                           # add this model's results to the list
        if sp == "train":                                   # for the training subjects, keep the fine-tuned model's predictions for the plot
            y_train_eval, pred_train = y, all_preds[ft_name]   # remember the truth and the predictions
        if sp == "test":                                    # for the test split, also keep predictions for plotting and saving
            y_test, preds_test = y, all_preds               # remember the truth and every model's predictions
            pd.DataFrame({"subject_id": ids, "a1c_true": y, "a1c_pred": all_preds[ft_name],   # a table: id, truth, prediction...
                          "n_days": sub_info.loc[ids, "n_days"].to_numpy()}                   # ...and how many days each subject had
                         ).to_csv(os.path.join(s["out_dir"], f"test_predictions_{tag}.csv"), index=False)   # write it to a CSV file

    metrics = pd.DataFrame(metric_rows)                     # list of dicts -> data frame (one row per split x model)
    first = ["split", "model", "n_subjects", "MAE", "MAE_ci_low", "MAE_ci_high",   # the column order we want in the saved table
             "RMSE", "R2", "r", "rho", "dMAE_vs_meanglucose", "dMAE_ci_low", "dMAE_ci_high"]   # (continuation of the list above: the rest of the column names)
    metrics = metrics[first]                                # reorder the columns (R: metrics[, first])
    metrics.to_csv(os.path.join(s["out_dir"], f"metrics_{tag}.csv"), index=False)   # save the metrics table

    print("[5/5] Plotting ...")                             # progress message
    plot_results(os.path.join(s["out_dir"], f"finetune_results_{tag}.png"), log,   # draw and save the 6-panel figure...
                 y_train_eval, pred_train, y_test, preds_test, ft_name, final_epoch,   # ...train and test truth/predictions, the model's name, the epoch used...
                 metrics, y_std)                            # ...the full metrics table, and the "always predict the mean" error level (= the SD of A1c)

    by_split = metrics[metrics["model"] == ft_name][["split", "MAE", "RMSE", "R2", "r"]]   # the fine-tuned model's rows for train, val and test
    print(f"\nFine-tuned model by split (A1c in % points; weights from epoch {final_epoch}):")   # "\n" at the start prints a blank line first
    print(by_split.to_string(index=False, float_format=lambda v: f"{v:.3f}"))   # print as text; float_format shows numbers with 3 decimals
    print(f"For reference: always predicting the mean A1c gives an RMSE of about {y_std:.2f}.")   # the level to beat
    print(f"\nTEST results (A1c in % points; weights from epoch {final_epoch}):")   # header for the test table below
    show = metrics[metrics["split"] == "test"][["model", "MAE", "RMSE", "R2", "r"]]   # test rows only, a few columns
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))   # print as text; float_format takes a tiny function (lambda v: ...) that shows numbers with 3 decimals
    row = metrics[(metrics["split"] == "test") & (metrics["model"] == ft_name)].iloc[0]   # the single row for the transformer on test; "&" combines two TRUE/FALSE columns; .iloc[0] = first row (R: [1, ])
    print(f"\nMAE change vs 'Mean glucose': {row['dMAE_vs_meanglucose']:+.3f} "   # "+.3f" always shows the sign
          f"(95% CI {row['dMAE_ci_low']:+.3f} to {row['dMAE_ci_high']:+.3f}); "   # (this and the next line continue the print() above: more text pieces to show)
          "an interval entirely below 0 means the transformer is reliably better.")   # (last piece; the closing bracket ends the print() call)
    print(f"Saved results to ./{s['out_dir']}/")            # tell the user where the files are


if __name__ == "__main__":                                  # TRUE only when you run this file directly (python finetune_a1c.py), not when another file imports it
    main()                                                  # start the program
