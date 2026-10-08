"""Deliberately vanilla one-step training.

Do not improve this. The claim is that the *standard* recipe is structurally
blind to the acoustics, so any cleverness here weakens the result. One-step MSE,
Adam, cosine schedule -- exactly what every paper in this space does.
`loss='cone'` (Experiment 5) is a deliberate departure from that recipe, not
part of the baseline claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import mse_loss, sobolev_loss, h1_loss, cone_window_loss


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "mse"            # 'mse' | 'sobolev' | 'h1' | 'cone'
    unroll: int = 1              # >1: backprop through this many autoregressive steps
    sobolev_order: int = 1
    sobolev_weight: float = 1e-2
    domain_length: float = 2 * np.pi
    val_fraction: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    cone_lambda: float = 1e3     # Experiment 5: weight of the cone term, mse-scale units
    cone_mach: float = 0.1       # analysis Mach number used to derive cone_c0 (record-keeping)
    cone_window: int = 16        # teacher-forced window length W (record-keeping; W = x.shape[1])
    cone_c0: float = 11.31       # sound speed used by the cone mask: sound_speed(u_rms, cone_mach)
    cone_dt: float = 0.04        # seconds between consecutive frames (dataset's dt_save)


def make_pairs(u, v, normalise=True):
    """(nt, n, n) histories -> one-step tensors x=(N,2,n,n), y=(N,2,n,n).

    Normalisation statistics are returned so predictions can be mapped back to
    physical units before any acoustic analysis -- the Lighthill source is
    quadratic, so analysing normalised fields silently rescales it.
    """
    x = np.stack([u[:-1], v[:-1]], axis=1)
    y = np.stack([u[1:], v[1:]], axis=1)
    stats = {"mean": 0.0, "std": 1.0}
    if normalise:
        stats = {"mean": float(x.mean()), "std": float(x.std()) or 1.0}
        x = (x - stats["mean"]) / stats["std"]
        y = (y - stats["mean"]) / stats["std"]
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32), stats)


def make_windows(u, v, steps, normalise=True):
    """(nt, n, n) histories -> x=(N,2,n,n) and the next `steps` targets y=(N,steps,2,n,n).

    For unrolled training; normalised exactly as `make_pairs` (same statistics), so
    steps=1 gives make_pairs' y with an extra axis.
    """
    f = np.stack([u, v], axis=1)                     # (nt, 2, n, n)
    stats = {"mean": 0.0, "std": 1.0}
    n = len(f) - steps
    if normalise:
        stats = {"mean": float(f[:-1].mean()), "std": float(f[:-1].std()) or 1.0}
        f = (f - stats["mean"]) / stats["std"]
    x = f[:n]
    y = np.stack([f[1 + j: 1 + j + n] for j in range(steps)], axis=1)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32), stats)


def make_tf_windows(u, v, W, stride, normalise=True):
    """(nt, n, n) histories -> teacher-forced windows x=(Nw, W, 2, n, n) (frames
    t..t+W-1) and y=(Nw, W, 2, n, n) (frames t+1..t+W), one window starting every
    `stride` frames.  Each y[:, j] is the true next frame of x[:, j] -- there is no
    rollout and no graph through time; a model applied frame-by-frame to x reproduces
    one-step teacher-forced prediction for every frame in the window at once.

    Normalised exactly as `make_pairs` (same statistics, computed over u[:-1], v[:-1]),
    so cone runs are comparable to the other losses.
    """
    f = np.stack([u, v], axis=1)                      # (nt, 2, n, n)
    stats = {"mean": 0.0, "std": 1.0}
    if normalise:
        stats = {"mean": float(f[:-1].mean()), "std": float(f[:-1].std()) or 1.0}
        f = (f - stats["mean"]) / stats["std"]
    nt = len(f)
    starts = list(range(0, nt - W, stride))
    x = np.stack([f[t:t + W] for t in starts], axis=0)
    y = np.stack([f[t + 1:t + 1 + W] for t in starts], axis=0)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32), stats)


def _infer_window_stride(x, w):
    """Recover the stride `make_tf_windows` used, from the windows themselves.

    Windows are equally spaced, deterministic slices of one time series, so the
    overlap between consecutive windows is exact: the smallest shift `s` for which
    x[0][s:] == x[1][:w-s] is the stride.  Used only to find the train/val boundary
    that avoids leaking frames between the two splits.
    """
    if len(x) < 2:
        return w
    first, nxt = x[0], x[1]
    for s in range(1, w):
        if torch.equal(first[s:], nxt[:w - s]):
            return s
    return w


def denormalise(arr, stats):
    return arr * stats["std"] + stats["mean"]


def train(model, x, y, cfg: TrainConfig, on_epoch_end=None, resume=None):
    """Train with the vanilla recipe.

    on_epoch_end(epoch, model, opt, sched, history) is called after every epoch, for
    checkpointing; it must not change training state.  `resume` is a dict saved from
    that callback (keys model, opt, sched, history, rng) and continues the run exactly
    where it stopped -- the recipe is unchanged either way.
    """
    torch.manual_seed(cfg.seed)
    n_val = max(1, int(len(x) * cfg.val_fraction))
    if x.ndim == 5:                                  # teacher-forced windows (cone): drop
        split = len(x) - n_val                        # train windows that overlap the val region
        stride = _infer_window_stride(x, x.shape[1])
        train_end = split
        while train_end > 0 and (train_end - 1) * stride + x.shape[1] > split * stride:
            train_end -= 1
        xt, yt, xv, yv = x[:train_end], y[:train_end], x[split:], y[split:]
    else:
        xt, yt, xv, yv = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]

    loader = DataLoader(TensorDataset(xt, yt), batch_size=cfg.batch_size, shuffle=True)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    history = []
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        history = list(resume["history"])
        torch.set_rng_state(resume["rng"])

    def criterion(p, t):
        if cfg.loss == "mse":
            return mse_loss(p, t)
        if cfg.loss == "sobolev":
            return sobolev_loss(p, t, cfg.sobolev_order, cfg.sobolev_weight, cfg.domain_length)
        if cfg.loss == "h1":
            return h1_loss(p, t, cfg.domain_length)
        if cfg.loss == "cone":
            return mse_loss(p, t) + cfg.cone_lambda * cone_window_loss(
                p - t, cfg.cone_dt, cfg.domain_length, cfg.cone_c0)
        raise ValueError(cfg.loss)

    def step_loss(xb, yb):
        if xb.ndim == 5:                             # teacher-forced windows (Experiment 5)
            b, w = xb.shape[0], xb.shape[1]
            p = model(xb.reshape(b * w, *xb.shape[2:])).reshape(b, w, *xb.shape[2:])
            return criterion(p, yb)
        if yb.ndim == 4:                             # one-step pairs
            return criterion(model(xb), yb)
        # unrolled: feed predictions back, keep the graph, average over the steps
        p, total = xb, 0.0
        for j in range(yb.shape[1]):
            p = model(p)
            total = total + criterion(p, yb[:, j])
        return total / yb.shape[1]

    for epoch in range(len(history), cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            loss = step_loss(xb, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        sched.step()

        model.eval()
        extra = {}
        with torch.no_grad():
            if xv.ndim == 5:                         # window mode: chunked one-step MSE + cone
                bv, wv = xv.shape[0], xv.shape[1]
                flat_x = xv.reshape(bv * wv, *xv.shape[2:])
                preds = [model(flat_x[i:i + 8].to(cfg.device)) for i in range(0, len(flat_x), 8)]
                pred = torch.cat(preds, dim=0).reshape(bv, wv, *xv.shape[2:])
                yv_dev = yv.to(cfg.device)
                val = mse_loss(pred, yv_dev).item()
                extra["val_cone"] = cone_window_loss(
                    pred - yv_dev, cfg.cone_dt, cfg.domain_length, cfg.cone_c0).item()
            else:
                yv1 = yv if yv.ndim == 4 else yv[:, 0]   # one-step MSE: comparable across runs
                val = mse_loss(model(xv.to(cfg.device)), yv1.to(cfg.device)).item()
        history.append({"epoch": epoch, "train": running / len(xt), "val": val, **extra})
        if epoch % max(1, cfg.epochs // 10) == 0:
            print(f"  epoch {epoch:3d}  train {running/len(xt):.3e}  val {val:.3e}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, opt, sched, history)
    return model, history


@torch.no_grad()
def predict_onestep(model, x, cfg: TrainConfig, batch=8):
    """One-step predictions over the whole record (teacher forced).

    One-step, not rolled out, on purpose: rollout divergence is a different
    (well-studied) failure mode, and mixing it in would let a reader attribute
    the acoustic error to accumulated drift instead of to the objective.
    """
    model.eval().to(cfg.device)
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(cfg.device)).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_rollout(model, x0, n_steps, cfg: TrainConfig):
    """Free autoregressive rollout: start from one true (normalised) snapshot x0 of shape
    (2, n, n) and feed every prediction back in.  Returns (n_valid, 2, n, n), stopping
    early at the first non-finite step.

    The control for `predict_onestep`: teacher forcing re-anchors each step to the truth,
    so step-to-step jitter is all its error can contain; a rollout has to carry the
    large scales itself.
    """
    model.eval().to(cfg.device)
    x = x0[None].to(cfg.device)
    out = []
    for _ in range(n_steps):
        x = model(x)
        if not torch.isfinite(x).all():
            break
        out.append(x[0].cpu().numpy())
    return np.stack(out) if out else np.empty((0,) + tuple(x0.shape), dtype=np.float32)


def load_model(path, n):
    """Rebuild a checkpoint written by scripts/02 (same constructors) -> (model, ckpt dict)."""
    from .fno import FNO2d
    from .unet import UNet2d
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FNO2d(modes=min(24, n // 4), width=48) if ck["arch"] == "fno" else UNet2d(base=48, depth=3)
    model.load_state_dict(ck["state_dict"])
    return model, ck
