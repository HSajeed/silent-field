"""Deliberately vanilla one-step training.

Do not improve this. The claim is that the *standard* recipe is structurally
blind to the acoustics, so any cleverness here weakens the result. One-step MSE,
Adam, cosine schedule -- exactly what every paper in this space does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import mse_loss, sobolev_loss


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "mse"            # 'mse' | 'sobolev'
    sobolev_order: int = 1
    sobolev_weight: float = 1e-2
    domain_length: float = 2 * np.pi
    val_fraction: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


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
        raise ValueError(cfg.loss)

    for epoch in range(len(history), cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            val = mse_loss(model(xv.to(cfg.device)), yv.to(cfg.device)).item()
        history.append({"epoch": epoch, "train": running / len(xt), "val": val})
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
