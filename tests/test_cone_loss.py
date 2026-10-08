"""Experiment 5's cone-weighted temporal-frequency loss: the window maker and
`cone_window_loss` isolate the error energy inside the radiating cone
omega >= c0*|k|, comparably to plain MSE."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.models.losses import cone_window_loss, mse_loss
from src.models.train import TrainConfig, make_pairs, make_tf_windows, train


def _mask(W, h, w, dt, length, c0):
    """Independent replica of the mask inside `cone_window_loss`, to check its output."""
    omega = torch.fft.fftfreq(W, d=dt) * 2 * math.pi
    omega = omega.abs()
    ky = torch.fft.fftfreq(h, d=length / h)[:, None] * 2 * math.pi
    kx = torch.fft.fftfreq(w, d=length / w)[None, :] * 2 * math.pi
    k = torch.sqrt(kx ** 2 + ky ** 2)
    return (omega[:, None, None] >= c0 * k[None, :, :]) & (k[None, :, :] > 0.0)


def test_white_noise_loss_matches_equally_weighted_cone_fraction():
    torch.manual_seed(0)
    B, W, n = 8, 16, 32
    dt, length, c0 = 1.0, 2 * math.pi, 1.5
    err = torch.randn(B, W, 2, n, n)
    frac = _mask(W, n, n, dt, length, c0).float().mean().item()
    mse = err.pow(2).mean().item()
    loss = cone_window_loss(err, dt, length, c0).item()
    assert abs(loss - frac * mse) < 0.1 * (frac * mse)


def test_constant_in_time_error_is_near_zero():
    torch.manual_seed(1)
    n, W = 16, 16
    dt, length, c0 = 1.0, 2 * math.pi, 1.0
    spatial = torch.randn(1, 1, 2, n, n)
    err = spatial.expand(1, W, 2, n, n).clone()
    mse = err.pow(2).mean().item()
    loss = cone_window_loss(err, dt, length, c0).item()
    assert loss < 1e-2 * mse


def _traveling_wave(j, n, W):
    """A pure plane wave cos(kx*x - omega_j*t), kx = 1, in channel 0 only."""
    omega = 2 * math.pi * j / W
    i = torch.arange(n, dtype=torch.float32)
    t = torch.arange(W, dtype=torch.float32)
    phase = 2 * math.pi * i[None, :] / n - omega * t[:, None]       # (W, n)
    field = torch.cos(phase)[:, None, :].expand(W, n, n).clone()    # same across y
    err = torch.zeros(1, W, 2, n, n)
    err[0, :, 0] = field
    return err


def test_plane_wave_inside_cone_carries_most_of_its_energy():
    n, W, dt, length, c0 = 32, 16, 1.0, 2 * math.pi, 1.0
    err = _traveling_wave(j=3, n=n, W=W)       # omega = 1.178 >= c0*k = 1.0
    mse = err.pow(2).mean().item()
    loss = cone_window_loss(err, dt, length, c0).item()
    assert loss / mse > 0.8


def test_plane_wave_outside_cone_carries_little_energy():
    n, W, dt, length, c0 = 32, 16, 1.0, 2 * math.pi, 1.0
    err = _traveling_wave(j=1, n=n, W=W)       # omega = 0.393 < c0*k = 1.0
    mse = err.pow(2).mean().item()
    loss = cone_window_loss(err, dt, length, c0).item()
    assert loss / mse < 0.2


def test_gradient_flows_and_is_finite():
    torch.manual_seed(2)
    err = torch.randn(2, 16, 2, 16, 16, requires_grad=True)
    loss = cone_window_loss(err, dt=1.0, length=2 * math.pi, c0=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert err.grad is not None
    assert torch.isfinite(err.grad).all()


def test_lambda_zero_equals_mse_exactly():
    torch.manual_seed(3)
    p = torch.randn(2, 16, 2, 8, 8)
    t = torch.randn(2, 16, 2, 8, 8)
    cone_lambda = 0.0
    combined = mse_loss(p, t) + cone_lambda * cone_window_loss(p - t, 1.0, 2 * math.pi, 1.0)
    assert torch.equal(combined, mse_loss(p, t))


def test_make_tf_windows_shapes_targets_and_stats_match_make_pairs():
    rng = np.random.default_rng(0)
    nt, n, W, stride = 40, 8, 6, 3
    u, v = rng.standard_normal((nt, n, n)), rng.standard_normal((nt, n, n))

    x, y, stats = make_tf_windows(u, v, W, stride)
    n_windows = len(range(0, nt - W, stride))
    assert x.shape == (n_windows, W, 2, n, n)
    assert y.shape == (n_windows, W, 2, n, n)
    for i in range(W - 1):
        assert torch.equal(y[:, i], x[:, i + 1])

    _, _, pair_stats = make_pairs(u, v)
    assert stats == pair_stats


def test_window_mode_train_runs_and_returns_finite_history_with_val_cone():
    rng = np.random.default_rng(4)
    nt, n, W, stride = 40, 8, 6, 3
    u, v = rng.standard_normal((nt, n, n)), rng.standard_normal((nt, n, n))
    x, y, _ = make_tf_windows(u, v, W, stride)

    torch.manual_seed(5)
    model = torch.nn.Conv2d(2, 2, 3, padding=1)
    cfg = TrainConfig(epochs=1, batch_size=2, loss="cone", device="cpu",
                      cone_lambda=1.0, cone_window=W, cone_c0=0.5, cone_dt=1.0,
                      domain_length=2 * math.pi)
    _, hist = train(model, x, y, cfg)

    assert len(hist) == 1
    assert math.isfinite(hist[-1]["train"])
    assert math.isfinite(hist[-1]["val"])
    assert "val_cone" in hist[-1]
    assert math.isfinite(hist[-1]["val_cone"])
