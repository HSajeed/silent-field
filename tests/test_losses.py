"""Sobolev loss: correct angular wavenumbers, a sane default scale, and the
relative high-k up-weighting that is the whole point of the Stage-4 loss."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.losses import _k_grid, sobolev_loss, mse_loss


def test_k_grid_length_2pi_is_mode_index_squared():
    h = w = 8
    k_sq = _k_grid(h, w, "cpu", length=2 * math.pi)
    m_y = (torch.fft.fftfreq(h, d=1.0) * h)[:, None]
    m_x = (torch.fft.rfftfreq(w, d=1.0) * w)[None, :]
    expected = m_x ** 2 + m_y ** 2
    assert torch.allclose(k_sq, expected, atol=1e-4)


def test_k_grid_length_4pi_is_half_mode_index_squared():
    h = w = 8
    k_sq = _k_grid(h, w, "cpu", length=4 * math.pi)
    m_y = (torch.fft.fftfreq(h, d=1.0) * h)[:, None]
    m_x = (torch.fft.rfftfreq(w, d=1.0) * w)[None, :]
    expected = (m_x / 2) ** 2 + (m_y / 2) ** 2
    assert torch.allclose(k_sq, expected, atol=1e-4)


def _penalty_mse_ratio(n, seed=0):
    torch.manual_seed(seed)
    diff = torch.randn(8, n, n)
    target = torch.zeros_like(diff)
    base = mse_loss(diff, target).item()
    full = sobolev_loss(diff, target, order=1, weight=1.0).item()
    return (full - base) / base


def test_white_noise_penalty_mse_ratio_is_order_one_64():
    ratio = _penalty_mse_ratio(64)
    assert 0.5 <= ratio <= 2.0


def test_white_noise_penalty_mse_ratio_is_order_one_128():
    ratio = _penalty_mse_ratio(128)
    assert 0.5 <= ratio <= 2.0


def _single_mode_error(n, mode, length=2 * math.pi, amp=1.0):
    """A pure sinusoidal error at an exact integer spatial frequency `mode`.

    Discrete orthogonality of the Fourier basis on a periodic grid makes
    mean(diff**2) == amp**2 / 2 exactly, independent of `mode`, so two such
    fields give identical MSE by construction -- any difference in the
    Sobolev loss must come from the k-dependent weighting.
    """
    coords = torch.arange(n, dtype=torch.float32) / n * length
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kx, ky = mode
    phase = 2 * math.pi * (kx * x + ky * y) / length
    return (amp * torch.cos(phase))[None]


def test_high_k_error_penalised_more_than_low_k_at_equal_mse():
    n = 64
    diff_low = _single_mode_error(n, mode=(1, 1))
    diff_high = _single_mode_error(n, mode=(20, 20))
    target = torch.zeros_like(diff_low)

    mse_low = mse_loss(diff_low, target).item()
    mse_high = mse_loss(diff_high, target).item()
    assert abs(mse_low - mse_high) / mse_low < 1e-4

    loss_low = sobolev_loss(diff_low, target, order=1, weight=1.0).item()
    loss_high = sobolev_loss(diff_high, target, order=1, weight=1.0).item()
    assert loss_high > loss_low


def test_sobolev_loss_is_differentiable():
    torch.manual_seed(0)
    pred = torch.randn(4, 32, 32, requires_grad=True)
    target = torch.randn(4, 32, 32)
    loss = sobolev_loss(pred, target, order=1, weight=1.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_h1_loss_is_relative_and_weights_high_k_more():
    from src.models.losses import h1_loss
    n = 32
    x = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="xy")
    target = (torch.cos(X) + torch.sin(2 * Y))[None, None]
    assert h1_loss(target, target).item() == 0.0
    assert abs(h1_loss(2 * target, target).item() - 1.0) < 1e-5     # relative error 1
    low = target + 0.1 * torch.cos(2 * X)[None, None]
    high = target + 0.1 * torch.cos(10 * X)[None, None]             # same L2 error, higher k
    assert h1_loss(high, target).item() > 3 * h1_loss(low, target).item()
