"""Shared spectral-grid utilities.

Everything in this project lives on a doubly-periodic square domain [0, L)^2 with
N points per side, so derivatives are taken by FFT and are exact to machine
precision.  That exactness is not a convenience -- it is the defence against the
objection that a large Lighthill-source error is really the discretisation error
of the differentiation scheme.

Index convention throughout:  arrays are (..., ny, nx), i.e. y is axis -2 and x
is axis -1.  Wavenumbers are angular (rad / length), so a mode with p full
periods across the box has k = 2*pi*p / L.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A doubly-periodic square spectral grid."""

    n: int
    length: float = 2.0 * np.pi

    # ---------------------------------------------------------------- geometry
    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.n) * self.dx

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical-space coordinates, shape (ny, nx)."""
        x = self.x
        return np.meshgrid(x, x, indexing="xy")  # -> (X, Y) both (ny, nx)

    # ------------------------------------------------------------- wavenumbers
    @property
    def k1d(self) -> np.ndarray:
        """Angular wavenumbers along one axis, FFT-ordered."""
        return 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)

    def wavenumbers(self) -> tuple[np.ndarray, np.ndarray]:
        """(kx, ky), each shape (ny, nx), broadcast for use with fft2."""
        k = self.k1d
        kx = k[None, :]
        ky = k[:, None]
        return np.broadcast_to(kx, (self.n, self.n)), np.broadcast_to(ky, (self.n, self.n))

    @property
    def k_sq(self) -> np.ndarray:
        kx, ky = self.wavenumbers()
        return kx**2 + ky**2

    @property
    def k_mag(self) -> np.ndarray:
        return np.sqrt(self.k_sq)

    @property
    def k_max(self) -> float:
        """Largest resolved wavenumber magnitude along an axis (Nyquist)."""
        return float(np.abs(self.k1d).max())

    # ------------------------------------------------------------- dealiasing
    def dealias_mask(self, rule: float = 2.0 / 3.0) -> np.ndarray:
        """Boolean mask implementing the 2/3 rule on each axis separately.

        The Lighthill source is quadratic in u, so forming u_i u_j without
        dealiasing folds unresolved high-k content back onto exactly the modes
        this project measures.  Always use this when multiplying fields.
        """
        kx, ky = self.wavenumbers()
        cut = rule * self.k_max
        return (np.abs(kx) <= cut) & (np.abs(ky) <= cut)


# ------------------------------------------------------------------ transforms
def fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fft2(field, axes=(-2, -1))


def ifft2(field_hat: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(field_hat, axes=(-2, -1)))


# ----------------------------------------------------------------- derivatives
def ddx(field: np.ndarray, grid: Grid) -> np.ndarray:
    """Exact periodic d/dx by FFT."""
    kx, _ = grid.wavenumbers()
    return ifft2(1j * kx * fft2(field))


def ddy(field: np.ndarray, grid: Grid) -> np.ndarray:
    """Exact periodic d/dy by FFT."""
    _, ky = grid.wavenumbers()
    return ifft2(1j * ky * fft2(field))


def gradient(field: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    return ddx(field, grid), ddy(field, grid)


def laplacian(field: np.ndarray, grid: Grid) -> np.ndarray:
    return ifft2(-grid.k_sq * fft2(field))


def divergence(u: np.ndarray, v: np.ndarray, grid: Grid) -> np.ndarray:
    return ddx(u, grid) + ddy(v, grid)


def dealias(field: np.ndarray, grid: Grid) -> np.ndarray:
    """Project a physical-space field onto its dealiased modes."""
    return ifft2(fft2(field) * grid.dealias_mask())


# ------------------------------------------------------------ vorticity <-> u
def velocity_from_vorticity(omega: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Invert omega = -laplacian(psi), then u = d psi/dy, v = -d psi/dx.

    The k = 0 mode of psi is undetermined and is set to zero, which fixes the
    mean flow to zero.
    """
    kx, ky = grid.wavenumbers()
    k_sq = grid.k_sq.copy()
    k_sq[0, 0] = 1.0  # avoid division by zero; mode zeroed below

    psi_hat = fft2(omega) / k_sq
    psi_hat[..., 0, 0] = 0.0

    u = ifft2(1j * ky * psi_hat)
    v = ifft2(-1j * kx * psi_hat)
    return u, v


def vorticity_from_velocity(u: np.ndarray, v: np.ndarray, grid: Grid) -> np.ndarray:
    """omega = dv/dx - du/dy."""
    return ddx(v, grid) - ddy(u, grid)


def project_solenoidal(u: np.ndarray, v: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Leray projection: remove the dilatational part of a vector field.

    Used by the divergence diagnostic to split a surrogate's prediction into the
    part that could have come from an incompressible flow and the spurious
    remainder.
    """
    kx, ky = grid.wavenumbers()
    k_sq = grid.k_sq.copy()
    k_sq[0, 0] = 1.0

    u_hat, v_hat = fft2(u), fft2(v)
    div_hat = (kx * u_hat + ky * v_hat) / k_sq
    return ifft2(u_hat - kx * div_hat), ifft2(v_hat - ky * div_hat)
