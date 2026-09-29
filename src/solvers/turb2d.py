"""Forced 2D Navier-Stokes on a doubly-periodic box, pseudospectral.

Vorticity-streamfunction form:

    d omega / dt  =  -(u d omega/dx + v d omega/dy)  +  nu laplacian(omega)
                     -  mu omega  +  f

with omega = -laplacian(psi), u = d psi/dy, v = -d psi/dx.

Time stepping is RK4 on the advective and forcing terms with an exact
integrating factor for the viscous and drag terms, which removes the stiff
linear part from the stability constraint.

The linear drag mu is not optional in 2D: the inverse cascade piles energy at
the largest scales until the box condenses into a pair of vortices, which
destroys the statistical stationarity that the (k, omega) estimate needs.

Why this and not a library: the solver is ~120 lines, and the project's whole
defence against "your source error is really your discretisation error" is that
every derivative is spectral and exact on a periodic domain.  Owning the solver
keeps that guarantee visible rather than buried in someone else's abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..spectral import Grid, fft2, ifft2, velocity_from_vorticity


@dataclass
class Turb2DConfig:
    n: int = 256
    length: float = 2.0 * np.pi
    viscosity: float = 1.0e-4      # nu
    drag: float = 0.05             # mu, arrests the inverse cascade
    forcing_k: float = 8.0         # wavenumber of energy injection
    forcing_amp: float = 1.0
    forcing_kind: str = "kolmogorov"  # 'kolmogorov' | 'ring' | 'none'
    dt: float = 2.0e-3
    seed: int = 0


class Turb2D:
    """Pseudospectral solver; state is vorticity in physical space."""

    def __init__(self, cfg: Turb2DConfig):
        self.cfg = cfg
        self.grid = Grid(n=cfg.n, length=cfg.length)
        self.kx, self.ky = self.grid.wavenumbers()
        self.k_sq = self.grid.k_sq
        self.mask = self.grid.dealias_mask()
        self.rng = np.random.default_rng(cfg.seed)
        self._forcing = self._build_forcing()

    # ------------------------------------------------------------------ setup
    def _build_forcing(self) -> np.ndarray:
        cfg = self.cfg
        X, Y = self.grid.meshgrid()
        if cfg.forcing_kind == "none":
            return np.zeros_like(X)
        if cfg.forcing_kind == "kolmogorov":
            # Deterministic, reproducible, and standard for forced 2D turbulence.
            return cfg.forcing_amp * cfg.forcing_k * np.sin(cfg.forcing_k * Y)
        if cfg.forcing_kind == "ring":
            # Random phases on a thin annulus in k -- more isotropic than
            # Kolmogorov forcing, at the cost of reproducibility across seeds.
            k_mag = self.grid.k_mag
            shell = (k_mag > cfg.forcing_k - 1.0) & (k_mag < cfg.forcing_k + 1.0)
            phase = self.rng.uniform(0, 2 * np.pi, size=k_mag.shape)
            f_hat = np.where(shell, np.exp(1j * phase), 0.0)
            f = ifft2(f_hat)
            return cfg.forcing_amp * f / (f.std() + 1e-30)
        raise ValueError(f"unknown forcing_kind {cfg.forcing_kind!r}")

    def initial_condition(self, energy: float = 1.0) -> np.ndarray:
        """Random solenoidal field with a k^4 exp(-2(k/k0)^2) energy spectrum."""
        k_mag = self.grid.k_mag
        k0 = self.cfg.forcing_k
        spec = k_mag**4 * np.exp(-2.0 * (k_mag / k0) ** 2)
        phase = self.rng.uniform(0, 2 * np.pi, size=k_mag.shape)
        om_hat = np.sqrt(spec) * np.exp(1j * phase)
        om_hat[0, 0] = 0.0
        om = ifft2(om_hat)
        return om / (om.std() + 1e-30) * energy

    # ------------------------------------------------------------------ rhs
    def _rhs(self, om_hat: np.ndarray) -> np.ndarray:
        """Nonlinear advection plus forcing, in spectral space (dealiased)."""
        k_sq = self.k_sq.copy()
        k_sq[0, 0] = 1.0
        psi_hat = om_hat / k_sq
        psi_hat[0, 0] = 0.0

        u = ifft2(1j * self.ky * psi_hat)
        v = ifft2(-1j * self.kx * psi_hat)
        om_x = ifft2(1j * self.kx * om_hat)
        om_y = ifft2(1j * self.ky * om_hat)

        adv_hat = fft2(u * om_x + v * om_y) * self.mask
        return -adv_hat + fft2(self._forcing)

    def step(self, om: np.ndarray, dt: float | None = None) -> np.ndarray:
        """One RK4 step with an integrating factor for the linear terms."""
        dt = self.cfg.dt if dt is None else dt
        lin = -self.cfg.viscosity * self.k_sq - self.cfg.drag
        e_full = np.exp(lin * dt)
        e_half = np.exp(lin * dt / 2.0)

        w = fft2(om)
        k1 = self._rhs(w)
        k2 = self._rhs(e_half * (w + 0.5 * dt * k1))
        k3 = self._rhs(e_half * w + 0.5 * dt * k2)
        k4 = self._rhs(e_full * w + dt * e_half * k3)

        w_new = (
            e_full * w
            + dt / 6.0 * (e_full * k1 + 2.0 * e_half * k2 + 2.0 * e_half * k3 + k4)
        )
        w_new *= self.mask
        return ifft2(w_new)

    # ------------------------------------------------------------------ driver
    def run(
        self,
        n_snapshots: int,
        save_every: int,
        spinup_steps: int = 0,
        omega0: np.ndarray | None = None,
        progress: bool = True,
    ):
        """Integrate and return (u, v) histories of shape (n_snapshots, n, n).

        `save_every * dt` is the sampling interval of the returned record and is
        what sets the frequency axis of every (k, omega) figure.  See
        `acoustic_sampling_interval` for how to choose it.
        """
        om = self.initial_condition() if omega0 is None else omega0.copy()

        for _ in range(spinup_steps):
            om = self.step(om)

        u_hist = np.empty((n_snapshots, self.grid.n, self.grid.n), dtype=np.float64)
        v_hist = np.empty_like(u_hist)

        for i in range(n_snapshots):
            for _ in range(save_every):
                om = self.step(om)
            u_hist[i], v_hist[i] = velocity_from_vorticity(om, self.grid)
            if progress and (i % max(1, n_snapshots // 20) == 0):
                print(f"  snapshot {i}/{n_snapshots}  u_rms={u_hist[i].std():.4f}")

        return u_hist, v_hist, om


def acoustic_sampling_interval(
    u_rms: float, grid: Grid, mach: float, safety: float = 4.0
) -> float:
    """Largest dt that keeps the radiating cone resolvable.

    The cone reaches up to omega = c0 * k.  To have any resolved content inside
    it at the smallest non-zero wavenumber k_min = 2 pi / L we need the temporal
    Nyquist frequency to exceed c0 * k_min:

        pi / dt  >  (u_rms / mach) * (2 pi / L)      =>      dt < mach * L / (2 u_rms)

    `safety` oversamples beyond that bound.  At M = 0.1 this is a mild
    constraint -- roughly a twentieth of an eddy turnover -- not the punishing
    one it first appears.
    """
    if mach <= 0 or u_rms <= 0:
        raise ValueError("mach and u_rms must be positive")
    return mach * grid.length / (2.0 * u_rms * safety)
