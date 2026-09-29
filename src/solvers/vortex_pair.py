"""Co-rotating vortex pair -- the acoustic validation gate.

Two identical vortices of circulation Gamma, each a distance a from the common
centroid, orbit one another at a constant angular velocity.  Each is advected by
the other's induced velocity, v = Gamma / (2 pi d) with d = 2a the separation,
so

    Omega = v / a = Gamma / (4 pi a^2),        T_orbit = 2 pi / Omega.

Two properties of the radiated sound are robust, memorable, and enough to
validate an acoustic pipeline without trusting any prefactor:

  1.  The configuration is a *rotating quadrupole*, so the radiated tone is at
      **twice** the orbital frequency: omega_acoustic = 2 Omega.
  2.  The far-field directivity has **four lobes** -- a cos(2 theta) pattern in
      the co-rotating frame.

This module generates the velocity field of the pair on a periodic grid so the
same Lighthill/Green's-function code used on turbulence can be pointed at a case
with a known answer.  If `scripts/00_validate_acoustics.py` does not recover
2 Omega and four lobes, nothing downstream is trustworthy.

On absolute amplitude: Mueller & Obermeier (1967) give the closed-form far-field
solution with its prefactor.  It is deliberately NOT reproduced here from
memory -- transcribe it from the paper when you want an amplitude comparison.
Frequency and directivity are sufficient for the validation gate.

The vortices are regularised as Lamb-Oseen (Gaussian) cores rather than point
vortices: a point vortex has a singular velocity field that cannot be
represented on a spectral grid without catastrophic Gibbs oscillations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spectral import Grid, velocity_from_vorticity


@dataclass
class VortexPair:
    """Two co-rotating Lamb-Oseen vortices on a periodic grid."""

    grid: Grid
    circulation: float = 1.0  # Gamma
    radius: float = 0.5       # a, distance of each vortex from the centroid
    core: float = 0.15        # Gaussian core size
    centre: tuple[float, float] | None = None

    def __post_init__(self):
        if self.centre is None:
            half = self.grid.length / 2.0
            object.__setattr__(self, "centre", (half, half)) if isinstance(
                self, tuple
            ) else setattr(self, "centre", (half, half))

    # ------------------------------------------------------------------ timing
    @property
    def omega_orbit(self) -> float:
        """Orbital angular velocity Omega = Gamma / (4 pi a^2)."""
        return self.circulation / (4.0 * np.pi * self.radius**2)

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega_orbit

    @property
    def omega_acoustic(self) -> float:
        """Radiated angular frequency: twice the orbital rate (quadrupole)."""
        return 2.0 * self.omega_orbit

    # ------------------------------------------------------------------ fields
    def positions(self, t: float) -> np.ndarray:
        """Vortex centres at time t, shape (2, 2)."""
        cx, cy = self.centre
        phase = self.omega_orbit * t
        return np.array(
            [
                [cx + self.radius * np.cos(phase), cy + self.radius * np.sin(phase)],
                [cx - self.radius * np.cos(phase), cy - self.radius * np.sin(phase)],
            ]
        )

    def vorticity(self, t: float) -> np.ndarray:
        """Superposed Lamb-Oseen vorticity, mean removed for periodicity.

        A periodic box cannot hold net circulation, so the mean is subtracted.
        This is a compromise inherent to validating an unbounded-domain problem
        on a periodic grid; keep the cores well inside the box and the effect on
        the near field is small.
        """
        X, Y = self.grid.meshgrid()
        omega = np.zeros_like(X)
        sigma2 = self.core**2
        for px, py in self.positions(t):
            r2 = (X - px) ** 2 + (Y - py) ** 2
            omega += self.circulation / (np.pi * sigma2) * np.exp(-r2 / sigma2)
        return omega - omega.mean()

    def velocity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return velocity_from_vorticity(self.vorticity(t), self.grid)

    def timeseries(self, n_steps: int, dt: float):
        """Velocity history, shape (n_steps, ny, nx) each component.

        The motion is known analytically, so no time integration is needed --
        the field is evaluated directly at each instant.  That removes the
        solver as a source of error in the validation gate.
        """
        u = np.empty((n_steps, self.grid.n, self.grid.n))
        v = np.empty_like(u)
        for i in range(n_steps):
            u[i], v[i] = self.velocity(i * dt)
        return u, v

    # ------------------------------------------------------------- suggestions
    def suggest_sampling(self, n_periods: int = 8, points_per_period: int = 64):
        """A (n_steps, dt) pair that resolves the acoustic tone with margin.

        The tone is at 2*Omega, so sampling must resolve at least that; 64
        points per orbit gives ample headroom and a clean spectral peak.
        """
        dt = self.period / points_per_period
        return n_periods * points_per_period, dt
