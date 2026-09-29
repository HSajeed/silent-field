"""The Lighthill acoustic source, computed from an incompressible velocity field.

Lighthill's analogy rearranges the exact Navier-Stokes equations into an
inhomogeneous wave equation

    (d2/dt2 - c0^2 laplacian) rho'  =  d2 T_ij / dx_i dx_j,
    T_ij = rho u_i u_j + (p' - c0^2 rho') delta_ij - tau_ij.

At low Mach and high Reynolds the Reynolds-stress term dominates, so the source
is

    S = rho0 * d2 (u_i u_j) / dx_i dx_j.

For an *incompressible* field (d u_i / d x_i = 0) this collapses to a product of
velocity gradients:

    S = rho0 * (d u_i / d x_j) (d u_j / d x_i)

which is (minus) the right-hand side of the pressure Poisson equation.  That
identity is what makes this project cheap: the acoustic source is a post-hoc
functional of an incompressible simulation, so no compressible solver is ever
needed.

Two facts about S drive the whole argument:

1.  It is a *double* derivative, so it amplifies high-k error by k^2.  (This part
    is covered by the existing spectral-bias literature -- acknowledge it.)
2.  It is a near-cancellation of large gradient terms, and only the part of it
    inside the radiating cone of (k, omega) space actually makes sound.  That
    part carries essentially none of the energy.  (This part is not covered, and
    is the contribution.)

Both formulations below are implemented because agreeing with each other to
machine precision is the cheapest available correctness check.
"""

from __future__ import annotations

import numpy as np

from ..spectral import Grid, dealias, ddx, ddy, fft2, ifft2


def lighthill_source_gradient_form(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    rho0: float = 1.0,
    do_dealias: bool = True,
) -> np.ndarray:
    """S = rho0 * (du_i/dx_j)(du_j/dx_i), valid for incompressible fields.

    In 2D with u = (u, v):
        S / rho0 = u_x^2 + 2 u_y v_x + v_y^2
    """
    u_x, u_y = ddx(u, grid), ddy(u, grid)
    v_x, v_y = ddx(v, grid), ddy(v, grid)

    s = u_x * u_x + 2.0 * u_y * v_x + v_y * v_y
    if do_dealias:
        s = dealias(s, grid)
    return rho0 * s


def lighthill_source_divergence_form(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    rho0: float = 1.0,
    do_dealias: bool = True,
) -> np.ndarray:
    """S = rho0 * d2 (u_i u_j) / dx_i dx_j, the general (compressible-valid) form.

    Dealiasing is applied to the quadratic products *before* differentiating,
    which is the correct order: aliasing is introduced by the multiplication.
    """
    uu, uv, vv = u * u, u * v, v * v
    if do_dealias:
        uu, uv, vv = (dealias(f, grid) for f in (uu, uv, vv))

    kx, ky = grid.wavenumbers()
    s_hat = -(kx * kx * fft2(uu) + 2.0 * kx * ky * fft2(uv) + ky * ky * fft2(vv))
    return rho0 * ifft2(s_hat)


def lighthill_source(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    rho0: float = 1.0,
    form: str = "divergence",
    do_dealias: bool = True,
) -> np.ndarray:
    """Dispatch to one of the two formulations.

    Use 'divergence' for anything that may not be exactly solenoidal -- notably a
    surrogate's prediction, whose spurious dilatation is part of what we are
    measuring.  The gradient form silently assumes incompressibility and would
    hide that error.
    """
    if form == "gradient":
        return lighthill_source_gradient_form(u, v, grid, rho0, do_dealias)
    if form == "divergence":
        return lighthill_source_divergence_form(u, v, grid, rho0, do_dealias)
    raise ValueError(f"unknown form {form!r}; expected 'gradient' or 'divergence'")


def source_timeseries(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    rho0: float = 1.0,
    form: str = "divergence",
) -> np.ndarray:
    """Apply `lighthill_source` across a leading time axis.

    u, v : (nt, ny, nx)  ->  (nt, ny, nx)
    """
    if u.ndim != 3:
        raise ValueError(f"expected (nt, ny, nx), got {u.shape}")
    return np.stack(
        [lighthill_source(u[t], v[t], grid, rho0, form) for t in range(u.shape[0])]
    )


def pressure_poisson(
    u: np.ndarray, v: np.ndarray, grid: Grid, rho0: float = 1.0
) -> np.ndarray:
    """Solve laplacian(p) = -S for the incompressible pressure.

    Provided mainly to make the identity between the Lighthill source and the
    pressure Poisson right-hand side checkable, and because the hydrodynamic
    pressure is a useful reference scale for the acoustic pressure.
    """
    s = lighthill_source_gradient_form(u, v, grid, rho0)
    k_sq = grid.k_sq.copy()
    k_sq[0, 0] = 1.0
    p_hat = -fft2(-s) / k_sq  # laplacian(p) = -S  ->  -k^2 p_hat = -S_hat
    p_hat[0, 0] = 0.0
    return ifft2(p_hat)
