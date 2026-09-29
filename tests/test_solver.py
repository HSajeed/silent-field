"""The solver must stay bounded and divergence-free."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.spectral import divergence
from src.solvers.turb2d import Turb2D, Turb2DConfig, acoustic_sampling_interval
from src.solvers.vortex_pair import VortexPair
from src.spectral import Grid

def test_solver_stable_and_solenoidal():
    s = Turb2D(Turb2DConfig(n=32, viscosity=5e-3, drag=0.1, forcing_k=4, dt=2e-3))
    u, v, _ = s.run(6, 20, spinup_steps=200, progress=False)
    assert np.isfinite(u).all() and u.std() > 0
    assert np.abs(divergence(u[-1], v[-1], s.grid)).max() < 1e-10

def test_vortex_pair_orbital_frequency():
    g = Grid(n=64)
    vp = VortexPair(grid=g, circulation=1.0, radius=0.6, core=0.2)
    assert abs(vp.omega_orbit - 1.0/(4*np.pi*0.36)) < 1e-12
    assert abs(vp.omega_acoustic - 2*vp.omega_orbit) < 1e-12

def test_sampling_bound_scales_with_mach():
    g = Grid(n=64)
    assert acoustic_sampling_interval(1.0, g, 0.2) > acoustic_sampling_interval(1.0, g, 0.1)
