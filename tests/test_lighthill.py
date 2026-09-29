"""Spectral derivatives and both Lighthill formulations against analytic results."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.spectral import Grid, ddx, ddy, laplacian, velocity_from_vorticity, divergence
from src.acoustics.lighthill import (lighthill_source_gradient_form,
                                     lighthill_source_divergence_form)

G = Grid(n=64)

def test_derivatives_exact():
    X, Y = G.meshgrid(); f = np.sin(3*X)*np.cos(2*Y)
    assert np.abs(ddx(f, G) - 3*np.cos(3*X)*np.cos(2*Y)).max() < 1e-12
    assert np.abs(ddy(f, G) + 2*np.sin(3*X)*np.sin(2*Y)).max() < 1e-12
    assert np.abs(laplacian(f, G) + 13*f).max() < 1e-11

def test_taylor_green_velocity():
    X, Y = G.meshgrid()
    om = -laplacian(np.cos(X)*np.cos(Y), G)
    u, v = velocity_from_vorticity(om, G)
    assert np.abs(u + np.cos(X)*np.sin(Y)).max() < 1e-12
    assert np.abs(v - np.sin(X)*np.cos(Y)).max() < 1e-12
    assert np.abs(divergence(u, v, G)).max() < 1e-12

def test_lighthill_forms_agree():
    """Algebraically identical for solenoidal fields; agreement is the check."""
    X, Y = G.meshgrid()
    om = -laplacian(np.cos(X)*np.cos(Y), G)
    u, v = velocity_from_vorticity(om, G)
    a = lighthill_source_gradient_form(u, v, G)
    b = lighthill_source_divergence_form(u, v, G)
    assert np.abs(a-b).max()/np.abs(b).max() < 1e-10

def test_lighthill_analytic():
    X, Y = G.meshgrid()
    u, v = -np.cos(X)*np.sin(Y), np.sin(X)*np.cos(Y)
    ux, uy = np.sin(X)*np.sin(Y), -np.cos(X)*np.cos(Y)
    vx, vy = np.cos(X)*np.cos(Y), -np.sin(X)*np.sin(Y)
    expect = ux**2 + 2*uy*vx + vy**2
    got = lighthill_source_gradient_form(u, v, G)
    assert np.abs(got - expect).max() < 1e-11
