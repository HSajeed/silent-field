#!/usr/bin/env python3
"""Assemble the README figures from saved metrics and data."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.diagnostics.spectra import time_averaged_energy_spectrum, fit_slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--fit-lo", type=float, default=12)
    ap.add_argument("--fit-hi", type=float, default=36)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, v, meta = load_dataset(args.data)
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    k, e, n_used = time_averaged_energy_spectrum(u, v, grid, stride=args.stride)
    good = (k > 0) & (e > 0)

    fit_lo, fit_hi = args.fit_lo, args.fit_hi
    slope = fit_slope(k, e, fit_lo, fit_hi)

    forcing_k = meta["forcing_k"]
    if fit_lo <= forcing_k <= fit_hi:
        print(f"WARNING: forcing_k={forcing_k} lies within the fit range [{fit_lo}, {fit_hi}] "
              f"-- this mixes the injection scale into the inertial-range fit!")

    print(f"slope over k in [{fit_lo}, {fit_hi}], time-averaged over {n_used} snapshots = {slope:.3f}")

    print("\nslope table (for transparency):")
    ranges = [(10, 30), (12, 36), (16, 40), (4, grid.n // 4)]
    for lo, hi in ranges:
        s = fit_slope(k, e, lo, hi)
        print(f"  [{lo:>3}, {hi:>3}] -> slope = {s:.3f}")

    print("\nlocal-slope profile (sliding window [k, 2k]):")
    for kk in range(10, 21, 2):
        s = fit_slope(k, e, kk, 2 * kk)
        print(f"  [{kk:>3}, {2*kk:>3}] -> slope = {s:.3f}")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(u[-1], origin="lower", cmap="RdBu_r"); ax[0].set_title("u")

    ax[1].loglog(k[good], e[good], label="E(k), time-averaged")

    m = (k >= fit_lo) & (k <= fit_hi) & (e > 0)
    if m.sum() >= 3:
        k_fit = k[m]
        c = np.exp(np.polyfit(np.log(k_fit), np.log(e[m]), 1)[1])
        ax[1].loglog(k_fit, c * k_fit ** slope, "r-", lw=2, label=f"fit slope={slope:.2f}")

    k_ref = k[good]
    if k_ref.size:
        c_ref = e[good][0] * k_ref[0] ** 3
        ax[1].loglog(k_ref, c_ref * k_ref ** -3, "k--", lw=1, label=r"$k^{-3}$")

    ax[1].axvline(forcing_k, color="g", ls=":", label=f"forcing k={forcing_k}")
    ax[1].set_xlabel("|k|"); ax[1].set_ylabel("E(k)")
    ax[1].legend(fontsize=7)
    ax[1].set_title(
        f"energy spectrum, fit range [{fit_lo:g}, {fit_hi:g}], "
        f"time-averaged ({n_used} snapshots), slope={slope:.2f}"
    )
    fig.tight_layout(); fig.savefig(out/"05_flow_overview.png", dpi=140)
    print(f"\nwrote {out/'05_flow_overview.png'}  slope={slope:.3f}")


if __name__ == "__main__":
    main()
