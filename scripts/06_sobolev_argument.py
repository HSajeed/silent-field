#!/usr/bin/env python3
"""STAGE 4 -- why the standard remedy for spectral bias cannot work here.

Requires no trained model.  The Stage-4 counterexample used to be a claim about a
training run; it turns out to be a statement about the *geometry* of the Sobolev
weight, and this script measures it directly.

The argument, in one line: a Sobolev objective is a sum of two separately normalised
terms, so the per-mode weight it applies is

    w_eff(k) = 1 + lambda * c * (1 + k^2)^order / <(1 + k^2)^order>

with c = N_cells / N_half on the rfft2 kx = 0 / Nyquist columns and N_cells / (2 N_half)
for every other mode (a half-plane coefficient there stands for a conjugate pair),
shell-averaged for the radial analysis,

and (1 + k^2)^order is three orders of magnitude below its own mean everywhere inside
the radiating cone, because the cone is at low k and 94% of that weight's mass is at
high k.  So at any lambda anyone actually trains at, w_eff is the 1, the objective's
weighting inside the cone is the L2 weighting, and the cone is left exactly as starved
as it was before.

The number this replaces: analysing the same trained objective as a pure
multiplicative (1 + k^2) reweighting makes Sobolev look ~19x worse than L2, which is
what the pre-2026-09-30 README reported.  That number is an artifact of the reading.
The trained objective is the additive one.

    python scripts/06_sobolev_argument.py --data data/turb2d.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.acoustics.komega import welch_komega_spectrum, sound_speed
from src.acoustics.radiating import (
    additive_sobolev_weight,
    effective_weight_in_cone,
    sobolev_halfplane_mean_weight,
    sobolev_lambda_sweep,
    spectral_loss_weight,
)

# Mirrors TrainConfig.sobolev_weight; spelled out so this script stays torch-free.
# tests/test_radiating.py asserts the two agree.
DEFAULT_SOBOLEV_LAMBDA = 1e-2
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def _cone_weight_report(spectrum, grid, c0, lam, order=1):
    """Where the cone sits in the Sobolev weight's range -- the geometric reason.

    Reported as a ratio to the weight's own mean over the rfft2 half-plane, which is
    the normalisation `losses.sobolev_loss` itself uses.
    """
    k = spectrum.k_resolved
    mask = spectrum.radiating_mask_resolved(c0)
    in_cone = np.unique(np.nonzero(mask.any(axis=0))[0])

    w = spectral_loss_weight(k, "sobolev", order)
    mean_w = sobolev_halfplane_mean_weight(grid, order)
    w_eff = additive_sobolev_weight(k, grid, lam, order)

    return {
        "n_k_in_cone": int(in_cone.size),
        "k_max_in_cone": float(k[in_cone].max()) if in_cone.size else 0.0,
        "sobolev_weight_in_cone_over_mean": (
            float(w[in_cone].min() / mean_w) if in_cone.size else None,
            float(w[in_cone].max() / mean_w) if in_cone.size else None,
        ),
        "objective_weight_in_cone_minus_one": (
            float(np.abs(w_eff[in_cone] - 1.0).max()) if in_cone.size else None,
        ),
        "fraction_of_resolved_k_below_cone_edge": float(in_cone.size / k.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--mach", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--plot-mach", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    u, v, meta = load_dataset(args.data)
    dt = meta["dt"]
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
    print(f"record {u.shape}, dt={dt:.5f}, u_rms={u_rms:.4f}")

    spec = welch_komega_spectrum(np.stack([u, v], axis=-1), grid, dt,
                                 n_segments=args.segments)
    print(f"spectrum: {spec.power.shape} binned, {spec.power_resolved.shape} resolved")

    results = {"u_rms": u_rms, "order": args.order, "default_lambda": DEFAULT_SOBOLEV_LAMBDA,
               "data": str(args.data), "argv": sys.argv, "rows": []}

    print("\n--- cone share of the objective, vs the lambda that is actually trained ---")
    print(f"{'Mach':>6} {'c0':>8} {'R_l2':>9} {'H1add(1e-2)':>13} {'x L2':>7} "
          f"{'H1mult':>9} {'x L2':>7} {'w_in_cone/mean':>16}")
    for m in args.mach:
        c0 = sound_speed(u_rms, m)
        l2 = effective_weight_in_cone(spec, c0, "l2", grid=grid, convention="additive")
        add = effective_weight_in_cone(spec, c0, "sobolev", lam=DEFAULT_SOBOLEV_LAMBDA,
                                       order=args.order, convention="additive", grid=grid)
        mult = effective_weight_in_cone(spec, c0, "h1", order=args.order,
                                        convention="multiplicative")
        rep = _cone_weight_report(spec, grid, c0, DEFAULT_SOBOLEV_LAMBDA, args.order)
        lo, hi = rep["sobolev_weight_in_cone_over_mean"]
        print(f"{m:>6.2g} {c0:>8.3f} {l2:>9.2e} {add:>13.2e} {add/l2:>7.3f} "
              f"{mult:>9.2e} {mult/l2:>7.3f}   {lo:.1e}..{hi:.1e}")

        results["rows"].append({
            "mach": float(m), "c0": c0,
            "l2": l2,
            "h1_add_default_lambda": add,
            "h1_add_ratio_to_l2": add / l2,
            "h1_multiplicative": mult,
            "h1_multiplicative_ratio_to_l2": mult / l2,
            "cone_weight_report": rep,
            "sweep": sobolev_lambda_sweep(spec, c0, grid, order=args.order,
                                          lambdas=LAMBDAS),
        })

    m0 = args.plot_mach
    c0_0 = sound_speed(u_rms, m0)
    sweep = results["rows"][args.mach.index(m0)]["sweep"]
    print(f"\nlambda sweep at M = {m0:g}  (R_l2 = {sweep[0]['cone_share']:.2e})")
    print(f"{'lambda':>12} {'cone share':>12} {'x L2':>10}")
    for r in sweep[2:]:
        print(f"{r['lambda']:>12.0e} {r['cone_share']:>12.2e} {r['ratio_to_l2']:>10.3f}")
    print(f"  H1 multiplicative (not trained): {sweep[1]['cone_share']:.2e}"
          f"  ({sweep[1]['ratio_to_l2']:.3f}x L2)")

    inert = [r for r in sweep[2:] if r["ratio_to_l2"] > 0.9]
    print(f"\nThe objective is indistinguishable from plain L2 on the cone for every "
          f"lambda up to {inert[-1]['lambda']:.0f}"
          f" ({inert[-1]['lambda']/DEFAULT_SOBOLEV_LAMBDA:.0f}x the default).")
    print("The cone is starved by where the weight's mass is, not by the exponent:")
    rep = results["rows"][args.mach.index(m0)]["cone_weight_report"]
    print(f"  the cone reaches k <= {rep['k_max_in_cone']:.2f}, i.e. "
          f"{100*rep['fraction_of_resolved_k_below_cone_edge']:.2f}% of the resolved "
          f"|k| values")
    print(f"  where (1+k^2)^order / <(1+k^2)^order> = "
          f"{rep['sobolev_weight_in_cone_over_mean'][0]:.1e}"
          f"..{rep['sobolev_weight_in_cone_over_mean'][1]:.1e}")
    print(f"  so |w_eff - 1| across the whole cone is at most "
          f"{rep['objective_weight_in_cone_minus_one'][0]:.1e}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "06_sobolev_argument.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'06_sobolev_argument.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

        ax = axes[0]
        lams = [r["lambda"] for r in sweep[2:]]
        shares = [r["cone_share"] for r in sweep[2:]]
        ax.plot(lams, shares, "o-", color="C0", label="trained objective, $1+\\lambda w$")
        ax.axhline(sweep[0]["cone_share"], color="k", ls="-", lw=1.2, label="L2 baseline")
        ax.axhline(sweep[1]["cone_share"], color="C3", ls="--", lw=1.2,
                   label="H$^1$ multiplicative (not trained)")
        ax.axvline(DEFAULT_SOBOLEV_LAMBDA, color="C2", ls=":", lw=1.5,
                   label=f"default $\\lambda$ = {DEFAULT_SOBOLEV_LAMBDA:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("share of objective in the cone")
        ax.set_title(f"Stage 4: the remedy is inert (M={m0:g})")
        ax.legend(fontsize=8, loc="lower left")

        ax = axes[1]
        for row, mk in zip(results["rows"], ["o", "s", "^", "D"]):
            lam_r = [r["lambda"] for r in row["sweep"][2:]]
            share_r = [r["cone_share"] for r in row["sweep"][2:]]
            ax.plot(lam_r, share_r, mk + "-", ms=4, label=f"M = {row['mach']:g}")
        ax.axvline(DEFAULT_SOBOLEV_LAMBDA, color="C2", ls=":", lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("share of objective in the cone")
        ax.set_title("across Mach numbers")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out / "06_sobolev_argument.png", dpi=150)
        print(f"wrote {out/'06_sobolev_argument.png'}")
    except Exception as e:  # plotting is optional; the numbers are the result
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
