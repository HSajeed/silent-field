#!/usr/bin/env python3
"""Mach scaling of the radiated power, against the compact 2D quadrupole law.

The flow is incompressible, so Mach is an analysis knob: c0 = U / M at fixed U.  The
2D compact-quadrupole law is P ~ rho U^7 l / c0^4 per unit span -- "M^7" when U varies
at fixed c0 (Schoder et al., arXiv:2608.19236, citing Ffowcs Williams 1969), which at
fixed U and varying c0 is P ~ M^4.  `tform.radiated_power` computes the T-form far-field
power and its compact limit (stress replaced by its tapered box integral), which scales
as c0^-4 exactly; the ratio P / P_compact measures non-compactness.

The cone shares R (velocity) and R_T are printed beside it, because the old Mach
exponents (10.7 / 7.1 / 5.7) were exponents of R -- an energy fraction, not a power.

    python scripts/08_mach_scaling.py --data data/turb2d.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.acoustics.komega import velocity_komega_spectrum, sound_speed
from src.acoustics.tform import radiated_power, cone_share_refined


def local_exponent(m, y):
    return np.gradient(np.log(y), np.log(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--n-mach", type=int, default=13)
    ap.add_argument("--mach-range", type=float, nargs=2, default=[0.02, 0.4])
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    u, v, meta = load_dataset(args.data)
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]
    U = float(np.sqrt(np.mean(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)))
    machs = np.geomspace(*args.mach_range, args.n_mach)
    c0s = [sound_speed(U, m) for m in machs]
    nt = u.shape[0]
    parts = {"full": (slice(None), args.segments),
             "half1": (slice(0, nt // 2), max(1, args.segments // 2)),
             "half2": (slice(nt // 2, None), max(1, args.segments // 2))}

    power = {}
    for part, (sl, seg) in parts.items():
        power[part] = radiated_power(u[sl], v[sl], grid, dt, c0s, n_segments=seg)
        print(f"[{part}] radiated power done", flush=True)
    r_t = cone_share_refined(u, v, grid, dt, c0s, n_segments=args.segments)["share"]
    spec = velocity_komega_spectrum(u, v, grid, dt, n_segments=args.segments)
    r_u = [spec.radiating_fraction(c) for c in c0s]

    P = np.array(power["full"]["power"])
    Pc = np.array(power["full"]["compact"])
    slopes = {p: local_exponent(machs, np.array(power[p]["power"])).tolist() for p in parts}
    rows = []
    print(f"{'Mach':>6} {'P/P(0.1)':>10} {'P/Pcomp':>8} {'dlnP/dlnM':>10} {'[halves]':>15} "
          f"{'dlnR/dlnM':>10} {'dlnR_T/dlnM':>11}")
    p01 = np.interp(np.log(0.1), np.log(machs), P)
    sl_r, sl_rt = local_exponent(machs, np.array(r_u)), local_exponent(machs, np.array(r_t))
    for j, m in enumerate(machs):
        h = [slopes["half1"][j], slopes["half2"][j]]
        rows.append({"mach": float(m), "c0": c0s[j], "power": float(P[j]),
                     "power_half1": power["half1"]["power"][j], "power_half2": power["half2"]["power"][j],
                     "compact": float(Pc[j]), "noncompact_ratio": float(P[j] / Pc[j]),
                     "slope": slopes["full"][j], "slope_half1": h[0], "slope_half2": h[1],
                     "R_u": r_u[j], "R_T": r_t[j],
                     "slope_R_u": float(sl_r[j]), "slope_R_T": float(sl_rt[j])})
        print(f"{m:>6.3f} {P[j]/p01:>10.3e} {P[j]/Pc[j]:>8.3f} {slopes['full'][j]:>10.2f} "
              f"[{min(h):>5.2f}, {max(h):>5.2f}] {sl_r[j]:>10.2f} {sl_rt[j]:>11.2f}")
    print("\nCompact 2D law at fixed U: slope 4 (= M^7 at fixed c0). P/Pcomp -> 1 when compact.")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem
    suffix = "" if stem == "turb2d" else f"_{stem}"
    (out / f"08_mach_scaling{suffix}.json").write_text(json.dumps(
        {"u_rms": U, "rows": rows,
         "settings": {"data": args.data, "segments": args.segments, "taper_alpha": 0.5,
                      "n_theta": 64, "argv": sys.argv}}, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].loglog(machs, P / p01, "o-", label="T-form radiated power $P$")
        ax[0].loglog(machs, Pc / p01, "--", label="compact limit ($\\propto M^4$)")
        ax[0].loglog(machs, np.array(r_u) / np.interp(np.log(0.1), np.log(machs), r_u), "s:",
                     label="cone share $R$ (velocity)")
        ax[0].set_xlabel("$M = U/c_0$ (fixed flow)"); ax[0].set_ylabel("normalised to $M=0.1$")
        ax[0].legend(fontsize=8)
        ax[1].semilogx(machs, slopes["full"], "o-", label="$d\\ln P/d\\ln M$")
        ax[1].fill_between(machs, np.minimum(slopes["half1"], slopes["half2"]),
                           np.maximum(slopes["half1"], slopes["half2"]), alpha=0.25, label="half records")
        ax[1].semilogx(machs, sl_r, "s:", label="$d\\ln R/d\\ln M$")
        ax[1].axhline(4, color="k", lw=1, ls="--", label="compact 2D (4 at fixed $U$ = $M^7$)")
        ax[1].set_xlabel("$M$"); ax[1].set_ylabel("local exponent"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"08_mach_scaling{suffix}.png", dpi=140)
        print(f"wrote {out / f'08_mach_scaling{suffix}.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
