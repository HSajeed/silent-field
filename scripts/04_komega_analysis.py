#!/usr/bin/env python3
"""STAGE 2 -- THE HEADLINE FIGURE. Requires no trained model.

Produces the (|k|, omega) energy spectrum with the sonic line and convective
ridge overlaid, and the number the whole project exists to report: the fraction
of an L2 objective that the radiating cone receives.

Also sweeps Mach number (free -- c0 is an analysis knob, not a simulation
parameter) and compares the L2 weighting against Sobolev/H1, which is the
Stage-4 argument in numerical form.

Every headline number is reported as a *band*, not a point estimate: alongside
the full-record value we compute the same quantity on the first and second
halves of the record (with half as many Welch segments) and report the
envelope [min, max] over all three plus the spread factor max/min. That band
is only a check against finite-record / windowing sensitivity within this one
realization -- it does not capture realization-to-realization or domain-size
uncertainty.

    python scripts/04_komega_analysis.py --data data/turb2d.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.acoustics.komega import welch_komega_spectrum, komega_spectrum, sound_speed
from src.acoustics.radiating import mach_sweep, effective_weight_in_cone
from src.acoustics.lighthill import source_timeseries


def _spectrum(field, grid, dt, n_segments):
    return (welch_komega_spectrum(field, grid, dt, n_segments=n_segments)
            if n_segments > 1 else komega_spectrum(field, grid, dt))


def _full_and_half_spectra(field, grid, dt, segments):
    """Full-record spectrum plus first- and second-half estimates.

    Halves get half as many Welch segments (rounded down, minimum 1), since
    they cover half the record length.
    """
    nt = field.shape[0]
    half = nt // 2
    half_segments = max(1, segments // 2)
    return {
        "full": _spectrum(field, grid, dt, segments),
        "half1": _spectrum(field[:half], grid, dt, half_segments),
        "half2": _spectrum(field[half:], grid, dt, half_segments),
    }


def _band(values):
    lo, hi = min(values), max(values)
    spread = (hi / lo) if lo > 0 else float("inf")
    return lo, hi, spread


def _fmt_c0(x):
    return f"{x:.3f}"


def _fmt_sci(x):
    return f"{x:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--machs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--plot-mach", type=float, default=0.1,
                     help="Mach number whose sonic line/cone is drawn on the figure.")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    u, v, meta = load_dataset(args.data)
    dt = meta["dt"]
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
    print(f"record {u.shape}, dt={dt:.5f}, u_rms={u_rms:.4f}")

    # --- velocity-field (k, omega) spectra: full record + both halves
    field = np.stack([u, v], axis=-1)
    vel_specs = _full_and_half_spectra(field, grid, dt, args.segments)
    spec = vel_specs["full"]
    print(f"spectrum: {spec.power.shape} (omega x k)")
    print(f"  omega up to {spec.omega.max():.3f}, k up to {spec.k_bins.max():.3f}")

    # --- Lighthill source (divergence form), computed frame by frame, same treatment
    src = source_timeseries(u, v, grid, form="divergence")
    src_specs = _full_and_half_spectra(src, grid, dt, args.segments)

    print("\n--- THE NUMBER: share of the objective inside the radiating cone ---")
    rows = []
    for m in args.machs:
        c0 = sound_speed(u_rms, m)

        l2_full = effective_weight_in_cone(vel_specs["full"], c0, "l2")
        l2_half1 = effective_weight_in_cone(vel_specs["half1"], c0, "l2")
        l2_half2 = effective_weight_in_cone(vel_specs["half2"], c0, "l2")
        l2_lo, l2_hi, l2_spread = _band([l2_full, l2_half1, l2_half2])
        h1 = effective_weight_in_cone(vel_specs["full"], c0, "h1")

        src_full = effective_weight_in_cone(src_specs["full"], c0, "l2")
        src_half1 = effective_weight_in_cone(src_specs["half1"], c0, "l2")
        src_half2 = effective_weight_in_cone(src_specs["half2"], c0, "l2")
        src_lo, src_hi, src_spread = _band([src_full, src_half1, src_half2])

        rows.append({
            "mach": float(m), "c0": c0,
            "l2": l2_full, "h1": h1,
            "l2_half1": l2_half1, "l2_half2": l2_half2,
            "l2_band": [l2_lo, l2_hi], "l2_spread": l2_spread,
            "src_l2": src_full,
            "src_l2_half1": src_half1, "src_l2_half2": src_half2,
            "src_l2_band": [src_lo, src_hi], "src_l2_spread": src_spread,
        })

    header = (f"{'Mach':>6} {'c0':>8} {'R_u':>9} {'R_u band':>19} {'F_u':>6} "
              f"{'R_S':>9} {'R_S band':>19} {'F_S':>6} {'H1':>9}")
    print(header)
    for r in rows:
        lo_u, hi_u = r["l2_band"]
        lo_s, hi_s = r["src_l2_band"]
        band_u = f"[{_fmt_sci(lo_u)}, {_fmt_sci(hi_u)}]"
        band_s = f"[{_fmt_sci(lo_s)}, {_fmt_sci(hi_s)}]"
        print(f"{r['mach']:>6.2g} {_fmt_c0(r['c0']):>8} {_fmt_sci(r['l2']):>9} "
              f"{band_u:>19} {r['l2_spread']:>6.1f} {_fmt_sci(r['src_l2']):>9} "
              f"{band_s:>19} {r['src_l2_spread']:>6.1f} {_fmt_sci(r['h1']):>9}")
    print("\nBand = envelope of {full record, first half, second half} estimates "
          "(halves use half as many Welch segments). It bounds finite-record and "
          "windowing sensitivity only -- it does NOT include realization-to-"
          "realization or domain-size uncertainty.")
    print("\nH1 (Sobolev) is the standard remedy for spectral bias. Note it does")
    print("not rescue the cone: it up-weights HIGH k, and the cone is at LOW k.")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.data).with_name(Path(args.data).stem + ".manifest.json")
    manifest_hash = None
    if manifest_path.exists():
        try:
            manifest_hash = json.loads(manifest_path.read_text()).get("hash")
        except Exception:
            manifest_hash = None

    settings = {
        "segments": args.segments,
        "window": "hann",
        "data": str(args.data),
        "dataset_manifest": str(manifest_path) if manifest_path.exists() else None,
        "dataset_manifest_hash": manifest_hash,
        "record_shape": list(field.shape),
        "dt": dt,
        "argv": sys.argv,
    }
    (out / "komega_metrics.json").write_text(json.dumps(
        {"u_rms": u_rms, "rows": rows, "settings": settings}, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        m = args.plot_mach
        c0 = sound_speed(u_rms, m)
        P = np.maximum(spec.power, spec.power.max()*1e-12)
        fig, ax = plt.subplots(figsize=(7,5.5))
        pc = ax.pcolormesh(spec.k_bins, spec.omega, P, norm=LogNorm(), shading="auto")
        ax.plot(spec.k_bins, spec.sonic_line(c0), "r-", lw=2,
                label=f"sonic line $\\omega=c_0|k|$ (M={m})")
        ax.plot(spec.k_bins, spec.convective_ridge(u_rms), "w--", lw=2,
                label="convective ridge $\\omega=U|k|$")
        ax.set_xlim(spec.k_bins[1], spec.k_bins.max())
        ax.set_ylim(spec.omega[1], spec.omega.max())
        ax.set_xlabel("$|k|$"); ax.set_ylabel("$\\omega$")
        R = spec.radiating_fraction(c0)
        ax.set_title(f"Energy in the radiating cone (M={m}): {R:.2e} of the total")
        ax.legend(loc="upper left"); fig.colorbar(pc, ax=ax, label="E(|k|,$\\omega$)")
        fig.tight_layout(); fig.savefig(out/"04_komega.png", dpi=150)
        print(f"\nwrote {out/'04_komega.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
