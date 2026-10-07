#!/usr/bin/env python3
"""STAGE 3 -- the error triple: field vs source vs far field.

The far field is radiated in the T-form (u_i u_j against the exact d_i d_j G): the
S-form also integrates a spurious box-edge term in a periodic box (Sarkar & Hussaini
1993; review_2026-10-01.md section 1), and the compact term radiates below c0*dk.

Reports the contrast the project rests on. The field error is the number papers
publish; the source and far-field errors are what the flow actually sounds like.

    python scripts/03_compute_sources.py --data data/turb2d.npz --model runs/fno_mse/model.pt --out runs/fno_mse
    python scripts/03_compute_sources.py --smoke --model tmp/smoke02/model.pt --out tmp/smoke03
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal.windows import tukey
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.diagnostics.errors import error_triple, relative_l2
from src.diagnostics.divergence import divergence_error, dilatational_energy_fraction
from src.acoustics.greens import (
    source_spectrum,
    far_field_pressure_ibp,
    observer_ring,
    select_band,
)
from src.acoustics.komega import sound_speed


def far_field_error(ut, vt, up, vp, grid, dt, c0, n_obs, n_band, band_mode, min_fraction,
                    baseline=None):
    """T-form far-field error of prediction (up, vp) against truth (ut, vt).

    Returns (errors dict, band metadata).  Radiates the spatially tapered T_ij = u_i u_j
    against the exact d_i d_j G, so no spurious box-edge surface term (Sarkar & Hussaini
    1993), and the band includes omega < c0*dk where the compact (k = 0) term dominates.

    baseline : optional (up_b, vp_b) scored against the same truth far field, band and
               observers.  errs["per_frequency"] lists, per band frequency, the complex
               error of the prediction (err_model) and of the baseline (err_copy), and
               the truth far-field power.
    """
    # Tukey(0.5) spatial taper: a hard box edge leaks convective-ridge content into
    # small k (see src/acoustics/tform.py); same taper as R_T in scripts/07.
    taper = np.outer(tukey(grid.n, 0.5, sym=False), tukey(grid.n, 0.5, sym=False))

    def stress_spectra(u, v):
        T = {"uu": u * u, "uv": u * v, "vv": v * v}
        hats = {}
        for key in T:
            T[key] -= T[key].mean(axis=0)     # steady part does not radiate; Hann leakage
            T[key] *= taper
            hats[key], om = source_spectrum(T[key], dt)
            T[key] = None
        return hats, om

    hat_t, om = stress_spectra(ut, vt)
    hat_p, _ = stress_spectra(up, vp)
    hat_b = stress_spectra(*baseline)[0] if baseline is not None else None
    truth_power = ((np.abs(hat_t["uu"]) ** 2 + 2 * np.abs(hat_t["uv"]) ** 2
                    + np.abs(hat_t["vv"]) ** 2).sum(axis=(1, 2)))

    band, meta = select_band(om, c0, grid, n_band, mode=band_mode, power=truth_power,
                             min_fraction=min_fraction)
    if not band:
        sys.exit(f"no radiating frequencies available: {meta.get('error')}")
    k_min = om[band[0]] / c0
    radius = max(30 * grid.length, 10 / k_min)
    meta["kr_range"] = [float(om[band[0]] / c0 * radius), float(om[band[-1]] / c0 * radius)]
    meta["radius"] = float(radius)
    if meta.get("fallback_reason"):
        print(f"[band] {meta['fallback_reason']}; using mode '{meta['mode']}'")
    print(f"[band] mode={meta['mode']}  {meta['n_selected']} frequencies, "
          f"omega in [{meta['omega_band'][0]:.3f}, {meta['omega_band'][1]:.2f}], "
          f"k in [{meta['k_band'][0]:.3f}, {meta['k_band'][1]:.2f}]")
    print(f"[band] c0*dk = {meta['omega_min_radiating']:.2f}; "
          f"{int((om[band] < meta['omega_min_radiating']).sum())} band frequencies below it")
    print(f"[band] kr in [{meta['kr_range'][0]:.0f}, {meta['kr_range'][1]:.0f}] "
          f"at r={radius:.0f}")
    if meta.get("source_power_fraction_captured") is not None:
        print(f"[band] captures {meta['source_power_fraction_captured']:.2%} of the "
              f"truth stress's omega>0 power")

    obs = observer_ring(radius, n_obs, (grid.length / 2, grid.length / 2))
    pt = far_field_pressure_ibp(hat_t, om, grid, obs, c0, band, far_field=False)
    del hat_t
    pp = far_field_pressure_ibp(hat_p, om, grid, obs, c0, band, far_field=False)
    pb = (far_field_pressure_ibp(hat_b, om, grid, obs, c0, band, far_field=False)
          if hat_b is not None else None)

    def freq_err(pred):
        return (np.linalg.norm((pred - pt).reshape(len(band), -1), axis=1)
                / np.linalg.norm(pt.reshape(len(band), -1), axis=1))

    def band_errs(pred):
        below = om[band] < meta["omega_min_radiating"]
        d = (np.abs(pred - pt) ** 2).reshape(len(band), -1).sum(axis=1)
        t = (np.abs(pt) ** 2).reshape(len(band), -1).sum(axis=1)
        agg = lambda m: float(np.sqrt(d[m].sum() / t[m].sum())) if m.any() else None
        out = {
            "far_field": relative_l2(np.abs(pred), np.abs(pt)),
            "far_field_complex": float(np.linalg.norm(pred - pt) / np.linalg.norm(pt)),
            "far_field_complex_below_c0dk": agg(below),
            "far_field_complex_above_c0dk": agg(~below),
        }
        assert np.isclose(agg(np.ones_like(below)), out["far_field_complex"], rtol=1e-9)
        return out

    errs = {
        **band_errs(pp),
        "far_field_primary": "complex",
        "far_field_form": "T (far_field_pressure_ibp, exact kernel, Tukey 0.5 spatial taper)",
        "far_field_radius": float(radius),
    }
    e_model = freq_err(pp)
    e_copy = freq_err(pb) if pb is not None else None
    t_pow = (np.abs(pt) ** 2).reshape(len(band), -1).sum(axis=1)
    errs["per_frequency"] = [
        {"omega": float(om[b]), "k": float(om[b] / c0),
         "below_c0dk": bool(om[b] < meta["omega_min_radiating"]),
         "truth_power": float(t_pow[i]), "err_model": float(e_model[i]),
         "err_copy": float(e_copy[i]) if e_copy is not None else None}
        for i, b in enumerate(band)]
    if pb is not None:
        errs["baseline_far_field"] = band_errs(pb)
    return errs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", default="runs/model.pt")
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--n-obs", type=int, default=36)
    ap.add_argument("--n-band", type=int, default=24,
                    help="number of radiating frequencies in the far-field band.")
    ap.add_argument("--band-mode", choices=("omega", "auto", "k", "top"), default="omega",
                    help="'omega' = log-spaced from the first bin to Nyquist, no radiation cut "
                         "(T-form: the compact term radiates at every omega); "
                         "'k' = the lowest radiating box modes; 'top' = the "
                         "strongest radiating frequencies of the truth source; "
                         "'auto' tries 'k' and falls back to 'top' if the band "
                         "captures under --min-cone-fraction of the radiating power.")
    ap.add_argument("--min-cone-fraction", type=float, default=0.01,
                    help="minimum share of the radiating source power the band must "
                         "capture before 'auto' accepts it.")
    ap.add_argument("--out", "--outdir", dest="out", default="figures",
                    help="folder for 03_errors.json (the run folder of the model)")
    ap.add_argument("--smoke", action="store_true",
                    help="first 64 snapshots and a 6-frequency band: a quick CPU check")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the copy-input baseline (prediction of step t+1 = snapshot t)")
    args = ap.parse_args()
    t0 = time.time()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required: pip install torch")

    from src.models.fno import FNO2d
    from src.models.unet import UNet2d
    from src.models.train import TrainConfig, make_pairs, predict_onestep, denormalise

    u, v, meta = load_dataset(args.data)
    if args.smoke:
        u, v, args.n_band = u[:64], v[:64], 6
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = FNO2d(modes=min(24, grid.n//4), width=48) if ckpt["arch"] == "fno" \
        else UNet2d(base=48, depth=3)
    model.load_state_dict(ckpt["state_dict"])
    stats = ckpt["stats"]

    x, y, _ = make_pairs(u, v)
    cfg = TrainConfig()
    pred = predict_onestep(model, x, cfg)
    pred = denormalise(pred, stats)
    up, vp = pred[:, 0], pred[:, 1]
    ut, vt = u[1:], v[1:]

    base = not args.no_baseline
    up_b, vp_b = u[:-1], v[:-1]
    trip = error_triple(up, vp, ut, vt, grid)
    trip_b = error_triple(up_b, vp_b, ut, vt, grid) if base else None

    u_rms = float(np.sqrt(np.mean(ut**2 + vt**2)))
    c0 = sound_speed(u_rms, args.mach)
    errs, meta = far_field_error(ut, vt, up, vp, grid, dt, c0, args.n_obs, args.n_band,
                                 args.band_mode, args.min_cone_fraction,
                                 baseline=(up_b, vp_b) if base else None)
    ffb = errs.pop("baseline_far_field", None)
    rows = errs["per_frequency"]

    def fmt(x):
        return f"{x:>12.2%}" if x is not None else f"{'n/a':>12}"

    def ratio(b, m):
        return f"{b / m:>11.1f}x" if b is not None and m else f"{'':>12}"

    def row(name, m, b):
        print(f"{name:<50}{fmt(m)}{fmt(b) if base else ''}"
              f"{ratio(b, m) if base else ''}")

    print("=" * 86)
    print(f"{'metric':<50}{'model':>12}" + (f"{'copy-input':>12}{'base/model':>12}" if base else ""))
    print("=" * 86)
    row("relative field error", trip["field"], trip_b and trip_b["field"])
    row("relative Lighthill source error", trip["source"], trip_b and trip_b["source"])
    row("far-field complex error (T-form), PRIMARY", errs["far_field_complex"],
        ffb and ffb["far_field_complex"])
    row("  complex, omega < c0*dk", errs["far_field_complex_below_c0dk"],
        ffb and ffb["far_field_complex_below_c0dk"])
    row("  complex, omega >= c0*dk", errs["far_field_complex_above_c0dk"],
        ffb and ffb["far_field_complex_above_c0dk"])
    row("far-field |p| error (magnitude only; blind to timing)", errs["far_field"],
        ffb and ffb["far_field"])
    print(f"{'spurious dilatational energy frac':<50}"
          f"{np.mean([dilatational_energy_fraction(up[i],vp[i],grid) for i in range(0,len(up),8)]):>10.2e}")
    print(f"{'div(u) error, truth':<50}"
          f"{np.mean([divergence_error(ut[i],vt[i],grid) for i in range(0,len(ut),8)]):>10.2e}")
    print(f"{'div(u) error, prediction':<50}"
          f"{np.mean([divergence_error(up[i],vp[i],grid) for i in range(0,len(up),8)]):>10.2e}")
    print("=" * 86)

    print("\nper-frequency complex far-field error")
    print(f"{'omega':>10}{'k':>10}{'<c0dk':>7}{'truth pow':>12}{'model':>10}"
          + (f"{'copy-input':>12}" if base else ""))
    for r in rows if len(rows) <= 12 else rows[::2]:
        print(f"{r['omega']:>10.4g}{r['k']:>10.4g}{'yes' if r['below_c0dk'] else 'no':>7}"
              f"{r['truth_power']:>12.3e}{r['err_model']:>10.2%}"
              + (f"{r['err_copy']:>12.2%}" if base else ""))
    print("\nThe field error is what gets published. The far-field complex error is what the flow sounds like.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"03_errors.json").write_text(json.dumps(
        {**trip, **errs, "baseline_copy_input": ({**trip_b, **ffb} if base else None),
         "mach": args.mach, "c0": c0, "far_field_band": meta,
         "model": str(args.model), "data": args.data, "smoke": args.smoke},
        indent=2))

    def pct(x):
        return f"{x:.2%}" if x is not None else "n/a"

    def line(name, t, f):
        print(f"  {name:<11} field {t['field']:.2%}  source {t['source']:.2%}  far-field complex "
              f"{pct(f['far_field_complex'])} (below {pct(f['far_field_complex_below_c0dk'])}, "
              f"above {pct(f['far_field_complex_above_c0dk'])})")

    print("\n" + "=" * 60)
    print("SUMMARY  scripts/03_compute_sources.py  (teacher-forced one-step)")
    print(f"  model        {args.model}  ({ckpt['arch'].upper()}, loss {ckpt.get('loss', '?')})")
    print(f"  data         {args.data}  ({len(up)} predicted steps{', SMOKE' if args.smoke else ''})")
    line("model", trip, errs)
    if base:
        line("copy-input", trip_b, ffb)
    print(f"  far-field |p| (magnitude only; blind to timing): model {pct(errs['far_field'])}"
          + (f", copy-input {pct(ffb['far_field'])}" if base else ""))
    print(f"  far field    T-form, M={args.mach}, {meta['n_selected']} freqs, "
          f"omega {meta['omega_band'][0]:.3g}-{meta['omega_band'][1]:.3g}, c0*dk {meta['omega_min_radiating']:.3g}")
    print(f"  device       {cfg.device}   wall time {(time.time() - t0) / 60:.1f} min")
    print(f"  results      {out / '03_errors.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
