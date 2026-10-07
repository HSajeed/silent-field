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


def far_field_error(ut, vt, up, vp, grid, dt, c0, n_obs, n_band, band_mode, min_fraction):
    """T-form far-field error of prediction (up, vp) against truth (ut, vt).

    Returns (errors dict, band metadata).  Radiates the spatially tapered T_ij = u_i u_j
    against the exact d_i d_j G, so no spurious box-edge surface term (Sarkar & Hussaini
    1993), and the band includes omega < c0*dk where the compact (k = 0) term dominates.
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
    errs = {
        "far_field": relative_l2(np.abs(pp), np.abs(pt)),
        "far_field_complex": float(np.linalg.norm(pp - pt) / np.linalg.norm(pt)),
        "far_field_form": "T (far_field_pressure_ibp, exact kernel, Tukey 0.5 spatial taper)",
        "far_field_radius": float(radius),
    }
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

    print("=" * 62)
    print(f"{'metric':<34}{'value':>14}")
    print("=" * 62)
    trip = error_triple(up, vp, ut, vt, grid)
    print(f"{'relative field error':<34}{trip['field']:>13.2%}")
    print(f"{'relative Lighthill source error':<34}{trip['source']:>13.2%}")

    print(f"{'spurious dilatational energy frac':<34}"
          f"{np.mean([dilatational_energy_fraction(up[i],vp[i],grid) for i in range(0,len(up),8)]):>13.2e}")
    print(f"{'div(u) error, truth':<34}"
          f"{np.mean([divergence_error(ut[i],vt[i],grid) for i in range(0,len(ut),8)]):>13.2e}")
    print(f"{'div(u) error, prediction':<34}"
          f"{np.mean([divergence_error(up[i],vp[i],grid) for i in range(0,len(up),8)]):>13.2e}")

    u_rms = float(np.sqrt(np.mean(ut**2 + vt**2)))
    c0 = sound_speed(u_rms, args.mach)
    errs, meta = far_field_error(ut, vt, up, vp, grid, dt, c0, args.n_obs, args.n_band,
                                 args.band_mode, args.min_cone_fraction)
    print(f"{'far-field |p| error (T-form)':<34}{errs['far_field']:>13.2%}")
    print(f"{'far-field complex error (T-form)':<34}{errs['far_field_complex']:>13.2%}")
    print("=" * 62)
    print("\nThe first row is what gets published. The rest is what the flow sounds like.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"03_errors.json").write_text(json.dumps(
        {**trip, **errs, "mach": args.mach, "c0": c0, "far_field_band": meta,
         "model": str(args.model), "data": args.data, "smoke": args.smoke},
        indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY  scripts/03_compute_sources.py  (teacher-forced one-step)")
    print(f"  model        {args.model}  ({ckpt['arch'].upper()}, loss {ckpt.get('loss', '?')})")
    print(f"  data         {args.data}  ({len(up)} predicted steps{', SMOKE' if args.smoke else ''})")
    print(f"  field error  {trip['field']:.2%}")
    print(f"  source error {trip['source']:.2%}")
    print(f"  far field    |p| {errs['far_field']:.2%}   complex {errs['far_field_complex']:.2%}"
          f"   (T-form, M={args.mach}, {meta['n_selected']} freqs, "
          f"omega {meta['omega_band'][0]:.3g}-{meta['omega_band'][1]:.3g})")
    print(f"  device       {cfg.device}   wall time {(time.time() - t0) / 60:.1f} min")
    print(f"  results      {out / '03_errors.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
