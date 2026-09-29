#!/usr/bin/env python3
"""STAGE 3 -- the error triple: field vs source vs far field.

Reports the contrast the project rests on. The field error is the number papers
publish; the source and far-field errors are what the flow actually sounds like.

    python scripts/03_compute_sources.py --data data/turb2d.npz --model runs/model.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.diagnostics.errors import error_triple, relative_l2
from src.diagnostics.divergence import divergence_error, dilatational_energy_fraction
from src.acoustics.lighthill import source_timeseries
from src.acoustics.greens import source_spectrum, far_field_pressure, observer_ring
from src.acoustics.komega import sound_speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", default="runs/model.pt")
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--n-obs", type=int, default=36)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required: pip install torch")

    from src.models.fno import FNO2d
    from src.models.unet import UNet2d
    from src.models.train import TrainConfig, make_pairs, predict_onestep, denormalise

    u, v, meta = load_dataset(args.data)
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = FNO2d(modes=min(24, grid.n//4), width=48) if ckpt["arch"] == "fno" \
        else UNet2d(base=48, depth=3)
    model.load_state_dict(ckpt["state_dict"])
    stats = ckpt["stats"]

    x, y, _ = make_pairs(u, v)
    pred = predict_onestep(model, x, TrainConfig())
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

    # far field
    u_rms = float(np.sqrt(np.mean(ut**2 + vt**2)))
    c0 = sound_speed(u_rms, args.mach)
    St = source_timeseries(ut, vt, grid)
    Sp = source_timeseries(up, vp, grid)
    sht, om = source_spectrum(St, dt)
    shp, _ = source_spectrum(Sp, dt)
    band = [i for i in range(1, len(om)) if om[i] > 0][:24]
    obs = observer_ring(30*grid.length, args.n_obs, (grid.length/2, grid.length/2))
    pt = far_field_pressure(sht, om, grid, obs, c0, True, band)
    pp = far_field_pressure(shp, om, grid, obs, c0, True, band)
    ff = relative_l2(np.abs(pp), np.abs(pt))
    print(f"{'far-field pressure error':<34}{ff:>13.2%}")
    print("=" * 62)
    print("\nThe first row is what gets published. The rest is what the flow sounds like.")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out/"03_errors.json").write_text(json.dumps(
        {**trip, "far_field": ff, "mach": args.mach, "c0": c0}, indent=2))


if __name__ == "__main__":
    main()
