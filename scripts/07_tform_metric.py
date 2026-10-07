#!/usr/bin/env python3
"""R_T -- the cone share of the Lighthill source built from T_ij, next to R_S.

R_S (scripts/04) is the cone share of S = d_i d_j T_ij on the box's own wavenumber
grid.  R_T evaluates S~ = -k_i k_j T~_ij of the tapered, box-truncated stress on a grid
`--pad` times finer, so it includes the sub-dk (compact) content a periodic box cannot
hold.  See `src/acoustics/tform.py`.  Same Welch segmentation as scripts/04; the band is
the envelope over {full record, first half, second half}.

    python scripts/07_tform_metric.py --data data/turb2d.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.acoustics.komega import welch_komega_spectrum, komega_spectrum, sound_speed
from src.acoustics.lighthill import source_timeseries
from src.acoustics.tform import cone_share_refined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--machs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--taper", type=float, default=0.5, help="Tukey alpha of the spatial taper")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    u, v, meta = load_dataset(args.data)
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]
    u_rms = float(np.sqrt(np.mean(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)))
    c0s = [sound_speed(u_rms, m) for m in args.machs]
    nt = u.shape[0]
    parts = {"full": (slice(None), args.segments),
             "half1": (slice(0, nt // 2), max(1, args.segments // 2)),
             "half2": (slice(nt // 2, None), max(1, args.segments // 2))}

    S = source_timeseries(u, v, grid)
    r_s = {}
    for part, (sl, seg) in parts.items():
        spec = (welch_komega_spectrum(S[sl], grid, dt, n_segments=seg) if seg > 1
                else komega_spectrum(S[sl], grid, dt))
        r_s[part] = [spec.radiating_fraction(c) for c in c0s]
    del S

    r_t = {}
    for part, (sl, seg) in parts.items():
        r_t[part] = cone_share_refined(u[sl], v[sl], grid, dt, c0s, pad=args.pad,
                                       n_segments=seg, taper_alpha=args.taper)
        print(f"[{part}] R_T done", flush=True)

    rows = []
    print(f"{'Mach':>5} {'R_S':>9} {'R_T':>9} {'R_T band':>21} {'F':>5} {'sub-dk':>9} {'R_T/R_S':>8}")
    for j, m in enumerate(args.machs):
        vals = [r_t[p]["share"][j] for p in parts]
        lo, hi = min(vals), max(vals)
        row = {"mach": m, "c0": c0s[j],
               "R_S": r_s["full"][j], "R_S_band": [min(r_s[p][j] for p in parts), max(r_s[p][j] for p in parts)],
               "R_T": vals[0], "R_T_half1": vals[1], "R_T_half2": vals[2],
               "R_T_band": [lo, hi], "R_T_spread": hi / lo if lo > 0 else float("inf"),
               "R_T_sub_dk": r_t["full"]["share_sub_dk"][j]}
        rows.append(row)
        print(f"{m:>5.2g} {row['R_S']:>9.2e} {row['R_T']:>9.2e} [{lo:.2e}, {hi:.2e}] "
              f"{row['R_T_spread']:>5.2f} {row['R_T_sub_dk']:>9.2e} {row['R_T']/row['R_S']:>8.2f}")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem
    name = "tform_metrics.json" if stem == "turb2d" else f"tform_metrics_{stem}.json"
    (out / name).write_text(json.dumps({
        "u_rms": u_rms, "rows": rows,
        "settings": {"data": args.data, "segments": args.segments, "pad": args.pad,
                     "taper_alpha": args.taper, "window": "hann", "argv": sys.argv}}, indent=2))
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
