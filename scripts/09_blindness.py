#!/usr/bin/env python3
"""STAGE 3 -- the blindness ratios B and B_S, with the controls review_2026-10-01.md §4.2 asks for.

    B   = R(error) / R(truth)      cone share of the velocity error vs of the signal
    B_S = R_T(error) / R_T(truth)  the same for the Lighthill stress (T-form, sub-dk included)

B = 1: the error is distributed over (k, omega) like the signal.  B >> 1: the error is
over-represented in the radiating cone.  "B >= 1" is not a theorem, and a large B can
come from step-to-step jitter rather than from the objective, so every number is shown
beside controls:

    model_tf        the model, teacher-forced one-step (what scripts/03 evaluates)
    model_rollout   the model, free rollout from the first true snapshot
    copy_input      predict t+1 = t (no model)
    white_matched   truth + solenoidal white noise, per-snapshot norm = the model's error
    shuffled_error  truth + the model's own error snapshots in random time order
                    (same spatial error, temporal coherence destroyed: the jitter test)

Also reported per variant: R, R_T and the T-form radiated power of the *prediction*
relative to the truth -- phase-free, so meaningful for a rollout that has decorrelated.

    python scripts/09_blindness.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01
    python scripts/09_blindness.py --smoke --model tmp/smoke02/model.pt --out tmp/smoke09
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid, project_solenoidal
from src.solvers.io import load_dataset
from src.diagnostics.errors import error_triple
from src.acoustics.komega import velocity_komega_spectrum, sound_speed
from src.acoustics.tform import cone_share_refined, radiated_power


def cone_share(u, v, grid, dt, c0, seg):
    return velocity_komega_spectrum(u, v, grid, dt, n_segments=seg).radiating_fraction(c0)


def white_like(err_u, err_v, grid, rng):
    """Solenoidal white noise with the per-snapshot L2 norm of the given error."""
    nu, nv = np.empty_like(err_u), np.empty_like(err_v)
    for t in range(err_u.shape[0]):
        a, b = project_solenoidal(rng.standard_normal(grid.n * grid.n).reshape(grid.n, grid.n),
                                  rng.standard_normal(grid.n * grid.n).reshape(grid.n, grid.n), grid)
        scale = np.sqrt((err_u[t] ** 2 + err_v[t] ** 2).sum() / (a ** 2 + b ** 2).sum())
        nu[t], nv[t] = a * scale, b * scale
    return nu, nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="first 64 snapshots, pad 2, 2 segments")
    args = ap.parse_args()
    t0 = time.time()

    import torch
    from src.models.train import (TrainConfig, make_pairs, predict_onestep, predict_rollout,
                                  denormalise, load_model)

    u, v, meta = load_dataset(args.data)
    if args.smoke:
        u, v, args.pad, args.segments = u[:64], v[:64], 2, 2
    u, v = u.astype(np.float64), v.astype(np.float64)
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]
    c0 = sound_speed(float(np.sqrt(np.mean(u[1:] ** 2 + v[1:] ** 2))), args.mach)
    rng = np.random.default_rng(args.seed)

    model, ckpt = load_model(args.model, grid.n)
    stats, cfg = ckpt["stats"], TrainConfig()
    x, _, _ = make_pairs(u.astype(np.float32), v.astype(np.float32), normalise=False)
    x = (x - stats["mean"]) / stats["std"]          # the model's training statistics
    tf = denormalise(predict_onestep(model, x, cfg), stats).astype(np.float64)
    x0 = torch.tensor(((np.stack([u[0], v[0]]) - stats["mean"]) / stats["std"]).astype(np.float32))
    ro = denormalise(predict_rollout(model, x0, len(u) - 1, cfg), stats).astype(np.float64)
    del x
    print(f"teacher-forced {len(tf)} steps; rollout {len(ro)} of {len(u) - 1} steps finite")

    ut_all, vt_all = u[1:], v[1:]
    e_u, e_v = tf[:, 0] - ut_all, tf[:, 1] - vt_all
    perm = rng.permutation(len(e_u))
    nu, nv = white_like(e_u, e_v, grid, rng)
    variants = {
        "model_tf": (tf[:, 0], tf[:, 1], len(tf)),
        "model_rollout": (ro[:, 0], ro[:, 1], len(ro)),
        "copy_input": (u[:-1], v[:-1], len(u) - 1),
        "white_matched": (ut_all + nu, vt_all + nv, len(ut_all)),
        "shuffled_error": (ut_all + e_u[perm], vt_all + e_v[perm], len(ut_all)),
    }

    truth_cache = {}

    def truth_stats(n):
        if n not in truth_cache:
            ut, vt = ut_all[:n], vt_all[:n]
            seg = args.segments if n >= 64 * args.segments else 1
            truth_cache[n] = {
                "seg": seg,
                "R": cone_share(ut, vt, grid, dt, c0, seg),
                "R_T": cone_share_refined(ut, vt, grid, dt, [c0], pad=args.pad, n_segments=seg)["share"][0],
                "P": radiated_power(ut, vt, grid, dt, [c0], n_segments=seg)["power"][0],
            }
        return truth_cache[n]

    rows = {}
    for name, (up, vp, n) in variants.items():
        if n < 16:
            rows[name] = {"n_steps": int(n), "skipped": "too few finite steps"}
            print(f"[{name}] skipped: {n} finite steps")
            continue
        ts = truth_stats(n)
        ut, vt, seg = ut_all[:n], vt_all[:n], ts["seg"]
        up, vp = up[:n], vp[:n]
        trip = error_triple(up, vp, ut, vt, grid)
        R_err = cone_share(up - ut, vp - vt, grid, dt, c0, seg)
        RT_err = cone_share_refined(up, vp, grid, dt, [c0], pad=args.pad, n_segments=seg,
                                    minus=(ut, vt))["share"][0]
        R_pred = cone_share(up, vp, grid, dt, c0, seg)
        RT_pred = cone_share_refined(up, vp, grid, dt, [c0], pad=args.pad, n_segments=seg)["share"][0]
        P_pred = radiated_power(up, vp, grid, dt, [c0], n_segments=seg)["power"][0]
        rows[name] = {
            "n_steps": int(n), "segments": seg,
            "field_err": trip["field"], "source_err": trip["source"],
            "R_err": R_err, "R_truth": ts["R"], "B": R_err / ts["R"],
            "R_T_err": RT_err, "R_T_truth": ts["R_T"], "B_S": RT_err / ts["R_T"],
            "R_pred_over_truth": R_pred / ts["R"], "R_T_pred_over_truth": RT_pred / ts["R_T"],
            "P_pred_over_truth": P_pred / ts["P"],
        }
        print(f"[{name}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "09_blindness.json").write_text(json.dumps(
        {"mach": args.mach, "c0": c0, "pad": args.pad, "segments": args.segments,
         "model": str(args.model), "arch": ckpt["arch"], "loss": ckpt.get("loss"), "unroll": ckpt.get("unroll", 1),
         "smoke": args.smoke, "seed": args.seed, "variants": rows}, indent=2))

    print("\n" + "=" * 100)
    print(f"SUMMARY  scripts/09_blindness.py  (M={args.mach}, {ckpt['arch'].upper()} {ckpt.get('loss', '')} unroll {ckpt.get('unroll', 1)}"
          f"{', SMOKE' if args.smoke else ''})")
    print(f"  {'variant':<15}{'steps':>6}{'field':>8}{'source':>9}{'B':>11}{'B_S':>11}"
          f"{'R/R0':>11}{'R_T/R_T0':>11}{'P/P0':>11}")
    for name, r in rows.items():
        if "skipped" in r:
            print(f"  {name:<15}{r['n_steps']:>6}  skipped"); continue
        print(f"  {name:<15}{r['n_steps']:>6}{r['field_err']:>8.1%}{r['source_err']:>9.1%}"
              f"{r['B']:>11.3g}{r['B_S']:>11.3g}{r['R_pred_over_truth']:>11.3g}"
              f"{r['R_T_pred_over_truth']:>11.3g}{r['P_pred_over_truth']:>11.3g}")
    print("  B, B_S: cone share of the error / of the truth.  X/X0: prediction / truth (phase-free).")
    print(f"  wall time {(time.time() - t0) / 60:.1f} min   results {out / '09_blindness.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
