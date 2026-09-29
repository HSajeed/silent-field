#!/usr/bin/env python3
"""STAGE 2 -- generate the forced 2D turbulence record.

Sampling is the thing to get right. See `acoustic_sampling_interval`: the
constraint is dt < M*L/(2*u_rms), which at M=0.1 is roughly a twentieth of an
eddy turnover -- mild, not punishing. The script computes it for you and warns
if --save-every violates it.

    python scripts/01_generate_data.py --n 128 --snapshots 1024
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solvers.turb2d import Turb2D, Turb2DConfig, acoustic_sampling_interval
from src.solvers.io import save_dataset, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--length", type=float, default=2.0 * np.pi)
    ap.add_argument("--snapshots", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--viscosity", type=float, default=5e-4)
    ap.add_argument("--drag", type=float, default=0.08)
    ap.add_argument("--forcing-k", type=float, default=8.0)
    ap.add_argument("--spinup", type=int, default=5000)
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--out", default="data/turb2d.npz")
    args = ap.parse_args()

    cfg = Turb2DConfig(n=args.n, length=args.length, dt=args.dt, viscosity=args.viscosity,
                       drag=args.drag, forcing_k=args.forcing_k)
    solver = Turb2D(cfg)
    dt_save = args.dt * args.save_every
    print(f"grid {args.n}^2   sampling dt = {dt_save:.5f}   record = "
          f"{args.snapshots * dt_save:.2f} time units")

    print("spinning up...")
    u, v, om = solver.run(args.snapshots, args.save_every, args.spinup)

    u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
    dt_max = acoustic_sampling_interval(u_rms, solver.grid, args.mach)
    print(f"\nu_rms = {u_rms:.4f}")
    print(f"acoustic sampling bound at M={args.mach}: dt < {dt_max:.5f}")
    if dt_save > dt_max:
        print(f"  *** WARNING: dt_save={dt_save:.5f} exceeds the bound. The radiating")
        print(f"      cone will be unresolved. Reduce --save-every to "
              f"{max(1,int(dt_max/args.dt))} or below.")
    else:
        print(f"  OK: oversampled by {dt_max/dt_save:.1f}x")

    meta = {"u_rms": u_rms, "mach_target": args.mach, "n": args.n,
            "dt_save": dt_save, "forcing_k": args.forcing_k}
    save_dataset(args.out, u, v, dt_save, solver.grid.length, meta)
    h = write_manifest(Path(args.out).with_suffix(".manifest.json"), cfg, meta)
    print(f"\nwrote {args.out}  (manifest {h})")


if __name__ == "__main__":
    main()
