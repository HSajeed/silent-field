#!/usr/bin/env python3
"""STAGE 3 -- train a deliberately standard surrogate.

Do not tune this for the acoustics. The claim is that the ORDINARY recipe is
blind, so a stock architecture with a stock objective is the strongest evidence.

    python scripts/02_train_surrogate.py --data data/turb2d.npz --model fno --out runs/fno_mse_01
    python scripts/02_train_surrogate.py --loss sobolev --out runs/fno_sob_01   # Stage 4
    python scripts/02_train_surrogate.py --loss h1 --out runs/fno_h1_01         # standard relative H1
    python scripts/02_train_surrogate.py --unroll 4 --out runs/fno_mse_unroll4  # backprop through 4 steps
    python scripts/02_train_surrogate.py --loss cone --out runs/fno_cone_01     # Experiment 5
    python scripts/02_train_surrogate.py --smoke --out tmp/smoke02              # CPU check

Checkpoints go to --ckpt-dir every epoch (fast local disk; /content/ckpt on Colab) and
are copied to --out every --save-every epochs and at the end (--out may be a mounted
Drive folder, where direct writes are slow).  Rerunning with the same --out resumes from
<out>/last.pt, so a dropped Colab runtime costs at most --save-every epochs.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solvers.io import load_dataset


def verify_sha256(path: Path):
    """Refuse a dataset that is not the author's file (registry in notebooks/nbtools/env.py)."""
    from notebooks.nbtools.env import DATASETS, sha256
    known = {info["file"]: info["sha256"] for info in DATASETS.values()}
    if path.name not in known:
        print(f"[data] {path.name} is not in the registry; sha256 not checked")
        return
    if sha256(path) != known[path.name]:
        sys.exit(f"[data] {path} sha256 does not match the author's copy -- refusing to train")
    print(f"[data] {path.name} sha256 verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", choices=["fno", "unet"], default="fno")
    ap.add_argument("--loss", choices=["mse", "sobolev", "h1", "cone"], default="mse")
    ap.add_argument("--unroll", type=int, default=1,
                    help="train through this many autoregressive steps (gradients flow through all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cone-lambda", type=float, default=1e3,
                    help="Experiment 5: weight of the cone term (mse-scale units)")
    ap.add_argument("--cone-mach", type=float, default=0.1,
                    help="Experiment 5: analysis Mach number used to set the cone's sound speed")
    ap.add_argument("--cone-window", type=int, default=16,
                    help="Experiment 5: teacher-forced window length W")
    ap.add_argument("--cone-stride", type=int, default=None,
                    help="Experiment 5: frames between window starts (default window//2)")
    ap.add_argument("--cone-batch", type=int, default=2,
                    help="Experiment 5: windows per optimiser step")
    ap.add_argument("--out", default="runs/fno_mse", help="run folder (model.pt, last.pt, history.json)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="fast scratch for per-epoch checkpoints (default /content/ckpt on Colab, else --out)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the weight initialisation and the batch order")
    ap.add_argument("--smoke", action="store_true",
                    help="first 64 snapshots, 2 epochs: a quick CPU check of the whole path")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required for training: pip install torch")

    from src.models.fno import FNO2d
    from src.models.unet import UNet2d
    from src.models.train import TrainConfig, make_pairs, make_windows, make_tf_windows, train
    from src.acoustics.komega import sound_speed

    if args.loss == "cone" and args.unroll != 1:
        sys.exit("--loss cone trains on teacher-forced windows already; --unroll != 1 is incompatible")

    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("/content/ckpt") if Path("/content").is_dir() else out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    verify_sha256(Path(args.data))
    u, v, meta = load_dataset(args.data)
    epochs = args.epochs
    cone_window, cone_stride = args.cone_window, args.cone_stride or args.cone_window // 2
    smoke_note = ""
    if args.smoke:
        u, v, epochs = u[:64], v[:64], 2
        if args.loss == "cone":
            # 64 snapshots leave ~6 windows at W=16/stride=8 -- too few once the 15%
            # val split and the boundary-overlap drop are applied.  Shrink the window
            # so --smoke still exercises a real train/val split with >=1 window each.
            cone_window, cone_stride = 8, 4
            smoke_note = f" (smoke: cone window/stride shrunk to {cone_window}/{cone_stride})"

    cone_c0 = sound_speed(meta["u_rms"], args.cone_mach)
    if args.loss == "cone":
        x, y, stats = make_tf_windows(u, v, cone_window, cone_stride)
    else:
        x, y, stats = make_pairs(u, v) if args.unroll == 1 else make_windows(u, v, args.unroll)
    batch_size = args.cone_batch if args.loss == "cone" else args.batch_size
    if args.loss == "cone":
        kind = f"{cone_window}-frame teacher-forced windows (stride {cone_stride})"
    elif args.unroll == 1:
        kind = "one-step pairs"
    else:
        kind = f"{args.unroll}-step windows"
    print(f"{len(x)} {kind}, {u.shape[-1]}^2 grid, loss={args.loss}{smoke_note}")

    torch.manual_seed(args.seed)                     # weight initialisation
    model = FNO2d(modes=min(24, u.shape[-1]//4), width=48) if args.model == "fno" \
        else UNet2d(base=48, depth=3)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.model.upper()}: {n_par/1e6:.2f}M parameters")

    cfg = TrainConfig(epochs=epochs, batch_size=batch_size, loss=args.loss,
                      unroll=args.unroll, domain_length=meta["grid_length"], seed=args.seed,
                      cone_lambda=args.cone_lambda, cone_mach=args.cone_mach,
                      cone_window=cone_window, cone_c0=cone_c0, cone_dt=meta["dt_save"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"device {cfg.device} (GPU: {gpu})")

    cone_params = (args.cone_lambda, args.cone_mach, cone_window, cone_stride, cone_c0)
    resume = None
    last = out / "last.pt"
    if last.exists():
        resume = torch.load(last, map_location="cpu", weights_only=False)
        if (resume["arch"], resume["loss"], resume["epochs"], resume.get("unroll", 1),
                resume.get("seed", 0), resume.get("cone_params", cone_params)) != (
                args.model, args.loss, epochs, args.unroll, args.seed, cone_params):
            sys.exit(f"{last} is a different run (arch/loss/epochs/cone params); use a new --out")
        print(f"resuming from {last} at epoch {len(resume['history'])}")

    def save(epoch, model, opt, sched, history):
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "history": history, "rng": torch.get_rng_state(),
                 "arch": args.model, "loss": args.loss, "epochs": epochs, "unroll": args.unroll,
                 "seed": args.seed, "cone_params": cone_params}
        torch.save(state, ckpt_dir / "last.pt")
        if ckpt_dir != out and ((epoch + 1) % args.save_every == 0 or epoch + 1 == epochs):
            shutil.copy2(ckpt_dir / "last.pt", last)

    model, hist = train(model, x, y, cfg, on_epoch_end=save, resume=resume)

    final = {"state_dict": model.state_dict(), "arch": args.model, "loss": args.loss,
             "unroll": args.unroll, "seed": args.seed, "cone_params": cone_params,
             "stats": stats, "meta": meta, "n": u.shape[-1], "history": hist}
    torch.save(final, ckpt_dir / "model.pt")
    if ckpt_dir != out:
        shutil.copy2(ckpt_dir / "model.pt", out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    wall = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print("SUMMARY  scripts/02_train_surrogate.py")
    print(f"  model        {args.model.upper()}  ({n_par/1e6:.2f}M params), loss {args.loss}, "
          f"unroll {args.unroll}, seed {args.seed}")
    print(f"  data         {args.data}  ({len(x)} pairs{', SMOKE' if args.smoke else ''})")
    if args.loss == "cone":
        print(f"  cone params  lambda {args.cone_lambda:.3g}  mach {args.cone_mach}  c0 {cone_c0:.3f}  "
              f"window {cone_window}  stride {cone_stride}  batch {args.cone_batch}{smoke_note}")
    print(f"  epochs       {len(hist)}   device {cfg.device} ({gpu})")
    print(f"  final        train {hist[-1]['train']:.3e}   val {hist[-1]['val']:.3e}  "
          f"(val = one-step MSE, normalised, comparable across losses)"
          + (f"   val_cone {hist[-1]['val_cone']:.3e}" if "val_cone" in hist[-1] else ""))
    print(f"  best val     {min(h['val'] for h in hist):.3e}")
    print(f"  wall time    {wall:.1f} min")
    print(f"  checkpoint   {out / 'model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
