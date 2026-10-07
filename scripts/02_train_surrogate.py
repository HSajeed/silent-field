#!/usr/bin/env python3
"""STAGE 3 -- train a deliberately standard surrogate.

Do not tune this for the acoustics. The claim is that the ORDINARY recipe is
blind, so a stock architecture with a stock objective is the strongest evidence.

    python scripts/02_train_surrogate.py --data data/turb2d.npz --model fno --out runs/fno_mse_01
    python scripts/02_train_surrogate.py --loss sobolev --out runs/fno_sob_01   # Stage 4
    python scripts/02_train_surrogate.py --loss h1 --out runs/fno_h1_01         # standard relative H1
    python scripts/02_train_surrogate.py --unroll 4 --out runs/fno_mse_unroll4  # backprop through 4 steps
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
    ap.add_argument("--loss", choices=["mse", "sobolev", "h1"], default="mse")
    ap.add_argument("--unroll", type=int, default=1,
                    help="train through this many autoregressive steps (gradients flow through all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="runs/fno_mse", help="run folder (model.pt, last.pt, history.json)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="fast scratch for per-epoch checkpoints (default /content/ckpt on Colab, else --out)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="first 64 snapshots, 2 epochs: a quick CPU check of the whole path")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required for training: pip install torch")

    from src.models.fno import FNO2d
    from src.models.unet import UNet2d
    from src.models.train import TrainConfig, make_pairs, make_windows, train

    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("/content/ckpt") if Path("/content").is_dir() else out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    verify_sha256(Path(args.data))
    u, v, meta = load_dataset(args.data)
    epochs = args.epochs
    if args.smoke:
        u, v, epochs = u[:64], v[:64], 2
    x, y, stats = make_pairs(u, v) if args.unroll == 1 else make_windows(u, v, args.unroll)
    print(f"{len(x)} {'one-step pairs' if args.unroll == 1 else f'{args.unroll}-step windows'}, "
          f"{u.shape[-1]}^2 grid, loss={args.loss}")

    model = FNO2d(modes=min(24, u.shape[-1]//4), width=48) if args.model == "fno" \
        else UNet2d(base=48, depth=3)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.model.upper()}: {n_par/1e6:.2f}M parameters")

    cfg = TrainConfig(epochs=epochs, batch_size=args.batch_size, loss=args.loss,
                      unroll=args.unroll, domain_length=meta["grid_length"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"device {cfg.device} (GPU: {gpu})")

    resume = None
    last = out / "last.pt"
    if last.exists():
        resume = torch.load(last, map_location="cpu", weights_only=False)
        if (resume["arch"], resume["loss"], resume["epochs"], resume.get("unroll", 1)) != \
                (args.model, args.loss, epochs, args.unroll):
            sys.exit(f"{last} is a different run (arch/loss/epochs); use a new --out")
        print(f"resuming from {last} at epoch {len(resume['history'])}")

    def save(epoch, model, opt, sched, history):
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "history": history, "rng": torch.get_rng_state(),
                 "arch": args.model, "loss": args.loss, "epochs": epochs, "unroll": args.unroll}
        torch.save(state, ckpt_dir / "last.pt")
        if ckpt_dir != out and ((epoch + 1) % args.save_every == 0 or epoch + 1 == epochs):
            shutil.copy2(ckpt_dir / "last.pt", last)

    model, hist = train(model, x, y, cfg, on_epoch_end=save, resume=resume)

    final = {"state_dict": model.state_dict(), "arch": args.model, "loss": args.loss,
             "unroll": args.unroll,
             "stats": stats, "meta": meta, "n": u.shape[-1], "history": hist}
    torch.save(final, ckpt_dir / "model.pt")
    if ckpt_dir != out:
        shutil.copy2(ckpt_dir / "model.pt", out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    wall = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print("SUMMARY  scripts/02_train_surrogate.py")
    print(f"  model        {args.model.upper()}  ({n_par/1e6:.2f}M params), loss {args.loss}, "
          f"unroll {args.unroll}")
    print(f"  data         {args.data}  ({len(x)} pairs{', SMOKE' if args.smoke else ''})")
    print(f"  epochs       {len(hist)}   device {cfg.device} ({gpu})")
    print(f"  final        train {hist[-1]['train']:.3e}   val {hist[-1]['val']:.3e}  "
          f"(val = one-step MSE, normalised, comparable across losses)")
    print(f"  best val     {min(h['val'] for h in hist):.3e}")
    print(f"  wall time    {wall:.1f} min")
    print(f"  checkpoint   {out / 'model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
