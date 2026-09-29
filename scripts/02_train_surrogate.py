#!/usr/bin/env python3
"""STAGE 3 -- train a deliberately standard surrogate.

Do not tune this for the acoustics. The claim is that the ORDINARY recipe is
blind, so a stock architecture with a stock objective is the strongest evidence.

    python scripts/02_train_surrogate.py --data data/turb2d.npz --model fno
    python scripts/02_train_surrogate.py --data data/turb2d.npz --loss sobolev  # Stage 4
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solvers.io import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", choices=["fno", "unet"], default="fno")
    ap.add_argument("--loss", choices=["mse", "sobolev"], default="mse")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="runs/model.pt")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required for training: pip install torch")

    from src.models.fno import FNO2d
    from src.models.unet import UNet2d
    from src.models.train import TrainConfig, make_pairs, train

    u, v, meta = load_dataset(args.data)
    x, y, stats = make_pairs(u, v)
    print(f"{len(x)} one-step pairs, {u.shape[-1]}^2 grid, loss={args.loss}")

    model = FNO2d(modes=min(24, u.shape[-1]//4), width=48) if args.model == "fno" \
        else UNet2d(base=48, depth=3)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.model.upper()}: {n_par/1e6:.2f}M parameters")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, loss=args.loss,
                       domain_length=meta["grid_length"])
    model, hist = train(model, x, y, cfg)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": args.model,
                "loss": args.loss, "stats": stats, "meta": meta,
                "n": u.shape[-1]}, out)
    out.with_suffix(".history.json").write_text(json.dumps(hist, indent=2))
    print(f"wrote {out}   final val {hist[-1]['val']:.3e}")


if __name__ == "__main__":
    main()
