"""Dataset persistence and run manifests.

Uses .npz by default so the repo has no hard dependency beyond numpy; swap in
zarr for records too large to hold in RAM (see `save_dataset` docstring).
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def save_dataset(path, u, v, dt, grid_length, meta=None):
    """Save a velocity history plus everything needed to interpret it.

    dt and grid_length are stored because every (k, omega) axis depends on them;
    a dataset without them is unusable and the mistake is easy to make.

    For records beyond a few GB, replace this with zarr and chunk along time:
        z = zarr.open(path, mode='w', shape=u.shape, chunks=(64, n, n))
    which lets the Welch segmentation in komega.py stream rather than load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.update({"dt": float(dt), "grid_length": float(grid_length),
                 "shape": list(u.shape)})
    np.savez_compressed(path, u=u.astype(np.float32), v=v.astype(np.float32),
                        meta=json.dumps(meta))
    return path


def load_dataset(path):
    d = np.load(Path(path), allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return d["u"].astype(np.float64), d["v"].astype(np.float64), meta


def write_manifest(path, config, extra=None):
    """Record exactly what produced a run. Every figure should be traceable."""
    cfg = asdict(config) if is_dataclass(config) else dict(config)
    payload = {"config": cfg, "extra": dict(extra or {})}
    blob = json.dumps(payload, sort_keys=True, indent=2, default=str)
    payload["hash"] = hashlib.sha256(blob.encode()).hexdigest()[:12]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2, default=str))
    return payload["hash"]
