"""Save heavy intermediates so a later chapter (or a reconnected Colab session) can
reuse them instead of recomputing.  Spectra are the expensive part: a full-record
(k, omega) spectrum of the baseline takes a minute or more.

Cache keys must say which code produced the object (e.g. "komega_new_full"), because the
old and new versions of `komega_spectrum` return different things.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FIELDS = ("power", "k_bins", "omega", "k_resolved", "power_resolved")


def save_spectrum(cache: Path, key: str, spec) -> Path:
    path = Path(cache) / f"{key}.npz"
    arrays = {f: getattr(spec, f) for f in _FIELDS if getattr(spec, f, None) is not None}
    np.savez(path, **arrays)
    return path


def load_spectrum(cache: Path, key: str, cls):
    """Rebuild a spectrum object of class `cls` (old or new KOmegaSpectrum)."""
    path = Path(cache) / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        kwargs = {f: d[f] for f in _FIELDS if f in d.files}
    return cls(**kwargs)


def spectrum(cache: Path, key: str, cls, build):
    """Load `key` from the cache, or call `build()` and store the result."""
    spec = load_spectrum(cache, key, cls)
    if spec is None:
        spec = build()
        save_spectrum(cache, key, spec)
    return spec


def array(cache: Path, key: str, build):
    """Same idea for a plain array or dict of arrays."""
    path = Path(cache) / f"{key}.npz"
    if path.exists():
        with np.load(path) as d:
            return d["arr"] if d.files == ["arr"] else {f: d[f] for f in d.files}
    out = build()
    if isinstance(out, dict):
        np.savez(path, **out)
    else:
        np.savez(path, arr=out)
    return out
