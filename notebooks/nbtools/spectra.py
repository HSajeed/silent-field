"""(k, omega) spectra of the datasets, shared by chapters 03-05 and cached.

Velocity spectra are computed one component at a time and summed. That is identical to
transforming the stacked 2-component field (`komega_spectrum` sums component powers)
but needs half the memory -- which is what lets the n = 256 dataset run on a 7 GB
machine.

`old=True` uses the initial commit's code (`src_old`), whose spectrum has no resolved
radial grid; cache keys carry an `_old` suffix so the two never mix.
"""
from __future__ import annotations

import numpy as np

from . import cache


def _modules(old: bool):
    if old:
        from .versions import old_src

        old_src()
        import src_old.acoustics.komega as komega
        import src_old.acoustics.lighthill as lighthill
    else:
        import src.acoustics.komega as komega
        import src.acoustics.lighthill as lighthill
    return komega, lighthill


def _spec(komega, field, grid, dt, segments):
    if segments > 1:
        return komega.welch_komega_spectrum(field, grid, dt, n_segments=segments)
    return komega.komega_spectrum(field, grid, dt)


def velocity_spectrum(u, v, grid, dt, segments=4, old=False):
    komega, _ = _modules(old)
    su = _spec(komega, u, grid, dt, segments)
    sv = _spec(komega, v, grid, dt, segments)
    kw = dict(power=su.power + sv.power, k_bins=su.k_bins, omega=su.omega)
    if getattr(su, "power_resolved", None) is not None:
        kw.update(k_resolved=su.k_resolved, power_resolved=su.power_resolved + sv.power_resolved)
    return komega.KOmegaSpectrum(**kw)


def load(env, name):
    from src.solvers.io import load_dataset
    from src.spectral import Grid

    u, v, meta = load_dataset(env.dataset(name))
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    return u, v, grid, meta["dt"]


def spectra_for(env, name, halves=False, old=False, source=True):
    """Velocity (and Lighthill-source) spectra of dataset `name`, from cache when possible.

    Returns a dict with U (u_rms), grid, dt and spectra under keys vel_full / src_full
    (plus *_half1 / *_half2 with `halves`). Halves use 2 Welch segments, the full record 4,
    as in scripts/04_komega_analysis.py.
    """
    komega, lighthill = _modules(old)
    suffix = "_old" if old else ""
    loaded = {}

    def data():
        if not loaded:
            loaded["u"], loaded["v"], loaded["grid"], loaded["dt"] = load(env, name)
        return loaded

    import json

    meta_path = env.cache / f"{name}_meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
    else:
        d = data()
        m = {"U": float(np.sqrt(np.mean(d["u"] ** 2 + d["v"] ** 2))), "n": int(d["u"].shape[-1]),
             "length": float(d["grid"].length), "dt": float(d["dt"]), "nt": int(d["u"].shape[0])}
        meta_path.write_text(json.dumps(m))
    from src.spectral import Grid

    out = {"U": m["U"], "grid": Grid(n=m["n"], length=m["length"]), "dt": m["dt"]}
    nt = m["nt"]
    parts = {"full": (slice(None), 4)}
    if halves:
        parts.update({"half1": (slice(0, nt // 2), 2), "half2": (slice(nt // 2, None), 2)})

    for part, (sl, seg) in parts.items():
        def build_v(sl=sl, seg=seg):
            d = data()
            return velocity_spectrum(d["u"][sl], d["v"][sl], d["grid"], d["dt"], seg, old)
        out[f"vel_{part}"] = cache.spectrum(env.cache, f"{name}_vel_{part}{suffix}", komega.KOmegaSpectrum, build_v)

    if source:
        S = {}
        for part, (sl, seg) in parts.items():
            def build_s(sl=sl, seg=seg):
                d = data()
                if "S" not in S:
                    S["S"] = lighthill.source_timeseries(d["u"], d["v"], d["grid"], form="divergence")
                return _spec(komega, S["S"][sl], d["grid"], d["dt"], seg)
            out[f"src_{part}"] = cache.spectrum(env.cache, f"{name}_src_{part}{suffix}", komega.KOmegaSpectrum, build_s)
    return out
