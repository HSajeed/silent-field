"""Shared helpers for the silent-field chapter notebooks.

    env       -- Colab/local detection, repo clone, Drive mount, dataset paths + checks
    versions  -- load the initial commit's `src/` as `src_old`, beside the current `src`
    cache     -- save heavy intermediates (spectra) so later chapters don't recompute
    style     -- plotting defaults and the old -> new comparison table

The notebooks import this package only after `env.setup()` has put the repo root on
sys.path, so nothing here imports `src` at module import time.
"""
