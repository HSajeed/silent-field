#!/usr/bin/env python3
"""Execute chapter notebooks headless (locally), as a check that every cell runs.

Executed copies go to .nb_cache/executed/ so the committed notebooks stay output-free.
Needs nbclient + ipykernel.

    python notebooks/run_notebooks.py            # all chapters
    python notebooks/run_notebooks.py 01 03      # selected chapters
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / ".nb_cache" / "executed"


def run(path: Path, timeout: int = 3600) -> float:
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, OUT / path.name)
    return time.time() - t0


def main(argv):
    books = sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    if argv:
        books = [b for b in books if any(b.name.startswith(a) for a in argv)]
    for b in books:
        print(f"running {b.name} ...", flush=True)
        dt = run(b)
        print(f"  ok in {dt/60:.1f} min -> {OUT / b.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
