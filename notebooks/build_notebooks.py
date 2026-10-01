#!/usr/bin/env python3
"""Build notebooks/*.ipynb from the percent-format sources in notebooks/_src/.

The .py sources are what gets reviewed and diffed; the .ipynb files are generated.
No dependency beyond the standard library.

Cell markers in a source file:
    # %% [markdown]     a markdown cell; every line's leading "# " is stripped
    # %%                a code cell
    # %% [bootstrap]    replaced by the shared setup cell in _src/_bootstrap.py

    python notebooks/build_notebooks.py            # build all
    python notebooks/build_notebooks.py 03         # build chapters whose name starts with 03
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_src"
COLAB = "https://colab.research.google.com/github/HSajeed/silent-field/blob/tier0-review-notebooks/notebooks/{name}"


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _lines(text)}


def parse(path: Path, bootstrap: str) -> list[dict]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf)
        if kind == "markdown":
            stripped = []
            for ln in buf:
                if ln.startswith("# "):
                    stripped.append(ln[2:])
                elif ln.strip() == "#":
                    stripped.append("")
                else:
                    stripped.append(ln)
            body = "\n".join(stripped)
            if body.strip():
                cells.append(_md(body))
        elif kind == "bootstrap":
            cells.append(_code(bootstrap))
        elif body.strip():
            cells.append(_code(body))

    for line in path.read_text().split("\n"):
        if line.startswith("# %%"):
            flush()
            tag = line[4:].strip()
            kind = {"[markdown]": "markdown", "[bootstrap]": "bootstrap"}.get(tag, "code")
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def build(src: Path, bootstrap: str) -> Path:
    name = src.stem + ".ipynb"
    cells = parse(src, bootstrap)
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB.format(name=name)})"
    cells.insert(0, _md(badge))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "cells": cells,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"{src.stem[:2]}-{i:03d}"
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


def main(argv):
    bootstrap = (SRC / "_bootstrap.py").read_text()
    sources = sorted(p for p in SRC.glob("[0-9][0-9]_*.py"))
    if argv:
        sources = [p for p in sources if any(p.name.startswith(a) for a in argv)]
    for src in sources:
        print("built", build(src, bootstrap).relative_to(HERE.parent))


if __name__ == "__main__":
    main(sys.argv[1:])
