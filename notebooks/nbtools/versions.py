"""Run the code as it was, next to the code as it is.

`old_src()` extracts `src/` from a past commit with `git archive` and imports it as the
package `src_old`.  Every module in `src/` uses relative imports, so `src_old` is fully
self-contained and both versions can be used in the same kernel:

    from nbtools.versions import old_src
    old = old_src()                                  # initial commit, pre-Tier-0
    from src_old.acoustics.komega import komega_spectrum as komega_old
    from src.acoustics.komega import komega_spectrum as komega_new
"""
from __future__ import annotations

import difflib
import importlib
import importlib.util
import subprocess
import sys
import tarfile
import io
from pathlib import Path

INITIAL_COMMIT = "2535bb4"  # 2026-09-29: post-follow-up (09-15), pre-Tier-0


def _repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                         check=True)
    return Path(out.stdout.strip())


def git_show(path: str, commit: str = INITIAL_COMMIT) -> str:
    """A file's text at a past commit."""
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True, cwd=_repo_root()).stdout


def diff(path: str, commit: str = INITIAL_COMMIT, context: int = 2, grep: str | None = None) -> str:
    """Unified diff of `path` between `commit` and the working copy.

    With `grep`, keep only hunks that mention the string -- handy for showing one change
    out of a large file.
    """
    old = git_show(path, commit).splitlines(keepends=True)
    new = (_repo_root() / path).read_text().splitlines(keepends=True)
    lines = list(difflib.unified_diff(old, new, f"{path} @ {commit}", f"{path} @ now", n=context))
    if grep is None:
        return "".join(lines)
    hunks, cur = [], []
    for ln in lines[2:]:
        if ln.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        hunks.append(cur)
    keep = [h for h in hunks if any(grep in ln for ln in h)]
    return "".join(lines[:2] + [ln for h in keep for ln in h])


def old_src(commit: str = INITIAL_COMMIT, name: str = "src_old"):
    """Import `src/` as it was at `commit`, under the package name `name`."""
    if name in sys.modules:
        return sys.modules[name]
    root = _repo_root()
    dest = root / ".nb_cache" / f"src_at_{commit}"
    pkg = dest / "src"
    if not (pkg / "__init__.py").exists():
        dest.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "archive", commit, "src"], capture_output=True, check=True,
                              cwd=root).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(dest)
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py",
                                                  submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    importlib.invalidate_caches()
    return module
