"""Where am I running, where is the repo, where is the data -- and is it the right data.

Colab: clones the private repo with a token from Colab Secrets (key `GH_TOKEN`), mounts
Google Drive and reads datasets from `MyDrive/silent-field/data/`.
Local: finds the repo by walking up from the working directory and reads `data/`.

Datasets are never regenerated here (see CLAUDE.md); they are checked instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "github.com/HSajeed/silent-field.git"
BRANCH = "tier0-review-notebooks"
COLAB_REPO = Path("/content/silent-field")
DRIVE_ROOT = Path("/content/drive/MyDrive/silent-field")

# sha256 of the dataset files as they exist on the author's machine (2026-10-01).
# The manifest hashes (111e13e318c9, ...) hash the *config*, not the data, so they
# cannot detect a corrupted or substituted file; these can.
DATASETS = {
    "turb2d": {
        "file": "turb2d.npz",
        "manifest_hash": "111e13e318c9",
        "sha256": "bf3a0036f26825de5994ddc43b6419c6274bb1c59c33ad5682733633fa35d0f0",
        "what": "baseline: L=2pi, n=128, forcing k=8, 1024 snapshots",
    },
    "turb2d_kf16": {
        "file": "turb2d_kf16.npz",
        "manifest_hash": "e5d144b83287",
        "sha256": "661cc7ed8d84e88aba3c4e97185db02bec2dc84a0646831a83e07f9154515e90",
        "what": "forcing-scale test: L=2pi, n=128, forcing k=16",
    },
    "turb2d_L4pi": {
        "file": "turb2d_L4pi.npz",
        "manifest_hash": "34d5ea324579",
        "sha256": "0ba32970c26ac21a02af8d3e579d737658b961fcbfb1d44d764f4f1e8e991f40",
        "what": "box-size test: L=4pi, n=256 (same dx), forcing k=8",
    },
}


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Env:
    colab: bool
    repo: Path
    data: Path
    cache: Path

    def dataset(self, name: str) -> Path:
        return self.data / DATASETS[name]["file"]

    def manifest(self, name: str) -> Path:
        # Manifests are tracked in git, so they come from the repo, not from Drive.
        return self.repo / "data" / DATASETS[name]["file"].replace(".npz", ".manifest.json")

    def has(self, name: str) -> bool:
        return self.dataset(name).exists()


def _find_local_repo(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src" / "spectral.py").exists() and (p / "scripts").is_dir():
            return p
    raise FileNotFoundError(
        "could not find the silent-field repo above the working directory; "
        "open the notebook from inside the repo's notebooks/ folder"
    )


def _token_url() -> str:
    from google.colab import userdata  # type: ignore

    return f"https://{userdata.get('GH_TOKEN')}@{REPO_URL}"


def _clone_in_colab(branch: str) -> Path:
    if (COLAB_REPO / ".git").exists():
        return COLAB_REPO
    # Full history is needed: the old-vs-new chapters read the initial commit.
    subprocess.run(["git", "clone", "-q", "--branch", branch, _token_url(), str(COLAB_REPO)], check=True)
    # Do not leave the token in .git/config.
    subprocess.run(["git", "-C", str(COLAB_REPO), "remote", "set-url", "origin", f"https://{REPO_URL}"],
                   check=True)
    return COLAB_REPO


def update_repo(branch: str = BRANCH) -> str:
    """Pull the latest commits of `branch` into the Colab clone (token used, not stored)."""
    if not in_colab():
        return "local run: update the repo with git yourself"
    out = subprocess.run(["git", "-C", str(COLAB_REPO), "pull", "-q", "--ff-only", _token_url(), branch],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.replace(_token_url(), "https://***@" + REPO_URL))
    return f"updated to {current_commit(COLAB_REPO)}"


def setup(branch: str = BRANCH, quiet: bool = False) -> Env:
    """Prepare the environment and return where everything lives.

    Puts the repo root and notebooks/ on sys.path and makes the repo root the working
    directory, so `from src... import` and `scripts/...` paths work as in the README.
    """
    colab = in_colab()
    if colab:
        repo = _clone_in_colab(branch)
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        data = DRIVE_ROOT / "data"
        cache = DRIVE_ROOT / "cache"
        try:
            import pytest  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    else:
        repo = _find_local_repo(Path.cwd().resolve())
        data = repo / "data"
        cache = repo / ".nb_cache"
    cache.mkdir(parents=True, exist_ok=True)

    for p in (str(repo), str(repo / "notebooks")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(repo)

    env = Env(colab=colab, repo=repo, data=data, cache=cache)
    if not quiet:
        where = "Google Colab" if colab else "local machine"
        print(f"running on : {where}")
        print(f"repo       : {repo}  (branch {current_branch(repo)}, commit {current_commit(repo)})")
        print(f"data       : {data}")
        for name, info in DATASETS.items():
            mark = "found" if env.has(name) else "MISSING"
            print(f"  {info['file']:<20} {mark:<8} {info['what']}")
    return env


def current_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def current_branch(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_dataset(env: Env, name: str, checksum: bool = True) -> dict:
    """Check a dataset before using it. Raises if it is not the file the logs describe.

    Two checks:
      * the file's sha256 matches the author's copy (catches corruption/substitution);
      * the dataset's own metadata agrees with its tracked manifest (u_rms, dt_save, n).
    The checksum result is remembered in the cache folder so it runs once per file.
    """
    import numpy as np

    info = DATASETS[name]
    path = env.dataset(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- upload {info['file']} to the data folder")

    stamp = env.cache / f"verified_{name}.json"
    stat = path.stat()
    if checksum:
        ok_before = False
        if stamp.exists():
            s = json.loads(stamp.read_text())
            ok_before = s.get("size") == stat.st_size and s.get("mtime") == stat.st_mtime
        if not ok_before:
            digest = sha256(path)
            if digest != info["sha256"]:
                raise ValueError(f"{info['file']}: sha256 {digest[:12]}... does not match the "
                                 f"author's copy {info['sha256'][:12]}...")
            stamp.write_text(json.dumps({"size": stat.st_size, "mtime": stat.st_mtime}))

    manifest = json.loads(env.manifest(name).read_text())
    if manifest["hash"] != info["manifest_hash"]:
        raise ValueError(f"manifest hash {manifest['hash']} != {info['manifest_hash']}")
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
    extra = manifest["extra"]
    for key in ("u_rms", "dt_save", "n"):
        if abs(float(meta[key]) - float(extra[key])) > 1e-9 * max(1.0, abs(float(extra[key]))):
            raise ValueError(f"{info['file']}: meta[{key}]={meta[key]} disagrees with manifest {extra[key]}")
    return {"file": info["file"], "manifest": manifest["hash"], "shape": meta["shape"],
            "dt_save": meta["dt_save"], "length": meta["grid_length"], "u_rms": meta["u_rms"],
            "config": manifest["config"]}
