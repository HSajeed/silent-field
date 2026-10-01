"""The notebook helpers: old/new code must coexist, dataset checks must catch a bad file,
and a history row must refuse to disagree with its log."""
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks"))
import json
import numpy as np
import pytest

from nbtools import env as nbenv
from nbtools.versions import old_src, diff, git_show
from nbtools.style import Comparison


def test_old_and_new_src_coexist():
    old_src()
    from src_old.acoustics.komega import KOmegaSpectrum as OldSpec
    from src.acoustics.komega import KOmegaSpectrum as NewSpec
    assert OldSpec is not NewSpec
    # The initial commit predates the resolved radial grid (Tier 0, item 3).
    assert not hasattr(OldSpec, "radiating_mask_resolved")
    assert hasattr(NewSpec, "radiating_mask_resolved")
    # Relative imports inside the old package resolve to the old package.
    import src_old.acoustics.komega as k_old
    assert k_old.Grid.__module__ == "src_old.spectral"


def test_diff_can_isolate_one_change():
    d = diff("scripts/03_compute_sources.py", grep="[:24]")
    assert "[:24]" in d and d.startswith("---")
    assert "band = [i for i in range" in git_show("scripts/03_compute_sources.py")


def test_comparison_refuses_to_rewrite_history():
    c = Comparison("t").add("R", old=8.691e-08, new=3.2e-08, logged=8.691e-08)
    assert "8.691e-08" in c.markdown()
    with pytest.raises(AssertionError):
        Comparison("t").add("R", old=9.0e-08, new=3.2e-08, logged=8.691e-08)


def test_verify_dataset_rejects_a_substituted_file(tmp_path):
    (tmp_path / "data").mkdir()
    fake = tmp_path / "data" / "turb2d.npz"
    np.savez(fake, u=np.zeros(1), v=np.zeros(1), meta=json.dumps({"u_rms": 1.0}))
    e = nbenv.Env(colab=False, repo=ROOT, data=tmp_path / "data", cache=tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        nbenv.verify_dataset(e, "turb2d")


@pytest.mark.skipif(not (ROOT / "data" / "turb2d.npz").exists(), reason="dataset not present")
def test_verify_dataset_accepts_the_baseline_metadata(tmp_path):
    e = nbenv.Env(colab=False, repo=ROOT, data=ROOT / "data", cache=tmp_path)
    info = nbenv.verify_dataset(e, "turb2d", checksum=False)
    assert info["manifest"] == "111e13e318c9"
    assert info["shape"] == [1024, 128, 128]
