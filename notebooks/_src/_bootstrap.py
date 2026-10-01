# Setup -- run this cell first.
# In Colab: clones the private repo (needs the Colab secret GH_TOKEN) and mounts Google
# Drive, where the datasets live in MyDrive/silent-field/data/.  Locally: finds the repo.
import subprocess, sys
from pathlib import Path

try:
    import google.colab  # noqa: F401
    _repo = Path("/content/silent-field")
    if not (_repo / ".git").exists():
        from google.colab import userdata
        _url = f"https://{userdata.get('GH_TOKEN')}@github.com/HSajeed/silent-field.git"
        subprocess.run(["git", "clone", "-q", "--branch", "tier0-review-notebooks", _url, str(_repo)],
                       check=True)
        subprocess.run(["git", "-C", str(_repo), "remote", "set-url", "origin",
                        "https://github.com/HSajeed/silent-field.git"], check=True)
    sys.path.insert(0, str(_repo / "notebooks"))
except ImportError:
    _here = Path.cwd()
    sys.path.insert(0, str(_here if (_here / "nbtools").exists() else _here / "notebooks"))

from nbtools import env as nbenv, style
from nbtools.style import Comparison, md

ENV = nbenv.setup()
style.use()