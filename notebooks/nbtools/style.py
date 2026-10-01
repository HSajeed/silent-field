"""Plot defaults and the old -> new comparison table used by every history cell."""
from __future__ import annotations

import math

OLD = "#c0392b"   # red: code as it was
NEW = "#1f6fb2"   # blue: code as it is
REF = "#555555"   # grey: references (sonic line, k^-3, ...)


def use():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.figsize": (7.0, 4.2),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def _fmt(x):
    if isinstance(x, str):
        return x
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.4g}"
    return f"{x:.3e}"


class Comparison:
    """Rows of `quantity | old | new | new/old | logged | source`.

    `logged` is the value a log or JSON file recorded at the time. When given, the
    corresponding computed value must match it to `rtol`, otherwise the cell fails --
    the notebook must reproduce history, not reinterpret it.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows = []

    def add(self, quantity, old=None, new=None, logged=None, check="old", rtol=1e-3, source=""):
        if logged is not None:
            got = old if check == "old" else new
            if not math.isclose(got, logged, rel_tol=rtol):
                raise AssertionError(f"{quantity}: computed {check} value {got!r} does not reproduce "
                                     f"the logged {logged!r} ({source})")
        ratio = (new / old) if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old else None
        self.rows.append((quantity, old, new, ratio, logged, source))
        return self

    def markdown(self) -> str:
        head = f"**{self.title}**\n\n| quantity | old | new | new/old | logged | source |\n|---|---|---|---|---|---|\n"
        body = "".join(
            f"| {q} | {_fmt(o)} | {_fmt(n)} | {_fmt(r)} | {_fmt(l)} | {s} |\n"
            for q, o, n, r, l, s in self.rows
        )
        return head + body

    def show(self):
        try:
            from IPython.display import Markdown, display

            display(Markdown(self.markdown()))
        except ImportError:
            print(self.markdown())
        return self


def md(text: str):
    """Display markdown from a code cell (used for computed summaries)."""
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)
