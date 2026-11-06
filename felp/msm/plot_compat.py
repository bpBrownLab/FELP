"""Compatibility shims for PyEMMA plotting on newer Matplotlib releases."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def _matplotlib_legend_positional_loc_compat() -> Iterator[None]:
    """Allow PyEMMA's old ``Figure.legend(handles, labels, loc, ...)`` call."""

    from matplotlib.figure import Figure

    original_legend = Figure.legend

    def legend_compat(self, *args, **kwargs):
        if len(args) >= 3 and "loc" not in kwargs:
            kwargs["loc"] = args[2]
            args = args[:2] + args[3:]
        return original_legend(self, *args, **kwargs)

    Figure.legend = legend_compat
    try:
        yield
    finally:
        Figure.legend = original_legend


def plot_pyemma_cktest(pyemma: Any, ck_result: Any, *, diag: bool = True) -> None:
    """Plot a PyEMMA CK test with Matplotlib 3.8+ legend compatibility."""

    with _matplotlib_legend_positional_loc_compat():
        pyemma.plots.plot_cktest(ck_result, diag=diag)
