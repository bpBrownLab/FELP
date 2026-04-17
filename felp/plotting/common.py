"""Shared plotting helpers for MSM and FELP visualizations."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from typing import Iterable

import numpy as np

from felp.core import K_BOLTZMANN_KJ_PER_MOL_K


def import_matplotlib():
    """Import matplotlib with a non-interactive backend for script use."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def apply_publication_style() -> None:
    """Apply a light publication-oriented matplotlib style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 4.5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "font.size": 10,
        }
    )


def prepare_output_dir(output_dir: str, overwrite: bool = False) -> None:
    """Create an output directory while respecting overwrite behavior."""

    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        elif os.listdir(output_dir):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}"
            )
    os.makedirs(output_dir, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    """Write a JSON payload to disk with stable formatting."""

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def read_csv_rows(path: str) -> list[dict[str, str]]:
    """Load a CSV file into a list of row dictionaries."""

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: str | None) -> float | None:
    """Convert a CSV string field into an optional float."""

    if value in (None, ""):
        return None
    return float(value)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize a probability vector while preserving zero-only inputs."""

    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.zeros_like(probabilities, dtype=float)
    return np.asarray(probabilities, dtype=float) / total


def free_energy_from_probabilities(
    probabilities: np.ndarray,
    temperature_kelvin: float,
) -> np.ndarray:
    """Convert probabilities to relative free energies in kJ/mol."""

    probabilities = np.asarray(probabilities, dtype=float)
    positive_mask = probabilities > 0.0
    free_energies = np.full_like(probabilities, fill_value=np.nan, dtype=float)
    if positive_mask.any():
        free_energies[positive_mask] = -(
            K_BOLTZMANN_KJ_PER_MOL_K
            * temperature_kelvin
            * np.log(probabilities[positive_mask])
        )
        minimum = np.nanmin(free_energies)
        free_energies[positive_mask] -= minimum
    return free_energies


def compute_weighted_histogram2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D histogram and normalize it to a probability surface."""

    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=weights)
    total = histogram.sum()
    if total > 0.0:
        histogram /= total
    return histogram, x_edges, y_edges


def probability_surface_to_free_energy(
    histogram: np.ndarray,
    temperature_kelvin: float,
) -> np.ndarray:
    """Convert a normalized 2D probability surface to relative free energy."""

    free_energy = np.full_like(histogram, fill_value=np.nan, dtype=float)
    positive_mask = histogram > 0.0
    if positive_mask.any():
        free_energy[positive_mask] = -(
            K_BOLTZMANN_KJ_PER_MOL_K
            * temperature_kelvin
            * np.log(histogram[positive_mask])
        )
        free_energy[positive_mask] -= np.nanmin(free_energy)
    return free_energy


_FEL_CAP_OVER_COLOR = "#e8e8e8"


def _copy_viridis_colormap():
    """Return a mutable viridis colormap without deprecated get_cmap when possible."""

    import matplotlib as mpl
    import matplotlib.cm as mpl_cm

    registry = getattr(mpl, "colormaps", None)
    if registry is not None:
        base = registry["viridis"]
    else:
        base = mpl_cm.get_cmap("viridis")
    return base.copy()


def scatter_state_projection(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    colorbar_label: str,
    title: str,
    xlabel: str = "Projection 1",
    ylabel: str = "Projection 2",
    *,
    fel_max_energy_kj_mol: float | None = None,
    symmetric_energy_cap: bool = False,
) -> None:
    """Draw a colored scatter plot for state-wise projected values."""

    import matplotlib.colors as mcolors

    values_plot = np.ma.masked_invalid(np.asarray(values, dtype=float))
    scatter_kwargs: dict = {
        "x": x,
        "y": y,
        "c": values_plot,
        "cmap": "viridis",
        "s": 60,
        "edgecolors": "black",
        "linewidths": 0.3,
    }
    colorbar_extend: str | None = None
    if fel_max_energy_kj_mol is not None:
        cap = float(fel_max_energy_kj_mol)
        cmap = _copy_viridis_colormap()
        cmap.set_bad(_FEL_CAP_OVER_COLOR)
        cmap.set_over(_FEL_CAP_OVER_COLOR)
        cmap.set_under(_FEL_CAP_OVER_COLOR)
        scatter_kwargs["cmap"] = cmap
        if symmetric_energy_cap:
            scatter_kwargs["norm"] = mcolors.Normalize(
                vmin=-cap, vmax=cap, clip=False
            )
            colorbar_extend = "both"
        else:
            scatter_kwargs["norm"] = mcolors.Normalize(vmin=0.0, vmax=cap, clip=False)
            colorbar_extend = "max"

    scatter = ax.scatter(**scatter_kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    colorbar = ax.figure.colorbar(scatter, ax=ax, extend=colorbar_extend or "neither")
    colorbar.set_label(colorbar_label)


def save_pcolormesh_landscape(
    plt,
    path: str,
    z: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    xlabel: str,
    ylabel: str,
    fel_max_energy_kj_mol: float | None = None,
) -> str:
    """Render a 2D surface on a histogram grid (e.g. probability or free energy)."""

    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(1, 1)
    z_plot = np.ma.masked_invalid(np.asarray(z, dtype=float).T)
    colorbar_extend: str | None = None
    if fel_max_energy_kj_mol is not None:
        cap = float(fel_max_energy_kj_mol)
        cmap = _copy_viridis_colormap()
        cmap.set_bad(_FEL_CAP_OVER_COLOR)
        cmap.set_over(_FEL_CAP_OVER_COLOR)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            shading="auto",
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0.0, vmax=cap, clip=False),
        )
        colorbar_extend = "max"
    else:
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            shading="auto",
            cmap="viridis",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, extend=colorbar_extend or "neither")
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def finite_top_indices(values: Iterable[float], count: int) -> np.ndarray:
    """Return indices of the largest finite absolute values."""

    array = np.asarray(list(values), dtype=float)
    finite_mask = np.isfinite(array)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return np.array([], dtype=int)
    ordered = finite_indices[np.argsort(np.abs(array[finite_mask]))[::-1]]
    return ordered[:count]
