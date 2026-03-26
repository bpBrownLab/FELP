"""Shared plotting helpers for MSM and FELP visualizations."""

from __future__ import annotations

import csv
import json
import os
import shutil
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

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def scatter_state_projection(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    colorbar_label: str,
    title: str,
) -> None:
    """Draw a colored scatter plot for state-wise projected values."""

    scatter = ax.scatter(
        x,
        y,
        c=values,
        cmap="viridis",
        s=60,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")
    colorbar = ax.figure.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)


def finite_top_indices(values: Iterable[float], count: int) -> np.ndarray:
    """Return indices of the largest finite absolute values."""

    array = np.asarray(list(values), dtype=float)
    finite_mask = np.isfinite(array)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return np.array([], dtype=int)
    ordered = finite_indices[np.argsort(np.abs(array[finite_mask]))[::-1]]
    return ordered[:count]
