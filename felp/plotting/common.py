"""Shared plotting helpers for MSM and FELP visualizations."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from typing import Iterable

import numpy as np

from felp.core import ConfigurationError, K_BOLTZMANN_KJ_PER_MOL_K


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


_FEL_BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)

DEFAULT_FEL_COLORMAP = "nipy_spectral_r"

# Diverging map for symmetric KL heatmaps: twilight_shifted arms + white center.
FELP_KL_DIVERGENCE_COLORMAP = "felp_kl_twilight"

# Per-cell signed KL values with |display| below this are plotted as zero (white center).
DEFAULT_KL_DISPLAY_THRESHOLD = 1.0e-5

_FELP_BUILTIN_CMAPS_REGISTERED = False


def _latex_probability_symbol(name: str) -> str:
    """Return a mathtext probability symbol for colorbar labels."""

    raw = str(name).strip()
    if raw.startswith("p_") and len(raw) > 2:
        subscript = raw[2:]
        return rf"p_{{\mathrm{{{subscript}}}}}"
    return rf"p_{{\mathrm{{{raw}}}}}"


def symmetric_kl_imbalance_mathtext_label(
    left: str,
    right: str,
    *,
    unit_label: str | None = None,
) -> str:
    """Matplotlib mathtext label for the signed per-cell KL imbalance."""

    p_left = _latex_probability_symbol(left)
    p_right = _latex_probability_symbol(right)
    math_label = (
        rf"$\frac{{1}}{{2}}\left[{p_left}\ln\frac{{{p_left}}}{{{p_right}}} "
        rf"- {p_right}\ln\frac{{{p_right}}}{{{p_left}}}\right]$"
    )
    if unit_label:
        return f"{math_label} ({unit_label})"
    return math_label


def resolve_kl_colorbar_limits(
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float] | None:
    """Resolve optional symmetric KL colorbar limits for diverging heatmaps.

    If only ``vmax`` is given, use ``[-vmax, vmax]``. If only ``vmin`` is given,
    use ``[vmin, -vmin]`` (``vmin`` must be negative).
    """

    if vmin is None and vmax is None:
        return None
    if vmin is not None and vmax is not None:
        vmin_value = float(vmin)
        vmax_value = float(vmax)
        if not math.isfinite(vmin_value) or not math.isfinite(vmax_value):
            raise ConfigurationError("kl colorbar limits must be finite floats.")
        if vmin_value >= vmax_value:
            raise ConfigurationError("kl-vmin must be less than kl-vmax.")
        return vmin_value, vmax_value
    if vmax is not None:
        vmax_value = float(vmax)
        if not math.isfinite(vmax_value) or vmax_value <= 0.0:
            raise ConfigurationError(
                "kl-vmax must be a positive finite float when kl-vmin is omitted."
            )
        return -vmax_value, vmax_value
    vmin_value = float(vmin)  # type: ignore[arg-type]
    if not math.isfinite(vmin_value) or vmin_value >= 0.0:
        raise ConfigurationError(
            "kl-vmin must be a negative finite float when kl-vmax is omitted."
        )
    return vmin_value, -vmin_value


def symmetric_kl_imbalance_plain_label(left: str, right: str) -> str:
    """Plain-text counterpart to :func:`symmetric_kl_imbalance_mathtext_label`."""

    return (
        f"0.5 * [{left} * log({left}/{right}) - "
        f"{right} * log({right}/{left})]"
    )


def apply_kl_display_threshold(
    grid: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Zero out tiny per-cell KL display values to avoid misleading color stretch.

    Matplotlib symmetric diverging maps otherwise set ``vmax`` to ``max(|grid|)``,
    which can be ~1e-6 kcal/mol when surfaces nearly match and makes noise look
    like strong imbalance.
    """

    if threshold <= 0.0:
        return np.asarray(grid, dtype=float), {
            "kl_display_threshold": 0.0,
            "kl_display_cells_zeroed": 0,
            "kl_display_max_abs_before_threshold": float(
                np.nanmax(np.abs(np.asarray(grid, dtype=float)))
            ),
        }
    array = np.asarray(grid, dtype=float)
    finite = np.isfinite(array)
    max_abs_before = float(np.nanmax(np.abs(array[finite]))) if finite.any() else 0.0
    small = finite & (np.abs(array) < threshold)
    masked = np.where(small, 0.0, array)
    return masked, {
        "kl_display_threshold": float(threshold),
        "kl_display_cells_zeroed": int(np.count_nonzero(small)),
        "kl_display_max_abs_before_threshold": max_abs_before,
        "kl_display_max_abs_after_threshold": float(
            np.nanmax(np.abs(masked[finite])) if finite.any() else 0.0
        ),
    }


def prepare_kl_display_grid(
    kl_grid: np.ndarray,
    *,
    temperature_kelvin: float,
    scale_by_kbt: bool,
    kbt_energy_scale: float = 1.0,
    kbt_energy_unit_label: str = "kJ/mol",
    display_threshold: float | None = DEFAULT_KL_DISPLAY_THRESHOLD,
    probability_labels: tuple[str, str] = ("p_pre", "p_rw"),
) -> tuple[np.ndarray, str, dict[str, object]]:
    """Scale optional kBT, apply display threshold, and return colorbar metadata."""

    array = np.asarray(kl_grid, dtype=float)
    p_left, p_right = probability_labels
    metadata: dict[str, object] = {
        "kl_display_units": "dimensionless",
        "kl_display_scale_factor": 1.0,
        "kl_display_equation": symmetric_kl_imbalance_plain_label(p_left, p_right),
    }
    if scale_by_kbt:
        scale = (
            K_BOLTZMANN_KJ_PER_MOL_K
            * float(temperature_kelvin)
            * float(kbt_energy_scale)
        )
        array = array * scale
        metadata["kl_display_units"] = str(kbt_energy_unit_label)
        metadata["kl_display_scale_factor"] = float(scale)
        colorbar_label = symmetric_kl_imbalance_mathtext_label(
            p_left,
            p_right,
            unit_label=str(kbt_energy_unit_label),
        )
    else:
        colorbar_label = symmetric_kl_imbalance_mathtext_label(p_left, p_right)

    threshold = (
        DEFAULT_KL_DISPLAY_THRESHOLD
        if display_threshold is None
        else float(display_threshold)
    )
    array, threshold_metadata = apply_kl_display_threshold(array, threshold=threshold)
    metadata.update(threshold_metadata)
    return array, colorbar_label, metadata


def _ensure_felp_builtin_colormaps_registered() -> None:
    """Register FELP-specific colormaps exactly once (idempotent).

    ``felp_kl_twilight`` follows the **left** and **right** arms of Matplotlib's
    ``twilight_shifted`` (indigo→blues toward the peak, then warm tans/browns→
    burgundy on the downward arm), but replaces the stock pinkish peak with
    **pure white** and uses **linear** progress in colormap parameter on each
    arm so the saturated tails are not compressed.
    """

    global _FELP_BUILTIN_CMAPS_REGISTERED
    if _FELP_BUILTIN_CMAPS_REGISTERED:
        return

    import matplotlib as mpl
    import matplotlib.colors as mcolors

    name = FELP_KL_DIVERGENCE_COLORMAP
    registry = getattr(mpl, "colormaps", None)
    if registry is not None and name in registry:
        _FELP_BUILTIN_CMAPS_REGISTERED = True
        return

    try:
        base = mpl.colormaps["twilight_shifted"]
    except (AttributeError, KeyError, TypeError):
        base = mpl.cm.get_cmap("twilight_shifted")

    white = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    # Left arm: dark indigo → blues (twilight parameter 0 .. ~0.46).
    left_u0, left_u1 = 0.0, 0.46
    # Right arm: warm descent from just past peak through brown/burgundy; stop
    # before ~0.94 where hues fold back toward purple (cyclical wrap).
    right_u0, right_u1 = 0.54, 0.91
    # Fraction of each half (near center) cross-faded to pure white.
    center_blend = 0.16

    n = 256
    rgba = np.zeros((n, 4), dtype=float)
    mid_left = (n - 1) // 2
    mid_right = n // 2

    for i in range(mid_left + 1):
        f = i / mid_left if mid_left else 0.0
        u = left_u0 + f * (left_u1 - left_u0)
        c = np.clip(np.asarray(base(u), dtype=float), 0.0, 1.0)
        if f <= 1.0 - center_blend:
            rgba[i] = c
        else:
            t = (f - (1.0 - center_blend)) / center_blend
            rgba[i] = (1.0 - t) * c + t * white

    for i in range(mid_right, n):
        g = (i - mid_right) / max(n - 1 - mid_right, 1)
        if g < center_blend:
            t = g / center_blend
            c0 = np.clip(np.asarray(base(right_u0), dtype=float), 0.0, 1.0)
            rgba[i] = (1.0 - t) * white + t * c0
        else:
            gf = (g - center_blend) / max(1.0 - center_blend, 1e-9)
            u = right_u0 + gf * (right_u1 - right_u0)
            c = np.clip(np.asarray(base(u), dtype=float), 0.0, 1.0)
            rgba[i] = c

    rgba[mid_left] = white
    rgba[mid_right] = white

    cmap = mcolors.ListedColormap(rgba, name=name)
    cmap.set_bad(color=(0.9, 0.9, 0.9, 1.0))

    if registry is not None:
        registry.register(cmap, name=name, force=True)
    else:
        mpl.cm.register_cmap(name=name, cmap=cmap)

    _FELP_BUILTIN_CMAPS_REGISTERED = True


def validate_registered_colormap(name: str) -> str:
    """Return a stripped matplotlib colormap name or raise ``ConfigurationError``."""

    _ensure_felp_builtin_colormaps_registered()

    normalized = str(name).strip()
    if not normalized:
        raise ConfigurationError("fel_colormap must be a non-empty string.")
    import matplotlib as mpl
    import matplotlib.cm as mpl_cm

    registry = getattr(mpl, "colormaps", None)
    if registry is not None:
        if normalized not in registry:
            raise ConfigurationError(
                f"Unknown matplotlib colormap {normalized!r} (not in matplotlib colormaps)."
            )
        return normalized
    try:
        mpl_cm.get_cmap(normalized)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"Unknown matplotlib colormap {normalized!r}."
        ) from exc
    return normalized


def _copy_colormap_for_fel_cap(name: str):
    """Return a mutable copy of ``name`` with norm overflow colors for energy caps.

    Values **above** the display cap map to **white** for any base colormap
    (``Normalize(..., clip=False)`` + ``set_over``). Values below the display
    range also map to white, which keeps symmetric capped ΔF plots clean.
    Invalid cells use the same white background so sparse FEL regions do not
    render as a separate gray artifact class.
    """

    import matplotlib as mpl
    import matplotlib.cm as mpl_cm

    registry = getattr(mpl, "colormaps", None)
    if registry is not None:
        base = registry[name]
    else:
        base = mpl_cm.get_cmap(name)
    cmap = base.copy()
    cmap.set_bad(_FEL_BACKGROUND_COLOR)
    cmap.set_over(_FEL_BACKGROUND_COLOR)
    cmap.set_under(_FEL_BACKGROUND_COLOR)
    return cmap


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
    symmetric_zero_center: bool = False,
    colorbar_vmin: float | None = None,
    colorbar_vmax: float | None = None,
    colormap: str | None = None,
    show_colorbar: bool = True,
    marker_size: float = 60.0,
    ramachandran_equal_aspect: bool = False,
    colorbar_fraction: float = 0.046,
    colorbar_pad: float = 0.04,
    colorbar_labelsize: float | None = None,
) -> None:
    """Draw a colored scatter plot for state-wise projected values."""

    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mcolors

    cmap_name = colormap if colormap is not None else DEFAULT_FEL_COLORMAP
    x_plot = np.asarray(x, dtype=float)
    y_plot = np.asarray(y, dtype=float)
    value_plot = np.asarray(values, dtype=float)
    position_mask = np.isfinite(x_plot) & np.isfinite(y_plot)
    x_points = x_plot[position_mask]
    y_points = y_plot[position_mask]
    point_values = value_plot[position_mask]
    finite_value_mask = np.isfinite(point_values)
    cmap = _copy_colormap_for_fel_cap(cmap_name)
    colorbar_extend: str | None = None
    norm = mcolors.Normalize()
    if fel_max_energy_kj_mol is not None:
        cap = float(fel_max_energy_kj_mol)
        if symmetric_energy_cap:
            norm = mcolors.Normalize(vmin=-cap, vmax=cap, clip=False)
            colorbar_extend = "both"
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=cap, clip=False)
            colorbar_extend = "max"
    elif symmetric_zero_center and finite_value_mask.any():
        if colorbar_vmin is not None and colorbar_vmax is not None:
            norm = mcolors.Normalize(
                vmin=float(colorbar_vmin),
                vmax=float(colorbar_vmax),
                clip=True,
            )
            colorbar_extend = "both"
        else:
            absmax = float(np.nanmax(np.abs(point_values[finite_value_mask])))
            if not math.isfinite(absmax) or absmax == 0.0:
                absmax = 1.0e-15
            norm = mcolors.Normalize(vmin=-absmax, vmax=absmax, clip=False)
    elif finite_value_mask.any():
        norm.autoscale(point_values[finite_value_mask])

    if x_points.size:
        facecolors = np.tile(
            np.asarray(_FEL_BACKGROUND_COLOR, dtype=float),
            (x_points.size, 1),
        )
        if finite_value_mask.any():
            facecolors[finite_value_mask] = cmap(norm(point_values[finite_value_mask]))
        ax.scatter(
            x_points,
            y_points,
            s=marker_size,
            facecolors=facecolors,
            edgecolors="black",
            linewidths=0.3,
        )
    if ramachandran_equal_aspect:
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if finite_value_mask.any() and show_colorbar:
        mappable = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(point_values[finite_value_mask])
        colorbar = ax.figure.colorbar(
            mappable,
            ax=ax,
            extend=colorbar_extend or "neither",
            fraction=colorbar_fraction,
            pad=colorbar_pad,
        )
        colorbar.set_label(
            colorbar_label,
            fontsize=colorbar_labelsize,
        )
        if colorbar_labelsize is not None:
            colorbar.ax.tick_params(labelsize=max(colorbar_labelsize - 1.0, 6.0))


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
    colormap: str | None = None,
    symmetric_zero_center: bool = False,
    colorbar_vmin: float | None = None,
    colorbar_vmax: float | None = None,
    plot_style: str = "default",
) -> str:
    """Render a 2D surface on a histogram grid (e.g. probability or free energy)."""

    import matplotlib.colors as mcolors

    cmap_name = colormap if colormap is not None else DEFAULT_FEL_COLORMAP
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor(_FEL_BACKGROUND_COLOR)
    z_plot = np.ma.masked_invalid(np.asarray(z, dtype=float).T)
    colorbar_extend: str | None = None
    if plot_style == "contour":
        x_centers = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
        y_centers = 0.5 * (np.asarray(y_edges[:-1]) + np.asarray(y_edges[1:]))
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="xy")
        if fel_max_energy_kj_mol is not None:
            cap = float(fel_max_energy_kj_mol)
            z_plot = np.ma.masked_where(z_plot > cap, z_plot)
            levels = np.linspace(0.0, cap, 41)
            norm = mcolors.Normalize(vmin=0.0, vmax=cap, clip=False)
            cmap = _copy_colormap_for_fel_cap(cmap_name)
            colorbar_extend = "max"
        elif symmetric_zero_center:
            if colorbar_vmin is not None and colorbar_vmax is not None:
                vmin = float(colorbar_vmin)
                vmax = float(colorbar_vmax)
                colorbar_extend = "both"
            else:
                flat = np.ma.filled(z_plot, np.nan)
                absmax = float(np.nanmax(np.abs(flat)))
                if not math.isfinite(absmax) or absmax == 0.0:
                    absmax = 1.0e-15
                vmin, vmax = -absmax, absmax
            levels = np.linspace(vmin, vmax, 41)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=colorbar_extend == "both")
            cmap = _copy_colormap_for_fel_cap(cmap_name)
        else:
            flat = np.ma.filled(z_plot, np.nan)
            finite = flat[np.isfinite(flat)]
            if finite.size:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
            else:
                vmin, vmax = 0.0, 1.0
            if vmin == vmax:
                delta = max(abs(vmin) * 0.05, 1.0e-15)
                vmin -= delta
                vmax += delta
            levels = np.linspace(vmin, vmax, 41)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            cmap = _copy_colormap_for_fel_cap(cmap_name)
        image = ax.contourf(
            x_grid,
            y_grid,
            z_plot,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=colorbar_extend or "neither",
        )
    elif plot_style != "default":
        raise ValueError(f"Unknown plot_style {plot_style!r}.")
    elif fel_max_energy_kj_mol is not None:
        cap = float(fel_max_energy_kj_mol)
        z_plot = np.ma.masked_where(z_plot >= cap, z_plot)
        cmap = _copy_colormap_for_fel_cap(cmap_name)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            shading="flat",
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0.0, vmax=cap, clip=False),
            antialiased=False,
        )
        colorbar_extend = "max"
    elif symmetric_zero_center:
        if colorbar_vmin is not None and colorbar_vmax is not None:
            vmin = float(colorbar_vmin)
            vmax = float(colorbar_vmax)
            colorbar_extend = "both"
            clip = True
        else:
            flat = np.ma.filled(z_plot, np.nan)
            absmax = float(np.nanmax(np.abs(flat)))
            if not math.isfinite(absmax) or absmax == 0.0:
                absmax = 1.0e-15
            vmin, vmax = -absmax, absmax
            clip = False
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            shading="flat",
            cmap=cmap_name,
            norm=norm,
        )
    else:
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            shading="flat",
            cmap=cmap_name,
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
