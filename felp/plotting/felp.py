"""Publication-style plotting utilities for FELP output summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os

import numpy as np

from felp.core import ConfigurationError, K_BOLTZMANN_KJ_PER_MOL_K
from felp.io import load_frame_results_csv
from felp.reweighting import is_reweighting_eligible_frame

from .common import (
    apply_publication_style,
    finite_top_indices,
    free_energy_from_probabilities,
    import_matplotlib,
    parse_optional_float,
    prepare_output_dir,
    probability_surface_to_free_energy,
    read_csv_rows,
    save_pcolormesh_landscape,
    scatter_state_projection,
    write_json,
)

_KJ_PER_KCAL = 4.184


def _plot_energy_display_scale_and_label(plot_energy_unit: str) -> tuple[float, str]:
    """Scale factor (multiply kJ/mol) and axis label for plot display units."""

    if plot_energy_unit == "kcal_mol":
        return 1.0 / _KJ_PER_KCAL, "kcal/mol"
    if plot_energy_unit == "kj_mol":
        return 1.0, "kJ/mol"
    raise ConfigurationError(
        f"plot_energy_unit must be 'kj_mol' or 'kcal_mol', not {plot_energy_unit!r}."
    )


@dataclass(frozen=True)
class FELPPlotConfig:
    """Configuration for plotting post-processing FELP outputs."""

    microstate_summary_csv: str
    output_dir: str
    raw_frame_results_csv: str | None = None
    temperature_kelvin: float = 300.0
    top_states: int = 40
    fel_bins: int = 72
    fel_max_points: int = 200_000
    fel_max_energy_kj_mol: float | None = None
    """Cap for relative free-energy color scales; always interpreted as kJ/mol."""

    plot_energy_unit: str = "kj_mol"
    """Display unit for energies on plots: ``kj_mol`` (default) or ``kcal_mol``."""

    overwrite: bool = False

    def validate(self) -> None:
        """Validate the FELP plotting configuration."""

        if not os.path.isfile(self.microstate_summary_csv):
            raise ConfigurationError(
                f"Microstate summary CSV does not exist: {self.microstate_summary_csv}"
            )
        if self.raw_frame_results_csv and not os.path.isfile(self.raw_frame_results_csv):
            raise ConfigurationError(
                f"Frame results CSV does not exist: {self.raw_frame_results_csv}"
            )
        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")
        if self.temperature_kelvin <= 0.0:
            raise ConfigurationError("temperature_kelvin must be positive.")
        if self.top_states < 1:
            raise ConfigurationError("top_states must be at least 1.")
        if self.fel_bins < 10:
            raise ConfigurationError("fel_bins must be at least 10.")
        if self.fel_max_points < 1:
            raise ConfigurationError("fel_max_points must be at least 1.")
        if self.fel_max_energy_kj_mol is not None and self.fel_max_energy_kj_mol <= 0.0:
            raise ConfigurationError("fel_max_energy_kj_mol must be positive when set.")
        _plot_energy_display_scale_and_label(self.plot_energy_unit)


def generate_felp_plots(config: FELPPlotConfig) -> dict[str, object]:
    """Generate a compact set of post-processing plots for FELP outputs."""

    config.validate()
    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    plt = import_matplotlib()
    apply_publication_style()

    energy_scale, energy_unit_label = _plot_energy_display_scale_and_label(
        config.plot_energy_unit
    )
    fel_max_display = (
        float(config.fel_max_energy_kj_mol) * energy_scale
        if config.fel_max_energy_kj_mol is not None
        else None
    )

    rows = read_csv_rows(config.microstate_summary_csv)
    state_ids = [row.get("state_id", row["label"]) for row in rows]
    labels = [row["label"] for row in rows]
    reference_probabilities = np.asarray(
        [float(row["reference_probability"]) for row in rows],
        dtype=float,
    )
    target_probabilities = np.asarray(
        [float(row["target_probability"]) for row in rows],
        dtype=float,
    )
    plot_x = np.asarray(
        [_as_nan(parse_optional_float(row.get("plot_x"))) for row in rows],
        dtype=float,
    )
    plot_y = np.asarray(
        [_as_nan(parse_optional_float(row.get("plot_y"))) for row in rows],
        dtype=float,
    )

    reference_free_energy_kj_mol = free_energy_from_probabilities(
        reference_probabilities,
        temperature_kelvin=config.temperature_kelvin,
    )
    target_free_energy_kj_mol = np.asarray(
        [
            _as_nan(parse_optional_float(row.get("free_energy_kj_mol")))
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(target_free_energy_kj_mol).any():
        target_free_energy_kj_mol = free_energy_from_probabilities(
            target_probabilities,
            temperature_kelvin=config.temperature_kelvin,
        )

    occupancy_shift = target_probabilities - reference_probabilities
    kl_ref_to_target, kl_contributions = _kl_divergence(
        reference_probabilities,
        target_probabilities,
    )
    kl_target_to_ref, _ = _kl_divergence(
        target_probabilities,
        reference_probabilities,
    )
    js_divergence = _jensen_shannon_divergence(
        reference_probabilities,
        target_probabilities,
    )
    overlap_coefficient = _overlap_coefficient(
        reference_probabilities,
        target_probabilities,
    )
    summary_metadata = _load_microstate_summary_metadata(rows)

    outputs = [
        _plot_occupancy_shift(
            plt=plt,
            labels=labels,
            occupancy_shift=occupancy_shift,
            output_dir=config.output_dir,
            top_states=config.top_states,
        ),
        _plot_probability_comparison(
            plt=plt,
            reference_probabilities=reference_probabilities,
            target_probabilities=target_probabilities,
            output_dir=config.output_dir,
        ),
        _plot_kl_contributions(
            plt=plt,
            labels=labels,
            contributions=kl_contributions,
            output_dir=config.output_dir,
            top_states=config.top_states,
        ),
    ]

    if np.isfinite(plot_x).any() and np.isfinite(plot_y).any():
        outputs.extend(
            _plot_projected_state_quantities(
                plt=plt,
                plot_x=plot_x,
                plot_y=plot_y,
                reference_probabilities=reference_probabilities,
                target_probabilities=target_probabilities,
                delta_free_energy=energy_scale
                * (target_free_energy_kj_mol - reference_free_energy_kj_mol),
                output_dir=config.output_dir,
                fel_max_energy_display=fel_max_display,
                energy_unit_label=energy_unit_label,
            )
        )

    phi_centers = np.asarray(
        [_as_nan(parse_optional_float(row.get("phi_center"))) for row in rows],
        dtype=float,
    )
    psi_centers = np.asarray(
        [_as_nan(parse_optional_float(row.get("psi_center"))) for row in rows],
        dtype=float,
    )
    outputs.extend(
        _plot_phi_psi_state_landscapes(
            plt=plt,
            phi_centers=phi_centers,
            psi_centers=psi_centers,
            reference_probabilities=reference_probabilities,
            target_probabilities=target_probabilities,
            temperature_kelvin=config.temperature_kelvin,
            output_dir=config.output_dir,
            energy_scale=energy_scale,
            energy_unit_label=energy_unit_label,
            fel_max_energy_display=fel_max_display,
        )
    )

    phi_psi_projection_available = False
    if config.raw_frame_results_csv:
        frame_results = load_frame_results_csv(config.raw_frame_results_csv)
        outputs.append(
            _plot_delta_u_distribution(
                plt=plt,
                frame_results=frame_results,
                output_dir=config.output_dir,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
            )
        )
        projected_outputs = _plot_phi_psi_projected_landscapes(
            plt=plt,
            frame_results=frame_results,
            state_ids=state_ids,
            reference_probabilities=reference_probabilities,
            temperature_kelvin=config.temperature_kelvin,
            bins=config.fel_bins,
            max_points=config.fel_max_points,
            output_dir=config.output_dir,
            energy_scale=energy_scale,
            energy_unit_label=energy_unit_label,
            fel_max_energy_display=fel_max_display,
        )
        if projected_outputs:
            phi_psi_projection_available = True
            outputs.extend(projected_outputs)
    if np.isfinite(phi_centers).any() and np.isfinite(psi_centers).any():
        outputs.extend(
            _plot_discrete_microstate_ramachandran(
                plt=plt,
                phi_degrees=phi_centers,
                psi_degrees=psi_centers,
                free_energy_kj_mol=target_free_energy_kj_mol,
                target_probability=target_probabilities,
                labels=labels,
                output_dir=config.output_dir,
                temperature_kelvin=config.temperature_kelvin,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_display,
            )
        )

    summary = {
        "microstate_summary_csv": config.microstate_summary_csv,
        "raw_frame_results_csv": config.raw_frame_results_csv,
        "output_dir": config.output_dir,
        "kl_reference_to_target": float(kl_ref_to_target),
        "kl_target_to_reference": float(kl_target_to_ref),
        "jensen_shannon_divergence": float(js_divergence),
        "overlap_coefficient": float(overlap_coefficient),
        "reference_probability_source": summary_metadata.get("reference_probability_source"),
        "prior_probability_mode": summary_metadata.get("prior_probability_mode"),
        "phi_psi_frame_projection_available": phi_psi_projection_available,
        "phi_psi_frame_projection_estimator": (
            "w_ij proportional to pi_reference(i) * exp(-beta * delta_u_ij) / n_i"
        ),
        "state_reweighting_estimator": (
            "pi_target(i) proportional to "
            "pi_reference(i) * <exp(-beta * delta_u)>_i"
        ),
        "fel_max_energy_kj_mol": config.fel_max_energy_kj_mol,
        "plot_energy_unit": config.plot_energy_unit,
        "generated_files": outputs,
    }
    write_json(os.path.join(config.output_dir, "plot_summary.json"), summary)
    return summary


def _as_nan(value: float | None) -> float:
    """Convert an optional float into a finite-or-NaN plotting value."""

    return float("nan") if value is None else float(value)


def _load_microstate_summary_metadata(rows: list[dict[str, str]]) -> dict[str, object]:
    """Best-effort parse of shared per-state metadata from the summary CSV."""

    for row in rows:
        raw = row.get("metadata")
        if raw in (None, ""):
            continue
        try:
            return dict(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return {}


def _safe_probabilities(values: np.ndarray) -> np.ndarray:
    """Clip probabilities away from zero for divergence calculations."""

    return np.clip(np.asarray(values, dtype=float), 1.0e-12, None)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute KL divergence and per-state contributions."""

    safe_p = _safe_probabilities(p)
    safe_q = _safe_probabilities(q)
    contributions = safe_p * np.log(safe_p / safe_q)
    return float(np.sum(contributions)), contributions


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence between two distributions."""

    safe_p = _safe_probabilities(p)
    safe_q = _safe_probabilities(q)
    midpoint = 0.5 * (safe_p + safe_q)
    kl_pm, _ = _kl_divergence(safe_p, midpoint)
    kl_qm, _ = _kl_divergence(safe_q, midpoint)
    return 0.5 * (kl_pm + kl_qm)


def _overlap_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the bounded discrete overlap coefficient between distributions."""

    return float(np.minimum(np.asarray(p, dtype=float), np.asarray(q, dtype=float)).sum())


def _plot_occupancy_shift(
    plt,
    labels: list[str],
    occupancy_shift: np.ndarray,
    output_dir: str,
    top_states: int,
) -> str:
    """Plot the states with the largest occupancy changes."""

    indices = finite_top_indices(occupancy_shift, top_states)
    ordered = indices[np.argsort(occupancy_shift[indices])]
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    positions = np.arange(len(ordered))
    colors = ["firebrick" if occupancy_shift[index] > 0 else "steelblue" for index in ordered]
    ax.barh(positions, occupancy_shift[ordered], color=colors)
    ax.set_yticks(positions, [labels[index] for index in ordered])
    ax.set_xlabel("Target - Reference Probability")
    ax.set_ylabel("State")
    ax.set_title("Largest Occupancy Shifts")
    fig.tight_layout()

    path = os.path.join(output_dir, "occupancy_shifts.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_probability_comparison(
    plt,
    reference_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    output_dir: str,
) -> str:
    """Plot reference vs target state probabilities."""

    fig, ax = plt.subplots(1, 1)
    ax.scatter(reference_probabilities, target_probabilities, c="black", s=30, alpha=0.7)
    diagonal_limit = max(
        float(np.max(reference_probabilities)) if reference_probabilities.size else 0.0,
        float(np.max(target_probabilities)) if target_probabilities.size else 0.0,
        1.0e-12,
    )
    ax.plot([0.0, diagonal_limit], [0.0, diagonal_limit], linestyle="--", color="gray")
    ax.set_xlabel("Reference Probability")
    ax.set_ylabel("Target Probability")
    ax.set_title("State Occupancy Comparison")
    fig.tight_layout()

    path = os.path.join(output_dir, "occupancy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_kl_contributions(
    plt,
    labels: list[str],
    contributions: np.ndarray,
    output_dir: str,
    top_states: int,
) -> str:
    """Plot the largest state-wise KL divergence contributions."""

    indices = finite_top_indices(contributions, top_states)
    ordered = indices[np.argsort(contributions[indices])]
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    positions = np.arange(len(ordered))
    ax.barh(positions, contributions[ordered], color="mediumpurple")
    ax.set_yticks(positions, [labels[index] for index in ordered])
    ax.set_xlabel("KL Contribution")
    ax.set_ylabel("State")
    ax.set_title("Largest KL Divergence Contributions")
    fig.tight_layout()

    path = os.path.join(output_dir, "kl_divergence_contributions.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_projected_state_quantities(
    plt,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    reference_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    delta_free_energy: np.ndarray,
    output_dir: str,
    *,
    fel_max_energy_display: float | None,
    energy_unit_label: str,
) -> list[str]:
    """Plot state quantities projected onto reduced-space coordinates."""

    mask = np.isfinite(plot_x) & np.isfinite(plot_y)
    outputs: list[str] = []
    for values, title, colorbar_label, filename, cap_delta in (
        (
            reference_probabilities,
            "Reference State Probabilities",
            "Reference Probability",
            "reference_probability_projection.png",
            False,
        ),
        (
            target_probabilities,
            "Target State Probabilities",
            "Target Probability",
            "target_probability_projection.png",
            False,
        ),
        (
            delta_free_energy,
            "Delta Free Energy",
            f"Delta Free Energy ({energy_unit_label})",
            "delta_free_energy_projection.png",
            True,
        ),
    ):
        fig, ax = plt.subplots(1, 1)
        scatter_kwargs: dict = {
            "ax": ax,
            "x": plot_x[mask],
            "y": plot_y[mask],
            "values": values[mask],
            "colorbar_label": colorbar_label,
            "title": title,
        }
        if cap_delta and fel_max_energy_display is not None:
            scatter_kwargs["fel_max_energy_kj_mol"] = fel_max_energy_display
            scatter_kwargs["symmetric_energy_cap"] = True
        scatter_state_projection(**scatter_kwargs)
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        outputs.append(path)
    return outputs


def _plot_delta_u_distribution(
    plt,
    frame_results,
    output_dir: str,
    *,
    energy_scale: float,
    energy_unit_label: str,
) -> str:
    """Plot the distribution of per-frame deltaU values."""

    delta_values: list[float] = []
    for result in frame_results:
        if not is_reweighting_eligible_frame(result):
            continue
        delta_values.append(float(result.delta_u_kj_mol) * energy_scale)
    delta_u = np.asarray(delta_values, dtype=float)

    fig, ax = plt.subplots(1, 1)
    if delta_u.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite ΔU samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        ax.hist(delta_u, bins=40, color="slategray", alpha=0.85)
    ax.set_xlabel(f"Delta U ({energy_unit_label})")
    ax.set_ylabel("Frame Count")
    ax.set_title("Per-Frame Delta U Distribution")
    fig.tight_layout()

    path = os.path.join(output_dir, "delta_u_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _regular_centers_to_edges(
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Infer regularly spaced histogram edges from state-center coordinates."""

    finite = np.asarray(sorted(np.unique(centers[np.isfinite(centers)])), dtype=float)
    if finite.size == 0:
        return None
    if finite.size == 1:
        width = 360.0
    else:
        diffs = np.diff(finite)
        if not np.allclose(diffs, diffs[0], atol=1.0e-6, rtol=1.0e-6):
            return None
        width = float(diffs[0])
    edges = np.concatenate(([finite[0] - width / 2.0], finite + width / 2.0))
    return finite, edges


def _build_regular_state_grid(
    phi_centers: np.ndarray,
    psi_centers: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a rectangular state grid when the summary rows form a complete mesh."""

    if not (
        phi_centers.size == psi_centers.size == values.size
        and np.isfinite(phi_centers).all()
        and np.isfinite(psi_centers).all()
    ):
        return None
    phi_grid = _regular_centers_to_edges(phi_centers)
    psi_grid = _regular_centers_to_edges(psi_centers)
    if phi_grid is None or psi_grid is None:
        return None
    phi_unique, phi_edges = phi_grid
    psi_unique, psi_edges = psi_grid
    if phi_unique.size * psi_unique.size != values.size:
        return None

    grid = np.full((phi_unique.size, psi_unique.size), np.nan, dtype=float)
    phi_index = {float(value): index for index, value in enumerate(phi_unique)}
    psi_index = {float(value): index for index, value in enumerate(psi_unique)}
    for phi, psi, value in zip(phi_centers, psi_centers, values):
        location = (phi_index[float(phi)], psi_index[float(psi)])
        if not math.isnan(grid[location]):
            return None
        grid[location] = float(value)
    if np.isnan(grid).any():
        return None
    return grid, phi_edges, psi_edges


def _subsample_indices(total: int, max_points: int) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, max_points, dtype=int)


def _plot_phi_psi_state_landscapes(
    plt,
    *,
    phi_centers: np.ndarray,
    psi_centers: np.ndarray,
    reference_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    temperature_kelvin: float,
    output_dir: str,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
) -> list[str]:
    """Build Ramachandran-like free energy surfaces from discrete state probabilities."""

    reference_grid_data = _build_regular_state_grid(
        phi_centers,
        psi_centers,
        reference_probabilities,
    )
    target_grid_data = _build_regular_state_grid(
        phi_centers,
        psi_centers,
        target_probabilities,
    )
    if reference_grid_data is None or target_grid_data is None:
        return []
    reference_grid, x_edges, y_edges = reference_grid_data
    target_grid, _, _ = target_grid_data
    reference_fe = energy_scale * probability_surface_to_free_energy(
        reference_grid, temperature_kelvin
    )
    reweighted_fe = energy_scale * probability_surface_to_free_energy(
        target_grid, temperature_kelvin
    )

    outputs: list[str] = []
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, "reference_phi_psi_state_free_energy_landscape.png"),
            reference_fe,
            x_edges,
            y_edges,
            title="Reference microstate free energy (phi, psi)",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=r"$\phi$ (deg)",
            ylabel=r"$\psi$ (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, "reweighted_phi_psi_state_free_energy_landscape.png"),
            reweighted_fe,
            x_edges,
            y_edges,
            title="Reweighted microstate free energy (phi, psi)",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=r"$\phi$ (deg)",
            ylabel=r"$\psi$ (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
        )
    )
    return outputs


def _sample_state_projection_points(
    state_samples: dict[str, list[tuple[float, float, float]]],
    max_points: int,
) -> dict[str, list[tuple[float, float, float]]]:
    """Deterministically subsample projection points while preserving state coverage."""

    total_count = sum(len(values) for values in state_samples.values())
    if total_count <= max_points:
        return state_samples

    sample_budget = max(max_points, len(state_samples))
    sampled: dict[str, list[tuple[float, float, float]]] = {}
    for state_id, values in state_samples.items():
        target_count = max(
            1,
            int(round(len(values) * sample_budget / total_count)),
        )
        target_count = min(target_count, len(values))
        sampled[state_id] = [values[index] for index in _subsample_indices(len(values), target_count)]
    return sampled


def _collect_phi_psi_projection_samples(
    *,
    frame_results,
    reference_probability_by_state: dict[str, float],
    temperature_kelvin: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect exact phi/psi frame samples with state-consistent reference weights."""

    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    eligible_delta_values: list[float] = []
    for result in frame_results:
        if not is_reweighting_eligible_frame(result):
            continue
        if result.microstate_id not in reference_probability_by_state:
            continue
        if (
            result.phi_degrees is None
            or result.psi_degrees is None
            or not math.isfinite(result.phi_degrees)
            or not math.isfinite(result.psi_degrees)
        ):
            continue
        grouped[result.microstate_id].append(
            (
                float(result.phi_degrees),
                float(result.psi_degrees),
                float(result.delta_u_kj_mol),
            )
        )
        eligible_delta_values.append(float(result.delta_u_kj_mol))

    if not eligible_delta_values:
        return None

    missing_positive_prior = [
        state_id
        for state_id, probability in reference_probability_by_state.items()
        if probability > 0.0 and not grouped.get(state_id)
    ]
    if missing_positive_prior:
        return None

    sampled = _sample_state_projection_points(grouped, max_points)
    beta = 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * temperature_kelvin)
    delta_shift = min(eligible_delta_values)

    phi_values: list[float] = []
    psi_values: list[float] = []
    reference_weights: list[float] = []
    reweighted_weights: list[float] = []
    for state_id, samples in sampled.items():
        sample_count = len(samples)
        if sample_count == 0:
            continue
        reference_probability = reference_probability_by_state.get(state_id, 0.0)
        for phi, psi, delta_u in samples:
            phi_values.append(phi)
            psi_values.append(psi)
            reference_weights.append(reference_probability / sample_count)
            reweighted_weights.append(
                reference_probability
                * math.exp(-beta * (delta_u - delta_shift))
                / sample_count
            )

    if not phi_values:
        return None

    return (
        np.asarray(phi_values, dtype=float),
        np.asarray(psi_values, dtype=float),
        np.asarray(reference_weights, dtype=float),
        np.asarray(reweighted_weights, dtype=float),
    )


def _plot_phi_psi_projected_landscapes(
    plt,
    *,
    frame_results,
    state_ids: list[str],
    reference_probabilities: np.ndarray,
    temperature_kelvin: float,
    bins: int,
    max_points: int,
    output_dir: str,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
) -> list[str]:
    """Project exact phi/psi frame samples using prior-consistent frame weights."""

    reference_probability_by_state = {
        state_id: float(probability)
        for state_id, probability in zip(state_ids, reference_probabilities)
    }
    sampled = _collect_phi_psi_projection_samples(
        frame_results=frame_results,
        reference_probability_by_state=reference_probability_by_state,
        temperature_kelvin=temperature_kelvin,
        max_points=max_points,
    )
    if sampled is None:
        return []

    phi, psi, reference_weights, reweighted_weights = sampled
    ram_range = ((-180.0, 180.0), (-180.0, 180.0))
    reference_hist, x_edges, y_edges = np.histogram2d(
        phi,
        psi,
        bins=bins,
        range=ram_range,
        weights=reference_weights,
    )
    reweighted_hist, _, _ = np.histogram2d(
        phi,
        psi,
        bins=bins,
        range=ram_range,
        weights=reweighted_weights,
    )
    if reference_hist.sum() > 0.0:
        reference_hist = reference_hist / float(reference_hist.sum())
    if reweighted_hist.sum() > 0.0:
        reweighted_hist = reweighted_hist / float(reweighted_hist.sum())
    reference_fe = energy_scale * probability_surface_to_free_energy(
        reference_hist, temperature_kelvin
    )
    reweighted_fe = energy_scale * probability_surface_to_free_energy(
        reweighted_hist, temperature_kelvin
    )

    outputs: list[str] = []
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, "reference_phi_psi_free_energy_landscape.png"),
            reference_fe,
            x_edges,
            y_edges,
            title="Reference projected free energy (phi, psi)",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=r"$\phi$ (deg)",
            ylabel=r"$\psi$ (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, "reweighted_phi_psi_free_energy_landscape.png"),
            reweighted_fe,
            x_edges,
            y_edges,
            title="Reweighted projected free energy (phi, psi)",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=r"$\phi$ (deg)",
            ylabel=r"$\psi$ (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
        )
    )
    return outputs


def _plot_discrete_microstate_ramachandran(
    plt,
    *,
    phi_degrees: np.ndarray,
    psi_degrees: np.ndarray,
    free_energy_kj_mol: np.ndarray,
    target_probability: np.ndarray,
    labels: list[str],
    output_dir: str,
    temperature_kelvin: float,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
) -> list[str]:
    """Scatter microstate centers on (phi, psi) colored by reweighted free energy."""

    mask = np.isfinite(phi_degrees) & np.isfinite(psi_degrees)
    if int(mask.sum()) < 1:
        return []

    fe = energy_scale * np.asarray(free_energy_kj_mol, dtype=float)
    if not np.isfinite(fe[mask]).any():
        values = energy_scale * free_energy_from_probabilities(
            np.asarray(target_probability, dtype=float),
            temperature_kelvin=temperature_kelvin,
        )
        colorbar_label = f"Relative free energy from P_target ({energy_unit_label})"
        title = "Microstate centers (phi, psi)"
    else:
        values = fe
        colorbar_label = f"Relative free energy ({energy_unit_label})"
        title = "Microstate centers (phi, psi)"

    fig, ax = plt.subplots(1, 1)
    scatter_kwargs: dict = {
        "ax": ax,
        "x": phi_degrees[mask],
        "y": psi_degrees[mask],
        "values": values[mask],
        "colorbar_label": colorbar_label,
        "title": title,
        "xlabel": r"$\phi$ center (deg)",
        "ylabel": r"$\psi$ center (deg)",
    }
    if fel_max_energy_display is not None:
        scatter_kwargs["fel_max_energy_kj_mol"] = fel_max_energy_display
    scatter_state_projection(**scatter_kwargs)
    if int(mask.sum()) <= 24:
        for index in np.flatnonzero(mask):
            ax.annotate(
                labels[index],
                (phi_degrees[index], psi_degrees[index]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.85,
            )
    fig.tight_layout()
    path = os.path.join(output_dir, "discrete_microstate_free_energy_ramachandran.png")
    fig.savefig(path)
    plt.close(fig)
    return [path]
