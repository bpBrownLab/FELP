"""Publication-style plotting utilities for FELP output summaries."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from felp.core import ConfigurationError

from .common import (
    apply_publication_style,
    finite_top_indices,
    free_energy_from_probabilities,
    import_matplotlib,
    parse_optional_float,
    prepare_output_dir,
    read_csv_rows,
    scatter_state_projection,
    write_json,
)


@dataclass(frozen=True)
class FELPPlotConfig:
    """Configuration for plotting post-processing FELP outputs."""

    microstate_summary_csv: str
    output_dir: str
    raw_frame_results_csv: str | None = None
    temperature_kelvin: float = 300.0
    top_states: int = 40
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


def generate_felp_plots(config: FELPPlotConfig) -> dict[str, object]:
    """Generate a compact set of post-processing plots for FELP outputs."""

    config.validate()
    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    plt = import_matplotlib()
    apply_publication_style()

    rows = read_csv_rows(config.microstate_summary_csv)
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

    reference_free_energy = free_energy_from_probabilities(
        reference_probabilities,
        temperature_kelvin=config.temperature_kelvin,
    )
    target_free_energy = np.asarray(
        [
            _as_nan(parse_optional_float(row.get("free_energy_kj_mol")))
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(target_free_energy).any():
        target_free_energy = free_energy_from_probabilities(
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
                delta_free_energy=target_free_energy - reference_free_energy,
                output_dir=config.output_dir,
            )
        )

    if config.raw_frame_results_csv:
        outputs.append(
            _plot_delta_u_distribution(
                plt=plt,
                raw_frame_results_csv=config.raw_frame_results_csv,
                output_dir=config.output_dir,
            )
        )

    summary = {
        "microstate_summary_csv": config.microstate_summary_csv,
        "raw_frame_results_csv": config.raw_frame_results_csv,
        "output_dir": config.output_dir,
        "kl_reference_to_target": float(kl_ref_to_target),
        "kl_target_to_reference": float(kl_target_to_ref),
        "jensen_shannon_divergence": float(js_divergence),
        "generated_files": outputs,
    }
    write_json(os.path.join(config.output_dir, "plot_summary.json"), summary)
    return summary


def _as_nan(value: float | None) -> float:
    """Convert an optional float into a finite-or-NaN plotting value."""

    return float("nan") if value is None else float(value)


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
) -> list[str]:
    """Plot state quantities projected onto reduced-space coordinates."""

    mask = np.isfinite(plot_x) & np.isfinite(plot_y)
    outputs: list[str] = []
    for values, title, colorbar_label, filename in (
        (
            reference_probabilities,
            "Reference State Probabilities",
            "Reference Probability",
            "reference_probability_projection.png",
        ),
        (
            target_probabilities,
            "Target State Probabilities",
            "Target Probability",
            "target_probability_projection.png",
        ),
        (
            delta_free_energy,
            "Delta Free Energy",
            "Delta Free Energy (kJ/mol)",
            "delta_free_energy_projection.png",
        ),
    ):
        fig, ax = plt.subplots(1, 1)
        scatter_state_projection(
            ax=ax,
            x=plot_x[mask],
            y=plot_y[mask],
            values=values[mask],
            colorbar_label=colorbar_label,
            title=title,
        )
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        outputs.append(path)
    return outputs


def _plot_delta_u_distribution(
    plt,
    raw_frame_results_csv: str,
    output_dir: str,
) -> str:
    """Plot the distribution of per-frame deltaU values."""

    rows = read_csv_rows(raw_frame_results_csv)
    delta_u = np.asarray([float(row["delta_u_kj_mol"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(1, 1)
    ax.hist(delta_u, bins=40, color="slategray", alpha=0.85)
    ax.set_xlabel("Delta U (kJ/mol)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Per-Frame Delta U Distribution")
    fig.tight_layout()

    path = os.path.join(output_dir, "delta_u_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path
