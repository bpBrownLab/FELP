"""Publication-style plotting utilities for MSM bundles."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from felp.core import ConfigurationError
from felp.msm import load_bundle

from .common import (
    apply_publication_style,
    compute_weighted_histogram2d,
    import_matplotlib,
    prepare_output_dir,
    probability_surface_to_free_energy,
    save_pcolormesh_landscape,
    scatter_state_projection,
    write_json,
)


@dataclass(frozen=True)
class MSMPlotConfig:
    """Configuration for plotting pre-processing MSM bundle outputs."""

    bundle_dir: str
    output_dir: str
    state_level: str = "cluster"
    temperature_kelvin: float = 300.0
    bins: int = 100
    max_state_bars: int = 40
    max_points: int = 50000
    overwrite: bool = False

    def validate(self) -> None:
        """Validate the plotting configuration."""

        if not os.path.isdir(self.bundle_dir):
            raise ConfigurationError(f"MSM bundle directory does not exist: {self.bundle_dir}")
        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")
        if self.state_level not in {"cluster", "metastable"}:
            raise ConfigurationError(f"Unsupported state level: {self.state_level}")
        if self.temperature_kelvin <= 0.0:
            raise ConfigurationError("temperature_kelvin must be positive.")
        if self.bins < 10:
            raise ConfigurationError("bins must be at least 10.")
        if self.max_state_bars < 1:
            raise ConfigurationError("max_state_bars must be at least 1.")
        if self.max_points < 1:
            raise ConfigurationError("max_points must be at least 1.")


def generate_msm_plots(config: MSMPlotConfig) -> dict[str, object]:
    """Generate a compact set of pre-processing plots for an MSM bundle."""

    config.validate()
    bundle = load_bundle(config.bundle_dir)
    definitions = bundle.definitions_for_level(config.state_level)
    assignments = bundle.assignments[config.state_level]

    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    plt = import_matplotlib()
    apply_publication_style()

    probabilities = np.asarray(
        [float(definition.reference_probability or 0.0) for definition in definitions],
        dtype=float,
    )
    if probabilities.sum() <= 0.0:
        counts = np.bincount(
            np.concatenate(assignments, axis=0),
            minlength=len(definitions),
        ).astype(float)
        probabilities = counts / max(float(counts.sum()), 1.0)
    else:
        probabilities = probabilities / probabilities.sum()

    outputs: list[str] = []
    probability_plot = _plot_state_probabilities(
        plt=plt,
        definitions=definitions,
        probabilities=probabilities,
        output_dir=config.output_dir,
        max_state_bars=config.max_state_bars,
    )
    outputs.append(probability_plot)

    embeddings = _load_reduced_embeddings(bundle)
    if embeddings and embeddings[0].shape[1] >= 2:
        flattened_coordinates = np.concatenate(embeddings, axis=0)
        flattened_assignments = np.concatenate(assignments, axis=0)
        frame_weights = _frame_weights_from_probabilities(
            state_indices=flattened_assignments,
            state_probabilities=probabilities,
            state_count=len(definitions),
        )

        outputs.extend(
            _plot_reference_projections(
                plt=plt,
                output_dir=config.output_dir,
                coordinates=flattened_coordinates[:, 0:2],
                frame_weights=frame_weights,
                state_indices=flattened_assignments,
                bins=config.bins,
                temperature_kelvin=config.temperature_kelvin,
                max_points=config.max_points,
            )
        )

        plot_x = np.asarray(
            [_as_nan(definition.plot_x) for definition in definitions],
            dtype=float,
        )
        plot_y = np.asarray(
            [_as_nan(definition.plot_y) for definition in definitions],
            dtype=float,
        )
        if np.isfinite(plot_x).any() and np.isfinite(plot_y).any():
            outputs.append(
                _plot_state_centers(
                    plt=plt,
                    output_dir=config.output_dir,
                    plot_x=plot_x,
                    plot_y=plot_y,
                    probabilities=probabilities,
                )
            )

    summary = {
        "bundle_dir": config.bundle_dir,
        "output_dir": config.output_dir,
        "state_level": config.state_level,
        "state_count": len(definitions),
        "generated_files": outputs,
    }
    write_json(os.path.join(config.output_dir, "plot_summary.json"), summary)
    return summary


def _load_reduced_embeddings(bundle) -> tuple[np.ndarray, ...]:
    """Load reduced trajectory embeddings from a bundle."""

    trajectories = []
    index = 0
    while True:
        key = f"reduced_trajectory_{index}"
        if key not in bundle.embedding:
            break
        trajectories.append(np.asarray(bundle.embedding[key], dtype=float))
        index += 1
    return tuple(trajectories)


def _as_nan(value: float | None) -> float:
    """Convert an optional plotting coordinate into a finite-or-NaN value."""

    return float("nan") if value is None else float(value)


def _plot_state_probabilities(
    plt,
    definitions,
    probabilities: np.ndarray,
    output_dir: str,
    max_state_bars: int,
) -> str:
    """Plot the most populated states in the bundle."""

    order = np.argsort(probabilities)[::-1][:max_state_bars]
    labels = [definitions[index].label for index in order]
    values = probabilities[order]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    positions = np.arange(len(order))
    ax.barh(positions, values[::-1], color="steelblue")
    ax.set_yticks(positions, labels[::-1])
    ax.set_xlabel("Reference Probability")
    ax.set_ylabel("State")
    ax.set_title("Most Populated States")
    fig.tight_layout()

    path = os.path.join(output_dir, "state_probabilities.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_reference_projections(
    plt,
    output_dir: str,
    coordinates: np.ndarray,
    frame_weights: np.ndarray,
    state_indices: np.ndarray,
    bins: int,
    temperature_kelvin: float,
    max_points: int,
) -> list[str]:
    """Plot raw and weighted projection surfaces from reduced coordinates."""

    outputs: list[str] = []
    x = np.asarray(coordinates[:, 0], dtype=float)
    y = np.asarray(coordinates[:, 1], dtype=float)

    raw_hist, x_edges, y_edges = compute_weighted_histogram2d(x, y, bins=bins)
    weighted_hist, _, _ = compute_weighted_histogram2d(
        x,
        y,
        bins=bins,
        weights=frame_weights,
    )

    raw_prob_path = os.path.join(output_dir, "raw_probability_projection.png")
    _plot_probability_heatmap(
        plt=plt,
        path=raw_prob_path,
        histogram=raw_hist,
        x_edges=x_edges,
        y_edges=y_edges,
        title="Raw Projection Probability",
        colorbar_label="Probability",
    )
    outputs.append(raw_prob_path)

    weighted_prob_path = os.path.join(output_dir, "stationary_probability_projection.png")
    _plot_probability_heatmap(
        plt=plt,
        path=weighted_prob_path,
        histogram=weighted_hist,
        x_edges=x_edges,
        y_edges=y_edges,
        title="Stationary Projection Probability",
        colorbar_label="Probability",
    )
    outputs.append(weighted_prob_path)

    free_energy = probability_surface_to_free_energy(
        weighted_hist,
        temperature_kelvin=temperature_kelvin,
    )
    free_energy_path = os.path.join(output_dir, "reference_free_energy_landscape.png")
    _plot_probability_heatmap(
        plt=plt,
        path=free_energy_path,
        histogram=free_energy,
        x_edges=x_edges,
        y_edges=y_edges,
        title="Reference Free Energy Landscape",
        colorbar_label="Free Energy (kJ/mol)",
    )
    outputs.append(free_energy_path)

    sample_indices = _sample_indices(len(state_indices), max_points)
    sampled_coordinates = coordinates[sample_indices]
    sampled_states = state_indices[sample_indices]
    fig, ax = plt.subplots(1, 1)
    scatter = ax.scatter(
        sampled_coordinates[:, 0],
        sampled_coordinates[:, 1],
        c=sampled_states,
        cmap="tab20",
        s=6,
        alpha=0.6,
        linewidths=0.0,
    )
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")
    ax.set_title("State Assignments")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("State Index")
    fig.tight_layout()
    state_map_path = os.path.join(output_dir, "state_assignment_projection.png")
    fig.savefig(state_map_path)
    plt.close(fig)
    outputs.append(state_map_path)

    return outputs


def _plot_probability_heatmap(
    plt,
    path: str,
    histogram: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    title: str,
    colorbar_label: str,
) -> None:
    """Render a 2D histogram-like heatmap."""

    save_pcolormesh_landscape(
        plt,
        path,
        histogram,
        x_edges,
        y_edges,
        title=title,
        colorbar_label=colorbar_label,
        xlabel="Projection 1",
        ylabel="Projection 2",
    )


def _plot_state_centers(
    plt,
    output_dir: str,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    probabilities: np.ndarray,
) -> str:
    """Plot state centers colored by their stationary probabilities."""

    mask = np.isfinite(plot_x) & np.isfinite(plot_y)
    fig, ax = plt.subplots(1, 1)
    scatter_state_projection(
        ax=ax,
        x=plot_x[mask],
        y=plot_y[mask],
        values=probabilities[mask],
        colorbar_label="Reference Probability",
        title="State Centers",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "state_center_probabilities.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _frame_weights_from_probabilities(
    state_indices: np.ndarray,
    state_probabilities: np.ndarray,
    state_count: int,
) -> np.ndarray:
    """Distribute stationary state probabilities back to individual frames."""

    counts = np.bincount(state_indices, minlength=state_count).astype(float)
    per_state_weights = np.zeros(state_count, dtype=float)
    occupied_mask = counts > 0.0
    per_state_weights[occupied_mask] = (
        state_probabilities[occupied_mask] / counts[occupied_mask]
    )
    return per_state_weights[state_indices]


def _sample_indices(total_count: int, max_points: int) -> np.ndarray:
    """Subsample point indices deterministically for plotting."""

    if total_count <= max_points:
        return np.arange(total_count, dtype=int)
    return np.linspace(0, total_count - 1, max_points, dtype=int)
