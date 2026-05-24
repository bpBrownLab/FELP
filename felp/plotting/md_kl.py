"""Standalone KL plots comparing MD trajectory data to FELP reweighted data."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import os

import numpy as np

from felp.core import (
    ConfigurationError,
    DependencyError,
    K_BOLTZMANN_KJ_PER_MOL_K,
    ReweightingQCConfig,
)
from felp.io import load_frame_results_csv
from felp.msm.phi_psi_spec import _target_phi_psi_indices

from .common import (
    DEFAULT_FEL_COLORMAP,
    DEFAULT_KL_DISPLAY_THRESHOLD,
    FELP_KL_DIVERGENCE_COLORMAP,
    apply_publication_style,
    free_energy_from_probabilities,
    prepare_kl_display_grid,
    resolve_kl_colorbar_limits,
    import_matplotlib,
    prepare_output_dir,
    probability_surface_to_free_energy,
    read_csv_rows,
    save_pcolormesh_landscape,
    scatter_state_projection,
    validate_registered_colormap,
    write_json,
)
from .frame_projection_weights import (
    frame_projection_weight_detail,
    load_frame_weights_csv,
    resolve_frame_weights_csv_path,
    state_reweight_factors_from_summary_rows,
)
from .felp import (
    _axis_range,
    _collect_cv_projection_samples,
    _excluded_frame_indices_by_state,
    _filter_qc_excluded_frame_results,
    _load_microstate_summary_metadata,
    _summary_weight_treatment,
    _surface_divergence_summary,
)

_KJ_PER_KCAL = 4.184


@dataclass(frozen=True)
class MDKLPlotConfig:
    """Configuration for MD-vs-reweighted probability-surface KL plots."""

    output_dir: str
    topology_file: str
    trajectory_files: tuple[tuple[str, ...], ...]
    reweighted_surface_csv: str | None = None
    raw_frame_results_csv: str | None = None
    frame_weights_csv: str | None = None
    """Optional frame_weights.csv beside the microstate summary / reweight output."""

    felp_output_dir: str | None = None
    temperature_kelvin: float = 300.0
    phi_psi_residue_number: int | None = None
    phi_psi_residue_name: str | None = None
    phi_psi_selection: str | None = None
    stride: int = 1
    frame_start: int = 0
    frame_stop: int | None = None
    chunk_size: int = 1024
    fel_bins: int = 180
    fel_max_points: int = 200_000
    fel_max_energy_kj_mol: float | None = None
    plot_energy_unit: str = "kj_mol"
    fel_colormap: str = DEFAULT_FEL_COLORMAP
    weight_treatment: str | None = None
    kl_scale_by_kbt: bool = False
    kl_energy_unit: str = "kcal_mol"
    kl_display_threshold: float | None = DEFAULT_KL_DISPLAY_THRESHOLD
    """Plot |KL cell| below this (in display units) as zero; use 0 to disable."""
    kl_colorbar_vmin: float | None = None
    kl_colorbar_vmax: float | None = None
    """Optional fixed KL heatmap limits (display units; symmetric if only one is set)."""
    probability_colormap: str | None = None
    divergence_colormap: str | None = None
    overwrite: bool = False

    def validate(self) -> None:
        """Validate and normalize paths/options."""

        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")
        if not os.path.isfile(self.topology_file):
            raise ConfigurationError(f"Topology file does not exist: {self.topology_file}")
        if not self.trajectory_files:
            raise ConfigurationError("At least one trajectory file is required.")
        for group in self.trajectory_files:
            if not group:
                raise ConfigurationError("Trajectory groups may not be empty.")
            for trajectory_file in group:
                if not os.path.isfile(trajectory_file):
                    raise ConfigurationError(
                        f"Trajectory file does not exist: {trajectory_file}"
                    )
        if self.stride < 1:
            raise ConfigurationError("stride must be at least 1.")
        if self.frame_start < 0:
            raise ConfigurationError("frame_start must be non-negative.")
        if self.frame_stop is not None and self.frame_stop <= self.frame_start:
            raise ConfigurationError("frame_stop must be greater than frame_start.")
        if self.chunk_size < 1:
            raise ConfigurationError("chunk_size must be at least 1.")
        if self.temperature_kelvin <= 0.0:
            raise ConfigurationError("temperature_kelvin must be positive.")
        if self.phi_psi_selection is None and self.phi_psi_residue_number is None:
            raise ConfigurationError(
                "Provide either --phi-psi-selection or --phi-psi-residue-number."
            )

        reweighted_surface = self.resolved_reweighted_surface_csv()
        if not os.path.isfile(reweighted_surface):
            raise ConfigurationError(
                f"Reweighted surface CSV does not exist: {reweighted_surface}"
            )
        if self.raw_frame_results_csv and not os.path.isfile(self.raw_frame_results_csv):
            raise ConfigurationError(
                f"Frame results CSV does not exist: {self.raw_frame_results_csv}"
            )
        if self.fel_bins < 10:
            raise ConfigurationError("fel_bins must be at least 10.")
        if self.fel_max_points < 1:
            raise ConfigurationError("fel_max_points must be at least 1.")
        if self.fel_max_energy_kj_mol is not None and self.fel_max_energy_kj_mol <= 0.0:
            raise ConfigurationError("fel_max_energy_kj_mol must be positive when set.")
        _plot_energy_display_scale_and_label(self.plot_energy_unit)
        object.__setattr__(
            self,
            "fel_colormap",
            validate_registered_colormap(self.fel_colormap),
        )
        if self.kl_energy_unit not in {"kj_mol", "kcal_mol"}:
            raise ConfigurationError("kl_energy_unit must be 'kj_mol' or 'kcal_mol'.")
        if self.weight_treatment is not None:
            object.__setattr__(
                self,
                "weight_treatment",
                ReweightingQCConfig.normalize_weight_treatment(self.weight_treatment),
            )

        if self.probability_colormap is not None:
            object.__setattr__(
                self,
                "probability_colormap",
                validate_registered_colormap(self.probability_colormap),
            )
        div_src = self.divergence_colormap or FELP_KL_DIVERGENCE_COLORMAP
        object.__setattr__(
            self,
            "divergence_colormap",
            validate_registered_colormap(div_src),
        )
        if self.kl_display_threshold is not None:
            threshold = float(self.kl_display_threshold)
            if threshold < 0.0 or not math.isfinite(threshold):
                raise ConfigurationError(
                    "kl_display_threshold must be a non-negative finite float."
                )
            object.__setattr__(self, "kl_display_threshold", threshold)
        resolve_kl_colorbar_limits(self.kl_colorbar_vmin, self.kl_colorbar_vmax)

    def resolved_reweighted_surface_csv(self) -> str:
        """Return the CSV path containing reweighted target probabilities."""

        if self.reweighted_surface_csv:
            return self.reweighted_surface_csv
        if self.felp_output_dir:
            if self.raw_frame_results_csv:
                summary_path = os.path.join(self.felp_output_dir, "microstate_summary.csv")
                if os.path.isfile(summary_path):
                    return summary_path
            return os.path.join(self.felp_output_dir, "reweighted_surface.csv")
        raise ConfigurationError(
            "Provide --reweighted-surface or --felp-output-dir for the FELP output."
        )


def generate_md_kl_plot(config: MDKLPlotConfig) -> dict[str, object]:
    """Generate standalone MD-vs-reweighted KL plots."""

    config.validate()
    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    plt = import_matplotlib()
    apply_publication_style()

    summary_csv = config.resolved_reweighted_surface_csv()
    state_discrete_fe_available = _state_summary_supports_discrete_fe_plot(summary_csv)
    if config.raw_frame_results_csv:
        try:
            (
                reweighted_surface,
                x_edges,
                y_edges,
                surface_metadata,
            ) = _load_regular_reweighted_probability_surface(summary_csv)
        except ConfigurationError:
            (
                reweighted_surface,
                x_edges,
                y_edges,
                surface_metadata,
            ) = _load_frame_projected_reweighted_surface(config)
            surface_metadata = {
                **surface_metadata,
                "reweighted_kl_probability_surface_source": "frame_results",
                "reweighted_display_probability_surface_source": "frame_results",
                "raw_frame_results_used_for_kl": True,
            }
        else:
            surface_metadata = {
                **surface_metadata,
                "reweighted_kl_probability_surface_source": "regular_surface_csv",
                "reweighted_display_probability_surface_source": "regular_surface_csv",
                "raw_frame_results_used_for_kl": False,
            }
    else:
        (
            reweighted_surface,
            x_edges,
            y_edges,
            surface_metadata,
        ) = _load_regular_reweighted_probability_surface(summary_csv)
        surface_metadata = {
            **surface_metadata,
            "reweighted_kl_probability_surface_source": "regular_surface_csv",
            "reweighted_display_probability_surface_source": "regular_surface_csv",
            "raw_frame_results_used_for_kl": False,
        }
    phi_degrees, psi_degrees = _load_md_phi_psi_degrees(config)
    md_surface, _, _ = np.histogram2d(
        phi_degrees,
        psi_degrees,
        bins=(x_edges, y_edges),
    )
    if md_surface.sum() > 0.0:
        md_surface = md_surface / float(md_surface.sum())
    else:
        raise ConfigurationError("No MD frames fell inside the inferred phi/psi grid.")

    energy_scale, energy_unit_label = _plot_energy_display_scale_and_label(
        config.plot_energy_unit
    )
    fel_max_energy_display = (
        float(config.fel_max_energy_kj_mol) * energy_scale
        if config.fel_max_energy_kj_mol is not None
        else None
    )
    md_free_energy = energy_scale * probability_surface_to_free_energy(
        md_surface,
        config.temperature_kelvin,
    )
    reweighted_free_energy = energy_scale * probability_surface_to_free_energy(
        reweighted_surface,
        config.temperature_kelvin,
    )
    divergence_metadata, kl_grid = _surface_divergence_summary(
        md_surface,
        reweighted_surface,
    )
    kbt_energy_scale, kbt_energy_unit_label = _kl_energy_display_scale_and_label(
        config.kl_energy_unit
    )
    kl_display_grid, kl_colorbar_label, kl_display_metadata = prepare_kl_display_grid(
        kl_grid,
        temperature_kelvin=config.temperature_kelvin,
        scale_by_kbt=config.kl_scale_by_kbt,
        kbt_energy_scale=kbt_energy_scale,
        kbt_energy_unit_label=kbt_energy_unit_label,
        display_threshold=config.kl_display_threshold,
        probability_labels=("p_MD", "p_rw"),
    )
    kl_limits = resolve_kl_colorbar_limits(
        config.kl_colorbar_vmin,
        config.kl_colorbar_vmax,
    )
    summary_metadata = {
        "md_frame_count": int(phi_degrees.size),
        "reweighted_surface_csv": config.resolved_reweighted_surface_csv(),
        "raw_frame_results_csv": config.raw_frame_results_csv,
        "topology_file": config.topology_file,
        "trajectory_files": [list(group) for group in config.trajectory_files],
        "phi_psi_residue_number": config.phi_psi_residue_number,
        "phi_psi_residue_name": config.phi_psi_residue_name,
        "phi_psi_selection": config.phi_psi_selection,
        "stride": int(config.stride),
        "frame_start": int(config.frame_start),
        "frame_stop": config.frame_stop,
        "temperature_kelvin": float(config.temperature_kelvin),
        "fel_bins": int(config.fel_bins),
        "fel_max_points": int(config.fel_max_points),
        "fel_colormap": config.fel_colormap,
        "fel_max_energy_kj_mol": config.fel_max_energy_kj_mol,
        "plot_energy_unit": config.plot_energy_unit,
        "plot_energy_scale": float(energy_scale),
        "kl_scale_by_kbt": bool(config.kl_scale_by_kbt),
        "kl_energy_unit": config.kl_energy_unit if config.kl_scale_by_kbt else None,
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        **surface_metadata,
        **kl_display_metadata,
        **(
            {
                "kl_colorbar_vmin": float(kl_limits[0]),
                "kl_colorbar_vmax": float(kl_limits[1]),
            }
            if kl_limits is not None
            else {}
        ),
        "kl_md_to_reweighted": divergence_metadata[
            "kl_pre_reweighted_to_reweighted_projected_fel"
        ],
        "kbt_kl_md_to_reweighted_kj_mol": (
            K_BOLTZMANN_KJ_PER_MOL_K
            * float(config.temperature_kelvin)
            * divergence_metadata["kl_pre_reweighted_to_reweighted_projected_fel"]
        ),
        "kbt_kl_md_to_reweighted_kcal_mol": (
            K_BOLTZMANN_KJ_PER_MOL_K
            * float(config.temperature_kelvin)
            * divergence_metadata["kl_pre_reweighted_to_reweighted_projected_fel"]
            / _KJ_PER_KCAL
        ),
        "kl_reweighted_to_md": divergence_metadata[
            "kl_reweighted_to_pre_reweighted_projected_fel"
        ],
        "kbt_kl_reweighted_to_md_kj_mol": (
            K_BOLTZMANN_KJ_PER_MOL_K
            * float(config.temperature_kelvin)
            * divergence_metadata["kl_reweighted_to_pre_reweighted_projected_fel"]
        ),
        "kbt_kl_reweighted_to_md_kcal_mol": (
            K_BOLTZMANN_KJ_PER_MOL_K
            * float(config.temperature_kelvin)
            * divergence_metadata["kl_reweighted_to_pre_reweighted_projected_fel"]
            / _KJ_PER_KCAL
        ),
        "jensen_shannon_md_reweighted": divergence_metadata[
            "jensen_shannon_projected_fel"
        ],
        "overlap_coefficient_md_reweighted": divergence_metadata[
            "projected_fel_overlap_coefficient"
        ],
        "kl_grid_definition": (
            "0.5 * [p_MD * log(p_MD / p_reweighted) - "
            "p_reweighted * log(p_reweighted / p_MD)]"
        ),
    }

    probability_cmap = config.probability_colormap
    outputs: list[str] = []
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(config.output_dir, "md_phi_psi_probability.png"),
            md_surface,
            x_edges,
            y_edges,
            title="MD probability",
            colorbar_label="Probability",
            xlabel="phi (deg)",
            ylabel="psi (deg)",
            colormap=probability_cmap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(config.output_dir, "md_phi_psi_free_energy_landscape.png"),
            md_free_energy,
            x_edges,
            y_edges,
            title="MD reference free energy",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel="phi (deg)",
            ylabel="psi (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=config.fel_colormap,
        )
    )
    outputs.append(
        _plot_discrete_phi_psi_values(
            plt,
            os.path.join(
                config.output_dir,
                "md_discrete_microstate_free_energy_ramachandran.png",
            ),
            md_free_energy,
            x_edges,
            y_edges,
            title="MD discrete free energy",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            colormap=config.fel_colormap,
            fel_max_energy_display=fel_max_energy_display,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(config.output_dir, "reweighted_phi_psi_probability.png"),
            reweighted_surface,
            x_edges,
            y_edges,
            title="Reweighted probability",
            colorbar_label="Probability",
            xlabel="phi (deg)",
            ylabel="psi (deg)",
            colormap=probability_cmap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(
                config.output_dir,
                "reweighted_phi_psi_free_energy_landscape.png",
            ),
            reweighted_free_energy,
            x_edges,
            y_edges,
            title="Reweighted free energy",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel="phi (deg)",
            ylabel="psi (deg)",
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=config.fel_colormap,
        )
    )
    if state_discrete_fe_available:
        outputs.append(
            _plot_reweighted_discrete_from_state_summary(
                plt,
                os.path.join(
                    config.output_dir,
                    "reweighted_discrete_microstate_free_energy_ramachandran.png",
                ),
                summary_csv=summary_csv,
                temperature_kelvin=config.temperature_kelvin,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_energy_display,
                colormap=config.fel_colormap,
            )
        )
    else:
        outputs.append(
            _plot_discrete_phi_psi_values(
                plt,
                os.path.join(
                    config.output_dir,
                    "reweighted_discrete_microstate_free_energy_ramachandran.png",
                ),
                reweighted_free_energy,
                x_edges,
                y_edges,
                title="Reweighted discrete free energy",
                colorbar_label=f"Relative free energy ({energy_unit_label})",
                colormap=config.fel_colormap,
                fel_max_energy_display=fel_max_energy_display,
            )
        )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(config.output_dir, "kl_md_vs_reweighted_phi_psi.png"),
            kl_display_grid,
            x_edges,
            y_edges,
            title="MD vs reweighted KL imbalance",
            colorbar_label=kl_colorbar_label,
            xlabel="phi (deg)",
            ylabel="psi (deg)",
            colormap=config.divergence_colormap,
            symmetric_zero_center=True,
            colorbar_vmin=kl_limits[0] if kl_limits is not None else None,
            colorbar_vmax=kl_limits[1] if kl_limits is not None else None,
        )
    )
    outputs.append(
        _plot_discrete_phi_psi_values(
            plt,
            os.path.join(
                config.output_dir,
                "kl_discrete_microstate_md_vs_reweighted_ramachandran.png",
            ),
            kl_display_grid,
            x_edges,
            y_edges,
            title="MD vs reweighted discrete KL imbalance",
            colorbar_label=kl_colorbar_label,
            colormap=config.divergence_colormap,
            symmetric_zero_center=True,
            colorbar_vmin=kl_limits[0] if kl_limits is not None else None,
            colorbar_vmax=kl_limits[1] if kl_limits is not None else None,
        )
    )

    md_surface_path = os.path.join(config.output_dir, "md_phi_psi_probability.csv")
    _write_probability_surface_csv(md_surface_path, md_surface, x_edges, y_edges)
    outputs.append(md_surface_path)
    summary_path = os.path.join(config.output_dir, "kl_md_vs_reweighted_summary.json")
    write_json(summary_path, summary_metadata)
    outputs.append(summary_path)

    return {
        "output_dir": config.output_dir,
        "outputs": outputs,
        **summary_metadata,
    }


def _state_summary_supports_discrete_fe_plot(summary_csv: str) -> bool:
    """Return True when the microstate summary has finite phi/psi centers."""

    rows = read_csv_rows(summary_csv)
    if not rows:
        return False
    for row in rows:
        try:
            x = _row_float(row, ("plot_x", "phi_center"))
            y = _row_float(row, ("plot_y", "psi_center"))
        except ConfigurationError:
            return False
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
    return True


def _plot_reweighted_discrete_from_state_summary(
    plt,
    path: str,
    *,
    summary_csv: str,
    temperature_kelvin: float,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
    colormap: str,
) -> str:
    """Scatter reweighted microstate centers using the same data as felp-plot."""

    rows = read_csv_rows(summary_csv)
    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    probabilities = np.asarray(
        [_row_float(row, ("target_probability", "probability")) for row in rows],
        dtype=float,
    )
    free_energy_kj_mol = np.asarray(
        [
            float(row["free_energy_kj_mol"])
            if row.get("free_energy_kj_mol") not in (None, "")
            else float("nan")
            for row in rows
        ],
        dtype=float,
    )
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 1:
        raise ConfigurationError(
            "Microstate summary has no finite phi/psi centers for discrete plotting."
        )

    if np.isfinite(free_energy_kj_mol[mask]).any():
        values = energy_scale * free_energy_kj_mol
        colorbar_label = f"Relative reweighted free energy ({energy_unit_label})"
    else:
        values = energy_scale * free_energy_from_probabilities(
            probabilities,
            temperature_kelvin=temperature_kelvin,
        )
        colorbar_label = (
            "Relative reweighted free energy from probability "
            f"({energy_unit_label})"
        )

    fig, ax = plt.subplots(1, 1)
    scatter_kwargs: dict[str, object] = {
        "ax": ax,
        "x": x_values[mask],
        "y": y_values[mask],
        "values": values[mask],
        "colorbar_label": colorbar_label,
        "title": "Reweighted microstate centers",
        "xlabel": "phi (deg)",
        "ylabel": "psi (deg)",
        "colormap": colormap,
    }
    if fel_max_energy_display is not None:
        scatter_kwargs["fel_max_energy_kj_mol"] = fel_max_energy_display
    scatter_state_projection(**scatter_kwargs)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_discrete_phi_psi_values(
    plt,
    path: str,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    colormap: str | None,
    fel_max_energy_display: float | None = None,
    symmetric_zero_center: bool = False,
    colorbar_vmin: float | None = None,
    colorbar_vmax: float | None = None,
) -> str:
    """Render a marker-based phi/psi grid using the shared state scatter style."""

    x_centers = 0.5 * (np.asarray(x_edges, dtype=float)[:-1] + x_edges[1:])
    y_centers = 0.5 * (np.asarray(y_edges, dtype=float)[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    fig, ax = plt.subplots(1, 1)
    scatter_state_projection(
        ax=ax,
        x=x_grid.ravel(),
        y=y_grid.ravel(),
        values=np.asarray(values, dtype=float).ravel(),
        colorbar_label=colorbar_label,
        title=title,
        xlabel="phi (deg)",
        ylabel="psi (deg)",
        fel_max_energy_kj_mol=fel_max_energy_display,
        symmetric_zero_center=symmetric_zero_center,
        colorbar_vmin=colorbar_vmin,
        colorbar_vmax=colorbar_vmax,
        colormap=colormap,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _kl_energy_display_scale_and_label(kl_energy_unit: str) -> tuple[float, str]:
    """Return kBT multiplier and unit label for KL heatmaps."""

    if kl_energy_unit == "kcal_mol":
        return 1.0 / _KJ_PER_KCAL, "kcal/mol"
    if kl_energy_unit == "kj_mol":
        return 1.0, "kJ/mol"
    raise ConfigurationError(
        f"kl_energy_unit must be 'kj_mol' or 'kcal_mol', not {kl_energy_unit!r}."
    )


def _plot_energy_display_scale_and_label(plot_energy_unit: str) -> tuple[float, str]:
    """Scale factor (multiply kJ/mol) and label for plotted energies."""

    if plot_energy_unit == "kcal_mol":
        return 1.0 / _KJ_PER_KCAL, "kcal/mol"
    if plot_energy_unit == "kj_mol":
        return 1.0, "kJ/mol"
    raise ConfigurationError(
        f"plot_energy_unit must be 'kj_mol' or 'kcal_mol', not {plot_energy_unit!r}."
    )


def _load_frame_projected_reweighted_surface(
    config: MDKLPlotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Project FELP reweighted frame weights onto a regular phi/psi grid."""

    summary_csv = config.resolved_reweighted_surface_csv()
    rows = read_csv_rows(summary_csv)
    state_ids = [row.get("state_id", row["label"]) for row in rows]
    reference_probabilities = np.asarray(
        [float(row["reference_probability"]) for row in rows],
        dtype=float,
    )
    reference_probability_by_state = {
        state_id: float(probability)
        for state_id, probability in zip(state_ids, reference_probabilities, strict=True)
    }
    summary_metadata = _load_microstate_summary_metadata(rows)
    reweighting_method = str(
        summary_metadata.get("reweighting_method", "independent") or "independent"
    )
    state_reweight_factors = state_reweight_factors_from_summary_rows(rows)
    frame_weights_csv_path = resolve_frame_weights_csv_path(
        microstate_summary_csv=summary_csv,
        explicit_path=config.frame_weights_csv,
    )
    frame_weights_by_index = (
        load_frame_weights_csv(frame_weights_csv_path)
        if frame_weights_csv_path is not None
        else None
    )
    weight_treatment = config.weight_treatment or _summary_weight_treatment(summary_metadata)
    frame_results = load_frame_results_csv(config.raw_frame_results_csv)
    frame_results = _filter_qc_excluded_frame_results(
        frame_results,
        _excluded_frame_indices_by_state(rows),
    )
    sampled = _collect_cv_projection_samples(
        frame_results=frame_results,
        reference_probability_by_state=reference_probability_by_state,
        temperature_kelvin=float(config.temperature_kelvin),
        weight_treatment=weight_treatment,
        max_points=config.fel_max_points,
        x_name="phi",
        y_name="psi",
        state_reweight_factors=state_reweight_factors,
        frame_weights_by_index=frame_weights_by_index,
        reweighting_method=reweighting_method,
    )
    if sampled is None:
        raise ConfigurationError(
            "Could not project reweighted frame_results onto phi/psi. Check that "
            "the frame results match the supplied microstate summary and include "
            "finite phi/psi values."
        )
    x_values, y_values, _reference_weights, reweighted_weights = sampled
    hist_range = _axis_range("phi", "psi")
    reweighted_surface, x_edges, y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=int(config.fel_bins),
        range=hist_range,
        weights=reweighted_weights,
    )
    total = float(reweighted_surface.sum())
    if total <= 0.0:
        raise ConfigurationError("Projected reweighted probability surface has zero mass.")
    reweighted_surface /= total
    return (
        reweighted_surface,
        x_edges,
        y_edges,
        {
            "reweighted_surface_source": "frame_results",
            "reweighting_method": reweighting_method,
            "reweighted_frame_projection_weight_treatment": weight_treatment,
            "reweighted_frame_projection_estimator_detail": frame_projection_weight_detail(
                reweighting_method=reweighting_method,
                used_frame_weights_csv=frame_weights_by_index is not None,
            ),
            "reweighted_frame_projection_state_mass_preserved": (
                frame_weights_by_index is not None
                or (
                    reweighting_method != "independent"
                    and bool(state_reweight_factors)
                )
            ),
            "frame_weights_path": frame_weights_csv_path,
            "reweighted_projected_frame_count": int(x_values.size),
            "reweighted_probability_total_before_normalization": total,
        },
    )


def _load_regular_reweighted_probability_surface(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ConfigurationError(f"Reweighted surface CSV is empty: {path}")

    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    probabilities = np.asarray(
        [_row_float(row, ("target_probability", "probability")) for row in rows],
        dtype=float,
    )
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(probabilities)
    if not finite.any():
        raise ConfigurationError(
            f"Reweighted surface CSV has no finite plot_x/plot_y probabilities: {path}"
        )
    x_centers = np.asarray(sorted(set(float(value) for value in x_values[finite])))
    y_centers = np.asarray(sorted(set(float(value) for value in y_values[finite])))
    if x_centers.size < 2 or y_centers.size < 2:
        raise ConfigurationError(
            "Reweighted surface must contain at least two unique x and y coordinates."
        )
    if int(finite.sum()) != int(x_centers.size * y_centers.size):
        raise ConfigurationError(
            "The reweighted CSV does not describe a complete regular grid. For MSM "
            "bundle or cluster-center outputs, rerun md-kl-plot with "
            "--raw-frame-results so FELP can project actual frame coordinates."
        )
    surface = np.zeros((x_centers.size, y_centers.size), dtype=float)
    x_index = {float(value): index for index, value in enumerate(x_centers)}
    y_index = {float(value): index for index, value in enumerate(y_centers)}
    for x, y, probability in zip(x_values[finite], y_values[finite], probabilities[finite]):
        if probability < 0.0:
            raise ConfigurationError("Reweighted probabilities must be non-negative.")
        surface[x_index[float(x)], y_index[float(y)]] += float(probability)
    total = float(surface.sum())
    if total <= 0.0:
        raise ConfigurationError("Reweighted probability surface has zero total mass.")
    surface /= total
    x_edges = _edges_from_centers(x_centers, angular=True)
    y_edges = _edges_from_centers(y_centers, angular=True)
    return (
        surface,
        x_edges,
        y_edges,
        {
            "reweighted_surface_source": "regular_surface_csv",
            "surface_shape": [int(surface.shape[0]), int(surface.shape[1])],
            "surface_row_count": int(len(rows)),
            "reweighted_probability_total_before_normalization": total,
        },
    )


def _row_float(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise ConfigurationError(
        f"CSV row is missing required numeric column; expected one of {names}."
    )


def _edges_from_centers(centers: np.ndarray, *, angular: bool) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ConfigurationError("At least two sorted centers are required.")
    deltas = np.diff(centers)
    if np.any(deltas <= 0.0):
        raise ConfigurationError("Surface centers must be strictly increasing.")
    inner = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * deltas[0]
    last = centers[-1] + 0.5 * deltas[-1]
    edges = np.concatenate(([first], inner, [last]))
    if angular and abs(edges[0] + 180.0) < 1.0e-6 and abs(edges[-1] - 180.0) < 1.0e-6:
        edges[0] = -180.0
        edges[-1] = 180.0
    if not np.all(np.isfinite(edges)):
        raise ConfigurationError("Surface grid edges must be finite.")
    return edges


def _load_md_phi_psi_degrees(config: MDKLPlotConfig) -> tuple[np.ndarray, np.ndarray]:
    try:
        import mdtraj as md
    except ImportError as exc:
        raise DependencyError("MDTraj is required for md-kl-plot trajectory loading.") from exc

    if config.phi_psi_selection is None and config.phi_psi_residue_number is None:
        raise ConfigurationError(
            "Provide either --phi-psi-selection or --phi-psi-residue-number."
        )
    residue_number = (
        int(config.phi_psi_residue_number)
        if config.phi_psi_residue_number is not None
        else 1
    )
    phi_indices, psi_indices, _label = _target_phi_psi_indices(
        config.topology_file,
        residue_number,
        config.phi_psi_residue_name,
        config.phi_psi_selection,
    )

    phi_chunks: list[np.ndarray] = []
    psi_chunks: list[np.ndarray] = []
    selected_index = 0
    chunk_size = max(1, int(config.chunk_size))
    for group in config.trajectory_files:
        group_source_index = 0
        for trajectory_file in group:
            iter_stride = int(config.stride) if len(group) == 1 else 1
            for chunk in md.iterload(
                trajectory_file,
                chunk=chunk_size,
                top=config.topology_file,
                stride=iter_stride,
            ):
                if len(group) == 1:
                    keep = np.arange(chunk.n_frames)
                    group_source_index += chunk.n_frames * int(config.stride)
                else:
                    source_indices = group_source_index + np.arange(chunk.n_frames)
                    keep = np.flatnonzero(source_indices % int(config.stride) == 0)
                    group_source_index += chunk.n_frames
                if keep.size == 0:
                    continue
                chunk = chunk[keep]
                n_frames = chunk.n_frames
                selected_indices = selected_index + np.arange(n_frames)
                selected_index += n_frames
                frame_mask = selected_indices >= int(config.frame_start)
                if config.frame_stop is not None:
                    frame_mask &= selected_indices < int(config.frame_stop)
                if not frame_mask.any():
                    continue
                if config.frame_stop is not None and selected_indices[0] >= config.frame_stop:
                    break
                chunk = chunk[frame_mask]
                dihedrals = md.compute_dihedrals(chunk, [phi_indices, psi_indices])
                degrees = _wrap_degrees(np.degrees(dihedrals))
                finite = np.isfinite(degrees[:, 0]) & np.isfinite(degrees[:, 1])
                if finite.any():
                    phi_chunks.append(degrees[finite, 0])
                    psi_chunks.append(degrees[finite, 1])

    if not phi_chunks:
        raise ConfigurationError("No finite phi/psi values were measured from the MD input.")
    return np.concatenate(phi_chunks), np.concatenate(psi_chunks)


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _write_probability_surface_csv(
    path: str,
    surface: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("plot_x", "plot_y", "probability"),
        )
        writer.writeheader()
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                probability = float(surface[ix, iy])
                if math.isfinite(probability):
                    writer.writerow(
                        {
                            "plot_x": f"{float(x):.12g}",
                            "plot_y": f"{float(y):.12g}",
                            "probability": f"{probability:.16g}",
                        }
                    )
