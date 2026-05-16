"""Publication-style plotting utilities for FELP output summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any

import numpy as np

from felp.core import ConfigurationError, K_BOLTZMANN_KJ_PER_MOL_K, ReweightingQCConfig
from felp.io import load_frame_results_csv
from felp.reweighting import is_reweighting_eligible_frame, psis_smooth_weights

from .common import (
    DEFAULT_FEL_COLORMAP,
    FELP_KL_DIVERGENCE_COLORMAP,
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
    validate_registered_colormap,
    write_json,
)

_KJ_PER_KCAL = 4.184
_OCCUPANCY_MARKER_FACE_COLOR = "#e6e6e6"
_ANGULAR_CV_NAMES = {"phi", "psi", "omega"}
_DEFAULT_DIHEDRAL_CV_NAMES = {"phi", "psi"}
_FEL_ESTIMATORS = {"histogram", "gaussian-kernel"}
_FEL_BANDWIDTH_RULES = {"scott", "silverman", "state-adaptive"}


def _normalize_cv_name(raw: str) -> str:
    """Extract the short CV name from a bare name or a full distance-cv spec.

    Accepts ``"ends"`` (returned as-is) or ``"ends:1:CA:10:CA"`` (returns
    ``"ends"``).  The distance-cv CLI spec format is
    ``name:res1:atom1:res2:atom2``, so the name is everything before the first
    colon **only** when the string contains at least four colons (i.e. it looks
    like a full spec).  Strings with fewer colons are returned as-is.
    """
    if raw.count(":") >= 4:
        return raw.split(":", 1)[0]
    return raw


def _plot_energy_display_scale_and_label(plot_energy_unit: str) -> tuple[float, str]:
    """Scale factor (multiply kJ/mol) and axis label for plot display units."""

    if plot_energy_unit == "kcal_mol":
        return 1.0 / _KJ_PER_KCAL, "kcal/mol"
    if plot_energy_unit == "kj_mol":
        return 1.0, "kJ/mol"
    raise ConfigurationError(
        f"plot_energy_unit must be 'kj_mol' or 'kcal_mol', not {plot_energy_unit!r}."
    )


def _parse_fel_bandwidth(raw: str) -> tuple[float, float] | None:
    """Parse an automatic or fixed 2D FEL kernel bandwidth specification."""

    value = str(raw).strip()
    if value == "auto":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {1, 2} or any(not part for part in parts):
        raise ConfigurationError(
            "fel_bandwidth must be 'auto', one positive float, or two comma-separated "
            "positive floats."
        )
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ConfigurationError(
            "fel_bandwidth must be 'auto', one positive float, or two comma-separated "
            "positive floats."
        ) from exc
    if any(value <= 0.0 or not math.isfinite(value) for value in parsed):
        raise ConfigurationError("fel_bandwidth values must be positive finite floats.")
    if len(parsed) == 1:
        return (parsed[0], parsed[0])
    return (parsed[0], parsed[1])


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

    fel_estimator: str = "histogram"
    """Projected frame-density estimator: ``histogram`` or ``gaussian-kernel``."""

    fel_bandwidth: str = "auto"
    """Gaussian-kernel bandwidth: ``auto``, one float, or ``x,y`` floats."""

    fel_bandwidth_rule: str = "state-adaptive"
    """Automatic bandwidth rule for Gaussian-kernel projected FELs."""

    fel_colormap: str = DEFAULT_FEL_COLORMAP
    """Matplotlib colormap for CV free-energy heatmaps and delta-FE scatter plots."""

    probability_colormap: str | None = None
    """Colormap for 2D probability heatmaps and reference/target probability scatters.

    When omitted, matches ``fel_colormap`` after validation.
    """

    divergence_colormap: str | None = None
    """Colormap for projected KL heatmaps (default: built-in ``felp_kl_twilight``)."""

    kl_scale_by_kbt: bool = False
    """If true, display projected KL grids multiplied by kBT in plot energy units."""

    weight_treatment: str | None = None
    """Optional override for frame-projected reweighting weights."""

    cv_x: str | None = None
    """Optional CV name (or full distance-cv spec) for the x-axis of 2D FEL/CV plots."""

    cv_y: str | None = None
    """Optional CV name (or full distance-cv spec) for the y-axis of 2D FEL/CV plots."""

    overwrite: bool = False

    def validate(self) -> None:
        """Validate (and normalize) the FELP plotting configuration.

        If ``cv_x`` or ``cv_y`` contain a full distance-cv spec string
        (e.g. ``ends:1:CA:10:CA``), the short name is extracted automatically.
        """

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
        fel_estimator = str(self.fel_estimator).strip()
        if fel_estimator not in _FEL_ESTIMATORS:
            raise ConfigurationError(
                "fel_estimator must be one of: histogram, gaussian-kernel."
            )
        object.__setattr__(self, "fel_estimator", fel_estimator)
        fel_bandwidth = str(self.fel_bandwidth).strip()
        _parse_fel_bandwidth(fel_bandwidth)
        object.__setattr__(self, "fel_bandwidth", fel_bandwidth)
        fel_bandwidth_rule = str(self.fel_bandwidth_rule).strip()
        if fel_bandwidth_rule not in _FEL_BANDWIDTH_RULES:
            raise ConfigurationError(
                "fel_bandwidth_rule must be one of: scott, silverman, state-adaptive."
            )
        object.__setattr__(self, "fel_bandwidth_rule", fel_bandwidth_rule)
        fel_cmap = validate_registered_colormap(self.fel_colormap)
        object.__setattr__(self, "fel_colormap", fel_cmap)
        prob_src = (
            self.probability_colormap if self.probability_colormap is not None else fel_cmap
        )
        object.__setattr__(
            self,
            "probability_colormap",
            validate_registered_colormap(prob_src),
        )
        div_src = (
            self.divergence_colormap
            if self.divergence_colormap is not None
            else FELP_KL_DIVERGENCE_COLORMAP
        )
        object.__setattr__(
            self,
            "divergence_colormap",
            validate_registered_colormap(div_src),
        )
        if self.weight_treatment is not None:
            weight_treatment = ReweightingQCConfig.normalize_weight_treatment(
                self.weight_treatment
            )
            if weight_treatment not in {"raw", "psis-smoothed"}:
                raise ConfigurationError(
                    "weight_treatment must be one of: raw, psis-smoothed "
                    "(alias: psis)."
                )
            object.__setattr__(self, "weight_treatment", weight_treatment)
        if self.cv_x is not None:
            object.__setattr__(self, "cv_x", _normalize_cv_name(self.cv_x))
        if self.cv_y is not None:
            object.__setattr__(self, "cv_y", _normalize_cv_name(self.cv_y))
        if (self.cv_x is None) != (self.cv_y is None):
            raise ConfigurationError("cv_x and cv_y must be provided together.")
        if self.cv_x is not None and self.cv_y is not None and self.cv_x == self.cv_y:
            raise ConfigurationError("cv_x and cv_y must be different CV names.")
        _plot_energy_display_scale_and_label(self.plot_energy_unit)


@dataclass(frozen=True)
class _CVAxisSelection:
    x_name: str
    y_name: str
    source: str


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
    summary_weight_treatment = _summary_weight_treatment(summary_metadata)
    frame_projection_weight_treatment = (
        config.weight_treatment or summary_weight_treatment
    )
    excluded_frame_indices_by_state = _excluded_frame_indices_by_state(rows)
    frame_results = (
        load_frame_results_csv(config.raw_frame_results_csv)
        if config.raw_frame_results_csv
        else None
    )
    cv_axes = _resolve_cv_axes(
        rows=rows,
        frame_results=frame_results,
        requested_x=config.cv_x,
        requested_y=config.cv_y,
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

    cv_axes_are_custom = (
        cv_axes is not None
        and not _DEFAULT_DIHEDRAL_CV_NAMES.issuperset({cv_axes.x_name, cv_axes.y_name})
    )
    if (
        np.isfinite(plot_x).any()
        and np.isfinite(plot_y).any()
        and not cv_axes_are_custom
    ):
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
                fel_colormap=config.fel_colormap,
                probability_colormap=config.probability_colormap,
            )
        )

    state_x = state_y = None
    if cv_axes is not None:
        state_x = _state_cv_array(rows, cv_axes.x_name)
        state_y = _state_cv_array(rows, cv_axes.y_name)
        outputs.extend(
            _plot_cv_state_landscapes(
                plt=plt,
                x_centers=state_x,
                y_centers=state_y,
                x_name=cv_axes.x_name,
                y_name=cv_axes.y_name,
                reference_probabilities=reference_probabilities,
                target_probabilities=target_probabilities,
                temperature_kelvin=config.temperature_kelvin,
                output_dir=config.output_dir,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_display,
                fel_colormap=config.fel_colormap,
                probability_colormap=config.probability_colormap,
            )
        )

    frame_projection_available = False
    frame_projection_metadata: dict[str, object] = {
        "cv_frame_projection_estimator": config.fel_estimator,
        "cv_frame_projection_kernel": None,
        "cv_frame_projection_bandwidth": None,
        "cv_frame_projection_bandwidth_rule": None,
        "cv_frame_projection_state_mass_preserved": False,
    }
    if frame_results is not None:
        plot_frame_results = _filter_qc_excluded_frame_results(
            frame_results,
            excluded_frame_indices_by_state,
        )
        outputs.append(
            _plot_delta_u_distribution(
                plt=plt,
                frame_results=plot_frame_results,
                output_dir=config.output_dir,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
            )
        )
        if cv_axes is not None:
            projected_outputs, frame_projection_metadata = _plot_cv_projected_landscapes(
                plt=plt,
                frame_results=plot_frame_results,
                state_ids=state_ids,
                reference_probabilities=reference_probabilities,
                target_probabilities=target_probabilities,
                temperature_kelvin=config.temperature_kelvin,
                weight_treatment=frame_projection_weight_treatment,
                fel_estimator=config.fel_estimator,
                fel_bandwidth=config.fel_bandwidth,
                fel_bandwidth_rule=config.fel_bandwidth_rule,
                bins=config.fel_bins,
                max_points=config.fel_max_points,
                output_dir=config.output_dir,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_display,
                fel_colormap=config.fel_colormap,
                probability_colormap=config.probability_colormap,
                divergence_colormap=config.divergence_colormap,
                kl_scale_by_kbt=config.kl_scale_by_kbt,
                x_name=cv_axes.x_name,
                y_name=cv_axes.y_name,
            )
            if projected_outputs:
                frame_projection_available = True
                outputs.extend(projected_outputs)
    if state_x is not None and state_y is not None and np.isfinite(state_x).any() and np.isfinite(state_y).any():
        outputs.extend(
            _plot_discrete_microstate_cv_projection(
                plt=plt,
                x_values=state_x,
                y_values=state_y,
                x_name=cv_axes.x_name if cv_axes is not None else "x",
                y_name=cv_axes.y_name if cv_axes is not None else "y",
                free_energy_kj_mol=reference_free_energy_kj_mol,
                probability=reference_probabilities,
                labels=labels,
                output_dir=config.output_dir,
                temperature_kelvin=config.temperature_kelvin,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_display,
                fel_colormap=config.fel_colormap,
                free_energy_label="reference",
                filename_prefix="reference",
            )
        )
        outputs.extend(
            _plot_discrete_microstate_cv_projection(
                plt=plt,
                x_values=state_x,
                y_values=state_y,
                x_name=cv_axes.x_name if cv_axes is not None else "x",
                y_name=cv_axes.y_name if cv_axes is not None else "y",
                free_energy_kj_mol=target_free_energy_kj_mol,
                probability=target_probabilities,
                labels=labels,
                output_dir=config.output_dir,
                temperature_kelvin=config.temperature_kelvin,
                energy_scale=energy_scale,
                energy_unit_label=energy_unit_label,
                fel_max_energy_display=fel_max_display,
                fel_colormap=config.fel_colormap,
                free_energy_label="reweighted",
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
        "reweighting_qc": summary_metadata.get("reweighting_qc"),
        "cv_axes": (
            [cv_axes.x_name, cv_axes.y_name] if cv_axes is not None else []
        ),
        "cv_axis_source": cv_axes.source if cv_axes is not None else None,
        "cv_frame_projection_available": frame_projection_available,
        "cv_frame_projection_estimator": frame_projection_metadata.get(
            "cv_frame_projection_estimator",
            config.fel_estimator,
        ),
        "cv_frame_projection_estimator_detail": (
            "w_ij proportional to pi_reference(i) * "
            f"{frame_projection_weight_treatment} exp(-beta * delta_u_ij) / n_i"
        ),
        "cv_frame_projection_weight_treatment": frame_projection_weight_treatment,
        "cv_frame_projection_weight_treatment_source": (
            "cli_override" if config.weight_treatment is not None else "microstate_summary"
        ),
        "cv_frame_projection_kernel": frame_projection_metadata.get(
            "cv_frame_projection_kernel"
        ),
        "cv_frame_projection_bandwidth": frame_projection_metadata.get(
            "cv_frame_projection_bandwidth"
        ),
        "cv_frame_projection_bandwidth_rule": frame_projection_metadata.get(
            "cv_frame_projection_bandwidth_rule"
        ),
        "cv_frame_projection_state_mass_preserved": frame_projection_metadata.get(
            "cv_frame_projection_state_mass_preserved",
            False,
        ),
        "cv_frame_projection_excluded_frame_count": sum(
            len(indices) for indices in excluded_frame_indices_by_state.values()
        ),
        "state_reweighting_estimator": (
            "pi_target(i) proportional to "
            f"pi_reference(i) * <{summary_weight_treatment} exp(-beta * delta_u)>_i"
        ),
        "state_reweighting_weight_treatment": summary_weight_treatment,
        "fel_max_energy_kj_mol": config.fel_max_energy_kj_mol,
        "fel_colormap": config.fel_colormap,
        "probability_colormap": config.probability_colormap,
        "divergence_colormap": config.divergence_colormap,
        "plot_energy_unit": config.plot_energy_unit,
        "generated_files": outputs,
    }
    summary.update(
        {
            key: value
            for key, value in frame_projection_metadata.items()
            if key not in summary
        }
    )
    if cv_axes is not None and (cv_axes.x_name, cv_axes.y_name) == ("phi", "psi"):
        summary.update(
            {
                "phi_psi_frame_projection_available": frame_projection_available,
                "phi_psi_frame_projection_estimator": summary[
                    "cv_frame_projection_estimator"
                ],
                "phi_psi_frame_projection_estimator_detail": (
                    summary["cv_frame_projection_estimator_detail"]
                ),
            }
        )
    write_json(os.path.join(config.output_dir, "plot_summary.json"), summary)
    return summary


def _as_nan(value: float | None) -> float:
    """Convert an optional float into a finite-or-NaN plotting value."""

    return float("nan") if value is None else float(value)


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _row_metadata(row: dict[str, str]) -> dict[str, Any]:
    return _parse_metadata(row.get("metadata"))


def _mapping_from_metadata(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _optional_mapping_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cv_filename_suffix(x_name: str, y_name: str) -> str:
    suffix = f"{x_name}_{y_name}".lower()
    suffix = re.sub(r"[^a-z0-9]+", "_", suffix).strip("_")
    return suffix or "cv_projection"


def _cv_axis_label(name: str, *, center: bool = False) -> str:
    if name == "phi":
        return r"$\phi$ center (deg)" if center else r"$\phi$ (deg)"
    if name == "psi":
        return r"$\psi$ center (deg)" if center else r"$\psi$ (deg)"
    if name == "omega":
        return r"$\omega$ center (deg)" if center else r"$\omega$ (deg)"
    return f"{name} center" if center else name


def _axis_range(x_name: str, y_name: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if x_name in _ANGULAR_CV_NAMES and y_name in _ANGULAR_CV_NAMES:
        return ((-180.0, 180.0), (-180.0, 180.0))
    return None


def _state_cv_center(row: dict[str, str], name: str) -> float:
    metadata = _row_metadata(row)
    for metadata_key in ("cv_centers", "state_cv_centers"):
        value = _optional_mapping_float(
            _mapping_from_metadata(metadata, metadata_key),
            name,
        )
        if value is not None:
            return value
    if name == "phi":
        return _as_nan(parse_optional_float(row.get("phi_center")))
    if name == "psi":
        return _as_nan(parse_optional_float(row.get("psi_center")))
    return _as_nan(parse_optional_float(row.get(f"{name}_center")))


def _state_cv_array(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([_state_cv_center(row, name) for row in rows], dtype=float)


def _available_state_cv_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        metadata = _row_metadata(row)
        for metadata_key in ("cv_centers", "state_cv_centers"):
            for name in _mapping_from_metadata(metadata, metadata_key):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        for name, column in (("phi", "phi_center"), ("psi", "psi_center")):
            if name not in seen and parse_optional_float(row.get(column)) is not None:
                seen.add(name)
                names.append(name)
        for column, value in row.items():
            if not column.endswith("_center"):
                continue
            name = column[: -len("_center")]
            if name and name not in seen and parse_optional_float(value) is not None:
                seen.add(name)
                names.append(name)
    return names


def _frame_cv_value(result, name: str) -> float | None:
    metadata = dict(getattr(result, "metadata", {}) or {})
    for metadata_key in (
        "assignment_measurements",
        "target_cv_values",
        "target_cv_values_degrees",
    ):
        value = _optional_mapping_float(
            _mapping_from_metadata(metadata, metadata_key),
            name,
        )
        if value is not None:
            return value
    if name == "phi":
        return result.phi_degrees
    if name == "psi":
        return result.psi_degrees
    return None


def _available_frame_cv_names(frame_results) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for result in frame_results:
        metadata = dict(getattr(result, "metadata", {}) or {})
        for metadata_key in ("assignment_measurements", "target_cv_values"):
            for name in _mapping_from_metadata(metadata, metadata_key):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        if result.phi_degrees is not None and "phi" not in seen:
            seen.add("phi")
            names.append("phi")
        if result.psi_degrees is not None and "psi" not in seen:
            seen.add("psi")
            names.append("psi")
    return names


def _resolve_cv_axes(
    *,
    rows: list[dict[str, str]],
    frame_results,
    requested_x: str | None,
    requested_y: str | None,
) -> _CVAxisSelection | None:
    if requested_x is not None and requested_y is not None:
        return _CVAxisSelection(requested_x, requested_y, "configured")

    state_names = _available_state_cv_names(rows)
    frame_names = _available_frame_cv_names(frame_results or ())
    combined = state_names + [name for name in frame_names if name not in state_names]

    custom = [n for n in combined if n not in _DEFAULT_DIHEDRAL_CV_NAMES]
    if len(custom) >= 2:
        return _CVAxisSelection(custom[0], custom[1], "inferred_custom_cv")
    if "phi" in combined and "psi" in combined:
        return _CVAxisSelection("phi", "psi", "inferred_phi_psi")
    if "plot_x" in combined and "plot_y" in combined:
        return _CVAxisSelection("plot_x", "plot_y", "inferred_plot_projection")
    if len(combined) >= 2:
        return _CVAxisSelection(combined[0], combined[1], "inferred_cv_metadata")
    return None


def _load_microstate_summary_metadata(rows: list[dict[str, str]]) -> dict[str, object]:
    """Best-effort parse of shared per-state metadata from the summary CSV."""

    for row in rows:
        metadata = _row_metadata(row)
        if metadata:
            return dict(metadata)
    return {}


def _summary_weight_treatment(metadata: dict[str, object]) -> str:
    """Return the recorded reweighting weight treatment."""

    raw_treatment = metadata.get("reweighting_weight_treatment")
    if isinstance(raw_treatment, str) and raw_treatment in {"raw", "psis-smoothed"}:
        return raw_treatment
    qc_metadata = metadata.get("reweighting_qc")
    if isinstance(qc_metadata, dict):
        qc_treatment = qc_metadata.get("weight_treatment")
        if isinstance(qc_treatment, str) and qc_treatment in {"raw", "psis-smoothed"}:
            return qc_treatment
    return "raw"


def _excluded_frame_indices_by_state(
    rows: list[dict[str, str]],
) -> dict[str, set[int]]:
    """Return QC-excluded frame indices keyed by state id."""

    excluded: dict[str, set[int]] = {}
    for row in rows:
        state_id = row.get("state_id", row.get("label", ""))
        if not state_id:
            continue
        metadata = _row_metadata(row)
        raw_indices = metadata.get("excluded_frame_indices")
        if not isinstance(raw_indices, list):
            continue
        indices: set[int] = set()
        for value in raw_indices:
            try:
                indices.add(int(value))
            except (TypeError, ValueError):
                continue
        if indices:
            excluded[state_id] = indices
    return excluded


def _filter_qc_excluded_frame_results(
    frame_results,
    excluded_frame_indices_by_state: dict[str, set[int]],
):
    """Drop frame results excluded by the reweighting QC summary."""

    if not excluded_frame_indices_by_state:
        return frame_results
    return [
        result
        for result in frame_results
        if result.frame_index
        not in excluded_frame_indices_by_state.get(result.microstate_id, set())
    ]


def _safe_probabilities(values: np.ndarray) -> np.ndarray:
    """Clip probabilities away from zero for divergence calculations."""

    return np.clip(np.asarray(values, dtype=float), 1.0e-12, None)


def _pointwise_kl_term(a: np.ndarray, b: np.ndarray, *, eps: float) -> np.ndarray:
    """Compute ``a * log(a/b)`` grid with ``0 * log(0/b) -> 0``."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    safe_b = np.maximum(b, eps)
    mask = a > 0.0
    out[mask] = a[mask] * np.log(np.maximum(a[mask], eps) / safe_b[mask])
    return out


def _symmetric_kl_imbalance_grid(p: np.ndarray, q: np.ndarray, *, eps: float) -> np.ndarray:
    """Signed per-cell imbalance: ``½ ( p log(p/q) − q log(q/p) )``.

    Zero wherever ``p == q``. Swapping ``p`` and ``q`` flips the sign, so
    diverging colormaps centered at zero represent both directions fairly.
    """

    tp = _pointwise_kl_term(p, q, eps=eps)
    tq = _pointwise_kl_term(q, p, eps=eps)
    return 0.5 * (tp - tq)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute KL divergence and per-state contributions."""

    safe_p = _safe_probabilities(p)
    safe_q = _safe_probabilities(q)
    contributions = safe_p * np.log(safe_p / safe_q)
    return float(np.sum(contributions)), contributions


def _surface_divergence_summary(
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute divergence metrics between two probability surfaces.

    Scalar entries use KL(P||Q) and KL(Q||P) on flat normalized grids.
    The returned 2D grid is the signed imbalance
    ``0.5 * (p*log(p/q) - q*log(q/p))`` per cell (zero when p equals q, sign
    flips when p and q are swapped), for diverging colormaps centered at zero.
    """

    p_flat = np.asarray(p, dtype=float).ravel()
    q_flat = np.asarray(q, dtype=float).ravel()
    p_flat = np.where(np.isfinite(p_flat), p_flat, 0.0)
    q_flat = np.where(np.isfinite(q_flat), q_flat, 0.0)
    if p_flat.sum() > 0.0:
        p_flat = p_flat / float(p_flat.sum())
    if q_flat.sum() > 0.0:
        q_flat = q_flat / float(q_flat.sum())
    kl_pq, _ = _kl_divergence(p_flat, q_flat)
    kl_qp, _ = _kl_divergence(q_flat, p_flat)
    js = _jensen_shannon_divergence(p_flat, q_flat)
    overlap = _overlap_coefficient(p_flat, q_flat)
    shape = np.asarray(p, dtype=float).shape
    sym_grid = _symmetric_kl_imbalance_grid(
        p_flat.reshape(shape),
        q_flat.reshape(shape),
        eps=1.0e-12,
    )
    return (
        {
            "kl_pre_reweighted_to_reweighted_projected_fel": float(kl_pq),
            "kl_reweighted_to_pre_reweighted_projected_fel": float(kl_qp),
            "jensen_shannon_projected_fel": float(js),
            "projected_fel_overlap_coefficient": float(overlap),
        },
        sym_grid,
    )


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
    ax.set_facecolor("white")
    positions = np.arange(len(ordered))
    colors = ["firebrick" if occupancy_shift[index] > 0 else "steelblue" for index in ordered]
    ax.barh(
        positions,
        occupancy_shift[ordered],
        color=colors,
        edgecolor="black",
        linewidth=0.85,
    )
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
    ax.set_facecolor("white")
    diagonal_limit = max(
        float(np.max(reference_probabilities)) if reference_probabilities.size else 0.0,
        float(np.max(target_probabilities)) if target_probabilities.size else 0.0,
        1.0e-12,
    )
    ax.plot(
        [0.0, diagonal_limit],
        [0.0, diagonal_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        reference_probabilities,
        target_probabilities,
        facecolors=_OCCUPANCY_MARKER_FACE_COLOR,
        edgecolors="black",
        s=30,
        linewidths=0.7,
        zorder=2,
    )
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
    fel_colormap: str,
    probability_colormap: str,
) -> list[str]:
    """Plot state quantities projected onto reduced-space coordinates."""

    mask = np.isfinite(plot_x) & np.isfinite(plot_y)
    outputs: list[str] = []
    for values, title, colorbar_label, filename, cap_delta, cmap in (
        (
            reference_probabilities,
            "Reference State Probabilities",
            "Reference Probability",
            "reference_probability_projection.png",
            False,
            probability_colormap,
        ),
        (
            target_probabilities,
            "Target State Probabilities",
            "Target Probability",
            "target_probability_projection.png",
            False,
            probability_colormap,
        ),
        (
            delta_free_energy,
            "Delta Free Energy",
            f"Delta Free Energy ({energy_unit_label})",
            "delta_free_energy_projection.png",
            True,
            fel_colormap,
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
            "colormap": cmap,
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
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a rectangular state grid when the summary rows form a complete mesh."""

    if not (
        x_centers.size == y_centers.size == values.size
        and np.isfinite(x_centers).all()
        and np.isfinite(y_centers).all()
    ):
        return None
    x_grid = _regular_centers_to_edges(x_centers)
    y_grid = _regular_centers_to_edges(y_centers)
    if x_grid is None or y_grid is None:
        return None
    x_unique, x_edges = x_grid
    y_unique, y_edges = y_grid
    if x_unique.size * y_unique.size != values.size:
        return None

    grid = np.full((x_unique.size, y_unique.size), np.nan, dtype=float)
    x_index = {float(value): index for index, value in enumerate(x_unique)}
    y_index = {float(value): index for index, value in enumerate(y_unique)}
    for x_value, y_value, value in zip(x_centers, y_centers, values):
        location = (x_index[float(x_value)], y_index[float(y_value)])
        if not math.isnan(grid[location]):
            return None
        grid[location] = float(value)
    if np.isnan(grid).any():
        return None
    return grid, x_edges, y_edges


def _subsample_indices(total: int, max_points: int) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, max_points, dtype=int)


def _plot_cv_state_landscapes(
    plt,
    *,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    x_name: str,
    y_name: str,
    reference_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    temperature_kelvin: float,
    output_dir: str,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
    fel_colormap: str,
    probability_colormap: str,
) -> list[str]:
    """Build 2D CV free energy surfaces from discrete state probabilities."""

    reference_grid_data = _build_regular_state_grid(
        x_centers,
        y_centers,
        reference_probabilities,
    )
    target_grid_data = _build_regular_state_grid(
        x_centers,
        y_centers,
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
    suffix = _cv_filename_suffix(x_name, y_name)
    legacy_phi_psi = (x_name, y_name) == ("phi", "psi")
    reference_prob_filename = (
        "reference_phi_psi_state_probability_landscape.png"
        if legacy_phi_psi
        else f"reference_{suffix}_state_probability_landscape.png"
    )
    reweighted_prob_filename = (
        "reweighted_phi_psi_state_probability_landscape.png"
        if legacy_phi_psi
        else f"reweighted_{suffix}_state_probability_landscape.png"
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reference_prob_filename),
            reference_grid,
            x_edges,
            y_edges,
            title=f"Pre-reweighted microstate probability ({x_name}, {y_name})",
            colorbar_label="Probability",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            colormap=probability_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reweighted_prob_filename),
            target_grid,
            x_edges,
            y_edges,
            title=f"Reweighted microstate probability ({x_name}, {y_name})",
            colorbar_label="Probability",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            colormap=probability_colormap,
        )
    )
    reference_filename = (
        "reference_phi_psi_state_free_energy_landscape.png"
        if legacy_phi_psi
        else f"reference_{suffix}_state_free_energy_landscape.png"
    )
    reweighted_filename = (
        "reweighted_phi_psi_state_free_energy_landscape.png"
        if legacy_phi_psi
        else f"reweighted_{suffix}_state_free_energy_landscape.png"
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reference_filename),
            reference_fe,
            x_edges,
            y_edges,
            title=f"Pre-reweighted microstate free energy ({x_name}, {y_name})",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=fel_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reweighted_filename),
            reweighted_fe,
            x_edges,
            y_edges,
            title=f"Reweighted microstate free energy ({x_name}, {y_name})",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=fel_colormap,
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


def _collect_cv_projection_state_samples(
    *,
    frame_results,
    state_ids: set[str],
    x_name: str,
    y_name: str,
) -> tuple[dict[str, list[tuple[float, float, float]]], list[float]] | None:
    """Collect finite frame CV samples keyed by microstate id."""

    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    eligible_delta_values: list[float] = []
    for result in frame_results:
        if not is_reweighting_eligible_frame(result):
            continue
        if result.microstate_id not in state_ids:
            continue
        x_value = _frame_cv_value(result, x_name)
        y_value = _frame_cv_value(result, y_name)
        if x_value is None or y_value is None:
            continue
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        grouped[result.microstate_id].append(
            (
                float(x_value),
                float(y_value),
                float(result.delta_u_kj_mol),
            )
        )
        eligible_delta_values.append(float(result.delta_u_kj_mol))

    if not eligible_delta_values:
        return None
    return grouped, eligible_delta_values


def _collect_cv_projection_samples(
    *,
    frame_results,
    reference_probability_by_state: dict[str, float],
    temperature_kelvin: float,
    weight_treatment: str,
    max_points: int,
    x_name: str,
    y_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect exact 2D CV frame samples with state-consistent reference weights."""

    collected = _collect_cv_projection_state_samples(
        frame_results=frame_results,
        state_ids=set(reference_probability_by_state),
        x_name=x_name,
        y_name=y_name,
    )
    if collected is None:
        return None
    grouped, eligible_delta_values = collected

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

    x_values: list[float] = []
    y_values: list[float] = []
    reference_weights: list[float] = []
    reweighted_weights: list[float] = []
    for state_id, samples in sampled.items():
        sample_count = len(samples)
        if sample_count == 0:
            continue
        reference_probability = reference_probability_by_state.get(state_id, 0.0)
        delta_values = np.asarray([sample[2] for sample in samples], dtype=float)
        local_delta_shift = float(delta_values.min())
        treated_weights = np.exp(-beta * (delta_values - local_delta_shift))
        if weight_treatment == "psis-smoothed":
            treated_weights, _ = psis_smooth_weights(treated_weights)
        treated_weights = (
            math.exp(-beta * (local_delta_shift - delta_shift)) * treated_weights
        )
        for (x_value, y_value, _), treated_weight in zip(
            samples,
            treated_weights,
            strict=True,
        ):
            x_values.append(x_value)
            y_values.append(y_value)
            reference_weights.append(reference_probability / sample_count)
            reweighted_weights.append(
                reference_probability * float(treated_weight) / sample_count
            )

    if not x_values:
        return None

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(reference_weights, dtype=float),
        np.asarray(reweighted_weights, dtype=float),
    )


def _weighted_effective_sample_size(weights: np.ndarray) -> float:
    """Return ESS for non-negative analysis weights."""

    finite_weights = np.asarray(weights[np.isfinite(weights)], dtype=float)
    positive_weights = finite_weights[finite_weights > 0.0]
    if positive_weights.size == 0:
        return 0.0
    weight_sum = float(positive_weights.sum())
    square_sum = float(np.square(positive_weights).sum())
    return float((weight_sum**2) / square_sum) if square_sum > 0.0 else 0.0


def _weighted_standard_deviation(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a finite weighted standard deviation with unweighted fallback."""

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        fallback = values[np.isfinite(values)]
        return float(fallback.std(ddof=0)) if fallback.size else 0.0
    finite_values = values[finite]
    finite_weights = weights[finite]
    weight_sum = float(finite_weights.sum())
    if weight_sum <= 0.0:
        return float(finite_values.std(ddof=0)) if finite_values.size else 0.0
    normalized_weights = finite_weights / weight_sum
    mean = float(np.sum(normalized_weights * finite_values))
    variance = float(np.sum(normalized_weights * np.square(finite_values - mean)))
    return math.sqrt(max(variance, 0.0))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float | None:
    """Return a weighted quantile for finite positive-weight values."""

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        fallback = values[np.isfinite(values)]
        if fallback.size == 0:
            return None
        return float(np.quantile(fallback, quantile))
    finite_values = values[finite]
    finite_weights = weights[finite]
    order = np.argsort(finite_values)
    ordered_values = finite_values[order]
    ordered_weights = finite_weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = float(cumulative[-1])
    if total <= 0.0:
        return None
    threshold = min(max(float(quantile), 0.0), 1.0) * total
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(index, ordered_values.size - 1)
    return float(ordered_values[index])


def _robust_weighted_scale(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a conservative robust scale for bandwidth selection."""

    std = _weighted_standard_deviation(values, weights)
    q25 = _weighted_quantile(values, weights, 0.25)
    q75 = _weighted_quantile(values, weights, 0.75)
    scales = [std] if std > 0.0 and math.isfinite(std) else []
    if q25 is not None and q75 is not None:
        iqr_scale = (q75 - q25) / 1.349
        if iqr_scale > 0.0 and math.isfinite(iqr_scale):
            scales.append(float(iqr_scale))
    return min(scales) if scales else 0.0


def _state_adaptive_kernel_bandwidth(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    reference_weights: np.ndarray,
    target_weights: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[tuple[float, float], dict[str, object]]:
    """Resolve a conservative local bandwidth for one microstate/state support."""

    x_bin_width = float(np.median(np.diff(x_edges)))
    y_bin_width = float(np.median(np.diff(y_edges)))
    minimum = (0.5 * x_bin_width, 0.5 * y_bin_width)
    reference_ess = _weighted_effective_sample_size(reference_weights)
    target_ess = _weighted_effective_sample_size(target_weights)
    positive_ess = [value for value in (reference_ess, target_ess) if value > 0.0]
    effective_sample_size = min(positive_ess) if positive_ess else 1.0
    factor = effective_sample_size ** (-1.0 / 6.0)
    combined_weights = np.asarray(reference_weights, dtype=float) + np.asarray(
        target_weights,
        dtype=float,
    )

    x_scale = _robust_weighted_scale(x_values, combined_weights)
    y_scale = _robust_weighted_scale(y_values, combined_weights)
    x_raw = x_scale * factor
    y_raw = y_scale * factor
    x_bandwidth = max(minimum[0], x_raw)
    y_bandwidth = max(minimum[1], y_raw)
    return (float(x_bandwidth), float(y_bandwidth)), {
        "bandwidth": [float(x_bandwidth), float(y_bandwidth)],
        "reference_ess": reference_ess,
        "target_ess": target_ess,
        "effective_sample_size": effective_sample_size,
        "robust_scale": [float(x_scale), float(y_scale)],
        "minimum_bandwidth": [float(minimum[0]), float(minimum[1])],
    }


def _resolve_kernel_bandwidth(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    reference_weights: np.ndarray,
    target_weights: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    bandwidth_spec: str,
    bandwidth_rule: str,
) -> tuple[tuple[float, float], dict[str, object]]:
    """Resolve a fixed or automatic Gaussian-kernel bandwidth."""

    fixed_bandwidth = _parse_fel_bandwidth(bandwidth_spec)
    x_bin_width = float(np.median(np.diff(x_edges)))
    y_bin_width = float(np.median(np.diff(y_edges)))
    reference_ess = _weighted_effective_sample_size(reference_weights)
    target_ess = _weighted_effective_sample_size(target_weights)
    if fixed_bandwidth is not None:
        return fixed_bandwidth, {
            "cv_frame_projection_bandwidth": list(fixed_bandwidth),
            "cv_frame_projection_bandwidth_rule": "fixed",
            "cv_frame_projection_bandwidth_source": "cli",
            "cv_frame_projection_reference_ess": reference_ess,
            "cv_frame_projection_target_ess": target_ess,
        }

    positive_ess = [value for value in (reference_ess, target_ess) if value > 0.0]
    effective_sample_size = min(positive_ess) if positive_ess else 1.0
    dimension = 2.0
    if bandwidth_rule == "silverman":
        factor = (effective_sample_size * (dimension + 2.0) / 4.0) ** (
            -1.0 / (dimension + 4.0)
        )
    else:
        factor = effective_sample_size ** (-1.0 / (dimension + 4.0))

    combined_weights = np.asarray(reference_weights, dtype=float) + np.asarray(
        target_weights,
        dtype=float,
    )
    x_std = _weighted_standard_deviation(x_values, combined_weights)
    y_std = _weighted_standard_deviation(y_values, combined_weights)
    x_bandwidth = max(x_bin_width, x_std * factor)
    y_bandwidth = max(y_bin_width, y_std * factor)
    return (float(x_bandwidth), float(y_bandwidth)), {
        "cv_frame_projection_bandwidth": [float(x_bandwidth), float(y_bandwidth)],
        "cv_frame_projection_bandwidth_rule": bandwidth_rule,
        "cv_frame_projection_bandwidth_source": "auto",
        "cv_frame_projection_reference_ess": reference_ess,
        "cv_frame_projection_target_ess": target_ess,
        "cv_frame_projection_effective_sample_size_for_bandwidth": (
            effective_sample_size
        ),
        "cv_frame_projection_minimum_bandwidth": [x_bin_width, y_bin_width],
    }


def _state_kernel_probability_grid(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    bandwidth: tuple[float, float],
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return a finite-grid normalized separable Gaussian kernel probability grid."""

    normalized_weights = np.asarray(weights, dtype=float)
    weight_sum = float(normalized_weights.sum())
    if weight_sum <= 0.0:
        return np.zeros((x_centers.size, y_centers.size), dtype=float)
    normalized_weights = normalized_weights / weight_sum

    x_bandwidth, y_bandwidth = bandwidth
    density = np.zeros((x_centers.size, y_centers.size), dtype=float)
    for start in range(0, x_values.size, chunk_size):
        stop = min(start + chunk_size, x_values.size)
        x_chunk = x_values[start:stop]
        y_chunk = y_values[start:stop]
        weight_chunk = normalized_weights[start:stop]
        x_kernel = np.exp(
            -0.5 * np.square((x_chunk[:, None] - x_centers[None, :]) / x_bandwidth)
        )
        y_kernel = np.exp(
            -0.5 * np.square((y_chunk[:, None] - y_centers[None, :]) / y_bandwidth)
        )
        density += (x_kernel * weight_chunk[:, None]).T @ y_kernel

    density_sum = float(density.sum())
    return density / density_sum if density_sum > 0.0 else density


def _gaussian_kernel_projected_surfaces(
    *,
    frame_results,
    reference_probability_by_state: dict[str, float],
    target_probability_by_state: dict[str, float],
    temperature_kelvin: float,
    weight_treatment: str,
    bins: int,
    max_points: int,
    x_name: str,
    y_name: str,
    bandwidth_spec: str,
    bandwidth_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | None:
    """Estimate state-mass-preserving projected reference/target probability grids."""

    state_ids = set(reference_probability_by_state) | set(target_probability_by_state)
    collected = _collect_cv_projection_state_samples(
        frame_results=frame_results,
        state_ids=state_ids,
        x_name=x_name,
        y_name=y_name,
    )
    if collected is None:
        return None
    grouped, _ = collected
    missing_positive_prior = [
        state_id
        for state_id in state_ids
        if (
            reference_probability_by_state.get(state_id, 0.0) > 0.0
            or target_probability_by_state.get(state_id, 0.0) > 0.0
        )
        and not grouped.get(state_id)
    ]
    if missing_positive_prior:
        return None

    sampled = _sample_state_projection_points(grouped, max_points)
    flattened = [sample for samples in sampled.values() for sample in samples]
    if not flattened:
        return None
    all_x_values = np.asarray([sample[0] for sample in flattened], dtype=float)
    all_y_values = np.asarray([sample[1] for sample in flattened], dtype=float)
    _, x_edges, y_edges = np.histogram2d(
        all_x_values,
        all_y_values,
        bins=bins,
        range=_axis_range(x_name, y_name),
    )
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    beta = 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * temperature_kelvin)
    reference_grid = np.zeros((x_centers.size, y_centers.size), dtype=float)
    target_grid = np.zeros_like(reference_grid)
    bandwidth_x_values: list[float] = []
    bandwidth_y_values: list[float] = []
    bandwidth_reference_weights: list[float] = []
    bandwidth_target_weights: list[float] = []

    state_payloads = []
    for state_id, samples in sampled.items():
        if not samples:
            continue
        state_x = np.asarray([sample[0] for sample in samples], dtype=float)
        state_y = np.asarray([sample[1] for sample in samples], dtype=float)
        delta_values = np.asarray([sample[2] for sample in samples], dtype=float)
        reference_probability = reference_probability_by_state.get(state_id, 0.0)
        target_probability = target_probability_by_state.get(state_id, 0.0)

        reference_weights = np.full(state_x.size, 1.0 / state_x.size, dtype=float)
        local_delta_shift = float(delta_values.min())
        target_weights = np.exp(-beta * (delta_values - local_delta_shift))
        if weight_treatment == "psis-smoothed":
            target_weights, _ = psis_smooth_weights(target_weights)
        target_weight_sum = float(target_weights.sum())
        target_conditional = (
            target_weights / target_weight_sum
            if target_weight_sum > 0.0
            else np.zeros_like(target_weights)
        )

        state_payloads.append(
            (
                state_id,
                state_x,
                state_y,
                reference_weights,
                target_weights,
                target_conditional,
                reference_probability,
                target_probability,
            )
        )
        bandwidth_x_values.extend(float(value) for value in state_x)
        bandwidth_y_values.extend(float(value) for value in state_y)
        bandwidth_reference_weights.extend(
            float(reference_probability * value) for value in reference_weights
        )
        bandwidth_target_weights.extend(
            float(target_probability * value) for value in target_conditional
        )

    fixed_bandwidth = _parse_fel_bandwidth(bandwidth_spec)
    adaptive_bandwidths: dict[str, tuple[float, float]] = {}
    adaptive_metadata: dict[str, dict[str, object]] = {}
    if fixed_bandwidth is None and bandwidth_rule == "state-adaptive":
        for (
            state_id,
            state_x,
            state_y,
            reference_weights,
            _target_weights,
            target_conditional,
            _reference_probability,
            _target_probability,
        ) in state_payloads:
            state_bandwidth, state_metadata = _state_adaptive_kernel_bandwidth(
                x_values=state_x,
                y_values=state_y,
                reference_weights=reference_weights,
                target_weights=target_conditional,
                x_edges=x_edges,
                y_edges=y_edges,
            )
            adaptive_bandwidths[state_id] = state_bandwidth
            adaptive_metadata[state_id] = state_metadata
        bandwidth_values = np.asarray(
            list(adaptive_bandwidths.values()),
            dtype=float,
        )
        bandwidth_metadata = {
            "cv_frame_projection_bandwidth": (
                np.median(bandwidth_values, axis=0).astype(float).tolist()
                if bandwidth_values.size
                else None
            ),
            "cv_frame_projection_bandwidth_rule": "state-adaptive",
            "cv_frame_projection_bandwidth_source": "state_local_robust_scott",
            "cv_frame_projection_bandwidth_detail": (
                "Each state's conditional frame density uses a robust Scott bandwidth "
                "from that state's local CV support; state probabilities remain fixed."
            ),
            "cv_frame_projection_bandwidth_summary": (
                {
                    "min": np.min(bandwidth_values, axis=0).astype(float).tolist(),
                    "median": np.median(bandwidth_values, axis=0).astype(float).tolist(),
                    "max": np.max(bandwidth_values, axis=0).astype(float).tolist(),
                }
                if bandwidth_values.size
                else {}
            ),
            "cv_frame_projection_state_bandwidths": adaptive_metadata,
            "cv_frame_projection_minimum_bandwidth": [
                0.5 * float(np.median(np.diff(x_edges))),
                0.5 * float(np.median(np.diff(y_edges))),
            ],
        }
        bandwidth = None
    else:
        bandwidth, bandwidth_metadata = _resolve_kernel_bandwidth(
            x_values=np.asarray(bandwidth_x_values, dtype=float),
            y_values=np.asarray(bandwidth_y_values, dtype=float),
            reference_weights=np.asarray(bandwidth_reference_weights, dtype=float),
            target_weights=np.asarray(bandwidth_target_weights, dtype=float),
            x_edges=x_edges,
            y_edges=y_edges,
            bandwidth_spec=bandwidth_spec,
            bandwidth_rule=bandwidth_rule,
        )

    for (
        state_id,
        state_x,
        state_y,
        reference_weights,
        target_weights,
        _target_conditional,
        reference_probability,
        target_probability,
    ) in state_payloads:
        state_bandwidth = adaptive_bandwidths.get(state_id, bandwidth)
        if state_bandwidth is None:
            continue
        reference_grid += reference_probability * _state_kernel_probability_grid(
            x_values=state_x,
            y_values=state_y,
            weights=reference_weights,
            x_centers=x_centers,
            y_centers=y_centers,
            bandwidth=state_bandwidth,
        )
        target_grid += target_probability * _state_kernel_probability_grid(
            x_values=state_x,
            y_values=state_y,
            weights=target_weights,
            x_centers=x_centers,
            y_centers=y_centers,
            bandwidth=state_bandwidth,
        )

    reference_total = float(reference_grid.sum())
    target_total = float(target_grid.sum())
    if reference_total > 0.0:
        reference_grid = reference_grid / reference_total
    if target_total > 0.0:
        target_grid = target_grid / target_total

    metadata = {
        **bandwidth_metadata,
        "cv_frame_projection_estimator": "state_preserving_gaussian_kernel",
        "cv_frame_projection_kernel": "gaussian",
        "cv_frame_projection_state_mass_preserved": True,
        "cv_frame_projection_boundary_treatment": "finite_grid_state_renormalized",
        "cv_frame_projection_sampled_state_count": len(state_payloads),
        "cv_frame_projection_note": (
            "Projected PMF is a Gaussian-kernel density estimate over CV space; "
            "state probabilities from microstate_summary.csv are preserved before "
            "final finite-grid normalization."
        ),
    }
    return reference_grid, target_grid, x_edges, y_edges, metadata


def _plot_cv_projected_landscapes(
    plt,
    *,
    frame_results,
    state_ids: list[str],
    reference_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    temperature_kelvin: float,
    weight_treatment: str,
    fel_estimator: str,
    fel_bandwidth: str,
    fel_bandwidth_rule: str,
    bins: int,
    max_points: int,
    output_dir: str,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
    fel_colormap: str,
    probability_colormap: str,
    divergence_colormap: str,
    kl_scale_by_kbt: bool,
    x_name: str,
    y_name: str,
) -> tuple[list[str], dict[str, object]]:
    """Project exact 2D CV frame samples using prior-consistent frame weights."""

    reference_probability_by_state = {
        state_id: float(probability)
        for state_id, probability in zip(state_ids, reference_probabilities)
    }
    target_probability_by_state = {
        state_id: float(probability)
        for state_id, probability in zip(state_ids, target_probabilities)
    }
    projection_metadata: dict[str, object] = {
        "cv_frame_projection_estimator": "histogram",
        "cv_frame_projection_kernel": None,
        "cv_frame_projection_bandwidth": None,
        "cv_frame_projection_bandwidth_rule": None,
        "cv_frame_projection_state_mass_preserved": False,
    }
    if fel_estimator == "gaussian-kernel":
        projected = _gaussian_kernel_projected_surfaces(
            frame_results=frame_results,
            reference_probability_by_state=reference_probability_by_state,
            target_probability_by_state=target_probability_by_state,
            temperature_kelvin=temperature_kelvin,
            weight_treatment=weight_treatment,
            bins=bins,
            max_points=max_points,
            x_name=x_name,
            y_name=y_name,
            bandwidth_spec=fel_bandwidth,
            bandwidth_rule=fel_bandwidth_rule,
        )
        if projected is None:
            return [], projection_metadata
        reference_hist, reweighted_hist, x_edges, y_edges, projection_metadata = projected
    else:
        sampled = _collect_cv_projection_samples(
            frame_results=frame_results,
            reference_probability_by_state=reference_probability_by_state,
            temperature_kelvin=temperature_kelvin,
            weight_treatment=weight_treatment,
            max_points=max_points,
            x_name=x_name,
            y_name=y_name,
        )
        if sampled is None:
            return [], projection_metadata

        x_values, y_values, reference_weights, reweighted_weights = sampled
        hist_range = _axis_range(x_name, y_name)
        reference_hist, x_edges, y_edges = np.histogram2d(
            x_values,
            y_values,
            bins=bins,
            range=hist_range,
            weights=reference_weights,
        )
        reweighted_hist, _, _ = np.histogram2d(
            x_values,
            y_values,
            bins=bins,
            range=hist_range,
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
    divergence_metadata, kl_contribution_grid = _surface_divergence_summary(
        reference_hist,
        reweighted_hist,
    )
    divergence_metadata["kl_grid_definition"] = (
        "0.5 * [p_pre * log(p_pre / p_reweighted) - "
        "p_reweighted * log(p_reweighted / p_pre)]"
    )
    kl_display_grid, kl_colorbar_label, kl_display_metadata = _kl_display_grid(
        kl_contribution_grid,
        temperature_kelvin=temperature_kelvin,
        energy_scale=energy_scale,
        energy_unit_label=energy_unit_label,
        scale_by_kbt=kl_scale_by_kbt,
    )
    divergence_metadata.update(kl_display_metadata)
    projection_metadata.update(divergence_metadata)

    outputs: list[str] = []
    suffix = _cv_filename_suffix(x_name, y_name)
    legacy_phi_psi = (x_name, y_name) == ("phi", "psi")
    reference_filename = (
        "reference_phi_psi_free_energy_landscape.png"
        if legacy_phi_psi
        else f"reference_{suffix}_free_energy_landscape.png"
    )
    reweighted_filename = (
        "reweighted_phi_psi_free_energy_landscape.png"
        if legacy_phi_psi
        else f"reweighted_{suffix}_free_energy_landscape.png"
    )
    reference_prob_filename = (
        "reference_phi_psi_projected_probability.png"
        if legacy_phi_psi
        else f"reference_{suffix}_projected_probability.png"
    )
    reweighted_prob_filename = (
        "reweighted_phi_psi_projected_probability.png"
        if legacy_phi_psi
        else f"reweighted_{suffix}_projected_probability.png"
    )
    kl_filename = (
        "kl_phi_psi_projected_fel_contributions.png"
        if legacy_phi_psi
        else f"kl_{suffix}_projected_fel_contributions.png"
    )
    kl_summary_filename = (
        "kl_phi_psi_projected_fel_summary.json"
        if legacy_phi_psi
        else f"kl_{suffix}_projected_fel_summary.json"
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reference_prob_filename),
            reference_hist,
            x_edges,
            y_edges,
            title=f"Pre-reweighted projected probability ({x_name}, {y_name})",
            colorbar_label="Probability",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            colormap=probability_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reweighted_prob_filename),
            reweighted_hist,
            x_edges,
            y_edges,
            title=f"Reweighted projected probability ({x_name}, {y_name})",
            colorbar_label="Probability",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            colormap=probability_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reference_filename),
            reference_fe,
            x_edges,
            y_edges,
            title=f"Pre-reweighted projected free energy ({x_name}, {y_name})",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=fel_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, reweighted_filename),
            reweighted_fe,
            x_edges,
            y_edges,
            title=f"Reweighted projected free energy ({x_name}, {y_name})",
            colorbar_label=f"Relative free energy ({energy_unit_label})",
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            fel_max_energy_kj_mol=fel_max_energy_display,
            colormap=fel_colormap,
        )
    )
    outputs.append(
        save_pcolormesh_landscape(
            plt,
            os.path.join(output_dir, kl_filename),
            kl_display_grid,
            x_edges,
            y_edges,
            title=f"Projected FEL KL contributions ({x_name}, {y_name})",
            colorbar_label=kl_colorbar_label,
            xlabel=_cv_axis_label(x_name),
            ylabel=_cv_axis_label(y_name),
            colormap=divergence_colormap,
            symmetric_zero_center=True,
        )
    )
    kl_summary_path = os.path.join(output_dir, kl_summary_filename)
    write_json(kl_summary_path, divergence_metadata)
    outputs.append(kl_summary_path)
    return outputs, projection_metadata


def _kl_display_grid(
    kl_grid: np.ndarray,
    *,
    temperature_kelvin: float,
    energy_scale: float,
    energy_unit_label: str,
    scale_by_kbt: bool,
) -> tuple[np.ndarray, str, dict[str, object]]:
    """Return the displayed KL grid and metadata for optional kBT scaling."""

    if not scale_by_kbt:
        return np.asarray(kl_grid, dtype=float), "Signed KL imbalance", {
            "kl_display_units": "dimensionless",
            "kl_display_scale_factor": 1.0,
        }
    scale = K_BOLTZMANN_KJ_PER_MOL_K * float(temperature_kelvin) * float(energy_scale)
    return (
        np.asarray(kl_grid, dtype=float) * scale,
        f"Signed KL contribution ({energy_unit_label})",
        {
            "kl_display_units": energy_unit_label,
            "kl_display_scale_factor": float(scale),
        },
    )


def _plot_discrete_microstate_cv_projection(
    plt,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_name: str,
    y_name: str,
    free_energy_kj_mol: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    output_dir: str,
    temperature_kelvin: float,
    energy_scale: float,
    energy_unit_label: str,
    fel_max_energy_display: float | None,
    fel_colormap: str,
    free_energy_label: str,
    filename_prefix: str | None = None,
) -> list[str]:
    """Scatter microstate centers on selected CV axes colored by free energy."""

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 1:
        return []

    fe = energy_scale * np.asarray(free_energy_kj_mol, dtype=float)
    if not np.isfinite(fe[mask]).any():
        values = energy_scale * free_energy_from_probabilities(
            np.asarray(probability, dtype=float),
            temperature_kelvin=temperature_kelvin,
        )
        colorbar_label = (
            f"Relative {free_energy_label} free energy from probability "
            f"({energy_unit_label})"
        )
        title = f"{free_energy_label.capitalize()} microstate centers ({x_name}, {y_name})"
    else:
        values = fe
        colorbar_label = (
            f"Relative {free_energy_label} free energy ({energy_unit_label})"
        )
        title = f"{free_energy_label.capitalize()} microstate centers ({x_name}, {y_name})"

    fig, ax = plt.subplots(1, 1)
    scatter_kwargs: dict = {
        "ax": ax,
        "x": x_values[mask],
        "y": y_values[mask],
        "values": values[mask],
        "colorbar_label": colorbar_label,
        "title": title,
        "xlabel": _cv_axis_label(x_name, center=True),
        "ylabel": _cv_axis_label(y_name, center=True),
        "colormap": fel_colormap,
    }
    if fel_max_energy_display is not None:
        scatter_kwargs["fel_max_energy_kj_mol"] = fel_max_energy_display
    scatter_state_projection(**scatter_kwargs)
    if int(mask.sum()) <= 24:
        for index in np.flatnonzero(mask):
            ax.annotate(
                labels[index],
                (x_values[index], y_values[index]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.85,
            )
    fig.tight_layout()
    suffix = _cv_filename_suffix(x_name, y_name)
    stem = "discrete_microstate_free_energy"
    if filename_prefix:
        stem = f"{filename_prefix}_{stem}"
    filename = f"{stem}_ramachandran.png" if (x_name, y_name) == ("phi", "psi") else f"{stem}_{suffix}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return [path]
