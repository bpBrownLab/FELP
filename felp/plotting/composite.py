"""Configurable multi-row Ramachandran composite figures for publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Mapping

import numpy as np

from felp.core import ConfigurationError

from .common import (
    DEFAULT_FEL_COLORMAP,
    FELP_KL_DIVERGENCE_COLORMAP,
    apply_publication_style,
    free_energy_from_probabilities,
    import_matplotlib,
    prepare_kl_display_grid,
    prepare_output_dir,
    probability_surface_to_free_energy,
    read_csv_rows,
    resolve_kl_colorbar_limits,
    scatter_state_projection,
    symmetric_kl_imbalance_mathtext_label,
    validate_registered_colormap,
    write_json,
)
from .felp import _surface_divergence_summary
from .md_kl import _edges_from_centers, _row_float

_KJ_PER_KCAL = 4.184

SUPPORTED_PANEL_TYPES = frozenset(
    {
        "reference_fel",
        "reweighted_fel",
        "md_fel",
        "kl_divergence",
        "delta_probability",
        "delta_free_energy",
    }
)


@dataclass(frozen=True)
class CompositeFigureStyle:
    """Typography and layout knobs for composite figures."""

    font_family: str = "DejaVu Sans"
    title_size: float = 11.0
    label_size: float = 10.0
    tick_size: float = 8.0
    column_header_size: float = 11.0
    row_label_size: float = 11.0
    colorbar_label_size: float = 8.0
    panel_width: float = 2.45
    panel_height: float = 2.45
    marker_size: float = 22.0
    dpi: float = 300.0
    hspace: float = 0.18
    wspace: float = 0.42
    left_margin: float = 0.07
    right_margin: float = 0.90
    top_margin: float = 0.93
    bottom_margin: float = 0.07
    row_label_pad: float = 0.012
    column_header_pad: float = 0.015
    show_panel_titles: bool = False
    show_row_labels: bool = True
    show_column_headers: bool = True
    show_reference_arrows: bool = True
    colorbar_mode: str = "bottom_row"
    row_label_rotation: float = 90.0
    xlabel: str = r"$\phi$ (deg)"
    ylabel: str = r"$\psi$ (deg)"
    fel_colorbar_label: str = "Free energy (kcal/mol)"
    kl_colorbar_label: str = "Symmetric KL"
    use_kl_mathtext_label: bool = False
    reference_colorbar_label: str = ""
    reweighted_colorbar_label: str = ""
    md_colorbar_label: str = ""


@dataclass(frozen=True)
class CompositeColumnSpec:
    """One column in the composite grid."""

    type: str
    header: str = ""
    span_rows: bool = False
    draw_arrows_to: str | None = None


@dataclass(frozen=True)
class CompositeFigureConfig:
    """Configuration for a multi-row Ramachandran composite figure."""

    output: str
    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[CompositeColumnSpec, ...]
    style: CompositeFigureStyle = field(default_factory=CompositeFigureStyle)
    reference: Mapping[str, Any] | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    data_root: str | None = None
    path_templates: Mapping[str, str] = field(default_factory=dict)
    overwrite: bool = False

    def validate(self) -> None:
        if not self.output:
            raise ConfigurationError("Composite figure output path is required.")
        if not self.rows:
            raise ConfigurationError("At least one row is required.")
        if not self.columns:
            raise ConfigurationError("At least one column is required.")
        for column in self.columns:
            if column.type not in SUPPORTED_PANEL_TYPES:
                raise ConfigurationError(
                    f"Unsupported panel type {column.type!r}. "
                    f"Expected one of: {sorted(SUPPORTED_PANEL_TYPES)}."
                )
        reference_columns = [column for column in self.columns if column.span_rows]
        if len(reference_columns) > 1:
            raise ConfigurationError("Only one column may set span_rows=true.")
        if self.style.colorbar_mode not in {"bottom_row", "every_panel", "none"}:
            raise ConfigurationError(
                "style.colorbar_mode must be 'bottom_row', 'every_panel', or 'none'."
            )
        if not self.reference and any(column.type == "reference_fel" for column in self.columns):
            raise ConfigurationError(
                "A reference block is required when reference_fel is included."
            )


def load_composite_figure_config(path: str) -> CompositeFigureConfig:
    """Load a composite figure specification from JSON."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Composite config must be a JSON object: {path}")

    style_payload = payload.get("style", {})
    style = CompositeFigureStyle(
        **{key: style_payload[key] for key in style_payload if hasattr(CompositeFigureStyle, key)}
    )
    columns = tuple(
        CompositeColumnSpec(
            type=str(column["type"]),
            header=str(column.get("header", "")),
            span_rows=bool(column.get("span_rows", False)),
            draw_arrows_to=column.get("draw_arrows_to"),
        )
        for column in payload.get("columns", [])
    )
    return CompositeFigureConfig(
        output=str(payload["output"]),
        rows=tuple(payload.get("rows", [])),
        columns=columns,
        style=style,
        reference=payload.get("reference"),
        defaults=payload.get("defaults", {}),
        data_root=payload.get("data_root"),
        path_templates=payload.get("path_templates", {}),
        overwrite=bool(payload.get("overwrite", False)),
    )


def generate_composite_figure(config: CompositeFigureConfig) -> dict[str, object]:
    """Render a publication-style composite figure from CSV-backed panel data."""

    config.validate()
    defaults = dict(config.defaults)
    temperature_kelvin = float(defaults.get("temperature_kelvin", 310.0))
    plot_energy_unit = str(defaults.get("plot_energy_unit", "kcal_mol"))
    energy_scale, energy_unit_label = _energy_scale_and_label(plot_energy_unit)
    fel_max_energy_display = _fel_max_energy_display(defaults, energy_scale)
    fel_colormap = validate_registered_colormap(
        str(defaults.get("fel_colormap", DEFAULT_FEL_COLORMAP))
    )
    kl_colormap = validate_registered_colormap(
        str(defaults.get("kl_colormap", FELP_KL_DIVERGENCE_COLORMAP))
    )
    kl_display_threshold = defaults.get("kl_display_threshold", 1.0e-5)
    kl_limits = resolve_kl_colorbar_limits(
        defaults.get("kl_colorbar_vmin"),
        defaults.get("kl_colorbar_vmax"),
    )

    row_contexts = [_resolve_row_context(row, config) for row in config.rows]
    reference_context = (
        _resolve_reference_context(config.reference, config)
        if config.reference is not None
        else None
    )

    panel_jobs: list[_PanelJob] = []
    for row_index, row_ctx in enumerate(row_contexts):
        for column_index, column in enumerate(config.columns):
            if column.span_rows and row_index > 0:
                continue
            panel_jobs.append(
                _PanelJob(
                    row_index=row_index,
                    column_index=column_index,
                    column=column,
                    row_context=row_ctx,
                    reference_context=reference_context,
                )
            )

    panel_payloads = [
        _build_panel_payload(
            job,
            temperature_kelvin=temperature_kelvin,
            energy_scale=energy_scale,
            fel_max_energy_display=fel_max_energy_display,
            kl_display_threshold=kl_display_threshold,
            kl_limits=kl_limits,
        )
        for job in panel_jobs
    ]
    payload_by_key = {
        (payload.row_index, payload.column_index): payload for payload in panel_payloads
    }

    plt = import_matplotlib()
    apply_publication_style()
    _apply_composite_style(config.style)

    n_rows = len(row_contexts)
    n_cols = len(config.columns)
    has_span_reference = any(column.span_rows for column in config.columns)

    fig_width = config.style.panel_width * n_cols + 1.0
    fig_height = config.style.panel_height * n_rows + 0.6
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=config.style.dpi)
    width_ratios = [1.0 for _ in config.columns]
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        hspace=config.style.hspace,
        wspace=config.style.wspace,
        left=config.style.left_margin,
        right=config.style.right_margin,
        top=config.style.top_margin,
        bottom=config.style.bottom_margin,
    )

    column_index_by_type = {
        column.type: index for index, column in enumerate(config.columns)
    }
    ref_row = n_rows // 2
    for row_index in range(n_rows):
        for column_index, column in enumerate(config.columns):
            if column.span_rows:
                if row_index != ref_row:
                    continue
                payload = payload_by_key[(0, column_index)]
                ax = fig.add_subplot(gs[ref_row, column_index])
                show_colorbar = True
                is_bottom = True
                is_left = True
            else:
                payload = payload_by_key[(row_index, column_index)]
                ax = fig.add_subplot(gs[row_index, column_index])
                show_colorbar = _panel_show_colorbar(
                    config.style.colorbar_mode,
                    row_index=row_index,
                    n_rows=n_rows,
                )
                is_bottom = row_index == n_rows - 1
                is_left = False
            _draw_panel(
                ax,
                payload,
                fel_colormap=fel_colormap,
                kl_colormap=kl_colormap,
                style=config.style,
                show_ylabel=is_left,
                show_xlabel=is_bottom,
                show_xticklabels=is_bottom,
                show_yticklabels=is_left,
                show_colorbar=show_colorbar,
            )

    if config.style.show_column_headers:
        _add_column_headers(fig, gs, config, n_rows)
    if config.style.show_row_labels:
        _add_row_labels(fig, gs, row_contexts, config.style, n_rows, n_cols - 1)

    if config.style.show_reference_arrows and "reference_fel" in column_index_by_type:
        arrow_target_type = next(
            (column.draw_arrows_to for column in config.columns if column.draw_arrows_to),
            "reweighted_fel",
        )
        if arrow_target_type in column_index_by_type:
            ref_col = column_index_by_type["reference_fel"]
            target_col = column_index_by_type[arrow_target_type]
            _draw_reference_arrows(
                fig,
                gs,
                ref_col=ref_col,
                target_col=target_col,
                n_rows=n_rows,
                span_reference=has_span_reference,
            )

    output_path = config.output
    output_dir = os.path.dirname(output_path) or "."
    prepare_output_dir(output_dir, overwrite=config.overwrite)
    fig.savefig(output_path, dpi=config.style.dpi)
    plt.close(fig)

    summary = {
        "output": output_path,
        "row_count": n_rows,
        "column_types": [column.type for column in config.columns],
        "rows": [
            {
                "label": row_ctx["label"],
                "microstate_summary": row_ctx["microstate_summary"],
                "md_probability_csv": row_ctx.get("md_probability_csv"),
            }
            for row_ctx in row_contexts
        ],
        "reference": reference_context,
    }
    summary_path = os.path.join(output_dir, "composite_figure_summary.json")
    write_json(summary_path, summary)
    return summary


@dataclass(frozen=True)
class _PanelJob:
    row_index: int
    column_index: int
    column: CompositeColumnSpec
    row_context: Mapping[str, Any]
    reference_context: Mapping[str, Any] | None


@dataclass(frozen=True)
class _PanelPayload:
    row_index: int
    column_index: int
    column: CompositeColumnSpec
    panel_type: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    colorbar_label: str
    title: str
    fel_max_energy_display: float | None = None
    symmetric_zero_center: bool = False
    kl_colorbar_vmin: float | None = None
    kl_colorbar_vmax: float | None = None


def _resolve_row_context(row: Mapping[str, Any], config: CompositeFigureConfig) -> dict[str, Any]:
    context: dict[str, Any] = {
        "data_root": config.data_root or "",
        "label": str(row.get("label", "")),
        "target": row.get("target", ""),
    }
    for key, template in config.path_templates.items():
        context[key] = _expand_template(template, context)
    for key, value in row.items():
        if isinstance(value, str):
            context[key] = _expand_template(value, context)
        else:
            context[key] = value

    reweight_dir = context.get("reweight_dir")
    md_kl_dir = context.get("md_kl_dir")
    if reweight_dir and "microstate_summary" not in context:
        context["microstate_summary"] = os.path.join(reweight_dir, "microstate_summary.csv")
    if md_kl_dir and "md_probability_csv" not in context:
        context["md_probability_csv"] = os.path.join(md_kl_dir, "md_phi_psi_probability.csv")
    elif reweight_dir and "md_probability_csv" not in context:
        context["md_probability_csv"] = os.path.join(
            reweight_dir,
            "md_vs_reweighted_kl",
            "md_phi_psi_probability.csv",
        )
    if "microstate_summary" not in context:
        raise ConfigurationError(
            f"Row {context.get('label') or row!r} must define microstate_summary or reweight_dir."
        )
    return context


def _resolve_reference_context(
    reference: Mapping[str, Any],
    config: CompositeFigureConfig,
) -> dict[str, Any]:
    context = dict(reference)
    base = {"data_root": config.data_root or ""}
    for key, value in reference.items():
        if isinstance(value, str):
            context[key] = _expand_template(value, base)
    if "microstate_summary" not in context:
        raise ConfigurationError("reference.microstate_summary is required.")
    context.setdefault("probability_column", "reference_probability")
    return context


def _expand_template(value: str, context: Mapping[str, Any]) -> str:
    if "{" not in value:
        return value
    try:
        return value.format(**context)
    except KeyError as exc:
        missing = exc.args[0]
        raise ConfigurationError(
            f"Unknown template key {missing!r} in path {value!r}."
        ) from exc


def _build_panel_payload(
    job: _PanelJob,
    *,
    temperature_kelvin: float,
    energy_scale: float,
    fel_max_energy_display: float | None,
    kl_display_threshold: float | None,
    kl_limits: tuple[float, float] | None,
) -> _PanelPayload:
    panel_type = job.column.type
    if panel_type == "reference_fel":
        if job.reference_context is None:
            raise ConfigurationError("reference_fel requires a reference block.")
        summary_csv = str(job.reference_context["microstate_summary"])
        probability_column = str(job.reference_context.get("probability_column", "reference_probability"))
        x, y, values, colorbar_label = _summary_free_energy_arrays(
            summary_csv,
            probability_column=probability_column,
            temperature_kelvin=temperature_kelvin,
            energy_scale=energy_scale,
            prefer_csv_free_energy=False,
        )
        title = "Reference"
    elif panel_type == "reweighted_fel":
        summary_csv = str(job.row_context["microstate_summary"])
        x, y, values, colorbar_label = _summary_free_energy_arrays(
            summary_csv,
            probability_column="target_probability",
            temperature_kelvin=temperature_kelvin,
            energy_scale=energy_scale,
            prefer_csv_free_energy=True,
        )
        title = "Reweighted"
    elif panel_type == "md_fel":
        md_csv = str(job.row_context["md_probability_csv"])
        x, y, values, colorbar_label = _md_free_energy_arrays(
            md_csv,
            temperature_kelvin=temperature_kelvin,
            energy_scale=energy_scale,
        )
        title = "MD"
    elif panel_type == "kl_divergence":
        md_csv = str(job.row_context["md_probability_csv"])
        summary_csv = str(job.row_context["microstate_summary"])
        x, y, values, colorbar_label = _kl_display_arrays(
            md_csv=md_csv,
            summary_csv=summary_csv,
            temperature_kelvin=temperature_kelvin,
            kl_display_threshold=kl_display_threshold,
            kl_limits=kl_limits,
        )
        title = "KL"
    elif panel_type == "delta_probability":
        summary_csv = str(job.row_context["microstate_summary"])
        x, y, values, colorbar_label = _delta_probability_arrays(summary_csv)
        title = "Delta probability"
    elif panel_type == "delta_free_energy":
        summary_csv = str(job.row_context["microstate_summary"])
        x, y, values, colorbar_label = _delta_free_energy_arrays(
            summary_csv,
            temperature_kelvin=temperature_kelvin,
            energy_scale=energy_scale,
        )
        title = "Delta free energy"
    else:
        raise ConfigurationError(f"Unsupported panel type {panel_type!r}.")

    kl_vmin = kl_limits[0] if panel_type == "kl_divergence" and kl_limits is not None else None
    kl_vmax = kl_limits[1] if panel_type == "kl_divergence" and kl_limits is not None else None
    symmetric_cap = panel_type in {"delta_probability", "delta_free_energy"}
    return _PanelPayload(
        row_index=job.row_index,
        column_index=job.column_index,
        column=job.column,
        panel_type=panel_type,
        x=x,
        y=y,
        values=values,
        colorbar_label=colorbar_label,
        title=title,
        fel_max_energy_display=fel_max_energy_display if panel_type.endswith("_fel") else None,
        symmetric_zero_center=panel_type == "kl_divergence" or symmetric_cap,
        kl_colorbar_vmin=kl_vmin,
        kl_colorbar_vmax=kl_vmax,
    )


def _summary_free_energy_arrays(
    summary_csv: str,
    *,
    probability_column: str,
    temperature_kelvin: float,
    energy_scale: float,
    prefer_csv_free_energy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows = read_csv_rows(summary_csv)
    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    probabilities = np.asarray([float(row[probability_column]) for row in rows], dtype=float)
    free_energy_kj_mol = np.asarray(
        [
            float(row["free_energy_kj_mol"])
            if row.get("free_energy_kj_mol") not in (None, "")
            else float("nan")
            for row in rows
        ],
        dtype=float,
    )
    if prefer_csv_free_energy and np.isfinite(free_energy_kj_mol).any():
        values = energy_scale * free_energy_kj_mol
    else:
        values = energy_scale * free_energy_from_probabilities(
            probabilities,
            temperature_kelvin=temperature_kelvin,
        )
    return x_values, y_values, values, ""


def _md_free_energy_arrays(
    md_csv: str,
    *,
    temperature_kelvin: float,
    energy_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    surface, x_edges, y_edges = _load_probability_surface_csv(md_csv)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    free_energy = energy_scale * probability_surface_to_free_energy(
        surface,
        temperature_kelvin,
    )
    return (
        x_grid.ravel(),
        y_grid.ravel(),
        free_energy.ravel(),
        "",
    )


def _kl_display_arrays(
    *,
    md_csv: str,
    summary_csv: str,
    temperature_kelvin: float,
    kl_display_threshold: float | None,
    kl_limits: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    md_surface, x_edges, y_edges = _load_probability_surface_csv(md_csv)
    reweighted_surface, _, _ = _load_summary_probability_surface(summary_csv)
    if md_surface.shape != reweighted_surface.shape:
        raise ConfigurationError(
            "MD and reweighted probability grids differ in shape for KL panel. "
            "Regenerate md-kl-plot after the state-grid KL fix."
        )
    _, kl_grid = _surface_divergence_summary(md_surface, reweighted_surface)
    kl_display_grid, colorbar_label, _ = prepare_kl_display_grid(
        kl_grid,
        temperature_kelvin=temperature_kelvin,
        scale_by_kbt=False,
        display_threshold=kl_display_threshold,
        probability_labels=("p_MD", "p_rw"),
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    return x_grid.ravel(), y_grid.ravel(), kl_display_grid.ravel(), colorbar_label


def _delta_probability_arrays(summary_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows = read_csv_rows(summary_csv)
    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    delta = np.asarray(
        [float(row["target_probability"]) - float(row["reference_probability"]) for row in rows],
        dtype=float,
    )
    return x_values, y_values, delta, "Target - reference probability"


def _delta_free_energy_arrays(
    summary_csv: str,
    *,
    temperature_kelvin: float,
    energy_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows = read_csv_rows(summary_csv)
    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    reference = free_energy_from_probabilities(
        np.asarray([float(row["reference_probability"]) for row in rows], dtype=float),
        temperature_kelvin=temperature_kelvin,
    )
    target = free_energy_from_probabilities(
        np.asarray([float(row["target_probability"]) for row in rows], dtype=float),
        temperature_kelvin=temperature_kelvin,
    )
    return (
        x_values,
        y_values,
        energy_scale * (target - reference),
        "Δ free energy",
    )


def _load_summary_probability_surface(
    summary_csv: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(summary_csv)
    x_values = np.asarray([_row_float(row, ("plot_x", "phi_center")) for row in rows])
    y_values = np.asarray([_row_float(row, ("plot_y", "psi_center")) for row in rows])
    probabilities = np.asarray(
        [_row_float(row, ("target_probability", "probability")) for row in rows],
        dtype=float,
    )
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(probabilities)
    x_centers = np.asarray(sorted(set(float(value) for value in x_values[finite])))
    y_centers = np.asarray(sorted(set(float(value) for value in y_values[finite])))
    surface = np.zeros((x_centers.size, y_centers.size), dtype=float)
    x_index = {float(value): index for index, value in enumerate(x_centers)}
    y_index = {float(value): index for index, value in enumerate(y_centers)}
    for x, y, probability in zip(x_values[finite], y_values[finite], probabilities[finite]):
        surface[x_index[float(x)], y_index[float(y)]] += float(probability)
    total = float(surface.sum())
    if total <= 0.0:
        raise ConfigurationError(f"Summary probabilities sum to zero: {summary_csv}")
    surface /= total
    x_edges = _edges_from_centers(x_centers, angular=True)
    y_edges = _edges_from_centers(y_centers, angular=True)
    return surface, x_edges, y_edges


def _load_probability_surface_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise ConfigurationError(f"Probability CSV is empty: {path}")
    x_values = np.asarray([float(row["plot_x"]) for row in rows])
    y_values = np.asarray([float(row["plot_y"]) for row in rows])
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(probabilities)
    x_centers = np.asarray(sorted(set(float(value) for value in x_values[finite])))
    y_centers = np.asarray(sorted(set(float(value) for value in y_values[finite])))
    surface = np.zeros((x_centers.size, y_centers.size), dtype=float)
    x_index = {float(value): index for index, value in enumerate(x_centers)}
    y_index = {float(value): index for index, value in enumerate(y_centers)}
    for x, y, probability in zip(x_values[finite], y_values[finite], probabilities[finite]):
        surface[x_index[float(x)], y_index[float(y)]] += float(probability)
    total = float(surface.sum())
    if total <= 0.0:
        raise ConfigurationError(f"Probability CSV has zero total mass: {path}")
    surface /= total
    x_edges = _edges_from_centers(x_centers, angular=True)
    y_edges = _edges_from_centers(y_centers, angular=True)
    return surface, x_edges, y_edges


def _panel_show_colorbar(colorbar_mode: str, *, row_index: int, n_rows: int) -> bool:
    if colorbar_mode == "none":
        return False
    if colorbar_mode == "every_panel":
        return True
    return row_index == n_rows - 1


def _display_colorbar_label(payload: _PanelPayload, style: CompositeFigureStyle) -> str:
    if payload.panel_type == "kl_divergence":
        if style.use_kl_mathtext_label:
            return symmetric_kl_imbalance_mathtext_label("p_MD", "p_rw")
        return style.kl_colorbar_label
    if payload.panel_type == "reference_fel" and style.reference_colorbar_label:
        return style.reference_colorbar_label
    if payload.panel_type == "reweighted_fel" and style.reweighted_colorbar_label:
        return style.reweighted_colorbar_label
    if payload.panel_type == "md_fel" and style.md_colorbar_label:
        return style.md_colorbar_label
    if payload.panel_type.endswith("_fel"):
        return style.fel_colorbar_label
    return payload.colorbar_label


def _draw_panel(
    ax,
    payload: _PanelPayload,
    *,
    fel_colormap: str,
    kl_colormap: str,
    style: CompositeFigureStyle,
    show_ylabel: bool,
    show_xlabel: bool,
    show_xticklabels: bool,
    show_yticklabels: bool,
    show_colorbar: bool,
) -> None:
    colormap = kl_colormap if payload.panel_type == "kl_divergence" else fel_colormap
    scatter_kwargs: dict[str, object] = {
        "ax": ax,
        "x": payload.x,
        "y": payload.y,
        "values": payload.values,
        "colorbar_label": _display_colorbar_label(payload, style),
        "title": payload.title if style.show_panel_titles else "",
        "xlabel": style.xlabel if show_xlabel else "",
        "ylabel": style.ylabel if show_ylabel else "",
        "colormap": colormap,
        "show_colorbar": show_colorbar,
        "marker_size": style.marker_size,
        "ramachandran_equal_aspect": False,
        "colorbar_fraction": 0.04,
        "colorbar_pad": 0.01,
        "colorbar_labelsize": style.colorbar_label_size,
    }
    if payload.fel_max_energy_display is not None and payload.panel_type.endswith("_fel"):
        scatter_kwargs["fel_max_energy_kj_mol"] = payload.fel_max_energy_display
    if payload.symmetric_zero_center:
        scatter_kwargs["symmetric_zero_center"] = True
        if payload.kl_colorbar_vmin is not None and payload.kl_colorbar_vmax is not None:
            scatter_kwargs["colorbar_vmin"] = payload.kl_colorbar_vmin
            scatter_kwargs["colorbar_vmax"] = payload.kl_colorbar_vmax
    scatter_state_projection(**scatter_kwargs)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.tick_params(
        labelsize=style.tick_size,
        labelbottom=show_xticklabels,
        labelleft=show_yticklabels,
    )
    if not show_xticklabels:
        ax.set_xlabel("")
    if not show_yticklabels:
        ax.set_ylabel("")


def _apply_composite_style(style: CompositeFigureStyle) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": style.font_family,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
        }
    )


def _get_colorbar_bbox(fig, ax):
    ax_pos = ax.get_position()
    for other_ax in fig.get_axes():
        if other_ax == ax:
            continue
        other_pos = other_ax.get_position()
        y_match = abs(other_pos.y0 - ax_pos.y0) < 0.05 and abs(other_pos.y1 - ax_pos.y1) < 0.05
        x_match = other_pos.x0 > ax_pos.x1 - 0.01
        width_match = other_pos.width < ax_pos.width * 0.4
        if y_match and x_match and width_match:
            return other_pos
    return None


def _add_column_headers(
    fig,
    gs,
    config: CompositeFigureConfig,
    n_rows: int,
) -> None:
    all_axes = fig.get_axes()
    for column_index, column in enumerate(config.columns):
        if not column.header:
            continue
        
        # Find the top plot axes in this column
        plot_ax = None
        for ax in all_axes:
            pos = ax.get_position()
            # Try to match the top row or the middle row if it's the reference column
            for r in range(n_rows):
                cell_pos = gs[r, column_index].get_position(fig)
                if abs(pos.x0 - cell_pos.x0) < 0.05 and abs(pos.y0 - cell_pos.y0) < 0.05:
                    plot_ax = ax
                    break
            if plot_ax is not None:
                break

        if plot_ax is not None:
            plot_pos = plot_ax.get_position()
            # If the column has a colorbar, center over the plot panel + colorbar area
            colorbar_bbox = _get_colorbar_bbox(fig, plot_ax)
            if colorbar_bbox is not None:
                x_center = (plot_pos.x0 + colorbar_bbox.x1) * 0.5
                y_top = max(plot_pos.y1, colorbar_bbox.y1)
            else:
                x_center = (plot_pos.x0 + plot_pos.x1) * 0.5
                y_top = plot_pos.y1
        else:
            bbox = gs[0, column_index].get_position(fig)
            x_center = (bbox.x0 + bbox.x1) * 0.5
            y_top = bbox.y1

        fig.text(
            x_center,
            y_top + config.style.column_header_pad,
            column.header,
            ha="center",
            va="bottom",
            fontsize=config.style.column_header_size,
            fontweight="bold",
        )


def _add_row_labels(
    fig,
    gs,
    row_contexts: list[Mapping[str, Any]],
    style: CompositeFigureStyle,
    n_rows: int,
    last_col: int,
) -> None:
    all_axes = fig.get_axes()
    for row_index, row_ctx in enumerate(row_contexts):
        label = str(row_ctx.get("label", ""))
        if not label:
            continue
        
        plot_ax = None
        for ax in all_axes:
            pos = ax.get_position()
            cell_pos = gs[row_index, last_col].get_position(fig)
            if abs(pos.x0 - cell_pos.x0) < 0.05 and abs(pos.y0 - cell_pos.y0) < 0.05:
                plot_ax = ax
                break

        if plot_ax is not None:
            plot_pos = plot_ax.get_position()
            colorbar_bbox = _get_colorbar_bbox(fig, plot_ax)
            if colorbar_bbox is not None:
                right_edge = colorbar_bbox.x1
            else:
                right_edge = plot_pos.x1
            y_pos = (plot_pos.y0 + plot_pos.y1) * 0.5
        else:
            bbox = gs[row_index, last_col].get_position(fig)
            right_edge = bbox.x1
            y_pos = (bbox.y0 + bbox.y1) * 0.5

        fig.text(
            right_edge + style.row_label_pad,
            y_pos,
            label,
            ha="left",
            va="center",
            rotation=style.row_label_rotation,
            fontsize=style.row_label_size,
            fontweight="bold",
        )


def _draw_reference_arrows(
    fig,
    gs,
    *,
    ref_col: int,
    target_col: int,
    n_rows: int,
    span_reference: bool,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    all_axes = fig.get_axes()
    ref_row = n_rows // 2
    ref_ax = None
    for ax in all_axes:
        pos = ax.get_position()
        cell_pos = gs[ref_row, ref_col].get_position(fig)
        if abs(pos.x0 - cell_pos.x0) < 0.05 and abs(pos.y0 - cell_pos.y0) < 0.05:
            ref_ax = ax
            break

    if ref_ax is None:
        return

    target_axes_by_row = {}
    for row_index in range(n_rows):
        for ax in all_axes:
            pos = ax.get_position()
            cell_pos = gs[row_index, target_col].get_position(fig)
            if abs(pos.x0 - cell_pos.x0) < 0.05 and abs(pos.y0 - cell_pos.y0) < 0.05:
                target_axes_by_row[row_index] = ax
                break

    ref_pos = ref_ax.get_position()
    colorbar_bbox = _get_colorbar_bbox(fig, ref_ax)
    if colorbar_bbox is not None:
        colorbar_right = colorbar_bbox.x1
    else:
        colorbar_right = ref_pos.x1

    # We add 0.045 to clear the vertically-oriented colorbar label text!
    x_start = colorbar_right + 0.045

    target_0 = target_axes_by_row.get(0)
    if target_0 is None:
        return
    t_pos_0 = target_0.get_position()
    
    # Since Column 1 has no y-labels or tickmarks, the target_left is just its x0 minus a tiny gap
    target_left = t_pos_0.x0 - 0.015
    
    x_split = x_start + 0.15 * (target_left - x_start)

    ref_y = (ref_pos.y0 + ref_pos.y1) * 0.5

    # Draw horizontal starting stem
    line_start = FancyArrowPatch(
        (x_start, ref_y),
        (x_split, ref_y),
        transform=fig.transFigure,
        arrowstyle="-",
        linewidth=1.5,
        color="0.2",
        clip_on=False,
        zorder=10,
    )
    fig.patches.append(line_start)

    # Gather y coordinates of targets to draw vertical backbone
    y_coords = []
    for row_index in range(n_rows):
        target_ax = target_axes_by_row.get(row_index)
        if target_ax is None:
            continue
        t_pos = target_ax.get_position()
        target_y = (t_pos.y0 + t_pos.y1) * 0.5
        y_coords.append(target_y)

    if not y_coords:
        return

    y_bottom = min(y_coords)
    y_top = max(y_coords)

    # Draw vertical backbone
    line_vertical = FancyArrowPatch(
        (x_split, y_bottom),
        (x_split, y_top),
        transform=fig.transFigure,
        arrowstyle="-",
        linewidth=1.5,
        color="0.2",
        clip_on=False,
        zorder=10,
    )
    fig.patches.append(line_vertical)

    # Draw individual arrows with arrowheads ending just before the Column 1 axes
    for row_index in range(n_rows):
        target_ax = target_axes_by_row.get(row_index)
        if target_ax is None:
            continue
        t_pos = target_ax.get_position()
        target_y = (t_pos.y0 + t_pos.y1) * 0.5
        
        arrow = FancyArrowPatch(
            (x_split + 0.006, target_y),
            (target_left, target_y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="0.2",
            clip_on=False,
            zorder=10,
        )
        fig.patches.append(arrow)


def _energy_scale_and_label(plot_energy_unit: str) -> tuple[float, str]:
    if plot_energy_unit == "kcal_mol":
        return 1.0 / _KJ_PER_KCAL, "kcal/mol"
    if plot_energy_unit == "kj_mol":
        return 1.0, "kJ/mol"
    raise ConfigurationError(
        f"plot_energy_unit must be 'kj_mol' or 'kcal_mol', not {plot_energy_unit!r}."
    )


def _fel_max_energy_display(defaults: Mapping[str, Any], energy_scale: float) -> float | None:
    if defaults.get("fel_max_energy_kcal_mol") is not None:
        return float(defaults["fel_max_energy_kcal_mol"])
    if defaults.get("fel_max_energy_kj_mol") is not None:
        return float(defaults["fel_max_energy_kj_mol"]) * energy_scale
    return None
