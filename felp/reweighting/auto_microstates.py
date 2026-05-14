"""Automatic CV microstate generation from reference trajectory statistics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import tempfile
from typing import Any, Iterable

import numpy as np

from felp.conversion import StructureConversionService
from felp.core import ConfigurationError, RunConfig, SnapshotJob
from felp.restraints import BackboneDihedralCVProtocol

from .coordinate_cv import coordinate_cv_extractors

_AUTO_MICROSTATE_VERSION = 1
_DEFAULT_TARGET_FRAMES_PER_STATE = 100
_DEFAULT_MIN_BINS_PER_NONCONSTANT_AXIS = 3
_DEFAULT_MAX_BINS_PER_AXIS = 64
_DEFAULT_MAX_TOTAL_STATES = 2500


@dataclass(frozen=True)
class AutoMicrostateFile:
    """Description of an automatically generated microstate definition file."""

    path: str
    cv_names: tuple[str, str]
    state_count: int
    frame_count: int
    metadata: dict[str, Any]


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
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
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _safe_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "cv"


def _configured_cv_names(config: RunConfig) -> tuple[str, ...]:
    return tuple(
        [spec.name for spec in config.restraints.cv_specs]
        + [spec.name for spec in config.restraints.distance_cv_specs]
    )


def _measure_configured_cvs(
    jobs: Iterable[SnapshotJob],
    config: RunConfig,
) -> dict[str, np.ndarray]:
    jobs = list(jobs)
    cv_names = _configured_cv_names(config)
    values: dict[str, list[float]] = {name: [] for name in cv_names}

    conversion_service = StructureConversionService(
        pyrosetta_init_options=config.rosetta.init_options,
        radii_set=config.input.radii_set,
    )
    template_structure = conversion_service.load_topology(config.input.topology_file)
    cv_protocol = BackboneDihedralCVProtocol(
        cv_specs=config.restraints.cv_specs,
        distance_cv_specs=config.restraints.distance_cv_specs,
        force_constant=config.restraints.effective_rosetta_force_constant(),
    )
    coordinate_extractors = coordinate_cv_extractors(template_structure, cv_protocol)

    for job in jobs:
        coordinates = np.asarray(job.coordinates_angstrom, dtype=float)
        for name in cv_names:
            values[name].append(float(coordinate_extractors[name](coordinates)))

    return {name: np.asarray(raw_values, dtype=float) for name, raw_values in values.items()}


def _freedman_diaconis_bin_count(
    values: np.ndarray,
    *,
    min_bins: int,
    max_bins: int,
) -> int:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 1

    value_range = float(finite.max() - finite.min())
    if value_range <= 0.0:
        return 1

    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr > 0.0:
        width = 2.0 * iqr / (finite.size ** (1.0 / 3.0))
        if width > 0.0:
            bins = int(math.ceil(value_range / width))
        else:
            bins = int(math.ceil(math.sqrt(finite.size)))
    else:
        bins = int(math.ceil(math.sqrt(finite.size)))

    bins = max(int(min_bins), bins)
    return min(int(max_bins), bins)


def _cap_joint_bin_count(
    x_bins: int,
    y_bins: int,
    frame_count: int,
    *,
    target_frames_per_state: int,
    max_total_states: int,
) -> tuple[int, int]:
    max_states_by_support = max(1, frame_count // int(target_frames_per_state))
    max_states = max(4, min(int(max_total_states), max_states_by_support))
    x_bins = max(1, int(x_bins))
    y_bins = max(1, int(y_bins))

    while x_bins * y_bins > max_states and (x_bins > 1 or y_bins > 1):
        if x_bins >= y_bins and x_bins > 1:
            x_bins -= 1
        elif y_bins > 1:
            y_bins -= 1
        else:
            break
    return x_bins, y_bins


def _edges_for_values(values: np.ndarray, bin_count: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ConfigurationError("Cannot auto-generate microstates without finite CV values.")

    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    pad = max(abs(lower), abs(upper), 1.0) * 1.0e-6
    if span <= 0.0:
        return np.asarray([lower - pad, upper + pad], dtype=float)
    return np.linspace(lower - pad, upper + pad, int(bin_count) + 1, dtype=float)


def _axis_stats(values: np.ndarray, bins: int, edges: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    q25, q50, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
    return {
        "min": float(finite.min()),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "bin_count": int(bins),
        "edges": [float(value) for value in edges],
    }


def build_auto_microstate_payload(
    *,
    cv_values: dict[str, np.ndarray],
    cv_names: tuple[str, str],
    target_frames_per_state: int = _DEFAULT_TARGET_FRAMES_PER_STATE,
    min_bins_per_axis: int = _DEFAULT_MIN_BINS_PER_NONCONSTANT_AXIS,
    max_bins_per_axis: int = _DEFAULT_MAX_BINS_PER_AXIS,
    max_total_states: int = _DEFAULT_MAX_TOTAL_STATES,
) -> dict[str, Any]:
    """Build reusable microstate JSON from observed 2D CV statistics."""

    target_frames_per_state = max(1, int(target_frames_per_state))
    min_bins_per_axis = max(1, int(min_bins_per_axis))
    max_bins_per_axis = max(min_bins_per_axis, int(max_bins_per_axis))
    max_total_states = max(1, int(max_total_states))

    x_name, y_name = cv_names
    x_values = cv_values[x_name]
    y_values = cv_values[y_name]
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite_mask.any():
        raise ConfigurationError(
            "Cannot auto-generate microstates because no frames have finite values "
            f"for both {x_name!r} and {y_name!r}."
        )

    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    x_bins = _freedman_diaconis_bin_count(
        x_values,
        min_bins=min_bins_per_axis,
        max_bins=max_bins_per_axis,
    )
    y_bins = _freedman_diaconis_bin_count(
        y_values,
        min_bins=min_bins_per_axis,
        max_bins=max_bins_per_axis,
    )
    uncapped_x_bins = int(x_bins)
    uncapped_y_bins = int(y_bins)
    x_bins, y_bins = _cap_joint_bin_count(
        x_bins,
        y_bins,
        int(finite_mask.sum()),
        target_frames_per_state=target_frames_per_state,
        max_total_states=max_total_states,
    )
    x_edges = _edges_for_values(x_values, x_bins)
    y_edges = _edges_for_values(y_values, y_bins)

    x_token = _safe_token(x_name)
    y_token = _safe_token(y_name)
    states: list[dict[str, Any]] = []
    for x_index in range(len(x_edges) - 1):
        for y_index in range(len(y_edges) - 1):
            x_lower = float(x_edges[x_index])
            x_upper = float(x_edges[x_index + 1])
            y_lower = float(y_edges[y_index])
            y_upper = float(y_edges[y_index + 1])
            state_id = f"{x_token}_{x_index:02d}_{y_token}_{y_index:02d}"
            states.append(
                {
                    "state_id": state_id,
                    "label": state_id,
                    "cv_ranges": {
                        x_name: [x_lower, x_upper],
                        y_name: [y_lower, y_upper],
                    },
                    "plot_x": (x_lower + x_upper) / 2.0,
                    "plot_y": (y_lower + y_upper) / 2.0,
                    "metadata": {
                        "auto_microstate": True,
                        "bin_indices": {x_name: x_index, y_name: y_index},
                    },
                }
            )

    metadata = {
        "generator": "felp.auto_microstates",
        "version": _AUTO_MICROSTATE_VERSION,
        "cv_names": [x_name, y_name],
        "frame_count": int(finite_mask.sum()),
        "state_count": len(states),
        "policy": {
            "marginal_rule": "freedman_diaconis",
            "target_frames_per_state": target_frames_per_state,
            "min_bins_per_axis": min_bins_per_axis,
            "max_total_states": max_total_states,
            "max_states_by_support": int(
                max(1, int(finite_mask.sum()) // target_frames_per_state)
            ),
            "max_bins_per_axis": max_bins_per_axis,
            "uncapped_bin_counts": {
                x_name: uncapped_x_bins,
                y_name: uncapped_y_bins,
            },
            "complete_joint_grid": True,
        },
        "axes": {
            x_name: _axis_stats(x_values, len(x_edges) - 1, x_edges),
            y_name: _axis_stats(y_values, len(y_edges) - 1, y_edges),
        },
    }
    return {"metadata": metadata, "states": states}


def write_auto_microstate_file(
    *,
    jobs: Iterable[SnapshotJob],
    config: RunConfig,
    output_path: str,
) -> AutoMicrostateFile:
    """Measure configured CVs and write an auto-generated microstate JSON file."""

    cv_names = _configured_cv_names(config)
    if len(cv_names) != 2:
        raise ConfigurationError(
            "Automatic microstate generation currently requires exactly two configured "
            f"CVs, but found {len(cv_names)}: {', '.join(cv_names) or '<none>'}."
        )

    cv_values = _measure_configured_cvs(jobs, config)
    payload = build_auto_microstate_payload(
        cv_values=cv_values,
        cv_names=cv_names,
        target_frames_per_state=config.microstates.auto_target_frames_per_state,
        min_bins_per_axis=config.microstates.auto_min_bins_per_axis,
        max_bins_per_axis=config.microstates.auto_max_bins_per_axis,
        max_total_states=config.microstates.auto_max_total_states,
    )
    _write_json_atomic(output_path, payload)
    metadata = dict(payload["metadata"])
    return AutoMicrostateFile(
        path=output_path,
        cv_names=tuple(cv_names),  # type: ignore[arg-type]
        state_count=len(payload["states"]),
        frame_count=int(metadata["frame_count"]),
        metadata=metadata,
    )
