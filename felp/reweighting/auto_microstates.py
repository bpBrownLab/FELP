"""Automatic CV microstate generation from reference trajectory statistics."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
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

from felp.reweighting.grid_microstates import _cap_joint_bin_count as _cap_nd_bin_counts

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
    cv_names: tuple[str, ...]
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
    names = (
        [spec.name for spec in config.restraints.cv_specs]
        + [spec.name for spec in config.restraints.distance_cv_specs]
        + [spec.name for spec in config.restraints.dihedral_cv_specs]
        + [spec.name for spec in config.restraints.difference_distance_cv_specs]
    )
    return tuple(names)


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
        dihedral_cv_specs=config.restraints.dihedral_cv_specs,
        difference_distance_cv_specs=config.restraints.difference_distance_cv_specs,
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
    cv_names: tuple[str, ...],
    target_frames_per_state: int = _DEFAULT_TARGET_FRAMES_PER_STATE,
    min_bins_per_axis: int = _DEFAULT_MIN_BINS_PER_NONCONSTANT_AXIS,
    max_bins_per_axis: int = _DEFAULT_MAX_BINS_PER_AXIS,
    max_total_states: int = _DEFAULT_MAX_TOTAL_STATES,
) -> dict[str, Any]:
    """Build reusable microstate JSON from observed 1D–3D CV statistics."""

    if not cv_names or len(cv_names) > 3:
        raise ConfigurationError(
            "Automatic microstate generation requires 1–3 configured CVs, "
            f"but found {len(cv_names)}: {', '.join(cv_names) or '<none>'}."
        )

    target_frames_per_state = max(1, int(target_frames_per_state))
    min_bins_per_axis = max(1, int(min_bins_per_axis))
    max_bins_per_axis = max(min_bins_per_axis, int(max_bins_per_axis))
    max_total_states = max(1, int(max_total_states))

    arrays = [np.asarray(cv_values[name], dtype=float) for name in cv_names]
    finite_mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        if array.shape != arrays[0].shape:
            raise ConfigurationError(
                "All configured CV arrays must have the same length for auto microstates."
            )
        finite_mask &= np.isfinite(array)
    if not finite_mask.any():
        raise ConfigurationError(
            "Cannot auto-generate microstates because no frames have finite values "
            f"for all configured CVs: {', '.join(cv_names)}."
        )

    filtered = [array[finite_mask] for array in arrays]
    uncapped_bin_counts: dict[str, int] = {}
    bin_counts: list[int] = []
    for name, values in zip(cv_names, filtered, strict=True):
        count = _freedman_diaconis_bin_count(
            values,
            min_bins=min_bins_per_axis,
            max_bins=max_bins_per_axis,
        )
        uncapped_bin_counts[name] = int(count)
        bin_counts.append(int(count))

    capped_bin_counts = _cap_nd_bin_counts(
        bin_counts,
        int(finite_mask.sum()),
        target_frames_per_state=target_frames_per_state,
        max_total_states=max_total_states,
    )
    edges_list = [
        _edges_for_values(values, bin_count)
        for values, bin_count in zip(filtered, capped_bin_counts, strict=True)
    ]

    states: list[dict[str, Any]] = []
    index_ranges = [range(len(edges) - 1) for edges in edges_list]
    for indices in itertools.product(*index_ranges):
        cv_ranges: dict[str, list[float]] = {}
        bin_indices: dict[str, int] = {}
        centers: list[float] = []
        state_id_parts: list[str] = []
        for name, index, edges in zip(cv_names, indices, edges_list, strict=True):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            cv_ranges[name] = [lower, upper]
            bin_indices[name] = int(index)
            centers.append((lower + upper) / 2.0)
            state_id_parts.extend([_safe_token(name), f"{int(index):02d}"])
        state: dict[str, Any] = {
            "state_id": "_".join(state_id_parts),
            "label": "_".join(state_id_parts),
            "cv_ranges": cv_ranges,
            "plot_x": centers[0],
            "plot_y": centers[1] if len(centers) > 1 else 0.0,
            "metadata": {
                "auto_microstate": True,
                "bin_indices": bin_indices,
            },
        }
        states.append(state)

    metadata = {
        "generator": "felp.auto_microstates",
        "version": _AUTO_MICROSTATE_VERSION,
        "cv_names": list(cv_names),
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
            "uncapped_bin_counts": uncapped_bin_counts,
            "complete_joint_grid": True,
        },
        "axes": {
            name: _axis_stats(values, len(edges) - 1, edges)
            for name, values, edges in zip(cv_names, filtered, edges_list, strict=True)
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
    if not cv_names or len(cv_names) > 3:
        raise ConfigurationError(
            "Automatic microstate generation requires 1–3 configured CVs, "
            f"but found {len(cv_names)}: {', '.join(cv_names) or '<none>'}."
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
        cv_names=tuple(cv_names),
        state_count=len(payload["states"]),
        frame_count=int(metadata["frame_count"]),
        metadata=metadata,
    )
