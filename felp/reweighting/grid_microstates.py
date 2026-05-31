"""Regular CV-grid discretization shared by FELP microstates and msm-fit."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from felp.core import ConfigurationError, MicrostateDefinition
from felp.msm.repartition import (
    feature_labels_from_spec,
    measurement_scales_for_labels,
)


def _safe_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "cv"


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
    bin_counts: Sequence[int],
    frame_count: int,
    *,
    target_frames_per_state: int,
    max_total_states: int,
) -> tuple[int, ...]:
    counts = [max(1, int(value)) for value in bin_counts]
    max_states_by_support = max(1, frame_count // int(target_frames_per_state))
    max_states = max(1, min(int(max_total_states), max_states_by_support))

    while math.prod(counts) > max_states and any(value > 1 for value in counts):
        axis = int(np.argmax(counts))
        if counts[axis] <= 1:
            break
        counts[axis] -= 1
    return tuple(counts)


def _edges_for_values(values: np.ndarray, bin_count: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ConfigurationError("Cannot build a CV grid without finite feature values.")

    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    pad = max(abs(lower), abs(upper), 1.0) * 1.0e-6
    if span <= 0.0:
        return np.asarray([lower - pad, upper + pad], dtype=float)
    return np.linspace(lower - pad, upper + pad, int(bin_count) + 1, dtype=float)


def _digitize_axis(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return bin indices in ``[0, len(edges) - 2]`` for one axis."""

    count = len(edges) - 1
    if count <= 0:
        raise ConfigurationError("Grid edges must define at least one bin.")
    indices = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(indices, 0, count - 1).astype(int)


def _state_id_from_indices(cv_names: Sequence[str], indices: Sequence[int]) -> str:
    parts: list[str] = []
    for name, index in zip(cv_names, indices, strict=True):
        parts.extend([_safe_token(str(name)), f"{int(index):02d}"])
    return "_".join(parts)


def _linear_index(multi_index: Sequence[int], shape: Sequence[int]) -> int:
    index = 0
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        index += int(multi_index[axis]) * stride
        stride *= int(shape[axis])
    return index


@dataclass(frozen=True)
class GridDiscretization:
    """Regular lattice discretization in feature space."""

    cv_names: tuple[str, ...]
    bin_counts: tuple[int, ...]
    edges: tuple[np.ndarray, ...]
    state_ids: tuple[str, ...]
    centers: np.ndarray
    bin_indices_by_state: tuple[dict[str, int], ...]

    @property
    def ndim(self) -> int:
        return len(self.cv_names)

    @property
    def state_count(self) -> int:
        return len(self.state_ids)

    def lookup_state_index(self, multi_index: Sequence[int]) -> int:
        return _linear_index(multi_index, self.bin_counts)


@dataclass(frozen=True)
class GridClusterResult:
    """Grid clustering output compatible with the MSM pipeline."""

    dtrajs: tuple[np.ndarray, ...]
    clustercenters: np.ndarray
    discretization: GridDiscretization


def infer_grid_bin_counts(
    feature_rows: np.ndarray,
    *,
    requested_bins: Sequence[int] | None,
    min_bins_per_axis: int,
    max_bins_per_axis: int,
    target_frames_per_state: int,
    max_total_states: int,
) -> tuple[int, ...]:
    """Choose per-axis bin counts from explicit values or Freedman–Diaconis heuristics."""

    if feature_rows.ndim != 2:
        raise ConfigurationError("Feature rows must be a 2D array.")
    ndim = feature_rows.shape[1]
    if ndim < 1:
        raise ConfigurationError("At least one feature dimension is required for grid clustering.")

    if requested_bins:
        if len(requested_bins) != ndim:
            raise ConfigurationError(
                f"grid_bins must provide exactly {ndim} values for {ndim} feature(s); "
                f"got {len(requested_bins)}."
            )
        counts = tuple(max(1, int(value)) for value in requested_bins)
        if math.prod(counts) > int(max_total_states):
            raise ConfigurationError(
                f"Requested grid size {math.prod(counts)} exceeds grid_max_total_states="
                f"{int(max_total_states)}."
            )
        return counts

    counts = tuple(
        _freedman_diaconis_bin_count(
            feature_rows[:, axis],
            min_bins=min_bins_per_axis,
            max_bins=max_bins_per_axis,
        )
        for axis in range(ndim)
    )

    finite_rows = feature_rows[np.all(np.isfinite(feature_rows), axis=1)]
    frame_count = int(finite_rows.shape[0]) if finite_rows.size else int(feature_rows.shape[0])
    return _cap_joint_bin_count(
        counts,
        frame_count,
        target_frames_per_state=target_frames_per_state,
        max_total_states=max_total_states,
    )


def build_grid_discretization(
    feature_rows: np.ndarray,
    cv_names: Sequence[str],
    *,
    bin_counts: Sequence[int],
) -> GridDiscretization:
    """Build a full rectilinear grid over all bin combinations."""

    if feature_rows.ndim != 2:
        raise ConfigurationError("Feature rows must be a 2D array.")
    names = tuple(str(name) for name in cv_names)
    if len(names) != feature_rows.shape[1]:
        raise ConfigurationError(
            "Feature label count must match the number of feature columns."
        )
    counts = tuple(max(1, int(value)) for value in bin_counts)
    if len(counts) != len(names):
        raise ConfigurationError("bin_counts length must match cv_names length.")

    edges = tuple(_edges_for_values(feature_rows[:, axis], counts[axis]) for axis in range(len(names)))
    state_ids: list[str] = []
    centers: list[list[float]] = []
    bin_indices_by_state: list[dict[str, int]] = []

    index_ranges = [range(count) for count in counts]
    for multi_index in itertools.product(*index_ranges):
        bounds = []
        center = []
        bin_indices: dict[str, int] = {}
        for axis, name in enumerate(names):
            lower = float(edges[axis][multi_index[axis]])
            upper = float(edges[axis][multi_index[axis] + 1])
            bounds.append((lower, upper))
            center.append((lower + upper) / 2.0)
            bin_indices[name] = int(multi_index[axis])
        state_ids.append(_state_id_from_indices(names, multi_index))
        centers.append(center)
        bin_indices_by_state.append(bin_indices)

    return GridDiscretization(
        cv_names=names,
        bin_counts=counts,
        edges=edges,
        state_ids=tuple(state_ids),
        centers=np.asarray(centers, dtype=float),
        bin_indices_by_state=tuple(bin_indices_by_state),
    )


def assign_rows_to_state_indices(
    feature_rows: np.ndarray,
    discretization: GridDiscretization,
) -> np.ndarray:
    """Assign feature rows to linear grid state indices."""

    if feature_rows.ndim != 2 or feature_rows.shape[1] != discretization.ndim:
        raise ConfigurationError("Feature row shape does not match grid dimensionality.")

    axis_indices = [
        _digitize_axis(feature_rows[:, axis], discretization.edges[axis])
        for axis in range(discretization.ndim)
    ]
    stacked = np.stack(axis_indices, axis=1)
    return np.asarray(
        [
            discretization.lookup_state_index(tuple(int(value) for value in row))
            for row in stacked
        ],
        dtype=int,
    )


def physical_cv_ranges_for_bin(
    feature_spec: dict[str, Any],
    cv_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, tuple[float, float]]:
    """Convert feature-space bin bounds into FELP physical CV ranges."""

    labels = tuple(str(name) for name in cv_names)
    scales = measurement_scales_for_labels(feature_spec, labels)
    cv_ranges: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(labels):
        scale = float(scales[index])
        if scale <= 0.0:
            raise ConfigurationError(f"Invalid measurement scale for CV {name!r}.")
        cv_ranges[name] = (float(lower[index]) / scale, float(upper[index]) / scale)
    return cv_ranges


def build_grid_microstate_definitions(
    discretization: GridDiscretization,
    *,
    feature_spec: dict[str, Any],
    reference_probabilities: dict[int, float],
) -> tuple[MicrostateDefinition, ...]:
    """Build FELP microstate definitions for every grid cell."""

    definitions: list[MicrostateDefinition] = []
    for state_index, state_id in enumerate(discretization.state_ids):
        bin_indices = discretization.bin_indices_by_state[state_index]
        lower = []
        upper = []
        for axis, name in enumerate(discretization.cv_names):
            idx = bin_indices[name]
            lower.append(float(discretization.edges[axis][idx]))
            upper.append(float(discretization.edges[axis][idx + 1]))
        cv_ranges = physical_cv_ranges_for_bin(
            feature_spec,
            discretization.cv_names,
            lower,
            upper,
        )
        center = discretization.centers[state_index]
        plot_x = float(center[0]) if center.size >= 1 else None
        plot_y = float(center[1]) if center.size >= 2 else None
        definitions.append(
            MicrostateDefinition(
                state_id=state_id,
                label=state_id,
                cv_ranges=cv_ranges,
                reference_probability=float(reference_probabilities.get(state_index, 0.0)),
                plot_x=plot_x,
                plot_y=plot_y,
                state_level="cluster",
                metadata={
                    "cluster_index": state_index,
                    "grid_cluster": True,
                    "active": state_index in reference_probabilities,
                    "bin_indices": dict(bin_indices),
                },
            )
        )
    return tuple(definitions)


def grid_cluster_trajectories(
    trajectories: Sequence[np.ndarray],
    *,
    feature_spec: dict[str, Any],
    grid_bins: Sequence[int] | None = None,
    min_bins_per_axis: int = 3,
    max_bins_per_axis: int = 64,
    target_frames_per_state: int = 100,
    max_total_states: int = 2500,
) -> GridClusterResult:
    """Discretize trajectories on a regular feature-space grid."""

    labels = feature_labels_from_spec(feature_spec)
    if not labels:
        metadata_columns = feature_spec.get("metadata", {}).get("columns")
        if isinstance(metadata_columns, list) and metadata_columns:
            labels = tuple(str(name) for name in metadata_columns)
    if not labels:
        raise ConfigurationError(
            "Grid clustering requires named feature columns in the feature spec."
        )

    arrays = [np.asarray(values, dtype=float) for values in trajectories]
    if not arrays:
        raise ConfigurationError("At least one trajectory is required for grid clustering.")
    ndim = arrays[0].shape[1] if arrays[0].ndim == 2 else 1
    if ndim < 1 or ndim > 3:
        raise ConfigurationError(
            f"Grid clustering supports 1–3 feature dimensions; got {ndim}."
        )
    for array in arrays:
        if array.ndim != 2 or array.shape[1] != ndim:
            raise ConfigurationError("All trajectories must share the same feature dimension.")

    stacked = np.concatenate(arrays, axis=0)
    bin_counts = infer_grid_bin_counts(
        stacked,
        requested_bins=grid_bins,
        min_bins_per_axis=min_bins_per_axis,
        max_bins_per_axis=max_bins_per_axis,
        target_frames_per_state=target_frames_per_state,
        max_total_states=max_total_states,
    )
    discretization = build_grid_discretization(stacked, labels, bin_counts=bin_counts)
    dtrajs = tuple(assign_rows_to_state_indices(array, discretization) for array in arrays)
    return GridClusterResult(
        dtrajs=dtrajs,
        clustercenters=np.asarray(discretization.centers, dtype=float),
        discretization=discretization,
    )


def grid_discretization_to_dict(discretization: GridDiscretization) -> dict[str, Any]:
    """Serialize a grid discretization for MSM bundle manifests."""

    return {
        "cv_names": list(discretization.cv_names),
        "bin_counts": list(discretization.bin_counts),
        "edges": [
            [float(value) for value in edge]
            for edge in discretization.edges
        ],
        "state_count": discretization.state_count,
    }
