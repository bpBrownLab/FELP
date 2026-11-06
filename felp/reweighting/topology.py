"""CV-grid topology for spatially coupled reweighting estimators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import numpy as np

from felp.core import ConfigurationError, MicrostateDefinition
from .atlas import MicrostateAtlas


def normalize_grid_connectivity(
    connectivity: str,
    *,
    chebyshev_radius: int = 1,
) -> tuple[str, int]:
    """Parse grid connectivity mode and Chebyshev radius.

    Supported values:

    - ``cardinal`` — face-sharing neighbors (2 in 1D, 4 in 2D, 6 in 3D).
    - ``chebyshev`` — use ``chebyshev_radius`` (default 1).
    - ``chebyshev1``, ``chebyshev2``, ... — Chebyshev ball with L_inf radius N.
    """

    raw = str(connectivity).strip().lower()
    if raw == "cardinal":
        return "cardinal", 0
    if raw == "chebyshev":
        return "chebyshev", max(1, int(chebyshev_radius))
    if raw.startswith("chebyshev"):
        suffix = raw[len("chebyshev") :]
        if suffix == "":
            return "chebyshev", max(1, int(chebyshev_radius))
        if suffix.isdigit():
            return "chebyshev", max(1, int(suffix))
    raise ConfigurationError(
        "grid_connectivity must be 'cardinal', 'chebyshev', or 'chebyshevN' "
        f"with positive integer N; got {connectivity!r}."
    )


def grid_neighbor_offsets(
    *,
    ndim: int,
    connectivity: str,
    chebyshev_radius: int,
) -> tuple[tuple[int, ...], ...]:
    """Return integer grid offsets for the requested connectivity stencil."""

    if ndim < 1:
        raise ConfigurationError(f"Grid neighbor offsets require ndim >= 1; got {ndim}.")

    if connectivity == "cardinal":
        offsets: list[tuple[int, ...]] = []
        for axis in range(ndim):
            for step in (-1, 1):
                offset = [0] * ndim
                offset[axis] = step
                offsets.append(tuple(offset))
        return tuple(offsets)

    if connectivity == "chebyshev":
        radius = max(1, int(chebyshev_radius))
        offsets = []
        for offset in itertools.product(*[range(-radius, radius + 1)] * ndim):
            if all(value == 0 for value in offset):
                continue
            if max(abs(value) for value in offset) <= radius:
                offsets.append(offset)
        return tuple(offsets)

    raise ConfigurationError(
        f"Unsupported grid connectivity mode {connectivity!r} for offset generation."
    )


@dataclass(frozen=True)
class GridTopology:
    """Neighbor relationships between discrete microstates on a regular CV grid."""

    neighbors: dict[str, tuple[str, ...]]
    source: str
    grid_shape: tuple[int, ...] | None = None
    cv_names: tuple[str, ...] | None = None
    connectivity: str = "cardinal"
    chebyshev_radius: int | None = None
    periodic_cv_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def neighbor_ids(self, state_id: str) -> tuple[str, ...]:
        return self.neighbors.get(state_id, ())


def _infer_cv_names(definitions: Iterable[MicrostateDefinition]) -> tuple[str, ...] | None:
    """Infer ordered CV axis names shared by grid microstate definitions."""

    names: list[str] = []
    for definition in definitions:
        if definition.cv_ranges:
            for key in sorted(definition.cv_ranges):
                if key not in names:
                    names.append(key)
            if names:
                return tuple(names)
        metadata = definition.metadata or {}
        bin_indices = metadata.get("bin_indices")
        if isinstance(bin_indices, dict) and bin_indices:
            ordered = [str(key) for key in sorted(bin_indices)]
            if ordered:
                return tuple(ordered)
        if definition.phi_range is not None and "phi" not in names:
            names.append("phi")
        if definition.psi_range is not None and "psi" not in names:
            names.append("psi")
        if len(names) >= 2:
            return tuple(names[:2])
    if names:
        return tuple(names)
    return None


def _normalize_index_map(
    raw_indices: dict[str, tuple[int, ...]],
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...]]:
    """Map raw bin indices to contiguous 0..N-1 lattice coordinates per axis."""

    if not raw_indices:
        return {}, ()
    ndim = len(next(iter(raw_indices.values())))
    axis_values = [
        sorted({indices[axis] for indices in raw_indices.values()}) for axis in range(ndim)
    ]
    lookups = [{value: index for index, value in enumerate(values)} for values in axis_values]
    index_map = {
        state_id: tuple(lookups[axis][indices[axis]] for axis in range(ndim))
        for state_id, indices in raw_indices.items()
    }
    shape = tuple(len(values) for values in axis_values)
    return index_map, shape


def _neighbors_from_index_map(
    index_map: dict[str, tuple[int, ...]],
    index_to_state: dict[tuple[int, ...], str],
    *,
    offsets: tuple[tuple[int, ...], ...],
    grid_shape: tuple[int, ...],
    periodic_axes: frozenset[int] = frozenset(),
) -> dict[str, tuple[str, ...]]:
    """Build neighbor lists from normalized grid coordinates and offset stencil."""

    ndim = len(next(iter(index_map.values()))) if index_map else 0
    neighbors: dict[str, list[str]] = defaultdict(list)
    for state_id, position in index_map.items():
        for offset in offsets:
            if len(offset) != ndim:
                continue
            neighbor_position = tuple(
                (
                    (position[axis] + offset[axis]) % grid_shape[axis]
                    if axis in periodic_axes
                    else position[axis] + offset[axis]
                )
                for axis in range(ndim)
            )
            neighbor = index_to_state.get(neighbor_position)
            if neighbor is not None and neighbor != state_id:
                neighbors[state_id].append(neighbor)
    return {key: tuple(sorted(set(values))) for key, values in neighbors.items()}


def _extract_bin_indices(
    definition: MicrostateDefinition,
    cv_names: tuple[str, ...],
) -> tuple[int, ...] | None:
    metadata = definition.metadata or {}
    bin_indices = metadata.get("bin_indices")
    if isinstance(bin_indices, dict):
        try:
            return tuple(int(bin_indices[name]) for name in cv_names)
        except KeyError:
            return None
    return None


def _topology_from_bin_indices(
    definitions: Iterable[MicrostateDefinition],
    cv_names: tuple[str, ...],
    *,
    offsets: tuple[tuple[int, ...], ...],
    connectivity: str,
    chebyshev_radius: int | None,
    periodic_axes: frozenset[int],
    periodic_cv_names: tuple[str, ...],
) -> GridTopology | None:
    raw_indices: dict[str, tuple[int, ...]] = {}
    for definition in definitions:
        indices = _extract_bin_indices(definition, cv_names)
        if indices is not None:
            raw_indices[definition.state_id] = indices
    if len(raw_indices) < 2:
        return None

    index_map, shape = _normalize_index_map(raw_indices)
    index_to_state = {position: state_id for state_id, position in index_map.items()}
    neighbors = _neighbors_from_index_map(
        index_map,
        index_to_state,
        offsets=offsets,
        grid_shape=shape,
        periodic_axes=periodic_axes,
    )
    return GridTopology(
        neighbors=neighbors,
        source="bin_indices",
        grid_shape=shape,
        cv_names=cv_names,
        connectivity=connectivity,
        chebyshev_radius=chebyshev_radius,
        periodic_cv_names=periodic_cv_names,
    )


def _topology_from_regular_cv_ranges(
    definitions: Iterable[MicrostateDefinition],
    cv_names: tuple[str, ...],
    *,
    offsets: tuple[tuple[int, ...], ...],
    connectivity: str,
    chebyshev_radius: int | None,
    periodic_axes: frozenset[int],
    periodic_cv_names: tuple[str, ...],
) -> GridTopology | None:
    definition_list = list(definitions)
    axis_centers: list[dict[float, None]] = [{} for _ in cv_names]
    raw_indices: dict[str, tuple[int, ...]] = {}

    for definition in definition_list:
        for axis, name in enumerate(cv_names):
            bounds = definition.cv_ranges.get(name)
            if bounds is None:
                return None
            center = (float(bounds[0]) + float(bounds[1])) / 2.0
            centers = axis_centers[axis]
            if center not in centers:
                centers[center] = None

    # Assign indices only after every center is known. Otherwise a shuffled
    # definition list can assign multiple bins the same provisional index.
    lookups = [
        {center: index for index, center in enumerate(sorted(centers))}
        for centers in axis_centers
    ]
    for definition in definition_list:
        raw_indices[definition.state_id] = tuple(
            lookups[axis][sum(map(float, definition.cv_ranges[name])) / 2.0]
            for axis, name in enumerate(cv_names)
        )

    unique_counts = [len(centers) for centers in axis_centers]
    if all(count < 2 for count in unique_counts) and not periodic_axes:
        return None
    if math.prod(unique_counts) != len(definition_list):
        return None

    index_map, shape = _normalize_index_map(raw_indices)
    index_to_state = {position: state_id for state_id, position in index_map.items()}
    neighbors = _neighbors_from_index_map(
        index_map,
        index_to_state,
        offsets=offsets,
        grid_shape=shape,
        periodic_axes=periodic_axes,
    )
    return GridTopology(
        neighbors=neighbors,
        source="cv_ranges",
        grid_shape=shape,
        cv_names=cv_names,
        connectivity=connectivity,
        chebyshev_radius=chebyshev_radius,
        periodic_cv_names=periodic_cv_names,
    )


def _topology_from_knn(
    definitions: Iterable[MicrostateDefinition],
    *,
    max_neighbors: int = 4,
) -> GridTopology:
    coords: list[tuple[str, np.ndarray]] = []
    for definition in definitions:
        vector: list[float] = []
        if definition.plot_x is not None:
            vector.append(float(definition.plot_x))
        if definition.plot_y is not None:
            vector.append(float(definition.plot_y))
        if len(vector) < 2 and definition.cv_centers:
            for name in sorted(definition.cv_centers):
                value = float(definition.cv_centers[name])
                if math.isfinite(value):
                    vector.append(value)
        if len(vector) < 1:
            continue
        coords.append((definition.state_id, np.asarray(vector, dtype=float)))

    neighbors: dict[str, list[str]] = defaultdict(list)
    if len(coords) < 2:
        return GridTopology(
            neighbors={},
            source="knn",
            warnings=("Insufficient plot coordinates for kNN topology.",),
        )

    positions = np.stack([values for _, values in coords], axis=0)
    state_ids = [state_id for state_id, _ in coords]
    for index, state_id in enumerate(state_ids):
        distances = np.sum((positions - positions[index]) ** 2, axis=1)
        order = np.argsort(distances)
        picked: list[str] = []
        for neighbor_index in order[1:]:
            neighbor_id = state_ids[int(neighbor_index)]
            if neighbor_id == state_id:
                continue
            picked.append(neighbor_id)
            if len(picked) >= max_neighbors:
                break
        neighbors[state_id] = picked

    return GridTopology(
        neighbors={key: tuple(values) for key, values in neighbors.items()},
        source="knn",
        warnings=(
            "kNN topology from plot coordinates is approximate for non-grid atlases.",
        ),
    )


def build_grid_topology(
    atlas: MicrostateAtlas,
    *,
    max_neighbors: int = 4,
    grid_connectivity: str = "cardinal",
    chebyshev_radius: int = 3,
    periodic_cv_names: Iterable[str] = (),
) -> GridTopology | None:
    """Build a neighbor graph for spatially coupled reweighting estimators."""

    definitions = atlas.definitions
    if not definitions:
        return None

    connectivity, radius = normalize_grid_connectivity(
        grid_connectivity,
        chebyshev_radius=chebyshev_radius,
    )
    stored_radius = radius if connectivity == "chebyshev" else None

    requested_periodic_cv_names = tuple(dict.fromkeys(
        str(name).strip() for name in periodic_cv_names if str(name).strip()
    ))

    cv_names = _infer_cv_names(definitions)
    if cv_names is not None:
        unknown_periodic_cv_names = tuple(
            name for name in requested_periodic_cv_names if name not in cv_names
        )
        if unknown_periodic_cv_names:
            raise ConfigurationError(
                "Periodic neighbor CVs are not present in the microstate grid: "
                + ", ".join(unknown_periodic_cv_names)
            )
        periodic_axes = frozenset(
            cv_names.index(name) for name in requested_periodic_cv_names
        )
        ndim = len(cv_names)
        offsets = grid_neighbor_offsets(
            ndim=ndim,
            connectivity=connectivity,
            chebyshev_radius=radius,
        )
        topology = _topology_from_bin_indices(
            definitions,
            cv_names,
            offsets=offsets,
            connectivity=connectivity,
            chebyshev_radius=stored_radius,
            periodic_axes=periodic_axes,
            periodic_cv_names=requested_periodic_cv_names,
        )
        if topology is not None:
            return topology
        topology = _topology_from_regular_cv_ranges(
            definitions,
            cv_names,
            offsets=offsets,
            connectivity=connectivity,
            chebyshev_radius=stored_radius,
            periodic_axes=periodic_axes,
            periodic_cv_names=requested_periodic_cv_names,
        )
        if topology is not None:
            return topology

    if requested_periodic_cv_names:
        raise ConfigurationError(
            "Periodic neighbor CVs require a regular CV grid; "
            "kNN fallback cannot apply periodic wrapping."
        )
    return _topology_from_knn(definitions, max_neighbors=max_neighbors)
