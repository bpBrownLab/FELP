"""CV-grid topology for spatially coupled reweighting estimators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from felp.core import MicrostateDefinition
from .atlas import MicrostateAtlas


@dataclass(frozen=True)
class GridTopology:
    """Neighbor relationships between discrete microstates on a 2D grid."""

    neighbors: dict[str, tuple[str, ...]]
    source: str
    grid_shape: tuple[int, int] | None = None
    cv_names: tuple[str, str] | None = None
    warnings: tuple[str, ...] = ()

    def neighbor_ids(self, state_id: str) -> tuple[str, ...]:
        return self.neighbors.get(state_id, ())


def _infer_cv_names(definitions: Iterable[MicrostateDefinition]) -> tuple[str, str] | None:
    names: list[str] = []
    for definition in definitions:
        if len(definition.cv_ranges) >= 2:
            for key in sorted(definition.cv_ranges):
                if key not in names:
                    names.append(key)
                if len(names) >= 2:
                    return names[0], names[1]
        if definition.phi_range is not None and "phi" not in names:
            names.append("phi")
        if definition.psi_range is not None and "psi" not in names:
            names.append("psi")
        if len(names) >= 2:
            return names[0], names[1]
    if "phi" in names or "psi" in names:
        if "phi" not in names:
            names.insert(0, "phi")
        if "psi" not in names:
            names.append("psi")
    if len(names) >= 2:
        return names[0], names[1]
    return None


def _topology_from_bin_indices(
    definitions: Iterable[MicrostateDefinition],
    cv_names: tuple[str, str],
) -> GridTopology | None:
    x_name, y_name = cv_names
    indices: dict[str, tuple[int, int]] = {}
    for definition in definitions:
        metadata = definition.metadata or {}
        bin_indices = metadata.get("bin_indices")
        if not isinstance(bin_indices, dict):
            continue
        if x_name not in bin_indices or y_name not in bin_indices:
            continue
        indices[definition.state_id] = (int(bin_indices[x_name]), int(bin_indices[y_name]))
    if len(indices) < 2:
        return None

    x_values = sorted({value[0] for value in indices.values()})
    y_values = sorted({value[1] for value in indices.values()})
    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    index_to_state = {
        (x_lookup[x_index], y_lookup[y_index]): state_id
        for state_id, (x_index, y_index) in indices.items()
    }
    neighbors: dict[str, list[str]] = defaultdict(list)
    for state_id, (x_index, y_index) in indices.items():
        grid_x = x_lookup[x_index]
        grid_y = y_lookup[y_index]
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = index_to_state.get((grid_x + dx, grid_y + dy))
            if neighbor is not None and neighbor != state_id:
                neighbors[state_id].append(neighbor)
    return GridTopology(
        neighbors={key: tuple(sorted(set(values))) for key, values in neighbors.items()},
        source="bin_indices",
        grid_shape=(len(x_values), len(y_values)),
        cv_names=cv_names,
    )


def _topology_from_regular_cv_ranges(
    definitions: Iterable[MicrostateDefinition],
    cv_names: tuple[str, str],
) -> GridTopology | None:
    x_name, y_name = cv_names
    x_centers: dict[float, list[MicrostateDefinition]] = defaultdict(list)
    y_centers: dict[float, list[MicrostateDefinition]] = defaultdict(list)
    for definition in definitions:
        x_bounds = definition.cv_ranges.get(x_name)
        y_bounds = definition.cv_ranges.get(y_name)
        if x_bounds is None or y_bounds is None:
            return None
        x_center = (float(x_bounds[0]) + float(x_bounds[1])) / 2.0
        y_center = (float(y_bounds[0]) + float(y_bounds[1])) / 2.0
        x_centers[x_center].append(definition)
        y_centers[y_center].append(definition)

    x_unique = sorted(x_centers)
    y_unique = sorted(y_centers)
    if len(x_unique) < 2 and len(y_unique) < 2:
        return None
    if len(x_unique) * len(y_unique) != len(list(definitions)):
        return None

    state_by_index: dict[tuple[int, int], str] = {}
    for definition in definitions:
        x_bounds = definition.cv_ranges[x_name]
        y_bounds = definition.cv_ranges[y_name]
        x_center = (float(x_bounds[0]) + float(x_bounds[1])) / 2.0
        y_center = (float(y_bounds[0]) + float(y_bounds[1])) / 2.0
        state_by_index[(x_unique.index(x_center), y_unique.index(y_center))] = (
            definition.state_id
        )

    neighbors: dict[str, list[str]] = defaultdict(list)
    for (grid_x, grid_y), state_id in state_by_index.items():
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = state_by_index.get((grid_x + dx, grid_y + dy))
            if neighbor is not None and neighbor != state_id:
                neighbors[state_id].append(neighbor)
    return GridTopology(
        neighbors={key: tuple(sorted(set(values))) for key, values in neighbors.items()},
        source="cv_ranges",
        grid_shape=(len(x_unique), len(y_unique)),
        cv_names=cv_names,
    )


def _topology_from_knn(
    definitions: Iterable[MicrostateDefinition],
    *,
    max_neighbors: int = 4,
) -> GridTopology:
    coords: list[tuple[str, float, float]] = []
    for definition in definitions:
        x_value = definition.plot_x
        y_value = definition.plot_y
        if x_value is None or y_value is None:
            if definition.cv_centers:
                items = sorted(definition.cv_centers.items())
                if len(items) >= 2:
                    x_value = float(items[0][1])
                    y_value = float(items[1][1])
        if x_value is None or y_value is None or not (
            math.isfinite(float(x_value)) and math.isfinite(float(y_value))
        ):
            continue
        coords.append((definition.state_id, float(x_value), float(y_value)))

    neighbors: dict[str, list[str]] = defaultdict(list)
    if len(coords) < 2:
        return GridTopology(
            neighbors={},
            source="knn",
            warnings=("Insufficient plot coordinates for kNN topology.",),
        )

    positions = np.asarray([(x, y) for _, x, y in coords], dtype=float)
    state_ids = [state_id for state_id, _, _ in coords]
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
) -> GridTopology | None:
    """Build a neighbor graph for spatially coupled reweighting estimators."""

    definitions = atlas.definitions
    if not definitions:
        return None

    cv_names = _infer_cv_names(definitions)
    if cv_names is not None:
        topology = _topology_from_bin_indices(definitions, cv_names)
        if topology is not None:
            return topology
        topology = _topology_from_regular_cv_ranges(definitions, cv_names)
        if topology is not None:
            return topology

    return _topology_from_knn(definitions, max_neighbors=max_neighbors)
