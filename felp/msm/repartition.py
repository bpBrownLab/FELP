"""Nearest-neighbor assignment from FELP CV measurements to MSM bundle states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from felp.core import ConfigurationError, MicrostateDefinition

from .artifacts import MSMBundle

# PyEMMA reports interatomic distances in nm; FELP distance CVs are Angstrom.
_ANGSTROM_TO_NM = 0.1

_DISTANCE_FEATURE_KINDS = frozenset({"distances", "iterable_distances", "backbone_ca_distances"})
_ANGLE_FEATURE_KINDS = frozenset({"dihedrals", "angles", "backbone_torsions", "sidechain_torsions"})


def feature_labels_from_spec(feature_spec: dict[str, Any]) -> tuple[str, ...]:
    """Return ordered feature labels declared in a bundle feature specification."""

    labels: list[str] = []
    for operation in feature_spec.get("operations", ()):
        if not isinstance(operation, dict):
            continue
        raw_labels = operation.get("labels")
        if isinstance(raw_labels, list) and raw_labels:
            labels.extend(str(label) for label in raw_labels)
            continue
        kind = str(operation.get("kind", ""))
        if kind == "backbone_torsions":
            labels.extend(("phi", "psi"))
        elif kind in _DISTANCE_FEATURE_KINDS and not labels:
            pair_count = len(operation.get("pairs", ()))
            labels.extend(f"distance_{index}" for index in range(max(pair_count, 1)))
    return tuple(labels)


def measurement_scales_for_labels(
    feature_spec: dict[str, Any],
    labels: tuple[str, ...],
) -> tuple[float, ...]:
    """Return multipliers that map FELP assignment CV units into MSM feature units."""

    label_kinds: dict[str, str] = {}
    for operation in feature_spec.get("operations", ()):
        if not isinstance(operation, dict):
            continue
        kind = str(operation.get("kind", ""))
        raw_labels = operation.get("labels")
        if isinstance(raw_labels, list) and raw_labels:
            for label in raw_labels:
                label_kinds[str(label)] = kind
            continue
        if kind == "backbone_torsions":
            label_kinds.setdefault("phi", kind)
            label_kinds.setdefault("psi", kind)

    scales: list[float] = []
    for label in labels:
        kind = label_kinds.get(label, "")
        if kind in _DISTANCE_FEATURE_KINDS:
            scales.append(_ANGSTROM_TO_NM)
        elif kind in _ANGLE_FEATURE_KINDS or label in {"phi", "psi", "omega"}:
            scales.append(1.0)
        elif label.startswith("distance"):
            scales.append(_ANGSTROM_TO_NM)
        else:
            scales.append(1.0)
    return tuple(scales)


def _centers_from_definitions(
    definitions: tuple[MicrostateDefinition, ...],
) -> np.ndarray:
    """Build a (n_states, 2) center matrix from definition plot coordinates."""

    centers = np.full((len(definitions), 2), np.nan, dtype=float)
    for index, definition in enumerate(definitions):
        if definition.plot_x is None or definition.plot_y is None:
            raise ConfigurationError(
                f"MSM state {definition.state_id!r} is missing plot_x/plot_y coordinates "
                "required for geometric repartition."
            )
        centers[index, 0] = float(definition.plot_x)
        centers[index, 1] = float(definition.plot_y)
    return centers


def _feature_centers_for_level(bundle: MSMBundle, state_level: str) -> np.ndarray:
    """Return feature-space centers for the requested bundle state level."""

    embedding = bundle.embedding
    definitions = bundle.definitions_for_level(state_level)

    if state_level == "cluster":
        if "feature_centers" in embedding:
            return np.asarray(embedding["feature_centers"], dtype=float)
        if str(bundle.manifest.reducer.get("kind", "none")) == "none" and "cluster_centers" in embedding:
            return np.asarray(embedding["cluster_centers"], dtype=float)
        raise ConfigurationError(
            "MSM bundle is missing feature-space cluster centers required for repartition. "
            "Re-run msm-fit with a current FELP version, or use reducer=none so "
            "cluster_centers are stored in raw feature space."
        )

    if "metastable_feature_centers" in embedding:
        return np.asarray(embedding["metastable_feature_centers"], dtype=float)

    if str(bundle.manifest.reducer.get("kind", "none")) == "none":
        return _centers_from_definitions(definitions)

    raise ConfigurationError(
        "MSM bundle is missing metastable feature-space centers required for repartition "
        "with a reduced feature representation. Re-run msm-fit with a current FELP version."
    )


def feature_vector_to_physical_cv_ranges(
    feature_spec: dict[str, Any],
    labels: tuple[str, ...],
    vector: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Convert a PyEMMA feature vector into FELP-compatible degenerate cv_ranges."""

    scales = measurement_scales_for_labels(feature_spec, labels)
    cv_ranges: dict[str, tuple[float, float]] = {}
    for index, label in enumerate(labels):
        scale = float(scales[index])
        if scale <= 0.0:
            raise ConfigurationError(
                f"Invalid measurement scale for MSM feature label {label!r}."
            )
        center = float(vector[index]) / scale
        cv_ranges[str(label)] = (center, center)
    return cv_ranges


def enrich_msm_definitions(
    bundle: MSMBundle,
    definitions: tuple[MicrostateDefinition, ...],
    *,
    state_level: str,
) -> tuple[MicrostateDefinition, ...]:
    """Attach physical-unit cv_ranges and plot coordinates to MSM bundle states."""

    labels = feature_labels_from_spec(bundle.manifest.feature_spec)
    if not labels:
        return definitions

    feature_centers = _feature_centers_for_level(bundle, state_level)
    if feature_centers.shape[0] != len(definitions):
        raise ConfigurationError(
            "MSM feature centers and state definitions are inconsistent."
        )

    enriched: list[MicrostateDefinition] = []
    for index, definition in enumerate(definitions):
        metadata = dict(definition.metadata)
        bin_indices = metadata.get("bin_indices")
        has_grid_ranges = bool(definition.cv_ranges) and any(
            abs(float(bounds[1]) - float(bounds[0])) > 1.0e-12
            for bounds in definition.cv_ranges.values()
        )
        if has_grid_ranges and isinstance(bin_indices, dict) and bin_indices:
            cv_ranges = {
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in definition.cv_ranges.items()
            }
        else:
            cv_ranges = feature_vector_to_physical_cv_ranges(
                bundle.manifest.feature_spec,
                labels,
                feature_centers[index],
            )
        plot_x = definition.plot_x
        if len(labels) >= 1 and plot_x is None:
            low, high = cv_ranges[labels[0]]
            plot_x = (float(low) + float(high)) / 2.0
        plot_y = definition.plot_y
        if len(labels) >= 2 and plot_y is None:
            low, high = cv_ranges[labels[1]]
            plot_y = (float(low) + float(high)) / 2.0
        enriched.append(
            MicrostateDefinition(
                state_id=definition.state_id,
                label=definition.label,
                cv_ranges=cv_ranges,
                reference_probability=definition.reference_probability,
                plot_x=plot_x,
                plot_y=plot_y,
                state_level=definition.state_level,
                metadata=metadata,
            )
        )
    return tuple(enriched)


@dataclass(frozen=True)
class MSMAssignmentIndex:
    """Nearest-centroid lookup from FELP CV measurements to MSM state ids."""

    cv_names: tuple[str, ...]
    centers: np.ndarray
    measurement_scales: tuple[float, ...]
    state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        if centers.ndim != 2:
            raise ConfigurationError("MSM assignment centers must be a 2D array.")
        if centers.shape[0] != len(self.state_ids):
            raise ConfigurationError(
                "MSM assignment center count must match the number of exported states."
            )
        if centers.shape[1] != len(self.cv_names):
            raise ConfigurationError(
                "MSM assignment center dimension must match the configured feature labels."
            )
        if len(self.measurement_scales) != len(self.cv_names):
            raise ConfigurationError(
                "MSM measurement scale count must match the configured feature labels."
            )
        object.__setattr__(self, "centers", centers)

    @classmethod
    def from_bundle(cls, bundle: MSMBundle, state_level: str) -> "MSMAssignmentIndex":
        """Build an assignment index for the requested bundle state level."""

        labels = feature_labels_from_spec(bundle.manifest.feature_spec)
        if len(labels) < 1:
            raise ConfigurationError(
                "MSM bundle feature_spec does not declare feature labels required "
                "for geometric repartition."
            )
        centers = _feature_centers_for_level(bundle, state_level)
        if centers.shape[1] != len(labels):
            raise ConfigurationError(
                f"MSM bundle feature centers have dimension {centers.shape[1]}, "
                f"but feature_spec declares {len(labels)} labels: {labels}."
            )
        definitions = bundle.definitions_for_level(state_level)
        if centers.shape[0] != len(definitions):
            raise ConfigurationError(
                "MSM bundle state definitions and feature centers are inconsistent."
            )
        scales = measurement_scales_for_labels(bundle.manifest.feature_spec, labels)
        return cls(
            cv_names=labels,
            centers=centers,
            measurement_scales=scales,
            state_ids=tuple(definition.state_id for definition in definitions),
        )

    def vector_from_measurements(self, measurements: dict[str, float]) -> np.ndarray | None:
        """Convert a FELP measurement dictionary into a scaled feature-space vector."""

        values: list[float] = []
        for name, scale in zip(self.cv_names, self.measurement_scales, strict=True):
            if name not in measurements:
                return None
            value = float(measurements[name])
            if not np.isfinite(value):
                return None
            values.append(value * float(scale))
        return np.asarray(values, dtype=float)

    def assign(self, measurements: dict[str, float]) -> str:
        """Return the nearest MSM state id for the provided measurements."""

        vector = self.vector_from_measurements(measurements)
        if vector is None:
            missing = [name for name in self.cv_names if name not in measurements]
            raise ConfigurationError(
                "Cannot assign frame to MSM states: missing or non-finite assignment "
                f"measurements for {missing or self.cv_names}."
            )
        distances = np.sum((self.centers - vector) ** 2, axis=1)
        return self.state_ids[int(np.argmin(distances))]

    def assign_batch(self, measurement_rows: np.ndarray) -> np.ndarray:
        """Assign many pre-scaled feature vectors to state indices."""

        if measurement_rows.ndim != 2 or measurement_rows.shape[1] != len(self.cv_names):
            raise ConfigurationError("MSM batch assignment received invalid measurement rows.")
        distances = np.sum(
            (self.centers[None, :, :] - measurement_rows[:, None, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances, axis=1)


def _extract_assignment_measurements(result: Any) -> dict[str, float] | None:
    """Return stored assignment CVs from a frame result."""

    metadata = getattr(result, "metadata", {}) or {}
    measurements = metadata.get("assignment_measurements")
    if isinstance(measurements, dict):
        return {str(name): float(value) for name, value in measurements.items()}
    phi_degrees = getattr(result, "phi_degrees", None)
    psi_degrees = getattr(result, "psi_degrees", None)
    if phi_degrees is not None and psi_degrees is not None:
        return {"phi": float(phi_degrees), "psi": float(psi_degrees)}
    return None


def repartition_frame_results_to_msm_bundle(
    frame_results: list,
    atlas: Any,
) -> list | None:
    """Fast nearest-centroid repartition for MSM bundle atlases."""

    index = getattr(atlas, "msm_assignment_index", None)
    if index is None:
        return None

    from dataclasses import replace

    vectors: list[np.ndarray] = []
    vector_indices: list[int] = []
    for result_index, result in enumerate(frame_results):
        vector = index.vector_from_measurements(
            _extract_assignment_measurements(result) or {}
        )
        if vector is None:
            continue
        vectors.append(vector)
        vector_indices.append(result_index)

    assigned_state_indices: dict[int, int] = {}
    if vectors:
        for result_index, state_index in zip(
            vector_indices,
            index.assign_batch(np.asarray(vectors, dtype=float)),
            strict=True,
        ):
            assigned_state_indices[result_index] = int(state_index)

    repartitioned: list = []
    unassigned_count = 0
    for result_index, result in enumerate(frame_results):
        state_index = assigned_state_indices.get(result_index)
        if state_index is None:
            unassigned_count += 1
            repartitioned.append(result)
            continue

        metadata = dict(getattr(result, "metadata", {}) or {})
        measurements = _extract_assignment_measurements(result)
        if measurements is None:
            unassigned_count += 1
            repartitioned.append(result)
            continue

        definition = atlas.definitions[state_index]
        assignment = atlas._assignment_from_definition(definition, measurements)
        metadata["assignment_measurements"] = assignment.measurements
        metadata["cv_centers"] = assignment.cv_centers
        if assignment.phi_center is not None:
            metadata["phi_center"] = assignment.phi_center
        if assignment.psi_center is not None:
            metadata["psi_center"] = assignment.psi_center
        metadata["plot_x"] = assignment.plot_x
        metadata["plot_y"] = assignment.plot_y
        repartitioned.append(
            replace(
                result,
                microstate_id=assignment.state_id,
                microstate_label=assignment.label,
                metadata=metadata,
            )
        )

    if unassigned_count > 0:
        import sys

        print(
            f"[FELP] msm_bundle repartition: {unassigned_count} frame(s) could not be "
            "assigned because assignment_measurements were missing or incomplete",
            file=sys.stderr,
        )
    return repartitioned


def compute_feature_centers(
    feature_trajectories: tuple[np.ndarray, ...],
    discrete_assignments: tuple[np.ndarray, ...],
    *,
    state_count: int,
) -> np.ndarray:
    """Compute per-state mean feature vectors in PyEMMA feature space."""

    flat_features = np.concatenate(
        [np.asarray(values, dtype=float) for values in feature_trajectories],
        axis=0,
    )
    flat_assignments = np.concatenate(
        [np.asarray(values, dtype=int) for values in discrete_assignments],
        axis=0,
    )
    centers = np.full((int(state_count), flat_features.shape[1]), np.nan, dtype=float)
    for state_index in range(int(state_count)):
        mask = flat_assignments == state_index
        if mask.any():
            centers[state_index] = flat_features[mask].mean(axis=0)
    if np.isnan(centers).any():
        raise ConfigurationError(
            "Could not compute feature-space centers for every MSM state."
        )
    return centers
