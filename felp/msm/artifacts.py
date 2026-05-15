"""Read and write fast-loading MSM bundle artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Optional

import numpy as np

from felp.core import (
    ConfigurationError,
    InputConfig,
    MicrostateDefinition,
    flatten_trajectory_groups,
    normalize_trajectory_groups,
)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
ASSIGNMENTS_NAME = "assignments.npz"
EMBEDDING_NAME = "embedding.npz"


def _definition_from_payload(payload: dict[str, Any]) -> MicrostateDefinition:
    return MicrostateDefinition(
        state_id=str(payload["state_id"]),
        label=str(payload.get("label", payload["state_id"])),
        phi_range=tuple(payload["phi_range"]) if payload.get("phi_range") is not None else None,
        psi_range=tuple(payload["psi_range"]) if payload.get("psi_range") is not None else None,
        cv_ranges={
            str(name): tuple(bounds)
            for name, bounds in dict(payload.get("cv_ranges", {})).items()
        },
        reference_probability=payload.get("reference_probability"),
        plot_x=payload.get("plot_x"),
        plot_y=payload.get("plot_y"),
        state_level=str(payload.get("state_level", "microstate")),
        metadata=dict(payload.get("metadata", {})),
    )


def _definition_to_payload(definition: MicrostateDefinition) -> dict[str, Any]:
    return {
        "state_id": definition.state_id,
        "label": definition.label,
        "phi_range": list(definition.phi_range) if definition.phi_range is not None else None,
        "psi_range": list(definition.psi_range) if definition.psi_range is not None else None,
        "cv_ranges": {
            name: list(bounds) for name, bounds in definition.cv_ranges.items()
        },
        "reference_probability": definition.reference_probability,
        "plot_x": definition.plot_x,
        "plot_y": definition.plot_y,
        "state_level": definition.state_level,
        "metadata": definition.metadata,
    }


@dataclass(frozen=True)
class MSMBundleManifest:
    """JSON manifest describing an exported MSM bundle."""

    topology_file: str
    trajectory_groups: tuple[tuple[str, ...], ...]
    stride: int
    dt_traj: str
    lag_time: int
    feature_spec: dict[str, Any]
    reducer: dict[str, Any]
    clustering: dict[str, Any]
    model: dict[str, Any]
    state_definitions: dict[str, tuple[MicrostateDefinition, ...]]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    assignments_file: str = ASSIGNMENTS_NAME
    embedding_file: str = EMBEDDING_NAME

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["trajectory_groups"] = [
            list(group) for group in self.trajectory_groups
        ]
        payload["state_definitions"] = {
            level: [_definition_to_payload(definition) for definition in definitions]
            for level, definitions in self.state_definitions.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MSMBundleManifest":
        """Rehydrate a manifest from its JSON representation."""

        state_definitions = {
            level: tuple(_definition_from_payload(item) for item in definitions)
            for level, definitions in payload.get("state_definitions", {}).items()
        }
        return cls(
            topology_file=payload["topology_file"],
            trajectory_groups=normalize_trajectory_groups(
                payload["trajectory_groups"]
                if "trajectory_groups" in payload
                else payload["trajectory_files"]
            ),
            stride=int(payload["stride"]),
            dt_traj=str(payload["dt_traj"]),
            lag_time=int(payload["lag_time"]),
            feature_spec=dict(payload.get("feature_spec", {})),
            reducer=dict(payload.get("reducer", {})),
            clustering=dict(payload.get("clustering", {})),
            model=dict(payload.get("model", {})),
            state_definitions=state_definitions,
            metadata=dict(payload.get("metadata", {})),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            assignments_file=payload.get("assignments_file", ASSIGNMENTS_NAME),
            embedding_file=payload.get("embedding_file", EMBEDDING_NAME),
        )

    @property
    def trajectory_files(self) -> tuple[str, ...]:
        """Return all trajectory fragment files in order."""

        return flatten_trajectory_groups(self.trajectory_groups)


@dataclass(frozen=True)
class MSMBundle:
    """Fast-loading MSM bundle used by FELP and the PyEMMA CLI."""

    bundle_dir: str
    manifest: MSMBundleManifest
    assignments: dict[str, tuple[np.ndarray, ...]]
    frame_source_indices: tuple[np.ndarray, ...]
    embedding: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def topology_file(self) -> str:
        """Return the topology file stored in the manifest."""

        return self.manifest.topology_file

    @property
    def trajectory_files(self) -> tuple[str, ...]:
        """Return the trajectory files stored in the manifest."""

        return self.manifest.trajectory_files

    @property
    def trajectory_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return grouped trajectory fragments stored in the manifest."""

        return self.manifest.trajectory_groups

    def available_state_levels(self) -> tuple[str, ...]:
        """Return the state levels available in the bundle."""

        return tuple(self.manifest.state_definitions.keys())

    def definitions_for_level(self, state_level: str) -> tuple[MicrostateDefinition, ...]:
        """Return the state definitions for the requested level."""

        if state_level not in self.manifest.state_definitions:
            raise ConfigurationError(
                f"State level '{state_level}' is not available in bundle {self.bundle_dir}."
            )
        return self.manifest.state_definitions[state_level]

    def validate_against_input(self, input_config: InputConfig) -> None:
        """Ensure the FELP run is aligned with the saved MSM bundle."""

        if tuple(input_config.trajectory_groups) != self.trajectory_groups:
            raise ConfigurationError(
                "FELP trajectory groups must match the MSM bundle trajectory groups exactly."
            )
        if os.path.abspath(input_config.topology_file) != os.path.abspath(self.topology_file):
            raise ConfigurationError(
                "FELP topology file must match the MSM bundle topology file exactly."
            )
        if input_config.stride != self.manifest.stride:
            raise ConfigurationError(
                "FELP stride must match the MSM bundle stride for frame-aligned assignments."
            )

    def lookup_state_id(
        self,
        trajectory_id: int,
        source_frame_index: int,
        state_level: str,
    ) -> str:
        """Return the state identifier for a given trajectory/frame pair."""

        assignments = self.assignments.get(state_level)
        if assignments is None:
            raise ConfigurationError(
                f"State level '{state_level}' does not have assignment arrays."
            )
        if trajectory_id >= len(assignments):
            raise ConfigurationError(f"Invalid trajectory index {trajectory_id}.")

        frame_indices = self.frame_source_indices[trajectory_id]
        position = int(np.searchsorted(frame_indices, source_frame_index))
        if position >= len(frame_indices) or int(frame_indices[position]) != source_frame_index:
            raise ConfigurationError(
                "Requested frame does not exist in the MSM bundle. "
                "Check frame ranges and stride alignment."
            )

        state_index = int(assignments[trajectory_id][position])
        definitions = self.definitions_for_level(state_level)
        if state_index < 0 or state_index >= len(definitions):
            raise ConfigurationError(
                f"State index {state_index} is out of range for level '{state_level}'."
            )
        return definitions[state_index].state_id

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the bundle contents."""

        return {
            "bundle_dir": self.bundle_dir,
            "schema_version": self.manifest.schema_version,
            "trajectory_count": len(self.trajectory_groups),
            "fragment_file_count": len(self.trajectory_files),
            "stride": self.manifest.stride,
            "lag_time": self.manifest.lag_time,
            "available_state_levels": list(self.available_state_levels()),
            "state_counts": {
                level: len(definitions)
                for level, definitions in self.manifest.state_definitions.items()
            },
        }


def save_bundle(
    bundle_dir: str,
    manifest: MSMBundleManifest,
    assignments: dict[str, tuple[np.ndarray, ...]],
    frame_source_indices: tuple[np.ndarray, ...],
    embedding: Optional[dict[str, np.ndarray]] = None,
) -> MSMBundle:
    """Write an MSM bundle to disk and return the loaded representation."""

    os.makedirs(bundle_dir, exist_ok=True)

    manifest_path = os.path.join(bundle_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)

    assignment_payload: dict[str, Any] = {}
    assignment_payload["trajectory_count"] = np.array([len(frame_source_indices)], dtype=int)
    for trajectory_id, frame_indices in enumerate(frame_source_indices):
        assignment_payload[f"frame_source_indices_{trajectory_id}"] = np.asarray(
            frame_indices, dtype=int
        )
    for state_level, trajectory_assignments in assignments.items():
        for trajectory_id, state_values in enumerate(trajectory_assignments):
            assignment_payload[f"{state_level}_assignments_{trajectory_id}"] = np.asarray(
                state_values, dtype=int
            )
    np.savez(os.path.join(bundle_dir, manifest.assignments_file), **assignment_payload)

    embedding_payload = dict(embedding or {})
    np.savez(os.path.join(bundle_dir, manifest.embedding_file), **embedding_payload)

    return load_bundle(bundle_dir)


def load_bundle(bundle_dir: str) -> MSMBundle:
    """Load an MSM bundle from disk."""

    manifest_path = os.path.join(bundle_dir, MANIFEST_NAME)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = MSMBundleManifest.from_dict(json.load(handle))

    assignments_file = os.path.join(bundle_dir, manifest.assignments_file)
    with np.load(assignments_file, allow_pickle=False) as assignment_archive:
        trajectory_count = int(assignment_archive["trajectory_count"][0])
        frame_source_indices = tuple(
            np.asarray(assignment_archive[f"frame_source_indices_{trajectory_id}"], dtype=int)
            for trajectory_id in range(trajectory_count)
        )
        assignments: dict[str, tuple[np.ndarray, ...]] = {}
        for state_level in manifest.state_definitions:
            assignments[state_level] = tuple(
                np.asarray(
                    assignment_archive[f"{state_level}_assignments_{trajectory_id}"],
                    dtype=int,
                )
                for trajectory_id in range(trajectory_count)
            )

    embedding_path = os.path.join(bundle_dir, manifest.embedding_file)
    embedding: dict[str, np.ndarray] = {}
    if os.path.isfile(embedding_path):
        with np.load(embedding_path, allow_pickle=False) as embedding_archive:
            embedding = {
                key: np.asarray(embedding_archive[key])
                for key in embedding_archive.files
            }

    return MSMBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        assignments=assignments,
        frame_source_indices=frame_source_indices,
        embedding=embedding,
    )
