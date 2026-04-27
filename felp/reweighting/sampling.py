"""Microstate-stratified sampling for expensive FELP frame evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import tempfile
from typing import Any, Iterable

from felp.conversion import StructureConversionService
from felp.core import ConfigurationError, RunConfig, SamplingConfig, SnapshotJob
from felp.restraints import BackboneDihedralCVProtocol

from .atlas import MicrostateAtlas

_MANIFEST_VERSION = 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


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
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _sampling_config_payload(config: SamplingConfig) -> dict[str, Any]:
    return {
        "microstate_sample_fraction": config.microstate_sample_fraction,
        "microstate_min_samples": config.microstate_min_samples,
        "microstate_max_samples": config.microstate_max_samples,
        "microstate_sampling_seed": config.microstate_sampling_seed,
    }


def _state_seed(seed: int, state_id: str) -> int:
    payload = f"{seed}:{state_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class SamplingManifest:
    """Persisted record of full assignments and selected evaluation frames."""

    version: int
    sampling_config: dict[str, Any]
    microstate_source: str
    state_level: str
    total_assigned_frame_count: int
    assigned_counts: dict[str, int]
    selected_counts: dict[str, int]
    selected_frame_indices: tuple[int, ...]
    frame_state_ids: dict[int, str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SamplingManifest":
        return cls(
            version=int(payload["version"]),
            sampling_config=dict(payload["sampling_config"]),
            microstate_source=str(payload["microstate_source"]),
            state_level=str(payload["state_level"]),
            total_assigned_frame_count=int(payload["total_assigned_frame_count"]),
            assigned_counts={
                str(state_id): int(count)
                for state_id, count in payload.get("assigned_counts", {}).items()
            },
            selected_counts={
                str(state_id): int(count)
                for state_id, count in payload.get("selected_counts", {}).items()
            },
            selected_frame_indices=tuple(
                int(frame_index)
                for frame_index in payload.get("selected_frame_indices", ())
            ),
            frame_state_ids={
                int(frame_index): str(state_id)
                for frame_index, state_id in payload.get("frame_state_ids", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sampling_config": dict(self.sampling_config),
            "microstate_source": self.microstate_source,
            "state_level": self.state_level,
            "total_assigned_frame_count": self.total_assigned_frame_count,
            "assigned_counts": dict(sorted(self.assigned_counts.items())),
            "selected_counts": dict(sorted(self.selected_counts.items())),
            "selected_frame_indices": list(self.selected_frame_indices),
            "frame_state_ids": {
                str(frame_index): state_id
                for frame_index, state_id in sorted(self.frame_state_ids.items())
            },
        }

    @property
    def selected_frame_index_set(self) -> set[int]:
        return set(self.selected_frame_indices)

    @property
    def selected_frame_count(self) -> int:
        return len(self.selected_frame_indices)

    @property
    def skipped_frame_count(self) -> int:
        return self.total_assigned_frame_count - self.selected_frame_count

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sampling_config": dict(self.sampling_config),
            "microstate_source": self.microstate_source,
            "state_level": self.state_level,
            "total_assigned_frame_count": self.total_assigned_frame_count,
            "selected_frame_count": self.selected_frame_count,
            "skipped_frame_count": self.skipped_frame_count,
            "assigned_counts_by_state": dict(sorted(self.assigned_counts.items())),
            "selected_counts_by_state": dict(sorted(self.selected_counts.items())),
        }


def assign_jobs_for_sampling(
    jobs: Iterable[SnapshotJob],
    *,
    atlas: MicrostateAtlas,
    config: RunConfig,
) -> dict[int, str]:
    """Return a full frame-index to microstate-id assignment map."""

    jobs = list(jobs)
    if config.microstates.source == "msm_bundle":
        assignments: dict[int, str] = {}
        for job in jobs:
            if job.precomputed_state_id is None:
                raise ConfigurationError(
                    "MSM-bundle sampling requires every SnapshotJob to carry a "
                    "precomputed_state_id."
                )
            atlas.assign_precomputed(job.precomputed_state_id, {})
            assignments[job.frame_index] = job.precomputed_state_id
        return assignments

    conversion_service = StructureConversionService(
        pyrosetta_init_options=config.rosetta.init_options,
        radii_set=config.input.radii_set,
    )
    template_structure = conversion_service.load_topology(config.input.topology_file)
    cv_protocol = BackboneDihedralCVProtocol(
        cv_specs=config.restraints.cv_specs,
        force_constant=config.restraints.effective_rosetta_force_constant(),
    )

    assignments = {}
    for job in jobs:
        structure = conversion_service.clone_structure_with_coordinates(
            template_structure,
            job.coordinates_angstrom,
        )
        measurement = cv_protocol.measure_structure(structure)
        assignment = atlas.assign(measurement)
        assignments[job.frame_index] = assignment.state_id
    return assignments


def select_microstate_stratified_frame_indices(
    frame_state_ids: dict[int, str],
    sampling_config: SamplingConfig,
) -> tuple[tuple[int, ...], dict[str, int], dict[str, int]]:
    """Select evaluation frame indices uniformly within each assigned state."""

    by_state: dict[str, list[int]] = defaultdict(list)
    for frame_index, state_id in frame_state_ids.items():
        by_state[state_id].append(int(frame_index))
    assigned_counts = {
        state_id: len(frame_indices) for state_id, frame_indices in by_state.items()
    }

    selected: list[int] = []
    selected_counts: dict[str, int] = {}
    for state_id in sorted(by_state):
        frame_indices = sorted(by_state[state_id])
        available = len(frame_indices)
        if sampling_config.is_noop:
            target_count = available
        else:
            target_count = math.ceil(
                available * sampling_config.microstate_sample_fraction
            )
            target_count = max(target_count, sampling_config.microstate_min_samples)
            if sampling_config.microstate_max_samples is not None:
                target_count = min(target_count, sampling_config.microstate_max_samples)
            target_count = min(target_count, available)
        if available > 0 and target_count == 0:
            raise ConfigurationError(
                f"Sampling selected zero frames for assigned microstate {state_id!r}. "
                "Increase --microstate-sample-fraction or --microstate-min-samples."
            )
        if target_count == available:
            state_selection = frame_indices
        else:
            rng = random.Random(
                _state_seed(sampling_config.microstate_sampling_seed, state_id)
            )
            state_selection = sorted(rng.sample(frame_indices, target_count))
        selected.extend(state_selection)
        selected_counts[state_id] = len(state_selection)

    return tuple(sorted(selected)), assigned_counts, selected_counts


def build_sampling_manifest(
    jobs: Iterable[SnapshotJob],
    *,
    atlas: MicrostateAtlas,
    config: RunConfig,
) -> SamplingManifest:
    """Build a deterministic sampling manifest for a full job set."""

    jobs = list(jobs)
    frame_state_ids = assign_jobs_for_sampling(jobs, atlas=atlas, config=config)
    selected_indices, assigned_counts, selected_counts = (
        select_microstate_stratified_frame_indices(
            frame_state_ids,
            config.sampling,
        )
    )
    return SamplingManifest(
        version=_MANIFEST_VERSION,
        sampling_config=_sampling_config_payload(config.sampling),
        microstate_source=config.microstates.source,
        state_level=config.microstates.state_level,
        total_assigned_frame_count=len(frame_state_ids),
        assigned_counts=assigned_counts,
        selected_counts=selected_counts,
        selected_frame_indices=selected_indices,
        frame_state_ids=frame_state_ids,
    )


def load_or_create_sampling_manifest(
    *,
    manifest_path: str,
    jobs: Iterable[SnapshotJob],
    atlas: MicrostateAtlas,
    config: RunConfig,
) -> SamplingManifest:
    """Load an existing manifest or create one that matches the current run."""

    expected = build_sampling_manifest(jobs, atlas=atlas, config=config)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            observed = SamplingManifest.from_dict(json.load(handle))
        if observed.to_dict() != expected.to_dict():
            raise ConfigurationError(
                "Existing sampling manifest does not match the current input, "
                "microstate, or sampling configuration. Use a fresh output "
                "directory or restore the original run configuration."
            )
        return observed
    _write_json_atomic(manifest_path, expected.to_dict())
    return expected


def selected_jobs_from_manifest(
    jobs: Iterable[SnapshotJob],
    manifest: SamplingManifest,
) -> list[SnapshotJob]:
    """Return jobs selected for expensive evaluation, preserving frame order."""

    selected = manifest.selected_frame_index_set
    return [job for job in jobs if job.frame_index in selected]


def ensure_existing_results_match_manifest(
    existing_frame_indices: Iterable[int],
    manifest: SamplingManifest,
) -> None:
    """Fail when resumed results contain frames outside the sampling manifest."""

    existing = set(existing_frame_indices)
    unexpected = sorted(existing - manifest.selected_frame_index_set)
    if unexpected:
        preview = ", ".join(str(frame_index) for frame_index in unexpected[:10])
        if len(unexpected) > 10:
            preview = f"{preview}, ..."
        raise ConfigurationError(
            "Existing frame results include frames that are not in the current "
            f"sampling manifest: {preview}."
        )


def manifest_counts_for_atlas(
    manifest: SamplingManifest,
    atlas: MicrostateAtlas,
) -> dict[str, int]:
    """Return assigned frame counts for states represented in the atlas."""

    state_ids = {definition.state_id for definition in atlas.definitions}
    return {
        state_id: count
        for state_id, count in manifest.assigned_counts.items()
        if state_id in state_ids
    }


def evaluated_counts_by_state(frame_state_ids: Iterable[str]) -> dict[str, int]:
    """Return evaluated frame counts keyed by state id."""

    return dict(Counter(frame_state_ids))
