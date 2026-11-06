"""Representative structure export for MSM bundle states."""

from __future__ import annotations

import os
import shutil

from felp.core import ConfigurationError, DependencyError

from .artifacts import load_bundle
from .config import MSMSampleExportConfig


def _import_mdtraj():
    try:
        import mdtraj as md
    except ImportError as exc:
        raise DependencyError(
            "mdtraj is required to export MSM state samples."
        ) from exc
    return md


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _group_to_mdtraj_input(group: tuple[str, ...]) -> str | list[str]:
    """Convert a trajectory group to the form expected by MDTraj."""

    if len(group) == 1:
        return group[0]
    return list(group)


def export_state_samples(config: MSMSampleExportConfig) -> dict[str, object]:
    """Export representative sampled structures for states in a bundle."""

    config.validate()
    md = _import_mdtraj()
    bundle = load_bundle(config.bundle_dir)
    definitions = bundle.definitions_for_level(config.state_level)
    selected_ids = set(config.state_ids) if config.state_ids else {
        definition.state_id for definition in definitions
    }
    definition_by_index = {index: definition for index, definition in enumerate(definitions)}
    assignments = bundle.assignments[config.state_level]

    if os.path.isdir(config.output_dir):
        if config.overwrite:
            shutil.rmtree(config.output_dir)
        elif os.listdir(config.output_dir):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {config.output_dir}"
            )
    os.makedirs(config.output_dir, exist_ok=True)

    requested_samples: dict[str, list[tuple[int, int]]] = {
        state_id: [] for state_id in selected_ids
    }
    for trajectory_id, dtraj in enumerate(assignments):
        for local_index, state_index in enumerate(dtraj):
            definition = definition_by_index.get(int(state_index))
            if definition is None or definition.state_id not in selected_ids:
                continue
            samples = requested_samples[definition.state_id]
            if len(samples) < config.samples_per_state:
                samples.append((trajectory_id, local_index))
        if all(len(samples) >= config.samples_per_state for samples in requested_samples.values()):
            break

    exported_count = 0
    for definition in definitions:
        if definition.state_id not in selected_ids:
            continue
        frame_locations = requested_samples[definition.state_id]
        if not frame_locations:
            continue

        frame_trajectories = []
        by_trajectory: dict[int, list[int]] = {}
        for trajectory_id, local_index in frame_locations:
            by_trajectory.setdefault(trajectory_id, []).append(local_index)
        for trajectory_id, local_indices in by_trajectory.items():
            trajectory = md.load(
                _group_to_mdtraj_input(bundle.trajectory_groups[trajectory_id]),
                top=bundle.topology_file,
                stride=bundle.manifest.stride,
            )
            for local_index in local_indices:
                frame_trajectories.append(trajectory[local_index])

        if not frame_trajectories:
            continue
        sampled_trajectory = md.join(frame_trajectories, check_topology=True)
        basename = _safe_name(definition.label)
        sampled_trajectory.save_xtc(
            os.path.join(config.output_dir, f"{basename}.xtc")
        )
        sampled_trajectory[0].save_pdb(
            os.path.join(config.output_dir, f"{basename}.pdb")
        )
        exported_count += 1

    missing = sorted(
        state_id for state_id, samples in requested_samples.items() if not samples
    )
    return {
        "bundle_dir": config.bundle_dir,
        "output_dir": config.output_dir,
        "state_level": config.state_level,
        "requested_state_count": len(selected_ids),
        "exported_state_count": exported_count,
        "missing_state_ids": missing,
    }
