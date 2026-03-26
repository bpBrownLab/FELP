"""Trajectory loading for FELP."""

from __future__ import annotations

from dataclasses import dataclass

from felp.core import InputConfig, SnapshotJob
from felp.msm import MSMBundle


def _import_mdtraj():
    import mdtraj as md

    return md


def _group_to_mdtraj_input(group: tuple[str, ...]) -> str | list[str]:
    """Convert a trajectory group to the form expected by MDTraj."""

    if len(group) == 1:
        return group[0]
    return list(group)


@dataclass
class TrajectoryLoader:
    """Load trajectory frames and emit serializable jobs."""

    config: InputConfig
    bundle: MSMBundle | None = None
    state_level: str = "cluster"

    def load_jobs(self) -> list[SnapshotJob]:
        """Load the configured trajectories and yield a job for each frame."""

        md = _import_mdtraj()
        if self.bundle is not None:
            self.bundle.validate_against_input(self.config)

        start = self.config.frame_start
        stop = self.config.frame_stop

        jobs: list[SnapshotJob] = []
        global_index = 0
        for trajectory_id, trajectory_group in enumerate(self.config.trajectory_groups):
            trajectory = md.load(
                _group_to_mdtraj_input(trajectory_group),
                top=self.config.topology_file,
                stride=self.config.stride,
            )
            for local_index, frame in enumerate(trajectory):
                if stop is not None and global_index >= stop:
                    return jobs

                source_frame_index = local_index * self.config.stride
                if global_index < start:
                    global_index += 1
                    continue

                stride_number = global_index * self.config.stride
                coordinates_angstrom = tuple(
                    tuple(float(value) for value in atom_xyz)
                    for atom_xyz in (frame.xyz[0] * 10.0)
                )
                precomputed_state_id = None
                if self.bundle is not None:
                    precomputed_state_id = self.bundle.lookup_state_id(
                        trajectory_id=trajectory_id,
                        source_frame_index=source_frame_index,
                        state_level=self.state_level,
                    )
                jobs.append(
                    SnapshotJob(
                        frame_index=global_index,
                        stride_number=stride_number,
                        coordinates_angstrom=coordinates_angstrom,
                        trajectory_id=trajectory_id,
                        source_frame_index=source_frame_index,
                        precomputed_state_id=precomputed_state_id,
                    )
                )
                global_index += 1
        return jobs
