"""Trajectory loading for FELP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

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
    load_chunk_size: int = 1024

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
            for source_frame_index, frame_xyz_nm in self._iter_group_coordinates(
                md,
                trajectory_group,
            ):
                if stop is not None and global_index >= stop:
                    return jobs

                if global_index < start:
                    global_index += 1
                    continue

                stride_number = global_index * self.config.stride
                coordinates_angstrom = np.ascontiguousarray(
                    frame_xyz_nm,
                    dtype=np.float64,
                )
                coordinates_angstrom *= 10.0
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

    def _iter_group_coordinates(
        self,
        md,
        trajectory_group: tuple[str, ...],
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield selected frame coordinates from one logical trajectory group.

        Files within a group are contiguous trajectory segments.  Single-file
        groups can let MDTraj apply stride during IO; multi-file groups stream
        unstrided chunks so stride is applied over the concatenated logical
        trajectory rather than restarting at each file boundary.
        """

        if not hasattr(md, "iterload"):
            trajectory = md.load(
                _group_to_mdtraj_input(trajectory_group),
                top=self.config.topology_file,
                stride=self.config.stride,
            )
            for local_index, frame in enumerate(trajectory):
                yield local_index * self.config.stride, frame.xyz[0]
            return

        chunk_size = max(1, int(self.load_chunk_size))
        if len(trajectory_group) == 1:
            local_index = 0
            for chunk in md.iterload(
                trajectory_group[0],
                chunk=chunk_size,
                top=self.config.topology_file,
                stride=self.config.stride,
            ):
                for frame_xyz_nm in chunk.xyz:
                    yield local_index * self.config.stride, frame_xyz_nm
                    local_index += 1
            return

        source_frame_index = 0
        stride = int(self.config.stride)
        for trajectory_file in trajectory_group:
            for chunk in md.iterload(
                trajectory_file,
                chunk=chunk_size,
                top=self.config.topology_file,
                stride=1,
            ):
                for frame_xyz_nm in chunk.xyz:
                    if source_frame_index % stride == 0:
                        yield source_frame_index, frame_xyz_nm
                    source_frame_index += 1
