"""State assignment policies for geometric and precomputed MSM workflows."""

from __future__ import annotations

from dataclasses import dataclass

from felp.core import (
    CVMeasurement,
    MicrostateAssignment,
    MicrostateAssignmentError,
    SnapshotJob,
)

from .atlas import MicrostateAtlas


class AssignmentPolicy:
    """Protocol-like base class for frame-to-state assignment policies."""

    name = "unknown"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: CVMeasurement,
    ) -> MicrostateAssignment:
        """Assign a job to a state."""

        raise NotImplementedError


@dataclass(frozen=True)
class CVAssignmentPolicy(AssignmentPolicy):
    """Assign frames using the configured CV measurement map."""

    name: str = "cv"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: CVMeasurement,
    ) -> MicrostateAssignment:
        return atlas.assign(measurement)


@dataclass(frozen=True)
class PhiPsiAssignmentPolicy(CVAssignmentPolicy):
    """Compatibility wrapper for legacy phi/psi assignment mode."""

    name: str = "phi_psi"


@dataclass(frozen=True)
class PrecomputedStateAssignmentPolicy(AssignmentPolicy):
    """Assign frames using precomputed discrete states carried by a snapshot job."""

    name: str = "msm_bundle"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: CVMeasurement,
    ) -> MicrostateAssignment:
        if job.precomputed_state_id is None:
            raise MicrostateAssignmentError(
                "Snapshot job is missing a precomputed state id."
            )
        return atlas.assign_precomputed(
            job.precomputed_state_id,
            measurement,
        )
