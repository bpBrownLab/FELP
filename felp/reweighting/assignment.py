"""State assignment policies for geometric and precomputed MSM workflows."""

from __future__ import annotations

from dataclasses import dataclass

from felp.core import MicrostateAssignment, MicrostateAssignmentError, PhiPsiMeasurement, SnapshotJob

from .atlas import MicrostateAtlas


class AssignmentPolicy:
    """Protocol-like base class for frame-to-state assignment policies."""

    name = "unknown"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: PhiPsiMeasurement,
    ) -> MicrostateAssignment:
        """Assign a job to a state."""

        raise NotImplementedError


@dataclass(frozen=True)
class PhiPsiAssignmentPolicy(AssignmentPolicy):
    """Assign frames by measuring phi/psi on the reference pose."""

    name: str = "phi_psi"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: PhiPsiMeasurement,
    ) -> MicrostateAssignment:
        return atlas.assign(measurement.phi_degrees, measurement.psi_degrees)


@dataclass(frozen=True)
class PrecomputedStateAssignmentPolicy(AssignmentPolicy):
    """Assign frames using precomputed discrete states carried by a snapshot job."""

    name: str = "msm_bundle"

    def assign(
        self,
        atlas: MicrostateAtlas,
        job: SnapshotJob,
        measurement: PhiPsiMeasurement,
    ) -> MicrostateAssignment:
        if job.precomputed_state_id is None:
            raise MicrostateAssignmentError(
                "Snapshot job is missing a precomputed state id."
            )
        return atlas.assign_precomputed(
            job.precomputed_state_id,
            measurement.phi_degrees,
            measurement.psi_degrees,
        )
