"""Reference-frame restraint builders for FELP."""

from __future__ import annotations

from dataclasses import dataclass

from felp.core import DihedralRestraintSpec, PhiPsiMeasurement


@dataclass
class PhiPsiRestraintProtocol:
    """Preserve a dipeptide-like state using phi/psi restraints."""

    target_residue_number: int
    force_constant: float

    def measure_pose(self, pose) -> PhiPsiMeasurement:
        """Measure the phi and psi angles of the target residue."""
        return PhiPsiMeasurement(
            phi_degrees=float(pose.phi(self.target_residue_number)),
            psi_degrees=float(pose.psi(self.target_residue_number)),
        )

    def build_from_reference_pose(self, pose) -> list[DihedralRestraintSpec]:
        """Build dihedral restraint specifications based on the current pose geometry."""
        if self.target_residue_number <= 1 or self.target_residue_number >= pose.total_residue():
            raise ValueError(
                "Phi/psi restraints require a residue with both a previous and next residue."
            )

        measurement = self.measure_pose(pose)
        residue_number = self.target_residue_number
        return [
            DihedralRestraintSpec(
                label="phi",
                residue_numbers=(
                    residue_number - 1,
                    residue_number,
                    residue_number,
                    residue_number,
                ),
                atom_names=("C", "N", "CA", "C"),
                target_angle_degrees=measurement.phi_degrees,
                force_constant=self.force_constant,
            ),
            DihedralRestraintSpec(
                label="psi",
                residue_numbers=(
                    residue_number,
                    residue_number,
                    residue_number,
                    residue_number + 1,
                ),
                atom_names=("N", "CA", "C", "N"),
                target_angle_degrees=measurement.psi_degrees,
                force_constant=self.force_constant,
            ),
        ]
