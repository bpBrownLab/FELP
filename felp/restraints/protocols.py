"""Reference-frame restraint builders for FELP."""

from __future__ import annotations

from dataclasses import dataclass

from felp.core import (
    CVMeasurement,
    CVSpec,
    DihedralRestraintSpec,
    default_backbone_cv_specs,
)


@dataclass
class BackboneDihedralCVProtocol:
    """Measure and restrain a configured set of backbone dihedral CVs."""

    cv_specs: tuple[CVSpec, ...]
    force_constant: float

    def _dihedral_definition(
        self,
        spec: CVSpec,
    ) -> tuple[tuple[int, int, int, int], tuple[str, str, str, str]]:
        residue_number = spec.residue_number
        if spec.kind == "phi":
            return (
                (residue_number - 1, residue_number, residue_number, residue_number),
                ("C", "N", "CA", "C"),
            )
        if spec.kind == "psi":
            return (
                (residue_number, residue_number, residue_number, residue_number + 1),
                ("N", "CA", "C", "N"),
            )
        return (
            (residue_number, residue_number, residue_number + 1, residue_number + 1),
            ("CA", "C", "N", "CA"),
        )

    def _validate_cv_position(self, pose, spec: CVSpec) -> None:
        total_residue = pose.total_residue()
        residue_number = spec.residue_number
        if spec.kind == "phi" and not 1 < residue_number <= total_residue:
            raise ValueError("Phi restraints require a residue with a previous residue.")
        if spec.kind in {"psi", "omega"} and not 1 <= residue_number < total_residue:
            raise ValueError(
                f"{spec.kind} restraints require a residue with a following residue."
            )

    def _measure_cv(self, pose, spec: CVSpec) -> float:
        self._validate_cv_position(pose, spec)
        if spec.kind == "phi":
            return float(pose.phi(spec.residue_number))
        if spec.kind == "psi":
            return float(pose.psi(spec.residue_number))
        return float(pose.omega(spec.residue_number))

    def measure_pose(self, pose) -> CVMeasurement:
        """Measure all configured backbone-dihedral CVs."""

        return CVMeasurement(
            {spec.name: self._measure_cv(pose, spec) for spec in self.cv_specs}
        )

    def build_from_reference_pose(self, pose) -> list[DihedralRestraintSpec]:
        """Build dihedral restraint specifications based on the current pose geometry."""

        measurement = self.measure_pose(pose)
        restraints: list[DihedralRestraintSpec] = []
        for spec in self.cv_specs:
            residue_numbers, atom_names = self._dihedral_definition(spec)
            restraints.append(
                DihedralRestraintSpec(
                    label=spec.name,
                    kind=spec.kind,
                    residue_numbers=residue_numbers,
                    atom_names=atom_names,
                    target_angle_degrees=measurement.values[spec.name],
                    force_constant=self.force_constant,
                )
            )
        return restraints


class PhiPsiRestraintProtocol(BackboneDihedralCVProtocol):
    """Compatibility wrapper for the legacy phi/psi-only restraint mode."""

    def __init__(self, target_residue_number: int, force_constant: float) -> None:
        self.target_residue_number = target_residue_number
        super().__init__(
            cv_specs=default_backbone_cv_specs(target_residue_number),
            force_constant=force_constant,
        )
