"""Reference-frame restraint builders for FELP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from felp.core import (
    CVMeasurement,
    CVSpec,
    DihedralRestraintSpec,
    default_backbone_cv_specs,
)

ROSETTA_BRIDGE_METADATA_ATTR = "_rosetta_bridge_metadata"


def resolve_structure_residue(structure, residue_number: int):
    """Resolve a pose-indexed residue from a ParmEd structure."""

    metadata = getattr(structure, ROSETTA_BRIDGE_METADATA_ATTR, None)
    residue_metadata = metadata.get("residues") if isinstance(metadata, dict) else None
    if isinstance(residue_metadata, list) and len(residue_metadata) == len(structure.residues):
        for residue, residue_meta in zip(structure.residues, residue_metadata):
            if isinstance(residue_meta, dict) and int(residue_meta.get("seqpos", 0)) == residue_number:
                return residue
    if 1 <= residue_number <= len(structure.residues):
        return structure.residues[residue_number - 1]
    for residue in structure.residues:
        if residue.number == residue_number:
            return residue
    raise KeyError(f"Could not locate residue {residue_number} in structure.")


def find_structure_atom(structure, residue_number: int, atom_name: str):
    """Resolve an atom from a ParmEd structure using pose-indexed residue numbering."""

    residue = resolve_structure_residue(structure, residue_number)
    normalized_name = atom_name.strip()
    for atom in getattr(residue, "atoms", ()):
        if atom.name.strip() == normalized_name:
            return atom
    for atom in structure.atoms:
        if atom.residue is residue and atom.name.strip() == normalized_name:
            return atom
    raise KeyError(
        f"Could not locate atom {atom_name} in residue {residue_number} "
        "(Rosetta metadata, pose order, or residue.number)."
    )


def measure_dihedral_degrees(points: list[tuple[float, float, float]]) -> float:
    """Measure a signed dihedral angle in degrees from four Cartesian points."""

    p0, p1, p2, p3 = (np.asarray(point, dtype=float) for point in points)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if math.isclose(b1_norm, 0.0):
        raise ValueError("Cannot measure a dihedral with coincident middle atoms.")
    b1 /= b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


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

    def measure_structure(self, structure) -> CVMeasurement:
        """Measure all configured backbone-dihedral CVs from a ParmEd structure."""

        values: dict[str, float] = {}
        for spec in self.cv_specs:
            residue_numbers, atom_names = self._dihedral_definition(spec)
            points = [
                (
                    float(atom.xx),
                    float(atom.xy),
                    float(atom.xz),
                )
                for atom in (
                    find_structure_atom(structure, residue_number, atom_name)
                    for residue_number, atom_name in zip(residue_numbers, atom_names)
                )
            ]
            values[spec.name] = measure_dihedral_degrees(points)
        return CVMeasurement(values)

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
