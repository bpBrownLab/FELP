"""Fast CV values from plain coordinate arrays (Angstrom), without ParmEd clones."""

from __future__ import annotations

from typing import Callable

import numpy as np

from felp.restraints import BackboneDihedralCVProtocol
from felp.restraints.protocols import (
    find_structure_atom,
    measure_dihedral_degrees,
    measure_distance_angstrom,
)


def structure_atom_index(structure, residue_number: int, atom_name: str) -> int:
    atom = find_structure_atom(structure, residue_number, atom_name)
    index = getattr(atom, "idx", None)
    if index is not None:
        return int(index)
    return int(structure.atoms.index(atom))


def coordinate_cv_extractors(
    template_structure,
    cv_protocol: BackboneDihedralCVProtocol,
) -> dict[str, Callable[[np.ndarray], float]]:
    """Precompute per-CV callables mapping (n_atoms, 3) coordinates to scalar values."""

    extractors: dict[str, Callable[[np.ndarray], float]] = {}
    for spec in cv_protocol.cv_specs:
        residue_numbers, atom_names = cv_protocol._dihedral_definition(spec)
        atom_indices = tuple(
            structure_atom_index(template_structure, residue_number, atom_name)
            for residue_number, atom_name in zip(residue_numbers, atom_names)
        )

        def measure_dihedral(
            coordinates: np.ndarray, atom_indices: tuple[int, ...] = atom_indices
        ) -> float:
            return measure_dihedral_degrees(
                [coordinates[index] for index in atom_indices]
            )

        extractors[spec.name] = measure_dihedral

    for spec in cv_protocol.distance_cv_specs:
        atom1_index = structure_atom_index(
            template_structure,
            spec.atom1.residue_number,
            spec.atom1.atom_name,
        )
        atom2_index = structure_atom_index(
            template_structure,
            spec.atom2.residue_number,
            spec.atom2.atom_name,
        )

        def measure_distance(
            coordinates: np.ndarray,
            atom1_index: int = atom1_index,
            atom2_index: int = atom2_index,
        ) -> float:
            return measure_distance_angstrom(
                coordinates[atom1_index],
                coordinates[atom2_index],
            )

        extractors[spec.name] = measure_distance
    return extractors
