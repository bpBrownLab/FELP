"""Fast CV values from plain coordinate arrays (Angstrom), without ParmEd clones."""

from __future__ import annotations

from typing import Callable

import numpy as np

from felp.restraints import BackboneDihedralCVProtocol
from felp.core import normalize_angle_degrees
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

    for spec in cv_protocol.dihedral_cv_specs:
        atom_indices = tuple(
            structure_atom_index(template_structure, atom.residue_number, atom.atom_name)
            for atom in spec.atoms
        )

        def measure_custom_dihedral(
            coordinates: np.ndarray, atom_indices: tuple[int, ...] = atom_indices
        ) -> float:
            return normalize_angle_degrees(
                measure_dihedral_degrees([coordinates[index] for index in atom_indices])
            )

        extractors[spec.name] = measure_custom_dihedral

    for spec in cv_protocol.difference_distance_cv_specs:
        positive_indices = (
            structure_atom_index(
                template_structure,
                spec.positive_atom1.residue_number,
                spec.positive_atom1.atom_name,
            ),
            structure_atom_index(
                template_structure,
                spec.positive_atom2.residue_number,
                spec.positive_atom2.atom_name,
            ),
        )
        negative_indices = (
            structure_atom_index(
                template_structure,
                spec.negative_atom1.residue_number,
                spec.negative_atom1.atom_name,
            ),
            structure_atom_index(
                template_structure,
                spec.negative_atom2.residue_number,
                spec.negative_atom2.atom_name,
            ),
        )

        def measure_positive(
            coordinates: np.ndarray, pair: tuple[int, int] = positive_indices
        ) -> float:
            return measure_distance_angstrom(coordinates[pair[0]], coordinates[pair[1]])

        def measure_negative(
            coordinates: np.ndarray, pair: tuple[int, int] = negative_indices
        ) -> float:
            return measure_distance_angstrom(coordinates[pair[0]], coordinates[pair[1]])

        def measure_difference(
            coordinates: np.ndarray,
            positive_pair: tuple[int, int] = positive_indices,
            negative_pair: tuple[int, int] = negative_indices,
        ) -> float:
            return measure_distance_angstrom(
                coordinates[positive_pair[0]], coordinates[positive_pair[1]]
            ) - measure_distance_angstrom(
                coordinates[negative_pair[0]], coordinates[negative_pair[1]]
            )

        extractors[spec.positive_name] = measure_positive
        extractors[spec.negative_name] = measure_negative
        extractors[spec.name] = measure_difference
    return extractors
