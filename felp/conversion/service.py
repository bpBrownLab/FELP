"""Structure conversion utilities for FELP."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np

from felp.core import ConversionError, DependencyError

PMD_TO_ROSETTA_1LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "CYX": "C",
    "CYM": "C",
    "ACE": "A[ACE]",
    "NME": "N[NME]",
}

POSE_ATOM_NAME_CONVERSION = {
    "1HH3": "H1",
    "2HH3": "H2",
    "3HH3": "H3",
    "1HH2": "H1",
    "2HH2": "H2",
    "3HH2": "H3",
    "1HA": "HA2",
    "2HA": "HA3",
    "HN2": "H",
}

_PYROSETTA_INITIALIZED = False
_PYROSETTA_INIT_OPTIONS: Optional[str] = None


def _import_parmed():
    try:
        import parmed
        from parmed import load_rosetta
        from parmed.tools.changeradii import ChRad
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[2]
        vendor_root = repo_root / "ParmEd"
        if vendor_root.is_dir():
            sys.path.insert(0, str(vendor_root))
            try:
                import parmed
                from parmed import load_rosetta
                from parmed.tools.changeradii import ChRad
            except ImportError as nested_exc:
                raise DependencyError(
                    "ParmEd is required for FELP conversion."
                ) from nested_exc
        else:
            raise DependencyError("ParmEd is required for FELP conversion.") from exc
    return parmed, load_rosetta, ChRad


def _import_pyrosetta():
    try:
        import pyrosetta
        from pyrosetta import Pose, pose_from_sequence
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.numeric import xyzVector_double_t
    except ImportError as exc:
        raise DependencyError("PyRosetta is required for FELP conversion.") from exc
    return pyrosetta, Pose, pose_from_sequence, AtomID, xyzVector_double_t


@dataclass
class StructureConversionService:
    """Owns PyRosetta and ParmEd conversion policies."""

    pyrosetta_init_options: str
    radii_set: str = "mbondi3"

    def ensure_pyrosetta_initialized(self) -> None:
        """Initialize PyRosetta exactly once per process with the configured options."""
        global _PYROSETTA_INITIALIZED, _PYROSETTA_INIT_OPTIONS

        pyrosetta, _, _, _, _ = _import_pyrosetta()
        options = self.pyrosetta_init_options.strip()
        if _PYROSETTA_INITIALIZED:
            if options and _PYROSETTA_INIT_OPTIONS != options:
                raise DependencyError(
                    "PyRosetta was already initialized with different options."
                )
            return
        pyrosetta.init(options=options)
        _PYROSETTA_INITIALIZED = True
        _PYROSETTA_INIT_OPTIONS = options

    def load_topology(self, topology_file: str):
        """Load a ParmEd Structure from a topology file and apply the configured radii set."""
        parmed, _, changeradii = _import_parmed()
        structure = parmed.load_file(topology_file)
        changeradii(structure, self.radii_set)
        return structure

    def clone_structure_with_coordinates(self, template_structure, coordinates_angstrom):
        """Create a lightweight copy of a ParmEd Structure with new coordinates."""
        structure = copy.copy(template_structure)
        structure.coordinates = np.asarray(coordinates_angstrom, dtype=float)
        return structure

    def pose_to_structure(self, pose):
        """Convert a PyRosetta Pose to a ParmEd Structure."""
        _, load_rosetta, _ = _import_parmed()
        structure = load_rosetta(pose)
        return structure

    def structure_to_pose(self, structure):
        """Convert a ParmEd Structure to a PyRosetta Pose.
        
        This relies on the PMD_TO_ROSETTA_1LETTER mapping to build a sequence,
        then copies coordinates over by matching atom names.
        """
        parmed, _, _ = _import_parmed()
        if not isinstance(structure, parmed.Structure):
            raise ConversionError("Object is not a ParmEd Structure.")

        self.ensure_pyrosetta_initialized()
        _, Pose, pose_from_sequence, AtomID, xyz_vector = _import_pyrosetta()

        if not structure.residues:
            raise ConversionError("Cannot convert an empty Structure to a Pose.")

        sequence = []
        residue_atom_maps = []
        for residue in structure.residues:
            residue_name = residue.name.upper()
            if residue_name not in PMD_TO_ROSETTA_1LETTER:
                raise ConversionError(
                    f"Unsupported residue for Pose conversion: {residue.name}"
                )
            sequence.append(PMD_TO_ROSETTA_1LETTER[residue_name])
            residue_atom_maps.append({atom.name.strip(): atom for atom in residue.atoms})

        pose = pose_from_sequence("".join(sequence).replace(" ", ""))
        if not isinstance(pose, Pose):
            raise ConversionError("Failed to build a PyRosetta Pose from the Structure.")

        for residue_index in range(1, pose.total_residue() + 1):
            pose_residue = pose.residue(residue_index)
            atom_map = residue_atom_maps[residue_index - 1]
            for atom_index in range(1, pose_residue.natoms() + 1):
                atom_name = pose_residue.atom_name(atom_index).strip()
                parmed_atom = atom_map.get(atom_name)
                if parmed_atom is None and atom_name in POSE_ATOM_NAME_CONVERSION:
                    parmed_atom = atom_map.get(POSE_ATOM_NAME_CONVERSION[atom_name])
                if parmed_atom is None:
                    continue
                atom_id = AtomID(atom_index, residue_index)
                coordinates = xyz_vector(
                    float(parmed_atom.xx),
                    float(parmed_atom.xy),
                    float(parmed_atom.xz),
                )
                pose.set_xyz(atom_id, coordinates)

        try:
            pose.conformation().detect_disulfides()
        except Exception:
            pass

        return pose

    def validate_round_trip(self, structure) -> dict[str, int]:
        """Test conversion fidelity by converting a Structure to a Pose and back."""
        pose = self.structure_to_pose(structure)
        round_tripped = self.pose_to_structure(pose)
        return {
            "input_atom_count": len(structure.atoms),
            "output_atom_count": len(round_tripped.atoms),
            "input_residue_count": len(structure.residues),
            "output_residue_count": len(round_tripped.residues),
        }
