"""Structure conversion utilities for FELP."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
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


def _path_is_within(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _purge_parmed_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "parmed" or module_name.startswith("parmed."):
            del sys.modules[module_name]


def _promote_sys_path_entry(path: Path) -> None:
    target = path.resolve()
    kept_entries: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == target:
                continue
        except Exception:
            pass
        if entry == str(path):
            continue
        kept_entries.append(entry)
    sys.path[:] = [str(path)] + kept_entries


def _load_vendored_parmed(vendor_root: Path) -> ModuleType:
    package_root = vendor_root / "parmed"
    init_file = package_root / "__init__.py"
    if not init_file.is_file():
        raise DependencyError(f"Vendored ParmEd package is missing {init_file}.")

    existing = sys.modules.get("parmed")
    if existing is not None and _path_is_within(getattr(existing, "__file__", None), vendor_root):
        return existing

    _purge_parmed_modules()
    _promote_sys_path_entry(vendor_root)
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(
        "parmed",
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise DependencyError(f"Unable to create an import spec for vendored ParmEd at {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["parmed"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        _purge_parmed_modules()
        raise
    return module


def _import_parmed():
    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / "ParmEd"
    if vendor_root.is_dir():
        try:
            parmed = _load_vendored_parmed(vendor_root)
            load_rosetta = parmed.load_rosetta
            ChRad = importlib.import_module("parmed.tools.changeradii").ChRad
        except Exception as exc:
            raise DependencyError("Vendored ParmEd is required for FELP conversion.") from exc
        return parmed, load_rosetta, ChRad
    try:
        parmed = importlib.import_module("parmed")
        from parmed import load_rosetta
        from parmed.tools.changeradii import ChRad
    except Exception as exc:
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
        """Convert a PyRosetta Pose to a ParmEd Structure via vendored ParmEd."""
        _, load_rosetta, _ = _import_parmed()
        try:
            return load_rosetta(pose)
        except Exception as exc:
            raise ConversionError(f"Failed to convert Pose to Structure: {exc}") from exc

    def structure_to_pose(self, structure):
        """Convert a ParmEd Structure to a PyRosetta Pose via vendored ParmEd."""
        parmed, _, _ = _import_parmed()
        if not isinstance(structure, parmed.Structure):
            raise ConversionError("Object is not a ParmEd Structure.")

        self.ensure_pyrosetta_initialized()
        if not structure.residues:
            raise ConversionError("Cannot convert an empty Structure to a Pose.")
        save_rosetta = getattr(parmed, "save_rosetta", None)
        if save_rosetta is None:
            raise DependencyError("Vendored ParmEd is missing the save_rosetta bridge.")
        try:
            return save_rosetta(structure)
        except Exception as exc:
            raise ConversionError(f"Failed to convert Structure to Pose: {exc}") from exc

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
