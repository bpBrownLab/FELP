"""Structure conversion utilities for FELP."""

from __future__ import annotations

from collections.abc import Iterable
import copy
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional

import numpy as np

from felp.core import (
    ConversionError,
    DependencyError,
    mutation_target_family,
    normalize_mutation_target_name,
    rosetta_residue_name_candidates,
)

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
    "1HB": "HB2",
    "2HB": "HB3",
    "1HG": "HG2",
    "2HG": "HG3",
    "1HD": "HD2",
    "2HD": "HD3",
    "1HE": "HE2",
    "2HE": "HE3",
    "1HZ": "HZ1",
    "2HZ": "HZ2",
    "3HZ": "HZ3",
    "1HG1": "HG11",
    "2HG1": "HG12",
    "3HG1": "HG13",
    "1HG2": "HG21",
    "2HG2": "HG22",
    "3HG2": "HG23",
    "1HD1": "HD11",
    "2HD1": "HD12",
    "3HD1": "HD13",
    "1HD2": "HD21",
    "2HD2": "HD22",
    "3HD2": "HD23",
    "1HE2": "HE21",
    "2HE2": "HE22",
    "1HH1": "HH11",
    "2HH1": "HH12",
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
        raise DependencyError(
            f"Vendored ParmEd package is missing {init_file}. "
            "If this is a git checkout, initialize the submodule from the repo root: "
            "git submodule update --init ParmEd"
        )

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


def _normalize_pose_atom_name(atom_name: str) -> str:
    """Map Rosetta-style atom names onto AMBER/OpenMM template names."""

    stripped = atom_name.strip()
    return POSE_ATOM_NAME_CONVERSION.get(stripped, stripped)


def _normalize_pose_derived_structure_names(structure) -> None:
    """Normalize Rosetta atom names on a just-converted ParmEd structure.

    ``load_rosetta()`` preserves the exact Rosetta atom names in both the
    structure and its bridge metadata. That is ideal for round-tripping back
    into a Pose, but some Rosetta hydrogen names (`1HB`, `2HD`, etc.) do not
    match AMBER/OpenMM residue templates (`HB2`, `HD3`, etc.). Normalize the
    structure-facing names while keeping the original ``pose_name`` metadata so
    the bridge can still reconstruct the original Pose atom ordering.
    """

    metadata = getattr(structure, "_rosetta_bridge_metadata", None)
    residue_metadata = metadata.get("residues") if isinstance(metadata, dict) else None
    if not isinstance(residue_metadata, list):
        return

    for residue, residue_meta in zip(structure.residues, residue_metadata):
        atom_metadata = residue_meta.get("atoms") if isinstance(residue_meta, dict) else None
        if not isinstance(atom_metadata, list):
            continue
        for atom, atom_meta in zip(getattr(residue, "atoms", ()), atom_metadata):
            if not isinstance(atom_meta, dict):
                continue
            pose_name = str(atom_meta.get("pose_name", atom.name)).strip()
            normalized_name = _normalize_pose_atom_name(pose_name or str(atom.name))
            atom.name = normalized_name
            atom_meta["structure_name"] = normalized_name


def _is_virtual_pose_derived_atom(atom, atom_meta=None) -> bool:
    """Return whether a Rosetta-derived ParmEd atom is virtual."""

    if isinstance(atom_meta, dict) and bool(atom_meta.get("is_virtual", False)):
        return True
    try:
        if int(getattr(atom, "atomic_number", 0) or 0) == 0:
            return True
    except Exception:
        pass
    return str(getattr(atom, "type", "")).strip().upper() in {"EP", "VIRT"}


def _strip_pose_derived_virtual_atoms(structure) -> None:
    """Remove Rosetta virtual atoms and all bonded terms attached to them."""

    metadata = getattr(structure, "_rosetta_bridge_metadata", None)
    residue_metadata = metadata.get("residues") if isinstance(metadata, dict) else None
    selection: list[int] = []

    if isinstance(residue_metadata, list):
        for residue_index, residue in enumerate(getattr(structure, "residues", ())):
            residue_meta = (
                residue_metadata[residue_index]
                if residue_index < len(residue_metadata)
                and isinstance(residue_metadata[residue_index], dict)
                else None
            )
            atom_metadata = residue_meta.get("atoms") if residue_meta is not None else None
            kept_atom_metadata: list[dict] = []
            for atom_index, atom in enumerate(getattr(residue, "atoms", ())):
                atom_meta = (
                    atom_metadata[atom_index]
                    if isinstance(atom_metadata, list)
                    and atom_index < len(atom_metadata)
                    and isinstance(atom_metadata[atom_index], dict)
                    else None
                )
                is_virtual = _is_virtual_pose_derived_atom(atom, atom_meta)
                selection.append(1 if is_virtual else 0)
                if isinstance(atom_metadata, list) and atom_index < len(atom_metadata) and not is_virtual:
                    kept_atom_metadata.append(atom_metadata[atom_index])
            if residue_meta is not None and isinstance(atom_metadata, list):
                residue_meta["atoms"] = kept_atom_metadata
    else:
        selection = [1 if _is_virtual_pose_derived_atom(atom) else 0 for atom in structure.atoms]

    if any(selection):
        structure.strip(selection)


def _repair_pose_derived_proline_ring_closure(structure) -> None:
    """Ensure converted proline residues include the backbone N-CD bond.

    PyRosetta residue conversion occasionally drops the intramolecular N-CD bond
    that distinguishes proline from other amino acids. OpenMM's AMBER templates
    require that bond to match the internal ``PRO`` residue template, so repair
    it here if the converted ParmEd structure is missing it.
    """

    parmed, _, _ = _import_parmed()
    for residue in getattr(structure, "residues", ()):
        if str(getattr(residue, "name", "")).strip().upper() != "PRO":
            continue
        atoms_by_name = {
            str(getattr(atom, "name", "")).strip().upper(): atom
            for atom in getattr(residue, "atoms", ())
        }
        nitrogen = atoms_by_name.get("N")
        delta_carbon = atoms_by_name.get("CD")
        if nitrogen is None or delta_carbon is None:
            continue
        if delta_carbon in getattr(nitrogen, "bond_partners", ()):
            continue
        structure.bonds.append(parmed.Bond(nitrogen, delta_carbon))


def _normalized_residue_name(name: object) -> str:
    text = str(name).strip().upper()
    if not text:
        return ""
    return normalize_mutation_target_name(text)


def _unique_residue_names(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = _normalized_residue_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _residue_type_is_available(residue_type_set, residue_name: str) -> bool:
    raw_candidate = str(residue_name).strip().upper()
    candidate = _normalized_residue_name(residue_name)
    if not raw_candidate and not candidate:
        return False
    for probe in (raw_candidate, candidate):
        if not probe:
            continue
        if hasattr(residue_type_set, "has_name"):
            try:
                if residue_type_set.has_name(probe):
                    return True
            except Exception:
                pass
        if hasattr(residue_type_set, "has_name3"):
            try:
                if residue_type_set.has_name3(probe):
                    return True
            except Exception:
                pass
        if hasattr(residue_type_set, "get_representative_type_base_name"):
            try:
                residue_type_set.get_representative_type_base_name(probe)
                return True
            except Exception:
                pass
    return False


def _resolve_available_rosetta_name(residue_type_set, residue_name: str) -> str | None:
    for candidate in rosetta_residue_name_candidates(residue_name):
        if _residue_type_is_available(residue_type_set, candidate):
            return candidate
    return None


def _structure_residue_type_candidates(structure, residue_index: int) -> tuple[str, ...]:
    residue = structure.residues[residue_index]
    metadata = getattr(structure, "_rosetta_bridge_metadata", None)
    residue_metadata = metadata.get("residues") if isinstance(metadata, dict) else None
    residue_meta = (
        residue_metadata[residue_index]
        if isinstance(residue_metadata, list) and residue_index < len(residue_metadata)
        and isinstance(residue_metadata[residue_index], dict)
        else None
    )
    candidates = list(rosetta_residue_name_candidates(getattr(residue, "name", "")))
    if residue_meta is not None:
        candidates.extend(
            (
                residue_meta.get("type_name", ""),
                residue_meta.get("name3", ""),
                residue_meta.get("base_name", ""),
            )
        )
    expanded: list[object] = []
    for candidate in candidates:
        expanded.extend(rosetta_residue_name_candidates(str(candidate)))
        expanded.append(candidate)
    return _unique_residue_names(expanded)


def _residue_family_hint(residue, residue_meta=None) -> str:
    candidates = [getattr(residue, "name", "")]
    if isinstance(residue_meta, dict):
        candidates.extend(
            (
                residue_meta.get("type_name", ""),
                residue_meta.get("name3", ""),
                residue_meta.get("base_name", ""),
            )
        )
    for candidate in candidates:
        raw_candidate = str(candidate).strip().upper()
        if not raw_candidate:
            continue
        if raw_candidate.startswith("ASP_P") or mutation_target_family(raw_candidate) == "ASP":
            return "ASP"
        if raw_candidate.startswith("GLU_P") or mutation_target_family(raw_candidate) == "GLU":
            return "GLU"
        if raw_candidate.startswith("HIS_") or mutation_target_family(raw_candidate) == "HIS":
            return "HIS"
        if mutation_target_family(raw_candidate) == "CYS":
            return "CYS"
        if mutation_target_family(raw_candidate) == "LYS":
            return "LYS"
    return ""


def _histidine_openmm_name(residue, residue_meta=None) -> str | None:
    if _residue_family_hint(residue, residue_meta) != "HIS":
        return None
    atom_names = {
        str(getattr(atom, "name", "")).strip().upper()
        for atom in getattr(residue, "atoms", ())
    }
    has_hd1 = "HD1" in atom_names
    has_he2 = "HE2" in atom_names
    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    if has_he2:
        return "HIE"
    return "HIS"


def _acid_proton_location(
    residue,
    *,
    oxygen1_name: str,
    oxygen2_name: str,
    hydrogen1_names: tuple[str, ...],
    hydrogen2_names: tuple[str, ...],
) -> str | None:
    atom_names = {
        str(getattr(atom, "name", "")).strip().upper()
        for atom in getattr(residue, "atoms", ())
    }
    has_oxygen1 = oxygen1_name in atom_names
    has_oxygen2 = oxygen2_name in atom_names
    if not has_oxygen1 or not has_oxygen2:
        return None
    if any(name in atom_names for name in hydrogen2_names):
        return "O2"
    if any(name in atom_names for name in hydrogen1_names):
        return "O1"
    return None


def _rename_residue_atom(residue, from_name: str, to_name: str, residue_meta=None) -> None:
    source = from_name.strip().upper()
    target = to_name.strip().upper()
    if source == target:
        return
    for atom in getattr(residue, "atoms", ()):
        if str(getattr(atom, "name", "")).strip().upper() == source:
            atom.name = target
            break
    if not isinstance(residue_meta, dict):
        return
    for atom_meta in residue_meta.get("atoms", ()) or ():
        if str(atom_meta.get("structure_name", "")).strip().upper() == source:
            atom_meta["structure_name"] = target


def _canonicalize_protonated_acid(
    residue,
    residue_meta,
    *,
    openmm_name: str,
    oxygen1_name: str,
    oxygen2_name: str,
    hydrogen1_names: tuple[str, ...],
    hydrogen2_names: tuple[str, ...],
    canonical_hydrogen_name: str,
) -> str | None:
    proton_location = _acid_proton_location(
        residue,
        oxygen1_name=oxygen1_name,
        oxygen2_name=oxygen2_name,
        hydrogen1_names=hydrogen1_names,
        hydrogen2_names=hydrogen2_names,
    )
    if proton_location is None:
        return None
    if proton_location == "O1":
        temp_name = "__TMP_O__"
        _rename_residue_atom(residue, oxygen2_name, temp_name, residue_meta)
        _rename_residue_atom(residue, oxygen1_name, oxygen2_name, residue_meta)
        _rename_residue_atom(residue, temp_name, oxygen1_name, residue_meta)
        for hydrogen_name in hydrogen1_names:
            _rename_residue_atom(
                residue,
                hydrogen_name,
                canonical_hydrogen_name,
                residue_meta,
            )
    else:
        for hydrogen_name in hydrogen2_names:
            _rename_residue_atom(
                residue,
                hydrogen_name,
                canonical_hydrogen_name,
                residue_meta,
            )
    residue.name = openmm_name
    if isinstance(residue_meta, dict):
        residue_meta["openmm_name"] = openmm_name
    return openmm_name


def _acid_openmm_name(residue, residue_meta=None) -> str | None:
    family = _residue_family_hint(residue, residue_meta)
    if family == "ASP":
        return _canonicalize_protonated_acid(
            residue,
            residue_meta,
            openmm_name="ASH",
            oxygen1_name="OD1",
            oxygen2_name="OD2",
            hydrogen1_names=("HD1", "HD11", "HD12"),
            hydrogen2_names=("HD2", "HD21", "HD22"),
            canonical_hydrogen_name="HD2",
        )
    if family == "GLU":
        return _canonicalize_protonated_acid(
            residue,
            residue_meta,
            openmm_name="GLH",
            oxygen1_name="OE1",
            oxygen2_name="OE2",
            hydrogen1_names=("HE1", "HE11", "HE12"),
            hydrogen2_names=("HE2", "HE21", "HE22"),
            canonical_hydrogen_name="HE2",
        )
    return None


def _assign_openmm_residue_names(structure) -> None:
    metadata = getattr(structure, "_rosetta_bridge_metadata", None)
    residue_metadata = metadata.get("residues") if isinstance(metadata, dict) else None
    for residue_index, residue in enumerate(getattr(structure, "residues", ())):
        residue_meta = (
            residue_metadata[residue_index]
            if isinstance(residue_metadata, list)
            and residue_index < len(residue_metadata)
            and isinstance(residue_metadata[residue_index], dict)
            else None
        )
        normalized_name = _histidine_openmm_name(residue, residue_meta)
        if normalized_name is None:
            normalized_name = _acid_openmm_name(residue, residue_meta)
        if normalized_name is None:
            continue
        residue.name = normalized_name
        if isinstance(residue_meta, dict):
            residue_meta["openmm_name"] = normalized_name


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

    def _rosetta_residue_type_set(self):
        self.ensure_pyrosetta_initialized()
        pyrosetta, _, _, _, _ = _import_pyrosetta()
        return pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set(
            "fa_standard"
        )

    def load_topology(self, topology_file: str):
        """Load a ParmEd Structure from a topology file and apply the configured radii set."""
        parmed, _, changeradii = _import_parmed()
        structure = parmed.load_file(topology_file)
        changeradii(structure, self.radii_set)
        return structure

    def validate_rosetta_residue_type(self, residue_name: str, *, context: str) -> None:
        """Fail early when a requested Rosetta residue type is not loaded."""

        normalized = _normalized_residue_name(residue_name)
        residue_type_set = self._rosetta_residue_type_set()
        if _resolve_available_rosetta_name(residue_type_set, normalized) is not None:
            return
        raise ConversionError(
            f"Rosetta fa_standard could not resolve {context} residue {normalized!r}. "
            "FELP resolves CLI residue names onto Rosetta-native residue types before mutation, "
            "so protonation-state studies require a compatible residue type to be loaded in the "
            "current PyRosetta session. Histidine/cysteine variants may map onto built-in "
            "aliases such as HIS_D/HIS_E/HIS_P/CYZ/CYD, while protonated Asp/Glu may require "
            "Rosetta pH-mode residue types or custom params loaded with "
            "--pyrosetta-init-options/-extra_res_fa."
        )

    def resolve_rosetta_residue_type_name(self, residue_name: str, *, context: str) -> str:
        """Return the concrete Rosetta residue type name to hand to MutateResidue."""

        normalized = _normalized_residue_name(residue_name)
        residue_type_set = self._rosetta_residue_type_set()
        resolved = _resolve_available_rosetta_name(residue_type_set, normalized)
        if resolved is not None:
            return resolved
        raise ConversionError(
            f"Rosetta fa_standard could not resolve {context} residue {normalized!r}. "
            "FELP needs a concrete Rosetta residue type name for mutation/protonation "
            "enforcement. Histidine and cysteine variants may map onto Rosetta-native "
            "aliases such as HIS_D/HIS_E/HIS_P/CYZ/CYD, while other protonation states "
            "may require pH-mode residue types or custom params via --pyrosetta-init-options/-extra_res_fa."
        )

    def validate_rosetta_structure_residue_types(self, structure, *, context: str) -> None:
        """Fail early when topology residue names cannot be converted into Rosetta types."""

        residue_type_set = self._rosetta_residue_type_set()
        unsupported: list[str] = []
        for residue_index, residue in enumerate(getattr(structure, "residues", ()), start=1):
            candidates = _structure_residue_type_candidates(structure, residue_index - 1)
            if any(_residue_type_is_available(residue_type_set, candidate) for candidate in candidates):
                continue
            residue_name = _normalized_residue_name(getattr(residue, "name", "")) or "<unknown>"
            unsupported.append(f"{residue_index}:{residue_name}")
        if not unsupported:
            return
        detail = ", ".join(unsupported[:8])
        if len(unsupported) > 8:
            detail += ", ..."
        raise ConversionError(
            f"Rosetta fa_standard could not resolve residue types in {context}: {detail}. "
            "FELP converts the reference topology through PyRosetta before mutation, so the "
            "topology residue names must match residue types loaded in the current PyRosetta "
            "session. Histidine/cysteine variants may map onto Rosetta-native aliases such as "
            "HIS_D/HIS_E/HIS_P/CYZ/CYD, but protonated Asp/Glu may require Rosetta pH-mode "
            "residue types or custom params loaded with --pyrosetta-init-options/-extra_res_fa."
        )

    def clone_structure_with_coordinates(self, template_structure, coordinates_angstrom):
        """Create a lightweight copy of a ParmEd Structure with new coordinates."""
        structure = copy.copy(template_structure)
        structure.coordinates = np.asarray(coordinates_angstrom, dtype=float)
        return structure

    def pose_to_structure(self, pose):
        """Convert a PyRosetta Pose to a ParmEd Structure via vendored ParmEd."""
        _, load_rosetta, _ = _import_parmed()
        try:
            structure = load_rosetta(pose)
            _normalize_pose_derived_structure_names(structure)
            _repair_pose_derived_proline_ring_closure(structure)
            return structure
        except Exception as exc:
            raise ConversionError(f"Failed to convert Pose to Structure: {exc}") from exc

    def prepare_structure_for_openmm(self, structure):
        """Mutate a pose-derived structure into an OpenMM-template-compatible form."""

        _normalize_pose_derived_structure_names(structure)
        _assign_openmm_residue_names(structure)
        _strip_pose_derived_virtual_atoms(structure)
        _repair_pose_derived_proline_ring_closure(structure)
        return structure

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
