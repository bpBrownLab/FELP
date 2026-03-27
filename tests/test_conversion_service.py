from __future__ import annotations

from pathlib import Path

import pytest

from felp.conversion import StructureConversionService
from felp.conversion import service as conversion_service_module
from felp.conversion.service import _import_parmed
from tests.fake_rosetta_bridge import (
    FakeAtomID,
    FakeAtomType,
    FakePDBInfo,
    FakePose,
    FakeResidue,
    FakeResidueType,
    build_fake_import,
)

VENDORED_PARMED, _, _ = _import_parmed()
rosetta_pose_module = VENDORED_PARMED.rosetta.pose


def test_import_parmed_uses_repo_fallback() -> None:
    parmed, _, _ = _import_parmed()
    vendor_root = Path(__file__).resolve().parents[1] / "ParmEd"
    try:
        Path(parmed.__file__).resolve().relative_to(vendor_root.resolve())
    except ValueError as exc:
        raise AssertionError(f"Expected vendored ParmEd under {vendor_root}, got {parmed.__file__}") from exc


def test_load_topology_uses_parmed(tmp_path: Path) -> None:
    source_pdb = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.028   1.410   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.420   2.470   0.000  1.00  0.00           O
TER
END
"""
    topology_path = tmp_path / "minimal_ala.pdb"
    topology_path.write_text(source_pdb, encoding="utf-8")

    service = StructureConversionService(
        pyrosetta_init_options="-ignore_unrecognized_res true -load_PDB_components false"
    )
    structure = service.load_topology(str(topology_path))
    assert len(structure.atoms) > 0
    assert len(structure.residues) > 0


def _build_roundtrip_types():
    return [
        FakeResidueType(
            unique_name="ALA:NtermProteinFull",
            name3_value="ALA",
            name1_value="A",
            atom_order=("N", "CA", "C"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            },
            charges={"N": -0.3, "CA": 0.1, "C": 0.5},
        ),
        FakeResidueType(
            unique_name="GLY:CtermProteinFull",
            name3_value="GLY",
            name1_value="G",
            atom_order=("N", "CA", "C", "1HA", "2HA"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "1HA": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
                "2HA": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            },
            charges={"N": -0.25, "CA": 0.05, "C": 0.45},
            canonical_alias_map={"1HA": "HA2", "2HA": "HA3"},
        ),
    ]


def _build_roundtrip_pose():
    residue_types = _build_roundtrip_types()
    residue1 = FakeResidue(
        residue_types[0],
        coordinates={"N": (0.0, 0.0, 0.0), "CA": (1.0, 0.0, 0.0), "C": (2.0, 0.0, 0.0)},
    )
    residue2 = FakeResidue(
        residue_types[1],
        coordinates={"N": (3.0, 0.0, 0.0), "CA": (4.0, 0.0, 0.0), "C": (5.0, 0.0, 0.0), "1HA": (4.0, 1.0, 0.0), "2HA": (4.0, -1.0, 0.0)},
    )
    pdb_info = FakePDBInfo(2)
    pdb_info.chain(1, "A")
    pdb_info.chain(2, "A")
    pdb_info.number(1, 101)
    pdb_info.number(2, 102)
    pose = FakePose(
        [residue1, residue2],
        chain_indices=[1, 1],
        bonds=[((1, 3), (2, 1))],
        pdb_info=pdb_info,
    )
    return residue_types, pose


def test_structure_conversion_service_round_trips_with_vendor_bridge(monkeypatch) -> None:
    residue_types, pose = _build_roundtrip_pose()
    fake_import = build_fake_import(residue_types)
    fake_pyrosetta, _, _, _, _, xyz_vector = fake_import
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    monkeypatch.setattr(
        conversion_service_module,
        "_import_pyrosetta",
        lambda: (fake_pyrosetta, FakePose, None, FakeAtomID, xyz_vector),
    )
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INITIALIZED", False)
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INIT_OPTIONS", None)

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.pose_to_structure(pose)
    rebuilt_pose = service.structure_to_pose(structure)
    round_tripped = service.pose_to_structure(rebuilt_pose)

    assert fake_pyrosetta.init_calls == ["-mute all"]
    assert len(structure.atoms) == len(round_tripped.atoms)
    assert len(structure.residues) == len(round_tripped.residues)
    assert [atom.name for atom in structure.atoms] == [atom.name for atom in round_tripped.atoms]
    assert [residue.chain for residue in structure.residues] == [residue.chain for residue in round_tripped.residues]
    assert [
        (round(atom.xx, 3), round(atom.xy, 3), round(atom.xz, 3))
        for atom in round_tripped.atoms
    ] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (4.0, 1.0, 0.0),
        (4.0, -1.0, 0.0),
    ]


def test_clone_structure_with_coordinates_preserves_bridge_metadata(monkeypatch) -> None:
    residue_types, pose = _build_roundtrip_pose()
    fake_import = build_fake_import(residue_types)
    fake_pyrosetta, _, _, _, _, xyz_vector = fake_import
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    monkeypatch.setattr(
        conversion_service_module,
        "_import_pyrosetta",
        lambda: (fake_pyrosetta, FakePose, None, FakeAtomID, xyz_vector),
    )
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INITIALIZED", False)
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INIT_OPTIONS", None)

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.pose_to_structure(pose)
    cloned = service.clone_structure_with_coordinates(
        structure,
        [(10.0, 10.0, 10.0) for _ in structure.atoms],
    )

    assert hasattr(cloned, "_rosetta_bridge_metadata")
    assert cloned.coordinates.shape == (len(structure.atoms), 3)
