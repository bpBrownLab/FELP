from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from felp.conversion import StructureConversionService
from felp.conversion import service as conversion_service_module
from felp.conversion.service import _import_parmed
from felp.core import ConversionError
from tests.fake_rosetta_bridge import (
    FakeAtomID,
    FakeAtomType,
    FakePDBInfo,
    FakePose,
    FakeResidue,
    FakeResidueType,
    build_fake_import,
)


def _rosetta_pose_module():
    parmed, _, _ = _import_parmed()
    return parmed.rosetta.pose


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


def _patch_conversion_pyrosetta(monkeypatch, residue_types):
    fake_import = build_fake_import(residue_types)
    fake_pyrosetta, _, _, _, _, xyz_vector = fake_import
    monkeypatch.setattr(
        conversion_service_module,
        "_import_pyrosetta",
        lambda: (fake_pyrosetta, FakePose, None, FakeAtomID, xyz_vector),
    )
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INITIALIZED", False)
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INIT_OPTIONS", None)
    return fake_pyrosetta


def test_validate_rosetta_residue_type_rejects_unloaded_protonation_state(monkeypatch) -> None:
    residue_types = [
        FakeResidueType(
            unique_name="ALA:NtermProteinFull",
            name3_value="ALA",
            name1_value="A",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        )
    ]
    _patch_conversion_pyrosetta(monkeypatch, residue_types)
    service = StructureConversionService(pyrosetta_init_options="-mute all")
    with pytest.raises(ConversionError, match="mutation target residue 'ASH'"):
        service.validate_rosetta_residue_type("ASH", context="mutation target")


def test_resolve_rosetta_residue_type_name_uses_rosetta_alias_candidates(monkeypatch) -> None:
    residue_types = [
        FakeResidueType(
            unique_name="HIS_E",
            name3_value="HIS",
            name1_value="H",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        )
    ]
    _patch_conversion_pyrosetta(monkeypatch, residue_types)
    service = StructureConversionService(pyrosetta_init_options="-mute all")
    assert (
        service.resolve_rosetta_residue_type_name(
            "HIE",
            context="mutation target",
        )
        == "HIS_E"
    )


def test_resolve_rosetta_residue_type_name_prefers_canonical_protonated_acid_variant(
    monkeypatch,
) -> None:
    residue_types = [
        FakeResidueType(
            unique_name="ASP_P2",
            name3_value="ASP",
            name1_value="D",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        ),
        FakeResidueType(
            unique_name="ASP_P1",
            name3_value="ASP",
            name1_value="D",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        ),
    ]
    _patch_conversion_pyrosetta(monkeypatch, residue_types)
    service = StructureConversionService(pyrosetta_init_options="-mute all")
    assert (
        service.resolve_rosetta_residue_type_name(
            "ASH",
            context="mutation target",
        )
        == "ASP_P2"
    )


def test_validate_rosetta_structure_residue_types_passes_when_exact_type_is_loaded(
    monkeypatch,
) -> None:
    residue_types = [
        FakeResidueType(
            unique_name="GLH:NtermProteinFull",
            name3_value="GLH",
            name1_value="E",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        )
    ]
    _patch_conversion_pyrosetta(monkeypatch, residue_types)
    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = SimpleNamespace(residues=[SimpleNamespace(name="GLH")])
    service.validate_rosetta_structure_residue_types(
        structure,
        context="the reference topology",
    )


def test_validate_rosetta_structure_residue_types_raises_on_unloaded_variant(
    monkeypatch,
) -> None:
    residue_types = [
        FakeResidueType(
            unique_name="GLU:NtermProteinFull",
            name3_value="GLU",
            name1_value="E",
            atom_order=("CA",),
            atom_types={"CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1)},
        )
    ]
    _patch_conversion_pyrosetta(monkeypatch, residue_types)
    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = SimpleNamespace(residues=[SimpleNamespace(name="GLH")])
    with pytest.raises(ConversionError, match="GLH"):
        service.validate_rosetta_structure_residue_types(
            structure,
            context="the reference topology",
        )


@pytest.mark.parametrize(
    ("atom_names", "expected_name"),
    [
        (("ND1", "NE2", "HD1"), "HID"),
        (("ND1", "NE2", "HE2"), "HIE"),
        (("ND1", "NE2", "HD1", "HE2"), "HIP"),
    ],
)
def test_prepare_structure_for_openmm_assigns_histidine_openmm_name(
    atom_names: tuple[str, ...],
    expected_name: str,
) -> None:
    parmed, _, _ = _import_parmed()
    structure = parmed.Structure()
    for atom_number, atom_name in enumerate(atom_names, start=1):
        atomic_number = 1 if atom_name.startswith("H") else 7
        structure.add_atom(
            parmed.Atom(
                name=atom_name,
                atomic_number=atomic_number,
                mass=1.0 if atomic_number == 1 else 14.0,
                number=atom_number,
            ),
            "HIS",
            1,
        )

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.prepare_structure_for_openmm(structure)

    assert structure.residues[0].name == expected_name


@pytest.mark.parametrize(
    ("residue_name", "atom_names", "expected_name", "expected_atoms"),
    [
        ("ASP", ("OD1", "OD2", "HD2"), "ASH", {"OD1", "OD2", "HD2"}),
        ("ASP_P1", ("OD1", "OD2", "HD1"), "ASH", {"OD1", "OD2", "HD2"}),
        ("GLU", ("OE1", "OE2", "HE2"), "GLH", {"OE1", "OE2", "HE2"}),
        ("GLU_P1", ("OE1", "OE2", "HE1"), "GLH", {"OE1", "OE2", "HE2"}),
    ],
)
def test_prepare_structure_for_openmm_assigns_protonated_acid_openmm_name(
    residue_name: str,
    atom_names: tuple[str, ...],
    expected_name: str,
    expected_atoms: set[str],
) -> None:
    parmed, _, _ = _import_parmed()
    structure = parmed.Structure()
    for atom_number, atom_name in enumerate(atom_names, start=1):
        atomic_number = 1 if atom_name.startswith("H") else 8
        if atom_name in {"OD1", "OD2", "OE1", "OE2"}:
            atomic_number = 8
        structure.add_atom(
            parmed.Atom(
                name=atom_name,
                atomic_number=atomic_number,
                mass=1.0 if atomic_number == 1 else 16.0,
                number=atom_number,
            ),
            residue_name,
            1,
        )

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.prepare_structure_for_openmm(structure)

    assert structure.residues[0].name == expected_name
    assert {atom.name for atom in structure.atoms} == expected_atoms


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
    monkeypatch.setattr(_rosetta_pose_module(), "_import_pyrosetta", lambda: fake_import)
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
    monkeypatch.setattr(_rosetta_pose_module(), "_import_pyrosetta", lambda: fake_import)
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


def test_prepare_structure_for_openmm_normalizes_proline_and_strips_virtual_atoms(
    monkeypatch,
) -> None:
    proline_type = FakeResidueType(
        unique_name="PRO:NtermProteinFull:CtermProteinFull",
        name3_value="PRO",
        name1_value="P",
        atom_order=(
            "N",
            "CA",
            "C",
            "CB",
            "1HB",
            "2HB",
            "CG",
            "1HG",
            "2HG",
            "CD",
            "1HD",
            "2HD",
            "V1",
        ),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "CB": FakeAtomType(element="C", atom_type_name="CH2", lj_radius=1.8, lj_wdepth=0.1),
            "1HB": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HB": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "CG": FakeAtomType(element="C", atom_type_name="CH2", lj_radius=1.8, lj_wdepth=0.1),
            "1HG": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HG": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "CD": FakeAtomType(element="C", atom_type_name="CH2", lj_radius=1.8, lj_wdepth=0.1),
            "1HD": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HD": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "V1": FakeAtomType(
                element="EP",
                atom_type_name="VIRT",
                lj_radius=0.0,
                lj_wdepth=0.0,
                is_virtual=True,
            ),
        },
    )
    pose = FakePose(
        [
            FakeResidue(
                proline_type,
                coordinates={
                    "N": (0.0, 0.0, 0.0),
                    "CA": (1.0, 0.0, 0.0),
                    "C": (2.0, 0.0, 0.0),
                    "CB": (1.0, 1.0, 0.0),
                    "1HB": (1.0, 1.5, 0.8),
                    "2HB": (1.0, 1.5, -0.8),
                    "CG": (0.0, 1.5, 0.0),
                    "1HG": (-0.6, 1.8, 0.8),
                    "2HG": (-0.6, 1.8, -0.8),
                    "CD": (-0.5, 0.4, 0.0),
                    "1HD": (-1.2, 0.2, 0.8),
                    "2HD": (-1.2, 0.2, -0.8),
                    "V1": (-0.2, 0.9, 0.0),
                },
            )
        ],
        chain_indices=[1],
        bonds=[
            ((1, 1), (1, 2)),
            ((1, 2), (1, 3)),
            ((1, 2), (1, 4)),
            ((1, 4), (1, 5)),
            ((1, 4), (1, 6)),
            ((1, 4), (1, 7)),
            ((1, 7), (1, 8)),
            ((1, 7), (1, 9)),
            ((1, 7), (1, 10)),
            ((1, 10), (1, 11)),
            ((1, 10), (1, 12)),
            ((1, 10), (1, 13)),
        ],
        pdb_info=FakePDBInfo(1),
    )

    fake_import = build_fake_import([proline_type])
    fake_pyrosetta, _, _, _, _, xyz_vector = fake_import
    monkeypatch.setattr(_rosetta_pose_module(), "_import_pyrosetta", lambda: fake_import)
    monkeypatch.setattr(
        conversion_service_module,
        "_import_pyrosetta",
        lambda: (fake_pyrosetta, FakePose, None, FakeAtomID, xyz_vector),
    )
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INITIALIZED", False)
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INIT_OPTIONS", None)

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.pose_to_structure(pose)
    assert [atom.name for atom in structure.atoms][-1] == "V1"

    structure = service.prepare_structure_for_openmm(structure)

    assert [atom.name for atom in structure.atoms] == [
        "N",
        "CA",
        "C",
        "CB",
        "HB2",
        "HB3",
        "CG",
        "HG2",
        "HG3",
        "CD",
        "HD2",
        "HD3",
    ]
    metadata = getattr(structure, "_rosetta_bridge_metadata")
    assert [atom_meta["structure_name"] for atom_meta in metadata["residues"][0]["atoms"]] == [
        "N",
        "CA",
        "C",
        "CB",
        "HB2",
        "HB3",
        "CG",
        "HG2",
        "HG3",
        "CD",
        "HD2",
        "HD3",
    ]
    assert any(
        {bond.atom1.name, bond.atom2.name} == {"N", "CD"}
        for bond in structure.bonds
        if bond.atom1.residue is structure.residues[0] and bond.atom2.residue is structure.residues[0]
    )
    assert all(
        bond.atom1.name != "V1" and bond.atom2.name != "V1"
        for bond in structure.bonds
    )


def test_prepare_structure_for_openmm_normalizes_terminal_hydrogen_names(
    monkeypatch,
) -> None:
    gly_type = FakeResidueType(
        unique_name="GLY:NtermProteinFull",
        name3_value="GLY",
        name1_value="G",
        atom_order=("N", "CA", "C", "1H", "2H", "3H"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "1H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "2H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "3H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
        },
    )
    pose = FakePose(
        [
            FakeResidue(
                gly_type,
                coordinates={
                    "N": (0.0, 0.0, 0.0),
                    "CA": (1.0, 0.0, 0.0),
                    "C": (2.0, 0.0, 0.0),
                    "1H": (-0.5, 0.0, 0.0),
                    "2H": (-0.5, 0.5, 0.0),
                    "3H": (-0.5, -0.5, 0.0),
                },
            )
        ],
        chain_indices=[1],
        pdb_info=FakePDBInfo(1),
    )
    fake_import = build_fake_import([gly_type])
    fake_pyrosetta, _, _, _, _, xyz_vector = fake_import
    monkeypatch.setattr(_rosetta_pose_module(), "_import_pyrosetta", lambda: fake_import)
    monkeypatch.setattr(
        conversion_service_module,
        "_import_pyrosetta",
        lambda: (fake_pyrosetta, FakePose, None, FakeAtomID, xyz_vector),
    )
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INITIALIZED", False)
    monkeypatch.setattr(conversion_service_module, "_PYROSETTA_INIT_OPTIONS", None)

    service = StructureConversionService(pyrosetta_init_options="-mute all")
    structure = service.prepare_structure_for_openmm(service.pose_to_structure(pose))

    assert [atom.name for atom in structure.atoms] == ["N", "CA", "C", "H1", "H2", "H3"]
