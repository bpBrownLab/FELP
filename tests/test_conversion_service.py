from __future__ import annotations

from pathlib import Path

import pytest

from felp.conversion import StructureConversionService
from felp.conversion.service import _import_parmed


def test_import_parmed_uses_repo_fallback() -> None:
    parmed, _, _ = _import_parmed()
    assert "ParmEd/parmed/__init__.py" in parmed.__file__


def test_load_topology_uses_parmed(tmp_path: Path) -> None:
    source_pdb = (
        Path(__file__).resolve().parents[1] / "ParmEd" / "test" / "files" / "trx.pdb"
    )
    topology_path = tmp_path / "trx_copy.pdb"
    topology_path.write_text(source_pdb.read_text(encoding="utf-8"), encoding="utf-8")

    service = StructureConversionService(
        pyrosetta_init_options="-ignore_unrecognized_res true -load_PDB_components false"
    )
    structure = service.load_topology(str(topology_path))
    assert len(structure.atoms) > 0
    assert len(structure.residues) > 0


def test_validate_round_trip_requires_pyrosetta() -> None:
    pytest.importorskip("pyrosetta")
