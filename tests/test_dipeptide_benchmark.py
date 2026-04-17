from __future__ import annotations

from pathlib import Path

import pytest

from felp.app import run_pipeline
from felp.conversion import StructureConversionService
from felp.conversion.service import _import_parmed
from felp.core import (
    EnergyConfig,
    InputConfig,
    MicrostateConfig,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
)


@pytest.mark.integration
def test_gly_to_ala_benchmark_workflow(tmp_path: Path) -> None:
    pyrosetta = pytest.importorskip("pyrosetta")
    pytest.importorskip("openmm")
    pytest.importorskip("mdtraj")

    parmed, load_rosetta, _ = _import_parmed()

    pyrosetta.init("-ignore_unrecognized_res true -load_PDB_components false")
    pose = pyrosetta.pose_from_sequence("GGG")
    structure = load_rosetta(pose)
    round_tripped_pose = StructureConversionService(
        pyrosetta_init_options="-ignore_unrecognized_res true -load_PDB_components false"
    ).structure_to_pose(structure)
    round_tripped_structure = load_rosetta(round_tripped_pose)
    assert len(round_tripped_structure.atoms) == len(structure.atoms)
    assert len(round_tripped_structure.residues) == len(structure.residues)

    pdb_path = tmp_path / "gly_tripeptide.pdb"
    structure.save(str(pdb_path), overwrite=True)

    config = RunConfig(
        input=InputConfig(
            topology_file=str(pdb_path),
            trajectory_files=((str(pdb_path),),),
            stride=1,
        ),
        perturbation=PerturbationConfig(
            target_residue_number=2,
            target_residue_name="ALA",
            original_residue_name="GLY",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(force_constant=10000.0),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(source="grid", phi_bins=6, psi_bins=6),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir=str(tmp_path / "out")),
    )
    config.validate()

    summary = run_pipeline(config)
    assert summary["frame_count"] == 1
    assert (tmp_path / "out" / "frame_results.csv").is_file()
    assert (tmp_path / "out" / "microstate_summary.csv").is_file()
    assert (tmp_path / "out" / "reweighted_surface.csv").is_file()
