from __future__ import annotations

from felp.core import (
    AtomReference,
    DistanceRestraintSpec,
    DistanceCVSpec,
    EnergyConfig,
    InputConfig,
    MicrostateConfig,
    MobilityConfig,
    MutationSpec,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
)
from felp.protocol_summary import build_protocol_summary


def test_protocol_summary_explicitly_reports_target_only_backbone_scope() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(target_residue_number=4, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(
            distance_cv_specs=(
                DistanceCVSpec(
                    name="ends",
                    atom1=AtomReference(1, "CA"),
                    atom2=AtomReference(10, "CA"),
                ),
            ),
            rosetta_target_backbone_restraints=("phi", "psi"),
        ),
        mobility=MobilityConfig(include_backbone=True),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(source="auto"),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    summary = build_protocol_summary(config)

    assert summary["mobility"]["selected_residue_scope"]["residue_numbers"] == [4]
    assert summary["rosetta_preparation"]["movemap"]["mobile_backbone_scope"] == (
        "backbone torsions on selected residues"
    )
    assert summary["rosetta_preparation"]["constraints"]["dihedral_constraints"][
        "rosetta_only_target_backbone_dihedrals"
    ] == ["phi", "psi"]
    assert summary["openmm_final_minimization"]["mass_frozen_dofs"][
        "mobile_backbone_scope"
    ] == "backbone atoms on selected residues"
    assert summary["analysis_restraints"]["runtime_restraint_specs"]["available"] is False
    assert "frame-specific" in summary["analysis_restraints"]["runtime_restraint_specs"]["note"]
    assert summary["analysis_restraints"]["configured_distance_templates"] == [
        {
            "label": "ends",
            "atom1": {"residue_number": 1, "atom_name": "CA"},
            "atom2": {"residue_number": 10, "atom_name": "CA"},
            "target_source": "measured from each reference frame",
        }
    ]


def test_protocol_summary_reports_actual_runtime_restraints_when_available() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(target_residue_number=4, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(
            distance_cv_specs=(
                DistanceCVSpec(
                    name="ends",
                    atom1=AtomReference(1, "CA"),
                    atom2=AtomReference(10, "CA"),
                ),
            ),
        ),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(source="auto"),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )
    restraint = DistanceRestraintSpec(
        label="ends",
        atom1=AtomReference(1, "CA"),
        atom2=AtomReference(10, "CA"),
        target_distance_angstrom=21.4,
        force_constant=10000.0,
    )

    summary = build_protocol_summary(
        config,
        restraint_specs=(restraint,),
        rosetta_restraint_specs=(restraint,),
        openmm_restraint_specs=(restraint,),
    )

    runtime = summary["openmm_final_minimization"]["restraints"][
        "runtime_restraint_specs"
    ]
    assert runtime["available"] is True
    assert runtime["count"] == 1
    assert runtime["specs"][0]["target_distance_angstrom"] == 21.4


def test_protocol_summary_reports_sequence_window_and_backbone_restraint_residues() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(target_residue_number=4, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(
            distance_cv_specs=(
                DistanceCVSpec(
                    name="ends",
                    atom1=AtomReference(1, "CA"),
                    atom2=AtomReference(10, "CA"),
                ),
            ),
            rosetta_target_backbone_restraints=("phi", "psi", "omega"),
            rosetta_backbone_restraint_residue_numbers=(3, 4, 5),
            rosetta_target_backbone_restraint_force_constant=2500.0,
        ),
        mobility=MobilityConfig(
            mode="sequence_window",
            sequence_window=1,
            include_backbone=True,
        ),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(source="auto"),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    summary = build_protocol_summary(config)
    selected_scope = summary["mobility"]["selected_residue_scope"]
    dihedral_constraints = summary["rosetta_preparation"]["constraints"][
        "dihedral_constraints"
    ]

    assert selected_scope["mode"] == "sequence_window"
    assert selected_scope["residue_numbers"] == [3, 4, 5]
    assert dihedral_constraints[
        "rosetta_only_backbone_restraint_residue_numbers"
    ] == [3, 4, 5]
    assert dihedral_constraints[
        "configured_rosetta_only_target_backbone_templates"
    ][0] == {
        "label": "rosetta_residue_3_phi",
        "kind": "phi",
        "residue_number": 3,
        "force_constant": 2500.0,
        "target_source": "measured from each reference frame",
    }


def test_protocol_summary_multi_mutation_reports_union_target_only() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            mutations=(
                MutationSpec(residue_number=3, target_residue_name="ALA"),
                MutationSpec(residue_number=11, target_residue_name="SER"),
            ),
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(source="auto"),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
        mobility=MobilityConfig(mode="target_only"),
    )
    summary = build_protocol_summary(config)
    mobility = summary["mobility"]
    assert mobility["mutation_residue_numbers"] == [3, 11]
    assert mobility["selected_residue_scope"]["residue_numbers"] == [3, 11]
