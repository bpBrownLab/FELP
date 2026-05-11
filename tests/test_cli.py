from __future__ import annotations

import json
from pathlib import Path

import pytest

from felp.core import ConfigurationError
from felp.cli import (
    build_felp_plot_config,
    build_msm_fit_config,
    build_msm_plot_config,
    build_parser,
    build_run_config,
)


def test_build_run_config_from_cli(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--phi-bins",
            "12",
            "--psi-bins",
            "18",
            "--prior-probability-mode",
            "raw_counts",
        ]
    )

    config = build_run_config(namespace)
    assert config.input.topology_file == str(topology)
    assert config.input.trajectory_groups == ((str(trajectory),),)
    assert config.perturbation.target_residue_number == 2
    assert config.perturbation.target_residue_name == "ALA"
    assert config.microstates.phi_bins == 12
    assert config.microstates.psi_bins == 18
    assert config.microstates.reference_probability_mode == "raw_counts"
    assert config.energy.backend == "openmm"
    assert [
        (spec.name, spec.kind, spec.residue_number)
        for spec in config.restraints.cv_specs
    ] == [("phi", "phi", 2), ("psi", "psi", 2)]
    assert config.restraints.cv_measurement_tolerance_degrees == 1.0
    assert config.restraints.rosetta_distance_constraint_sd_angstrom == 0.001
    assert config.restraints.rosetta_restraint_force_constant is None
    assert config.restraints.openmm_restraint_force_constant is None
    assert config.restraints.effective_rosetta_force_constant() == 10000.0
    assert config.restraints.effective_openmm_force_constant() == 10000.0
    assert config.rosetta.perturbation_mode == "relax"
    assert config.rosetta.repack_trials == 1
    assert config.rosetta.symmetric_gly_tables is False
    assert config.sampling.microstate_sample_fraction == 1.0
    assert config.sampling.microstate_min_samples == 1
    assert config.sampling.microstate_max_samples is None
    assert config.sampling.microstate_sampling_seed == 0


def test_build_run_config_restraint_overrides_and_cv_tolerance(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--restraint-force-constant",
            "5000",
            "--rosetta-restraint-force-constant",
            "8000",
            "--openmm-restraint-force-constant",
            "12000",
            "--cv-measurement-tolerance-degrees",
            "2.5",
        ]
    )
    config = build_run_config(namespace)
    assert config.restraints.force_constant == 5000.0
    assert config.restraints.rosetta_restraint_force_constant == 8000.0
    assert config.restraints.openmm_restraint_force_constant == 12000.0
    assert config.restraints.cv_measurement_tolerance_degrees == 2.5
    assert config.restraints.effective_rosetta_force_constant() == 8000.0
    assert config.restraints.effective_openmm_force_constant() == 12000.0


def test_build_run_config_openmm_cpu_flag(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--cpu",
        ]
    )
    config = build_run_config(namespace)
    assert config.energy.platform_name == "CPU"
    assert config.energy.cpu_threads == 1


def test_build_run_config_cpu_threads_override(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--cpu",
            "--cpu-threads",
            "4",
        ]
    )
    config = build_run_config(namespace)
    assert config.energy.platform_name == "CPU"
    assert config.energy.cpu_threads == 4


def test_build_run_config_cpu_threads_must_be_positive(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--cpu",
            "--cpu-threads",
            "0",
        ]
    )
    with pytest.raises(ConfigurationError, match="cpu_threads"):
        build_run_config(namespace)


def test_build_run_config_cpu_conflicts_with_platform_name(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--cpu",
            "--platform-name",
            "CUDA",
        ]
    )
    with pytest.raises(ConfigurationError, match="--cpu"):
        build_run_config(namespace)


def test_build_run_config_gpu_requires_openmm_backend(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--energy-backend",
            "rosetta",
            "--gpu",
        ]
    )
    with pytest.raises(ConfigurationError, match="--gpu"):
        build_run_config(namespace)


def test_build_run_config_mute_pyrosetta(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--mute-pyrosetta",
            "--prior-probability-mode",
            "raw_counts",
        ]
    )

    config = build_run_config(namespace)
    assert config.rosetta.mute_tracers is True


def test_build_run_config_rosetta_perturbation_controls(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "GLY",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--rosetta-perturbation-mode",
            "repack",
            "--rosetta-repack-trials",
            "5",
            "--rosetta-symmetric-gly-tables",
        ]
    )

    config = build_run_config(namespace)
    assert config.rosetta.perturbation_mode == "repack"
    assert config.rosetta.repack_trials == 5
    assert config.rosetta.symmetric_gly_tables is True


def test_build_run_config_microstate_sampling_controls(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--microstate-sample-fraction",
            "0.25",
            "--microstate-min-samples",
            "5",
            "--microstate-max-samples",
            "20",
            "--microstate-sampling-seed",
            "123",
        ]
    )

    config = build_run_config(namespace)
    assert config.sampling.microstate_sample_fraction == 0.25
    assert config.sampling.microstate_min_samples == 5
    assert config.sampling.microstate_max_samples == 20
    assert config.sampling.microstate_sampling_seed == 123


def test_build_run_config_rejects_invalid_sampling_controls(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--prior-probability-mode",
            "raw_counts",
            "--microstate-sample-fraction",
            "0",
        ]
    )

    with pytest.raises(ConfigurationError, match="microstate_sample_fraction"):
        build_run_config(namespace)


def test_build_run_config_prior_mode_alias(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory-file",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--ignore-loaded-reference-probabilities",
        ]
    )

    config = build_run_config(namespace)
    assert config.microstates.reference_probability_mode == "raw_counts"


def test_build_run_config_with_msm_bundle(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    bundle_dir = tmp_path / "bundle"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")
    bundle_dir.mkdir()

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "ALA",
            "--output-dir",
            str(tmp_path / "out"),
            "--microstate-source",
            "msm_bundle",
            "--msm-bundle-dir",
            str(bundle_dir),
            "--state-level",
            "metastable",
        ]
    )

    config = build_run_config(namespace)
    assert config.microstates.source == "msm_bundle"
    assert config.microstates.msm_bundle_dir == str(bundle_dir)
    assert config.microstates.state_level == "metastable"
    assert config.microstates.reference_probability_mode == "loaded"


def test_build_run_config_with_custom_backbone_cvs_and_mutation_aliases(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    microstates = tmp_path / "states.json"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")
    microstates.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "helical",
                        "label": "helical",
                        "cv_ranges": {
                            "phi": [-120.0, -30.0],
                            "omega": [150.0, 180.0],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--target-residue-number",
            "2",
            "--target-residue-name",
            "his_d",
            "--microstate-source",
            "file",
            "--microstate-file",
            str(microstates),
            "--cv",
            "phi",
            "--cv",
            "omega",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    config = build_run_config(namespace)
    assert config.perturbation.target_residue_name == "HID"
    assert [
        (spec.name, spec.kind, spec.residue_number)
        for spec in config.restraints.cv_specs
    ] == [("phi", "phi", 2), ("omega", "omega", 2)]
    assert config.microstates.reference_probability_mode == "loaded"


def test_build_run_config_with_distance_cv_and_radius_mobility(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--target-residue-number",
            "61",
            "--target-residue-name",
            "LEU",
            "--prior-probability-mode",
            "raw_counts",
            "--cv",
            "phi",
            "--cv",
            "psi",
            "--distance-cv",
            "q61_sw1:61:CA:35:CA",
            "--mobility-mode",
            "radius",
            "--mobility-radius-angstrom",
            "8.0",
            "--mobility-include-backbone",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    config = build_run_config(namespace)

    assert config.restraints.distance_cv_specs[0].name == "q61_sw1"
    assert config.restraints.distance_cv_specs[0].atom1.residue_number == 61
    assert config.restraints.distance_cv_specs[0].atom2.atom_name == "CA"
    assert config.mobility.mode == "radius"
    assert config.mobility.radius_angstrom == 8.0
    assert config.mobility.include_backbone is True


def test_build_run_config_uses_auto_microstates_for_distance_cv_without_backbone_cvs(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--target-residue-number",
            "4",
            "--target-residue-name",
            "PRO",
            "--prior-probability-mode",
            "raw_counts",
            "--distance-cv",
            "ends:1:CA:10:CA",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    config = build_run_config(namespace)

    assert config.microstates.source == "auto"
    assert config.microstates.reference_probability_mode == "raw_counts"
    assert config.restraints.cv_specs == ()
    assert [spec.name for spec in config.restraints.distance_cv_specs] == ["ends"]


def test_build_run_config_allows_distance_only_custom_microstate_file(
    tmp_path: Path,
) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    microstates = tmp_path / "microstates.json"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")
    microstates.write_text(
        (
            '{"states":[{"state_id":"compact","label":"compact",'
            '"cv_ranges":{"ends":[0.0,12.0],"turn":[0.0,6.0]}}]}'
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--target-residue-number",
            "4",
            "--target-residue-name",
            "PRO",
            "--microstate-source",
            "file",
            "--microstate-file",
            str(microstates),
            "--distance-cv",
            "ends:1:CA:10:CA",
            "--distance-cv",
            "turn:3:CA:6:CA",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    config = build_run_config(namespace)

    assert config.restraints.cv_specs == ()
    assert [spec.name for spec in config.restraints.distance_cv_specs] == [
        "ends",
        "turn",
    ]


def test_build_msm_fit_config_from_cli(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    trajectory = tmp_path / "input.xtc"
    feature_spec = tmp_path / "features.json"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")
    feature_spec.write_text('{"operations": []}\n', encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "msm-fit",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--feature-spec-file",
            str(feature_spec),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--reducer",
            "none",
            "--clustering-method",
            "regspace",
            "--msm-lag",
            "5",
            "--metastable-states",
            "3",
        ]
    )

    config = build_msm_fit_config(namespace)
    assert config.feature_spec_file == str(feature_spec)
    assert config.reducer == "none"
    assert config.clustering_method == "regspace"
    assert config.msm_lag == 5
    assert config.metastable_states == 3


def test_build_msm_fit_config_with_repeatable_trajectory_groups(tmp_path: Path) -> None:
    topology = tmp_path / "input.pdb"
    traj_a = tmp_path / "traj_a.xtc"
    traj_b = tmp_path / "traj_b.xtc"
    traj_c = tmp_path / "traj_c.xtc"
    feature_spec = tmp_path / "features.json"
    topology.write_text("placeholder\n", encoding="utf-8")
    traj_a.write_text("placeholder\n", encoding="utf-8")
    traj_b.write_text("placeholder\n", encoding="utf-8")
    traj_c.write_text("placeholder\n", encoding="utf-8")
    feature_spec.write_text('{"operations": []}\n', encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "msm-fit",
            "--topology-file",
            str(topology),
            "--trajectory",
            str(traj_a),
            str(traj_b),
            "--trajectory",
            str(traj_c),
            "--feature-spec-file",
            str(feature_spec),
            "--output-dir",
            str(tmp_path / "bundle"),
        ]
    )

    config = build_msm_fit_config(namespace)
    assert config.trajectory_groups == (
        (str(traj_a), str(traj_b)),
        (str(traj_c),),
    )


def test_build_plot_configs_from_cli(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    summary_csv = tmp_path / "microstate_summary.csv"
    frame_csv = tmp_path / "frame_results.csv"
    bundle_dir.mkdir()
    summary_csv.write_text("label,reference_probability,target_probability\n", encoding="utf-8")
    frame_csv.write_text("delta_u_kj_mol\n", encoding="utf-8")

    parser = build_parser()
    msm_namespace = parser.parse_args(
        [
            "msm-plot",
            "--msm-bundle-dir",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "msm_plots"),
            "--state-level",
            "metastable",
        ]
    )
    felp_namespace = parser.parse_args(
        [
            "felp-plot",
            "--microstate-summary",
            str(summary_csv),
            "--raw-frame-results",
            str(frame_csv),
            "--output-dir",
            str(tmp_path / "felp_plots"),
            "--fel-bins",
            "36",
            "--fel-max-energy-kj-mol",
            "12",
            "--plot-energy-unit",
            "kcal_mol",
            "--plot-cv-x",
            "ends",
            "--plot-cv-y",
            "turn",
        ]
    )

    msm_config = build_msm_plot_config(msm_namespace)
    felp_config = build_felp_plot_config(felp_namespace)
    assert msm_config.state_level == "metastable"
    assert msm_config.output_dir.endswith("msm_plots")
    assert felp_config.raw_frame_results_csv == str(frame_csv)
    assert felp_config.output_dir.endswith("felp_plots")
    assert felp_config.fel_bins == 36
    assert felp_config.fel_max_energy_kj_mol == 12.0
    assert felp_config.plot_energy_unit == "kcal_mol"
    assert felp_config.cv_x == "ends"
    assert felp_config.cv_y == "turn"


def test_build_felp_plot_config_normalizes_cv_spec_strings(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"
    summary_csv.write_text("label,reference_probability,target_probability\n", encoding="utf-8")

    parser = build_parser()
    namespace = parser.parse_args(
        [
            "felp-plot",
            "--microstate-summary",
            str(summary_csv),
            "--output-dir",
            str(tmp_path / "felp_plots"),
            "--plot-cv-x",
            "ends:1:CA:10:CA",
            "--plot-cv-y",
            "turn:3:CA:6:CA",
        ]
    )
    config = build_felp_plot_config(namespace)
    assert config.cv_x == "ends"
    assert config.cv_y == "turn"
