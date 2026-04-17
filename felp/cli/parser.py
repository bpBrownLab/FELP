"""Central argument parsing for FELP."""

from __future__ import annotations

import argparse
from typing import Optional

from felp.core import (
    CVSpec,
    EnergyConfig,
    InputConfig,
    MicrostateConfig,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
    ConfigurationError,
    default_backbone_cv_specs,
)
from felp.msm import MSMFitConfig, MSMSampleExportConfig
from felp.plotting import FELPPlotConfig, MSMPlotConfig


def _default_cv_name(kind: str, residue_number: int, target_residue_number: int) -> str:
    """Return the legacy-friendly output name for a parsed CV spec."""

    return kind if residue_number == target_residue_number else f"{kind}_{residue_number}"


def _parse_cv_specs(
    raw_tokens: Optional[list[str]],
    target_residue_number: int,
) -> tuple[CVSpec, ...]:
    """Parse repeatable ``--cv`` tokens into typed backbone CV specs."""

    if not raw_tokens:
        return default_backbone_cv_specs(target_residue_number)

    cv_specs: list[CVSpec] = []
    seen_names: set[str] = set()
    for token in raw_tokens:
        kind_token, separator, residue_token = token.partition(":")
        if not separator:
            residue_number = target_residue_number
        else:
            residue_text = residue_token.strip()
            try:
                residue_number = int(residue_text)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Invalid --cv value {token!r}. Expected kind or kind:<residue>."
                ) from exc
        spec = CVSpec(
            kind=kind_token.strip().lower(),
            residue_number=residue_number,
            name=_default_cv_name(
                kind_token.strip().lower(),
                residue_number,
                target_residue_number,
            ),
        )
        if spec.name in seen_names:
            raise ConfigurationError(
                f"Duplicate backbone CV specification {spec.name!r}. "
                "Each configured CV name must be unique."
            )
        seen_names.add(spec.name)
        cv_specs.append(spec)
    return tuple(cv_specs)


def _add_common_microstate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by FELP commands that consume state definitions."""

    group = parser.add_argument_group("Microstates")
    group.add_argument(
        "--microstate-source",
        choices=("grid", "file", "msm_bundle"),
        default="grid",
        help=(
            "How to define microstates for the reference landscape. "
            "Use grid for first passes; msm_bundle requires a pre-fit bundle aligned "
            "to your --trajectory grouping."
        ),
    )
    group.add_argument(
        "--microstate-file",
        help="JSON or YAML file containing explicit microstate definitions.",
    )
    group.add_argument(
        "--phi-bins",
        type=int,
        default=36,
        help="Number of phi bins when using grid microstates.",
    )
    group.add_argument(
        "--psi-bins",
        type=int,
        default=36,
        help="Number of psi bins when using grid microstates.",
    )
    group.add_argument(
        "--msm-bundle-dir",
        help="Directory containing a previously exported MSM bundle.",
    )
    group.add_argument(
        "--state-level",
        choices=("cluster", "metastable"),
        default="cluster",
        help="Which state level to use when loading an MSM bundle.",
    )
    group.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="Allow frames outside the supplied microstate definitions.",
    )
    group.add_argument(
        "--ignore-loaded-reference-probabilities",
        action="store_true",
        help="Ignore reference probabilities embedded in a microstate file.",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the main FELP run command."""

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument(
        "--topology-file",
        required=True,
        help="Reference topology file understood by ParmEd.",
    )
    input_group.add_argument(
        "--trajectory",
        "--trajectory-file",
        action="append",
        nargs="+",
        required=True,
        dest="trajectory_groups",
        metavar="<trajectory>",
        help=(
            "Repeatable grouped trajectory input. Files within one flag are "
            "treated as contiguous chunks of a single trajectory, while "
            "repeated flags define independent trajectories."
        ),
    )
    input_group.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride used when loading the input trajectories.",
    )
    input_group.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Zero-based frame index to start from after stride is applied.",
    )
    input_group.add_argument(
        "--frame-stop",
        type=int,
        help="Exclusive frame index to stop at after stride is applied.",
    )
    input_group.add_argument(
        "--radii",
        default="mbondi3",
        dest="radii_set",
        help="ParmEd radii set to apply when loading the reference topology.",
    )

    perturb_group = parser.add_argument_group("Perturbation")
    perturb_group.add_argument(
        "--target-residue-number",
        required=True,
        type=int,
        help=(
            "Rosetta pose residue index (1..N) to mutate, as in pose.residue(i). "
            "This is sequential pose numbering, not necessarily PDB residue numbers."
        ),
    )
    perturb_group.add_argument(
        "--target-residue-name",
        required=True,
        help=(
            "Three-letter mutation target. Supports canonical amino acids plus "
            "common protonation/disulfide variants such as HID/HIE/HIP/CYM/CYX."
        ),
    )
    perturb_group.add_argument(
        "--original-residue-name",
        help=(
            "Optional three-letter residue expected at --target-residue-number before "
            "mutation (e.g. GLY for GLY→ALA). When set, the first frame checks the "
            "pose matches this name at the Rosetta index."
        ),
    )

    rosetta_group = parser.add_argument_group("PyRosetta")
    rosetta_group.add_argument(
        "--pyrosetta-init-options",
        default="-ignore_unrecognized_res true -load_PDB_components false",
        help="Options string passed to pyrosetta.init.",
    )
    rosetta_group.add_argument(
        "--rosetta-database",
        help="Optional explicit PyRosetta database path.",
    )
    rosetta_group.add_argument(
        "--rosetta-score-function",
        default="ref2015",
        help="Rosetta score function for minimization and optional scoring.",
    )
    rosetta_group.add_argument(
        "--rosetta-relax-iterations",
        default=5,
        type=int,
        help="Number of FastRelax iterations to perform.",
    )
    rosetta_group.add_argument(
        "--no-cartesian-relax",
        action="store_true",
        help="Disable Cartesian FastRelax.",
    )

    restraint_group = parser.add_argument_group("Restraints")
    restraint_group.add_argument(
        "--restraint-mode",
        choices=("phi_psi", "backbone_dihedrals"),
        default="phi_psi",
        help=(
            "How to preserve the common reference frame during minimization. "
            "Backbone-dihedral runs are configured explicitly with --cv."
        ),
    )
    restraint_group.add_argument(
        "--cv",
        action="append",
        dest="cv_tokens",
        metavar="<kind[:residue]>",
        help=(
            "Repeatable fixed backbone CV specification. Use phi, psi, or omega; "
            "append :<residue> to target a different pose residue than "
            "--target-residue-number. Defaults to phi and psi on the target residue."
        ),
    )
    restraint_group.add_argument(
        "--restraint-force-constant",
        type=float,
        default=10000.0,
        help="Stiff force constant applied to phi/psi state-locking restraints.",
    )

    energy_group = parser.add_argument_group("Energy")
    energy_group.add_argument(
        "--energy-backend",
        choices=("openmm", "rosetta"),
        default="openmm",
        help="Energy backend used for the final rescoring stage.",
    )
    energy_group.add_argument(
        "--protein-forcefield",
        default="amber/protein.ff14SBonlysc.xml",
        help="OpenMM protein force-field XML file.",
    )
    energy_group.add_argument(
        "--solvent-forcefield",
        default="implicit/gbn2.xml",
        help="OpenMM solvent/implicit-solvent XML file.",
    )
    energy_group.add_argument(
        "--temperature-kelvin",
        type=float,
        default=300.0,
        help="Temperature used for reweighting and OpenMM integrator setup.",
    )
    energy_group.add_argument(
        "--friction-per-ps",
        type=float,
        default=1.0,
        help="Langevin friction coefficient for OpenMM.",
    )
    energy_group.add_argument(
        "--timestep-fs",
        type=float,
        default=2.0,
        help="Langevin integrator timestep for OpenMM.",
    )
    energy_group.add_argument(
        "--nonbonded-cutoff-nm",
        type=float,
        default=1.0,
        help="OpenMM nonbonded cutoff in nanometers.",
    )
    energy_group.add_argument(
        "--minimization-tolerance-kj-per-mol-nm",
        type=float,
        default=1.0,
        help="OpenMM minimization tolerance.",
    )
    energy_group.add_argument(
        "--minimization-max-iterations",
        type=int,
        default=0,
        help="Maximum number of OpenMM minimization iterations. Zero means default behavior.",
    )
    energy_group.add_argument(
        "--platform-name",
        help="Optional OpenMM platform name, such as CPU or CUDA.",
    )

    _add_common_microstate_arguments(parser)

    parallel_group = parser.add_argument_group("Parallel")
    parallel_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parallel_group.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of frames submitted per worker task batch.",
    )
    parallel_group.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method.",
    )
    parallel_group.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Terminal progress display for frame processing. "
            "'auto' enables a bar for multi-worker runs (and long serial runs when tqdm is installed)."
        ),
    )
    parallel_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the run on the first per-frame processing failure.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        required=True,
        help="Directory where FELP outputs will be written.",
    )
    output_group.add_argument(
        "--write-pose-pdbs",
        action="store_true",
        help="Write relaxed reference and perturbed PDB files per frame.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    output_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing frame results file when possible.",
    )


def _add_msm_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for fitting a PyEMMA model and exporting a bundle."""

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument(
        "--topology-file",
        required=True,
        help="Topology file understood by MDTraj/PyEMMA.",
    )
    input_group.add_argument(
        "--trajectory",
        "--trajectory-file",
        action="append",
        nargs="+",
        required=True,
        dest="trajectory_groups",
        metavar="<trajectory>",
        help=(
            "Repeatable grouped trajectory input. Files within one flag are "
            "treated as contiguous chunks of a single trajectory, while "
            "repeated flags define independent trajectories for MSM fitting."
        ),
    )
    input_group.add_argument(
        "--feature-spec-file",
        required=True,
        help="JSON or YAML feature specification for the PyEMMA featurizer.",
    )
    input_group.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the MSM bundle will be written.",
    )
    input_group.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride used during featurization, clustering, and bundle export.",
    )
    input_group.add_argument(
        "--dt-traj",
        default="1 step",
        help="Frame time spacing passed to PyEMMA.",
    )

    reduction_group = parser.add_argument_group("Dimensionality Reduction")
    reduction_group.add_argument(
        "--reducer",
        choices=("none", "tica", "pca"),
        default="tica",
        help="Dimensionality reduction stage applied before clustering.",
    )
    reduction_group.add_argument(
        "--reducer-lag",
        type=int,
        default=10,
        help="Lag time used for TICA.",
    )
    reduction_group.add_argument(
        "--variance-cutoff",
        type=float,
        default=0.95,
        help="Variance/kinetic variance cutoff used by PCA or TICA.",
    )

    clustering_group = parser.add_argument_group("Clustering")
    clustering_group.add_argument(
        "--clustering-method",
        choices=("kmeans", "regspace"),
        default="kmeans",
        help="Clustering method used to discretize the reduced trajectories.",
    )
    clustering_group.add_argument(
        "--k-clusters",
        type=int,
        default=100,
        help="Number of k-means clusters.",
    )
    clustering_group.add_argument(
        "--k-iter",
        type=int,
        default=10,
        help="Maximum number of k-means iterations.",
    )
    clustering_group.add_argument(
        "--regspace-dmin",
        type=float,
        default=0.5,
        help="Minimum cluster spacing for regspace clustering.",
    )

    model_group = parser.add_argument_group("MSM Model")
    model_group.add_argument(
        "--msm-lag",
        type=int,
        default=1,
        help="Lag time used to estimate the discrete MSM.",
    )
    model_group.add_argument(
        "--metastable-states",
        type=int,
        help="Optional number of PCCA metastable states to export alongside cluster states.",
    )
    model_group.add_argument(
        "--hmm-states",
        type=int,
        help="Optional number of hidden metastable states to estimate with a hidden MSM.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-pyemma-checkpoints",
        action="store_true",
        help="Save optional PyEMMA-native checkpoints next to the exported bundle.",
    )
    output_group.add_argument(
        "--write-plots",
        action="store_true",
        help="Write a compact set of diagnostic MSM plots into the bundle directory.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing MSM bundle directory.",
    )


def _add_msm_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by MSM bundle inspection commands."""

    parser.add_argument(
        "--msm-bundle-dir",
        required=True,
        help="Directory containing a previously exported MSM bundle.",
    )


def _add_msm_plot_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for publication-style MSM bundle plots."""

    _add_msm_bundle_arguments(parser)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plot files should be written.",
    )
    parser.add_argument(
        "--state-level",
        choices=("cluster", "metastable"),
        default="cluster",
        help="Which state level to visualize from the bundle.",
    )
    parser.add_argument(
        "--temperature-kelvin",
        type=float,
        default=300.0,
        help="Temperature used when converting probabilities to free energies.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins used for 2D projection histograms.",
    )
    parser.add_argument(
        "--max-state-bars",
        type=int,
        default=40,
        help="Maximum number of states to display in probability bar plots.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of points to scatter in state projection plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing plot output directory.",
    )


def _add_felp_plot_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for publication-style FELP result plots."""

    parser.add_argument(
        "--microstate-summary",
        required=True,
        help="Microstate summary CSV produced by a FELP run or summarize command.",
    )
    parser.add_argument(
        "--raw-frame-results",
        help="Optional frame-level FELP CSV used for deltaU histograms.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plot files should be written.",
    )
    parser.add_argument(
        "--temperature-kelvin",
        type=float,
        default=300.0,
        help="Temperature used when converting probabilities to free energies.",
    )
    parser.add_argument(
        "--top-states",
        type=int,
        default=40,
        help="Maximum number of states to highlight in bar plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing plot output directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the central argument parser for the FELP CLI."""
    parser = argparse.ArgumentParser(description="FELP: Free Energy Landscape Perturbation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the FELP workflow on one or more reference trajectories.",
    )
    _add_run_arguments(run_parser)

    validate_parser = subparsers.add_parser(
        "validate-microstates",
        help="Load microstates and print a concise validation summary.",
    )
    _add_common_microstate_arguments(validate_parser)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Recompute microstate summaries from an existing frame results CSV.",
    )
    summarize_parser.add_argument(
        "--raw-frame-results",
        required=True,
        help="Frame-level CSV produced by a previous FELP run.",
    )
    summarize_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary CSV files should be written.",
    )
    summarize_parser.add_argument(
        "--temperature-kelvin",
        type=float,
        default=300.0,
        help="Temperature used when converting probabilities to free energies.",
    )
    _add_common_microstate_arguments(summarize_parser)

    msm_fit_parser = subparsers.add_parser(
        "msm-fit",
        help="Fit a PyEMMA model and export an FELP-readable MSM bundle.",
    )
    _add_msm_fit_arguments(msm_fit_parser)

    msm_inspect_parser = subparsers.add_parser(
        "msm-inspect",
        help="Inspect a previously exported MSM bundle.",
    )
    _add_msm_bundle_arguments(msm_inspect_parser)

    sample_parser = subparsers.add_parser(
        "msm-export-samples",
        help="Export representative structures for bundle states.",
    )
    _add_msm_bundle_arguments(sample_parser)
    sample_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where sampled state trajectories should be written.",
    )
    sample_parser.add_argument(
        "--state-level",
        choices=("cluster", "metastable"),
        default="cluster",
        help="Which state level to sample from the bundle.",
    )
    sample_parser.add_argument(
        "--state-id",
        dest="state_ids",
        nargs="+",
        default=(),
        help="Optional explicit subset of state ids to export.",
    )
    sample_parser.add_argument(
        "--samples-per-state",
        type=int,
        default=10,
        help="Maximum number of frames to export per state.",
    )
    sample_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing sample output directory.",
    )

    msm_plot_parser = subparsers.add_parser(
        "msm-plot",
        help="Generate publication-style plots from an MSM bundle.",
    )
    _add_msm_plot_arguments(msm_plot_parser)

    felp_plot_parser = subparsers.add_parser(
        "felp-plot",
        help="Generate publication-style plots from FELP outputs.",
    )
    _add_felp_plot_arguments(felp_plot_parser)

    return parser


def build_run_config(namespace: argparse.Namespace) -> RunConfig:
    """Construct and validate a RunConfig from the parsed CLI namespace."""
    config = RunConfig(
        input=InputConfig(
            topology_file=namespace.topology_file,
            trajectory_files=tuple(
                tuple(group) for group in namespace.trajectory_groups
            ),
            stride=namespace.stride,
            frame_start=namespace.frame_start,
            frame_stop=namespace.frame_stop,
            radii_set=namespace.radii_set,
        ),
        perturbation=PerturbationConfig(
            target_residue_number=namespace.target_residue_number,
            target_residue_name=namespace.target_residue_name,
            original_residue_name=namespace.original_residue_name,
        ),
        rosetta=RosettaConfig(
            init_options=namespace.pyrosetta_init_options,
            database_path=namespace.rosetta_database,
            score_function=namespace.rosetta_score_function,
            relax_iterations=namespace.rosetta_relax_iterations,
            cartesian=not namespace.no_cartesian_relax,
        ),
        restraints=RestraintConfig(
            mode=namespace.restraint_mode,
            force_constant=namespace.restraint_force_constant,
            cv_specs=_parse_cv_specs(
                namespace.cv_tokens,
                namespace.target_residue_number,
            ),
        ),
        energy=EnergyConfig(
            backend=namespace.energy_backend,
            protein_forcefield=namespace.protein_forcefield,
            solvent_forcefield=namespace.solvent_forcefield,
            temperature_kelvin=namespace.temperature_kelvin,
            friction_per_ps=namespace.friction_per_ps,
            timestep_fs=namespace.timestep_fs,
            nonbonded_cutoff_nm=namespace.nonbonded_cutoff_nm,
            minimization_tolerance_kj_per_mol_nm=namespace.minimization_tolerance_kj_per_mol_nm,
            minimization_max_iterations=namespace.minimization_max_iterations,
            platform_name=namespace.platform_name,
            rosetta_score_function=namespace.rosetta_score_function,
        ),
        microstates=MicrostateConfig(
            source=namespace.microstate_source,
            target_residue_number=namespace.target_residue_number,
            definition_file=namespace.microstate_file,
            phi_bins=namespace.phi_bins,
            psi_bins=namespace.psi_bins,
            msm_bundle_dir=namespace.msm_bundle_dir,
            state_level=namespace.state_level,
            allow_unassigned=namespace.allow_unassigned,
            use_loaded_reference_probabilities=not namespace.ignore_loaded_reference_probabilities,
        ),
        parallel=ParallelConfig(
            workers=namespace.workers,
            chunk_size=namespace.chunk_size,
            start_method=namespace.start_method,
            progress_mode=namespace.progress,
            fail_fast=namespace.fail_fast,
        ),
        output=OutputConfig(
            output_dir=namespace.output_dir,
            write_pose_pdbs=namespace.write_pose_pdbs,
            overwrite=namespace.overwrite,
            resume=namespace.resume,
        ),
    )
    config.validate()
    return config


def build_microstate_config(
    namespace: argparse.Namespace, target_residue_number: int = 2
) -> MicrostateConfig:
    """Construct a MicrostateConfig from the parsed CLI namespace."""
    return MicrostateConfig(
        source=namespace.microstate_source,
        target_residue_number=target_residue_number,
        definition_file=namespace.microstate_file,
        phi_bins=namespace.phi_bins,
        psi_bins=namespace.psi_bins,
        msm_bundle_dir=namespace.msm_bundle_dir,
        state_level=namespace.state_level,
        allow_unassigned=namespace.allow_unassigned,
        use_loaded_reference_probabilities=not namespace.ignore_loaded_reference_probabilities,
    )


def build_msm_fit_config(namespace: argparse.Namespace) -> MSMFitConfig:
    """Construct and validate an MSMFitConfig from the parsed CLI namespace."""

    config = MSMFitConfig(
        topology_file=namespace.topology_file,
        trajectory_files=tuple(
            tuple(group) for group in namespace.trajectory_groups
        ),
        feature_spec_file=namespace.feature_spec_file,
        output_dir=namespace.output_dir,
        stride=namespace.stride,
        dt_traj=namespace.dt_traj,
        reducer=namespace.reducer,
        reducer_lag=namespace.reducer_lag,
        variance_cutoff=namespace.variance_cutoff,
        clustering_method=namespace.clustering_method,
        k_clusters=namespace.k_clusters,
        k_iter=namespace.k_iter,
        regspace_dmin=namespace.regspace_dmin,
        msm_lag=namespace.msm_lag,
        metastable_states=namespace.metastable_states,
        hmm_states=namespace.hmm_states,
        save_pyemma_checkpoints=namespace.save_pyemma_checkpoints,
        write_plots=namespace.write_plots,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


def build_msm_sample_export_config(
    namespace: argparse.Namespace,
) -> MSMSampleExportConfig:
    """Construct and validate a sample export config from the CLI namespace."""

    config = MSMSampleExportConfig(
        bundle_dir=namespace.msm_bundle_dir,
        output_dir=namespace.output_dir,
        state_level=namespace.state_level,
        state_ids=tuple(namespace.state_ids),
        samples_per_state=namespace.samples_per_state,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


def build_msm_plot_config(namespace: argparse.Namespace) -> MSMPlotConfig:
    """Construct and validate an MSMPlotConfig from the CLI namespace."""

    config = MSMPlotConfig(
        bundle_dir=namespace.msm_bundle_dir,
        output_dir=namespace.output_dir,
        state_level=namespace.state_level,
        temperature_kelvin=namespace.temperature_kelvin,
        bins=namespace.bins,
        max_state_bars=namespace.max_state_bars,
        max_points=namespace.max_points,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


def build_felp_plot_config(namespace: argparse.Namespace) -> FELPPlotConfig:
    """Construct and validate a FELPPlotConfig from the CLI namespace."""

    config = FELPPlotConfig(
        microstate_summary_csv=namespace.microstate_summary,
        raw_frame_results_csv=namespace.raw_frame_results,
        output_dir=namespace.output_dir,
        temperature_kelvin=namespace.temperature_kelvin,
        top_states=namespace.top_states,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


class MasterArgumentParser:
    """Compatibility wrapper for the pre-rewrite parser entry point."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Compatibility parser for legacy FELP scripts."
        )
        _add_run_arguments(self.parser)

    def parse_args(self, args: Optional[list[str]] = None) -> argparse.Namespace:
        return self.parser.parse_args(args)
