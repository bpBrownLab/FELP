"""Central argument parsing for FELP."""

from __future__ import annotations

import argparse
from typing import Optional

from felp.core import (
    AtomReference,
    CVSpec,
    DistanceCVSpec,
    EnergyConfig,
    InputConfig,
    MicrostateConfig,
    MutationSpec,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    MobilityConfig,
    FrameReweightingConfig,
    LaplacianConfig,
    NeighborPoolConfig,
    ReweightingConfig,
    ReweightingQCConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
    SamplingConfig,
    ConfigurationError,
    default_backbone_cv_specs,
)
from felp.energy import resolve_openmm_gpu_platform_name
from felp.msm import MSMFitConfig, MSMSampleExportConfig
from felp.plotting import (
    ESSCompareConfig,
    FELPPlotConfig,
    MDKLPlotConfig,
    MSMPlotConfig,
)
from felp.plotting.composite import load_composite_figure_config
from felp.plotting.common import DEFAULT_FEL_COLORMAP, FELP_KL_DIVERGENCE_COLORMAP

_KJ_PER_KCAL = 4.184


def _resolve_fel_max_energy_kj_mol(
    fel_max_energy_kj_mol: float | None,
    fel_max_energy_kcal_mol: float | None,
) -> float | None:
    """Return the free-energy display cap in kJ/mol from CLI energy-cap flags."""

    if fel_max_energy_kj_mol is not None and fel_max_energy_kcal_mol is not None:
        raise ConfigurationError(
            "Specify only one of --fel-max-energy-kj-mol and --fel-max-energy-kcal-mol."
        )
    if fel_max_energy_kcal_mol is not None:
        if fel_max_energy_kcal_mol <= 0.0:
            raise ConfigurationError(
                "fel_max_energy_kcal_mol must be positive when set."
            )
        return float(fel_max_energy_kcal_mol) * _KJ_PER_KCAL
    return fel_max_energy_kj_mol


def _default_cv_name(kind: str, residue_number: int, target_residue_number: int) -> str:
    """Return the legacy-friendly output name for a parsed CV spec."""

    return kind if residue_number == target_residue_number else f"{kind}_{residue_number}"


def _parse_cv_specs(
    raw_tokens: Optional[list[str]],
    target_residue_number: int,
    *,
    include_legacy_default: bool = True,
) -> tuple[CVSpec, ...]:
    """Parse repeatable ``--cv`` tokens into typed backbone CV specs."""

    if not raw_tokens:
        if not include_legacy_default:
            return ()
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


def _parse_distance_cv_specs(
    raw_tokens: Optional[list[str]],
) -> tuple[DistanceCVSpec, ...]:
    """Parse repeatable ``--distance-cv`` tokens into typed distance CV specs."""

    if not raw_tokens:
        return ()
    specs: list[DistanceCVSpec] = []
    seen_names: set[str] = set()
    for token in raw_tokens:
        parts = [part.strip() for part in token.split(":")]
        if len(parts) != 5 or not all(parts):
            raise ConfigurationError(
                f"Invalid --distance-cv value {token!r}. Expected "
                "name:res1:atom1:res2:atom2."
            )
        name, residue1_text, atom1, residue2_text, atom2 = parts
        try:
            residue1 = int(residue1_text)
            residue2 = int(residue2_text)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid --distance-cv value {token!r}. Residues must be integers."
            ) from exc
        spec = DistanceCVSpec(
            name=name,
            atom1=AtomReference(residue_number=residue1, atom_name=atom1),
            atom2=AtomReference(residue_number=residue2, atom_name=atom2),
        )
        if spec.name in seen_names:
            raise ConfigurationError(
                f"Duplicate distance CV specification {spec.name!r}. "
                "Each configured CV name must be unique."
            )
        seen_names.add(spec.name)
        specs.append(spec)
    return tuple(specs)


def _parse_mutation_cli_token(token: str) -> MutationSpec:
    """Parse ``--mutation`` values as ``POS:ORIG:TO``."""

    raw = str(token).strip()
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ConfigurationError(
            f"Invalid --mutation {token!r}. Expected residue_index:original:target "
            "(leave original empty to skip chemistry checks before mutation, "
            "e.g. ``12::ALA``)."
        )
    residue_text, original_text, target_text = parts
    try:
        residue_number = int(residue_text)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid residue index in --mutation {token!r}."
        ) from exc
    if not target_text:
        raise ConfigurationError(f"Mutation target residue name may not be empty in {token!r}.")
    original = original_text if original_text else None
    return MutationSpec(
        residue_number=residue_number,
        target_residue_name=target_text,
        original_residue_name=original,
    )


def _add_common_microstate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by FELP commands that consume state definitions."""

    group = parser.add_argument_group("Microstates")
    group.add_argument(
        "--microstate-source",
        choices=("grid", "file", "msm_bundle", "auto"),
        default="grid",
        help=(
            "How to define microstates for the reference landscape. "
            "Use grid for classic phi/psi first passes, file for reusable custom "
            "CV states, auto to infer a reusable 1–3 CV JSON from the reference "
            "trajectory, or msm_bundle for a pre-fit bundle aligned to your "
            "--trajectory grouping."
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
        "--auto-microstate-target-frames-per-state",
        type=int,
        default=100,
        help=(
            "Target support used to cap automatically generated microstates. "
            "The auto grid will not exceed roughly frame_count / this value states "
            "before also applying --auto-microstate-max-states."
        ),
    )
    group.add_argument(
        "--auto-microstate-min-bins-per-axis",
        type=int,
        default=3,
        help="Minimum nonconstant-axis bin count for auto microstates.",
    )
    group.add_argument(
        "--auto-microstate-max-bins-per-axis",
        type=int,
        default=64,
        help="Maximum per-axis bin count for auto microstates.",
    )
    group.add_argument(
        "--auto-microstate-max-states",
        type=int,
        default=2500,
        help=(
            "Maximum total states for auto microstates. Raise this for very "
            "large trajectories when the default grid is still too coarse."
        ),
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
        "--prior-probability-mode",
        choices=("loaded", "raw_counts"),
        default="loaded",
        help=(
            "How FELP chooses the reference microstate prior. "
            "'loaded' uses probabilities embedded in the microstate file or MSM "
            "bundle; 'raw_counts' uses empirical frame counts from the evaluated "
            "reference frames."
        ),
    )
    group.add_argument(
        "--ignore-loaded-reference-probabilities",
        action="store_true",
        help=(
            "Deprecated alias for '--prior-probability-mode raw_counts'. "
            "When set, raw frame counts are used as the reference prior."
        ),
    )


def _add_reweighting_qc_arguments(parser: argparse.ArgumentParser) -> None:
    """Add by-state reweighting QC options."""

    group = parser.add_argument_group("Reweighting QC")
    group.add_argument(
        "--reweighting-qc-policy",
        choices=("none", "flag", "exclude-frames"),
        default="flag",
        help=(
            "By-state Zwanzig degeneracy policy. 'flag' records unreliable state "
            "estimates without changing probabilities; 'exclude-frames' removes "
            "dominant in-state frame outliers before reweighting."
        ),
    )
    group.add_argument(
        "--reweighting-qc-method",
        choices=("auto", "threshold"),
        default="auto",
        help=(
            "By-state degeneracy detector. 'auto' uses a PSIS-style Pareto-tail "
            "diagnostic with sample-size dependent reliability thresholds; "
            "'threshold' uses the explicit ESS/weight/leave-one-out cutoffs below."
        ),
    )
    group.add_argument(
        "--reweighting-weight-treatment",
        choices=("raw", "psis", "psis-smoothed"),
        default="raw",
        help=(
            "Treatment applied to in-state Boltzmann weights before computing "
            "Zwanzig averages. 'psis-smoothed' smooths and truncates the upper "
            "importance-weight tail to reduce variance in heavy-tailed states "
            "('psis' is accepted as a short alias)."
        ),
    )
    group.add_argument(
        "--reweighting-qc-min-frame-count",
        type=int,
        default=20,
        help=(
            "Minimum evaluated frames required before threshold-based by-state QC "
            "can flag a state."
        ),
    )
    group.add_argument(
        "--reweighting-qc-max-frame-weight-fraction",
        type=float,
        default=0.5,
        help=(
            "Flag states where one frame contributes more than this fraction of "
            "the state's in-state Boltzmann weight."
        ),
    )
    group.add_argument(
        "--reweighting-qc-min-state-ess",
        type=float,
        default=5.0,
        help="Flag states whose in-state importance-sampling ESS is below this value.",
    )
    group.add_argument(
        "--reweighting-qc-max-leave-one-out-shift-kj-mol",
        type=float,
        default=10.0,
        help=(
            "Flag states where removing the dominant frame shifts the state's "
            "Zwanzig free-energy estimate by more than this many kJ/mol."
        ),
    )
    group.add_argument(
        "--reweighting-qc-max-excluded-frames-per-state",
        type=int,
        default=5,
        help=(
            "Maximum dominant frames to remove per state when "
            "--reweighting-qc-policy=exclude-frames."
        ),
    )


def _add_reweighting_method_arguments(parser: argparse.ArgumentParser) -> None:
    """Add modular reweighting backend selection and tuning options."""

    group = parser.add_argument_group("Reweighting methods")
    group.add_argument(
        "--reweighting-method",
        choices=("independent", "neighbor_pool", "laplacian", "frame"),
        default="independent",
        help="Reweighting backend used for microstate target populations.",
    )
    group.add_argument(
        "--reweighting-methods",
        nargs="+",
        choices=("independent", "neighbor_pool", "laplacian", "frame"),
        help=(
            "Optional list of backends to run in one invocation. Writes each method "
            "under output_dir/reweighting_<method>/ plus reweighting_comparison.json."
        ),
    )
    group.add_argument(
        "--neighbor-pool-tau",
        type=float,
        default=None,
        help=(
            "Neighbor-pool blending strength tau for ESS-weighted partial pooling. "
            "When omitted, tau is chosen using --neighbor-pool-tau-mode."
        ),
    )
    group.add_argument(
        "--neighbor-pool-solver",
        choices=("blend", "variational"),
        default="blend",
        help=(
            "Neighbor-pool backend. 'blend' (default) ESS-weighted partial pooling "
            "toward ESS-weighted neighbor-local factors (recommended). "
            "'variational' solves a coupled grid GMRF on log-factors; tau semantics "
            "differ from blend and auto-tau often collapses to independent on sparse data."
        ),
    )
    group.add_argument(
        "--neighbor-pool-tau-mode",
        choices=("lcurve", "gcv", "variational", "mean_ess", "median_ess"),
        default="lcurve",
        help=(
            "Automatic tau selection when --neighbor-pool-tau is omitted. "
            "With solver=blend (default): 'lcurve' (recommended) picks the corner "
            "of the log(misfit) vs log(roughness) tradeoff curve. 'gcv' uses the "
            "legacy weighted sum with median frame count. With solver=variational: "
            "use 'variational' (or 'gcv' as alias) for profile energy minimization. "
            "'mean_ess' and 'median_ess' set tau from ESS summaries only."
        ),
    )
    group.add_argument(
        "--neighbor-pool-min-local-weight",
        type=float,
        default=0.0,
        help="Minimum local weight retained in neighbor-pool blending.",
    )
    group.add_argument(
        "--neighbor-pool-max-neighbors",
        type=int,
        default=4,
        help=(
            "Maximum neighbors used when kNN topology fallback is required "
            "(non-grid atlases such as MSM bundles)."
        ),
    )
    group.add_argument(
        "--neighbor-pool-grid-connectivity",
        default="chebyshev3",
        help=(
            "Neighbor stencil for regular CV-grid microstates (1D/2D/3D). 'cardinal' "
            "uses face-sharing bins. 'chebyshev' uses a Chebyshev (L_inf) ball in "
            "bin-index space with radius set by --neighbor-pool-chebyshev-radius. "
            "Aliases 'chebyshev1', 'chebyshev2', 'chebyshev3', etc. are accepted. "
            "Ignored for kNN fallback topologies."
        ),
    )
    group.add_argument(
        "--neighbor-pool-chebyshev-radius",
        type=int,
        default=3,
        help=(
            "Chebyshev radius for --neighbor-pool-grid-connectivity=chebyshev. "
            "Radius 1 includes cardinal and diagonal neighbors (8 in 2D); radius 2 "
            "adds the next shell (24 in 2D); radius 3 adds the third shell (48 in 2D). "
            "Ignored for cardinal connectivity and for kNN fallback."
        ),
    )
    group.add_argument(
        "--laplacian-lambda",
        type=float,
        default=1.0,
        dest="laplacian_lambda_kj_mol2",
        help=(
            "Laplacian neighbor smoothness strength on grid DeltaG. With the default "
            "--laplacian-lambda-scale-by-kbt, this is dimensionless and the optimizer "
            "uses lambda * (k_B T)^2 * sum_edges (DeltaG_i - DeltaG_j)^2 in (kJ/mol)^2."
        ),
    )
    group.add_argument(
        "--laplacian-lambda-scale-by-kbt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scale --laplacian-lambda by (k_B T)^2 so the coupling is energy-based "
            "(default: enabled). Use --no-laplacian-lambda-scale-by-kbt to supply "
            "lambda directly in (kJ/mol)^2."
        ),
    )
    group.add_argument(
        "--frame-qc-max-excluded-frames",
        type=int,
        default=5,
        help="Maximum globally dominant frames excluded for frame-level SNIS.",
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
        "--mutation",
        action="append",
        default=None,
        dest="mutation_tokens",
        metavar="RES:FROM:TO",
        help=(
            "Repeatable mutation site using sequential pose numbering: "
            "``residue_index:original:target``. Leave ``original`` blank (e.g. "
            "``12::ALA``) to skip the topology/pose originality check before mutating "
            "(see also --original-residue-name on the legacy path). "
            "When set, omit --target-residue-number/--target-residue-name."
        ),
    )
    perturb_group.add_argument(
        "--target-residue-number",
        type=int,
        default=None,
        help=(
            "Rosetta pose residue index (1..N) for a single mutation, as in pose.residue(i). "
            "Ignored when repeatable --mutation is used. Requires --target-residue-name."
        ),
    )
    perturb_group.add_argument(
        "--target-residue-name",
        default=None,
        help=(
            "Three-letter Rosetta mutation target. Supports canonical amino acids plus "
            "common protonation/disulfide variants such as HID/HIE/HIP/CYM/CYX/ASH/GLH. "
            "For protonation studies, run one topology/trajectory per protonation state: "
            "the topology must use force-field residue names, while this value must match "
            "a residue type loaded in the current PyRosetta session. Generic HIS lets "
            "Rosetta optH choose the neutral tautomer, which FELP then maps to HID/HIE "
            "for OpenMM from the actual ring hydrogens. Use explicit HIE/HID/HIP when "
            "you need Rosetta to keep a specific histidine state."
        ),
    )
    perturb_group.add_argument(
        "--original-residue-name",
        help=(
            "Optional three-letter residue expected at --target-residue-number before "
            "mutation (e.g. GLY for GLY→ALA). When set, FELP enforces the reference state "
            "to this exact compatible residue type before scoring and checks both the "
            "topology family and converted Pose residue name3() at that sequential index. "
            "For protonation studies, generic HIS allows Rosetta optH tautomer selection, "
            "while explicit HIE/HID/HIP asks FELP to use the corresponding Rosetta residue "
            "type up front and fail if Rosetta later changes that identity."
        ),
    )

    rosetta_group = parser.add_argument_group("PyRosetta")
    rosetta_group.add_argument(
        "--pyrosetta-init-options",
        default="-ignore_unrecognized_res true -load_PDB_components false",
        help=(
            "Options string passed to pyrosetta.init. Use this to load extra residue "
            "params or protonation-state residue types when Rosetta needs custom "
            "chemistry definitions (for example via -extra_res_fa ...). FELP "
            "automatically enables -no_optH false for generic HIS runs and adds "
            "-keep_input_protonation_state/-pH_mode for some explicit protonation-state "
            "requests unless you already set those flags yourself."
        ),
    )
    rosetta_group.add_argument(
        "--rosetta-database",
        help="Optional explicit PyRosetta database path.",
    )
    rosetta_group.add_argument(
        "--rosetta-score-function",
        default="ref2015",
        help="Rosetta score function for relax/repack preparation and optional scoring.",
    )
    rosetta_group.add_argument(
        "--rosetta-perturbation-mode",
        choices=("relax", "repack"),
        default="relax",
        help=(
            "Rosetta local preparation mode for reference/perturbed poses: "
            "'relax' uses FastRelax with backbone CVs locked, while 'repack' "
            "uses sidechain repacking only."
        ),
    )
    rosetta_group.add_argument(
        "--rosetta-relax-iterations",
        default=5,
        type=int,
        help="Number of FastRelax iterations to perform when --rosetta-perturbation-mode=relax.",
    )
    rosetta_group.add_argument(
        "--rosetta-repack-trials",
        default=1,
        type=int,
        help=(
            "Number of Rosetta packing trials to request when "
            "--rosetta-perturbation-mode=repack. Values >1 ask PackRotamersMover "
            "to keep the best local repack result."
        ),
    )
    rosetta_group.add_argument(
        "--no-cartesian-relax",
        action="store_true",
        help="Disable Cartesian FastRelax when --rosetta-perturbation-mode=relax.",
    )
    rosetta_group.add_argument(
        "--rosetta-symmetric-gly-tables",
        action="store_true",
        help=(
            "Append '-symmetric_gly_tables true' to pyrosetta.init so Rosetta "
            "uses symmetric glycine Ramachandran tables when supported."
        ),
    )
    rosetta_group.add_argument(
        "--mute-pyrosetta",
        action="store_true",
        help=(
            "Reduce Rosetta tracer noise (FastRelax, packing, etc.) by adding "
            "'-mute all' to pyrosetta.init unless your --pyrosetta-init-options already set -mute."
        ),
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
        "--distance-cv",
        action="append",
        dest="distance_cv_tokens",
        metavar="<name:res1:atom1:res2:atom2>",
        help=(
            "Repeatable distance CV restraint between two pose-indexed atoms, in "
            "name:res1:atom1:res2:atom2 form. Use primitive distance CVs to compose "
            "protein-oriented projected CVs initially."
        ),
    )
    restraint_group.add_argument(
        "--restraint-force-constant",
        type=float,
        default=10000.0,
        help=(
            "Default harmonic strength for backbone dihedral CVs: Rosetta "
            "CircularHarmonicFunc weight (via the restraint protocol) and OpenMM "
            "CustomTorsionForce k. Rosetta *distance* AtomPair constraints use a "
            "separate width; see --rosetta-distance-constraint-sd-angstrom. Use "
            "--rosetta-restraint-force-constant / --openmm-restraint-force-constant "
            "to override one backend."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-restraint-force-constant",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Optional override for Rosetta backbone dihedral strength during "
            "CV/restraint protocol construction. Does not set Rosetta atom-pair "
            "distance widths; use --rosetta-distance-constraint-sd-angstrom. "
            "When omitted, --restraint-force-constant is used."
        ),
    )
    restraint_group.add_argument(
        "--openmm-restraint-force-constant",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Optional override for OpenMM harmonic spring k (dihedral and distance "
            "Custom*Forces) during minimization. When omitted, --restraint-force-constant "
            "is used."
        ),
    )
    restraint_group.add_argument(
        "--cv-measurement-tolerance-degrees",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed |measured CV − restraint target| after Rosetta relax or "
            "OpenMM minimization before the frame fails validation."
        ),
    )
    restraint_group.add_argument(
        "--distance-measurement-tolerance-angstrom",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed distance-CV drift in Angstrom after Rosetta relax or "
            "OpenMM minimization before the frame fails validation."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-distance-constraint-sd-angstrom",
        type=float,
        default=0.001,
        metavar="SD",
        help=(
            "Width (σ, Å) for Rosetta AtomPairConstraint HarmonicFunc during "
            "relax/repack — not an OpenMM-style spring constant. Smaller σ is stiffer. "
            "Default 0.001 Å is intentionally tight; increase slightly if minimization "
            "becomes unstable."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-restraint-target-backbone",
        action="store_true",
        help=(
            "During Rosetta preparation, add restraint-only backbone dihedral "
            "constraints on --target-residue-number without adding those dihedrals "
            "as FELP CVs. Defaults to phi/psi unless "
            "--rosetta-target-backbone-restraint-kind is supplied."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-target-backbone-restraint-kind",
        action="append",
        choices=("phi", "psi", "omega"),
        default=[],
        metavar="KIND",
        help=(
            "Repeatable target-backbone dihedral to restrain during Rosetta "
            "preparation. Supplying this option implies "
            "--rosetta-restraint-target-backbone."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-backbone-restraint-residue",
        action="append",
        type=int,
        default=[],
        metavar="RESIDUE",
        help=(
            "Repeatable pose residue index whose backbone dihedrals should be "
            "restrained during Rosetta preparation without becoming FELP CVs. "
            "Use with --rosetta-target-backbone-restraint-kind; when kinds are "
            "omitted, phi/psi are restrained. For example, repeat this for 3, 4, "
            "and 5 to restrain a local sequence window around target residue 4."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-target-backbone-restraint-force-constant",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Optional Rosetta-only force constant for target-backbone dihedral "
            "restraints. When omitted, the effective Rosetta restraint force "
            "constant is used."
        ),
    )
    restraint_group.add_argument(
        "--rosetta-target-backbone-restraint-tolerance-degrees",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Optional post-Rosetta validation tolerance for Rosetta-only "
            "target-backbone restraints. When omitted, "
            "--cv-measurement-tolerance-degrees is used."
        ),
    )

    mobility_group = parser.add_argument_group("Mobility")
    mobility_group.add_argument(
        "--mobility-mode",
        choices=("target_only", "radius", "sequence_window"),
        default="target_only",
        help=(
            "Which residues may relax/minimize after mutation. target_only allows only "
            "the target residue; radius includes residues near the target in space; "
            "sequence_window includes residues within +/- --mobility-sequence-window "
            "positions of the target."
        ),
    )
    mobility_group.add_argument(
        "--mobility-radius-angstrom",
        type=float,
        default=0.0,
        help="Radius in Angstrom used when --mobility-mode=radius.",
    )
    mobility_group.add_argument(
        "--mobility-sequence-window",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of sequence-neighbor residues on each side of the target to "
            "include when --mobility-mode=sequence_window. N=1 selects target-1, "
            "target, and target+1."
        ),
    )
    mobility_group.add_argument(
        "--mobility-include-backbone",
        action="store_true",
        help="Allow backbone motion for residues selected by the mobility policy.",
    )
    mobility_group.add_argument(
        "--mobility-include-sidechains",
        dest="mobility_include_sidechains",
        action="store_true",
        default=True,
        help="Allow sidechain motion for residues selected by the mobility policy.",
    )
    mobility_group.add_argument(
        "--no-mobility-include-sidechains",
        dest="mobility_include_sidechains",
        action="store_false",
        help="Disallow sidechain motion for residues selected by the mobility policy.",
    )

    sampling_group = parser.add_argument_group("Evaluation sampling")
    sampling_group.add_argument(
        "--microstate-sample-fraction",
        type=float,
        default=1.0,
        help=(
            "Uniform random fraction of frames to evaluate within each assigned "
            "microstate. The default 1.0 preserves all frames."
        ),
    )
    sampling_group.add_argument(
        "--microstate-min-samples",
        type=int,
        default=1,
        help=(
            "Minimum number of frames to evaluate per assigned microstate when "
            "subsampling, capped by the number available in that state."
        ),
    )
    sampling_group.add_argument(
        "--microstate-max-samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of frames to evaluate per assigned microstate "
            "after applying --microstate-sample-fraction."
        ),
    )
    sampling_group.add_argument(
        "--microstate-sampling-seed",
        type=int,
        default=0,
        help="Random seed for deterministic microstate-stratified frame sampling.",
    )
    _add_reweighting_qc_arguments(parser)
    _add_reweighting_method_arguments(parser)

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
        help=(
            "Optional explicit OpenMM platform name (e.g. CPU, CUDA, OpenCL). "
            "Mutually exclusive with --cpu and --gpu."
        ),
    )
    openmm_device = energy_group.add_mutually_exclusive_group()
    openmm_device.add_argument(
        "--cpu",
        action="store_true",
        help="Use the OpenMM CPU platform for energy minimization and scoring.",
    )
    openmm_device.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Use an OpenMM GPU platform: CUDA if available, otherwise OpenCL. "
            "Only applies when --energy-backend is openmm."
        ),
    )
    energy_group.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help=(
            "Number of OpenMM CPU-platform threads per worker process. Defaults to 1 "
            "because FELP already parallelizes across frames via --workers, and "
            "letting each OpenMM context spin up hundreds of threads on a many-core "
            "host has been observed to cause numerical instability (NaN coordinates). "
            "Only applied when the OpenMM CPU platform is in use."
        ),
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
        help=(
            "Frames per worker task when using --workers>1. Values >1 cut pickling/scheduling "
            "overhead but force that many frames to run serially inside one worker before the "
            "task finishes—so use modest sizes (e.g. 8–32) unless batches are very fast. "
            "Progress is reported by finished frame count, not by batch start."
        ),
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
        "--debug-frame-outputs",
        action="store_true",
        help=(
            "Write frame_debug artifacts for processed frames: OpenMM input/minimized "
            "PDBs and per-frame energy metadata. Use --debug-frame-index to restrict "
            "large runs to specific frames."
        ),
    )
    output_group.add_argument(
        "--debug-frame-index",
        action="append",
        type=int,
        default=[],
        metavar="INDEX",
        help=(
            "Frame index to include in --debug-frame-outputs. Repeatable. When omitted, "
            "debug output is written for every processed frame."
        ),
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
        required=False,
        help=(
            "JSON or YAML feature specification for the PyEMMA featurizer. "
            "Omit if using --phi-psi-residue-number."
        ),
    )
    input_group.add_argument(
        "--phi-psi-residue-number",
        type=int,
        help=(
            "Build a phi/psi dihedral feature spec automatically for this residue "
            "(disambiguation matches --trajectory / FELP conventions: resSeq, then "
            "1-based topology index). Implies --feature-spec-file is not used; "
            "writes <output-dir>/inputs/phi_psi_feature_spec.json."
        ),
    )
    input_group.add_argument(
        "--phi-psi-residue-name",
        default="GLY",
        help=(
            "Expected residue name when resolving --phi-psi-residue-number "
            "(e.g. GLY for capped dipeptides)."
        ),
    )
    input_group.add_argument(
        "--phi-psi-selection",
        help=(
            "Optional MDTraj selection string identifying exactly one internal residue "
            "for phi/psi (overrides residue-number / name resolution)."
        ),
    )
    input_group.add_argument(
        "--phi-psi-representation",
        choices=("raw", "cossin"),
        default="raw",
        help="Feature representation for auto phi/psi: degrees (raw) or sin/cos pairs.",
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
        choices=("kmeans", "regspace", "grid"),
        default="kmeans",
        help=(
            "Clustering method used to discretize trajectories. 'grid' assigns frames "
            "to a regular rectilinear lattice in feature space (requires --reducer none)."
        ),
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
    clustering_group.add_argument(
        "--grid-bins",
        type=int,
        nargs="+",
        help=(
            "Per-axis bin counts for grid clustering (1–3 axes). Omit to infer counts "
            "from the reference trajectory using Freedman–Diaconis heuristics."
        ),
    )
    clustering_group.add_argument(
        "--grid-target-frames-per-state",
        type=int,
        default=100,
        help="Target support per grid cell when auto-selecting bin counts.",
    )
    clustering_group.add_argument(
        "--grid-min-bins-per-axis",
        type=int,
        default=3,
        help="Minimum bins per feature axis for auto grid clustering.",
    )
    clustering_group.add_argument(
        "--grid-max-bins-per-axis",
        type=int,
        default=64,
        help="Maximum bins per feature axis for auto grid clustering.",
    )
    clustering_group.add_argument(
        "--grid-max-total-states",
        type=int,
        default=2500,
        help="Maximum total grid cells when auto-selecting bin counts.",
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
        help=(
            "Write the validation report under <output-dir>/validation/ "
            "(implied timescales, CK tests, stationary distribution, transition matrix, "
            "2D feature projections when applicable). Use --its-lags to fix ITS lag times."
        ),
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing MSM bundle directory.",
    )

    validation_group = parser.add_argument_group("Validation plots (--write-plots)")
    validation_group.add_argument(
        "--its-max-lag",
        type=int,
        default=80,
        help="Maximum lag (steps) for implied timescales when --its-lags is omitted.",
    )
    validation_group.add_argument(
        "--its-lag-points",
        type=int,
        default=35,
        help="Number of lag samples between 1 and --its-max-lag when --its-lags is omitted.",
    )
    validation_group.add_argument(
        "--its-lags",
        type=int,
        nargs="+",
        help=(
            "Explicit lag times (steps) for the implied timescale plot, e.g. "
            "--its-lags 1 2 3 4 5 6 7 8 9 10. Overrides --its-max-lag/--its-lag-points."
        ),
    )
    validation_group.add_argument(
        "--its-processes",
        type=int,
        default=8,
        help="Number of slow ITS processes to plot and export numerically.",
    )
    validation_group.add_argument(
        "--its-n-jobs",
        type=int,
        default=1,
        help=(
            "Worker processes for PyEMMA implied-timescale estimation. "
            "Defaults to 1 because PyEMMA multiprocessing can be slow or fragile "
            "in CLI/IDE environments."
        ),
    )
    validation_group.add_argument(
        "--its-errors",
        choices=("none", "bayes", "calc"),
        default="none",
        help=(
            "Implied-timescale uncertainty in PyEMMA: none (point estimates, default), "
            "bayes (Bayesian sampling), or calc (error propagation)."
        ),
    )
    validation_group.add_argument(
        "--ck-nsets",
        type=int,
        default=4,
        help="Number of coarse sets for the discrete MSM Chapman–Kolmogorov test.",
    )
    validation_group.add_argument(
        "--ck-mlags",
        type=int,
        default=10,
        help="Number of lag multiples for CK validation.",
    )
    validation_group.add_argument(
        "--ck-n-jobs",
        type=int,
        default=1,
        help="Worker processes for PyEMMA CK validation.",
    )
    validation_group.add_argument(
        "--skip-ck",
        action="store_true",
        help="Skip Chapman-Kolmogorov validation plots.",
    )
    validation_group.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label for bundle-style 2D plots (auto-derived from feature spec if omitted).",
    )
    validation_group.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label for bundle-style 2D plots (auto-derived from feature spec if omitted).",
    )
    validation_group.add_argument(
        "--validation-cluster-scatter-max-points",
        type=int,
        default=25000,
        help=(
            "Max frames on validation/microstate_clusters.png scatter. "
            "Use 0 to plot all frames (can be slow and produce large files)."
        ),
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
    parser.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label for 2D projection plots (auto-derived from feature spec if omitted).",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label for 2D projection plots (auto-derived from feature spec if omitted).",
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
    parser.add_argument(
        "--fel-bins",
        type=int,
        default=72,
        help="Bin count per axis for 2D CV free-energy heatmaps.",
    )
    parser.add_argument(
        "--fel-max-points",
        type=int,
        default=200_000,
        help="Maximum frames used when building 2D CV heatmaps (uniform subsampling).",
    )
    parser.add_argument(
        "--fel-estimator",
        choices=("histogram", "gaussian-kernel"),
        default="histogram",
        help=(
            "Estimator for frame-projected CV free-energy landscapes. "
            "'histogram' preserves the existing weighted-bin estimator; "
            "'gaussian-kernel' uses a state-mass-preserving kernel density estimator."
        ),
    )
    parser.add_argument(
        "--fel-bandwidth",
        default="auto",
        help=(
            "Bandwidth for --fel-estimator=gaussian-kernel: 'auto', one positive "
            "float, or two comma-separated positive floats for x,y."
        ),
    )
    parser.add_argument(
        "--fel-bandwidth-rule",
        choices=("scott", "silverman", "state-adaptive", "bin-width"),
        default="state-adaptive",
        help=(
            "Automatic bandwidth rule for --fel-bandwidth=auto. "
            "'state-adaptive' uses per-state Scott widths capped to a small "
            "multiple of the plotting bin width (recommended default). "
            "'bin-width' sets bandwidth to a multiple of the histogram bin width "
            "(0.75 by default, or the value passed to --fel-bandwidth). "
            "'scott' and 'silverman' use one global bandwidth with the same cap."
        ),
    )
    parser.add_argument(
        "--fel-bandwidth-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for automatic Gaussian-kernel bandwidths. Use values below "
            "1.0 (for example 0.6) for sharper surfaces."
        ),
    )
    parser.add_argument(
        "--fel-bandwidth-max-bins",
        type=float,
        default=1.25,
        help=(
            "Upper cap on automatic kernel bandwidth in units of histogram-bin "
            "width per axis. Lower values reduce blobby over-smoothing."
        ),
    )
    parser.add_argument(
        "--fel-bandwidth-min-bins",
        type=float,
        default=0.5,
        help=(
            "Lower floor on automatic kernel bandwidth in units of histogram-bin "
            "width per axis. For visibly smoothed KDE surfaces, use at least 1.0–1.5."
        ),
    )
    parser.add_argument(
        "--fel-surface-smooth-sigma-bins",
        type=float,
        default=0.0,
        help=(
            "Optional display-only Gaussian smoothing of projected 2D probability "
            "grids after estimation, in units of histogram bins (requires scipy). "
            "Does not change microstate reweighting; use for figure aesthetics only."
        ),
    )
    parser.add_argument(
        "--fel-cmap",
        default=DEFAULT_FEL_COLORMAP,
        help=(
            "Matplotlib colormap for CV free-energy heatmaps, discrete microstate "
            "FEL scatter, and delta free-energy scatter (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--fel-plot-style",
        choices=("default", "contour"),
        default="default",
        help=(
            "Renderer for 2D free-energy/probability landscapes. 'default' draws one "
            "color per histogram bin (pcolormesh, no inter-bin interpolation). "
            "'contour' draws filled iso-energy contours on the same bin grid."
        ),
    )
    parser.add_argument(
        "--probability-cmap",
        default=None,
        help=(
            "Matplotlib colormap for 2D probability heatmaps (frame-projected and "
            "state-grid) and reference/target probability scatters. "
            "Defaults to --fel-cmap when omitted."
        ),
    )
    parser.add_argument(
        "--divergence-cmap",
        default=None,
        dest="divergence_cmap",
        help=(
            "Matplotlib colormap for the projected symmetric-KL heatmap "
            f"(default: {FELP_KL_DIVERGENCE_COLORMAP}, registered by FELP)."
        ),
    )
    parser.add_argument(
        "--kl-scale-by-kbt",
        action="store_true",
        help=(
            "Display projected KL heatmaps multiplied by kBT. This provides an "
            "energy-scale view of KL contributions using --temperature-kelvin and "
            "--plot-energy-unit, but the divergence is still computed from probabilities."
        ),
    )
    parser.add_argument(
        "--kl-display-threshold",
        type=float,
        default=1.0e-5,
        help=(
            "Plot per-cell signed KL values with magnitude below this threshold "
            "as zero (white at the diverging colormap center). Units match the KL "
            "colorbar (kcal/mol or kJ/mol when --kl-scale-by-kbt is set, otherwise "
            "dimensionless). Default: 1e-5. Use 0 to disable masking."
        ),
    )
    parser.add_argument(
        "--kl-vmin",
        type=float,
        default=None,
        help=(
            "Fixed minimum for KL imbalance colorbars (display units). For "
            "symmetric maps, omit and set only --kl-vmax to use [-vmax, vmax]."
        ),
    )
    parser.add_argument(
        "--kl-vmax",
        type=float,
        default=None,
        help=(
            "Fixed maximum for KL imbalance colorbars (display units). For "
            "symmetric maps, omit --kl-vmin to use [-vmax, vmax]."
        ),
    )
    parser.add_argument(
        "--reweighting-weight-treatment",
        choices=("raw", "psis", "psis-smoothed"),
        default=None,
        help=(
            "Override the weight treatment used for frame-projected FEL plots. "
            "By default, felp-plot reads the treatment recorded in the "
            "microstate summary. This does not recompute state probabilities; "
            "use summarize/run for that."
        ),
    )
    parser.add_argument(
        "--plot-cv-x",
        dest="plot_cv_x",
        help=(
            "CV name for the x-axis in FEL/CV plots. "
            "Pass either the short name (e.g. 'ends' or 'psi_114') or the full distance-cv "
            "spec string (e.g. 'ends:1:CA:10:CA'); only the name part is used. "
            "For 1D microstates, this alone selects the plotting axis. "
            "Defaults to custom CVs when available, otherwise phi/psi."
        ),
    )
    parser.add_argument(
        "--plot-cv-y",
        dest="plot_cv_y",
        help=(
            "CV name for the y-axis in 2D FEL plots. "
            "Pass either the short name (e.g. 'turn') or the full distance-cv "
            "spec string (e.g. 'turn:3:CA:6:CA'); only the name part is used. "
            "Omit for 1D microstate summaries with a single CV."
        ),
    )
    parser.add_argument(
        "--fel-max-energy-kj-mol",
        type=float,
        default=None,
        help=(
            "Cap relative free-energy color scales at this value (kJ/mol). "
            "Bins or points above the cap are drawn in a flat background color; "
            "delta free-energy projections use a symmetric cap (± this value). "
            "Mutually exclusive with --fel-max-energy-kcal-mol."
        ),
    )
    parser.add_argument(
        "--fel-max-energy-kcal-mol",
        type=float,
        default=None,
        help=(
            "Same as --fel-max-energy-kj-mol, but the cap is given in kcal/mol "
            "(converted internally to kJ/mol). Convenient with --plot-energy-unit "
            "kcal_mol. Mutually exclusive with --fel-max-energy-kj-mol."
        ),
    )
    parser.add_argument(
        "--plot-energy-unit",
        choices=("kj_mol", "kcal_mol"),
        default="kj_mol",
        help=(
            "Energy unit for plot axes and colorbars. CSV inputs remain in kJ/mol; "
            "values are converted for display when set to kcal_mol "
            "(1 kcal/mol = 4.184 kJ/mol). Use --fel-max-energy-kcal-mol to cap the "
            "color scale directly in kcal/mol."
        ),
    )


def _add_md_kl_plot_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for standalone MD-vs-reweighted KL plots."""

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument(
        "--felp-output-dir",
        help=(
            "FELP output directory containing microstate_summary.csv and/or "
            "reweighted_surface.csv. Use --reweighted-surface to point to a CSV directly."
        ),
    )
    input_group.add_argument(
        "--reweighted-surface",
        "--microstate-summary",
        dest="reweighted_surface",
        help=(
            "CSV containing plot_x, plot_y, and target_probability columns "
            "(for example FELP reweighted_surface.csv or microstate_summary.csv)."
        ),
    )
    input_group.add_argument(
        "--raw-frame-results",
        help=(
            "Frame-level FELP CSV used to project the reweighted distribution from "
            "actual frame phi/psi coordinates. Strongly recommended for MSM bundle "
            "or other non-grid microstate summaries."
        ),
    )
    input_group.add_argument(
        "--topology-file",
        required=True,
        help="Topology file for the reference MD trajectory, understood by MDTraj.",
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
            "Repeatable grouped MD trajectory input. Amber NetCDF (.nc) and other "
            "MDTraj-readable formats are supported."
        ),
    )
    input_group.add_argument(
        "--phi-psi-residue-number",
        type=int,
        help=(
            "Residue used to compute the MD phi/psi distribution. Resolution uses "
            "resSeq, then 1-based topology index. Not required with --phi-psi-selection."
        ),
    )
    input_group.add_argument(
        "--phi-psi-residue-name",
        help="Optional expected residue name when resolving --phi-psi-residue-number.",
    )
    input_group.add_argument(
        "--phi-psi-selection",
        help=(
            "Optional MDTraj selection string identifying exactly one internal residue "
            "for phi/psi; overrides residue-number/name resolution."
        ),
    )
    input_group.add_argument(
        "--distance-cv",
        action="append",
        dest="distance_cv_tokens",
        metavar="NAME:RES1:ATOM1:RES2:ATOM2",
        help=(
            "Repeatable distance CV for MD trajectory measurement, e.g. "
            "ends:1:CA:10:CA. Use with --plot-cv-x/--plot-cv-y instead of phi/psi."
        ),
    )
    input_group.add_argument(
        "--plot-cv-x",
        dest="plot_cv_x",
        help=(
            "CV name for the x-axis (e.g. ends). Required with --plot-cv-y when "
            "using --distance-cv."
        ),
    )
    input_group.add_argument(
        "--plot-cv-y",
        dest="plot_cv_y",
        help=(
            "CV name for the y-axis (e.g. turn). Required with --plot-cv-x when "
            "using --distance-cv."
        ),
    )
    input_group.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride used when reading the MD trajectory.",
    )
    input_group.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Zero-based selected-frame index to start after applying stride.",
    )
    input_group.add_argument(
        "--frame-stop",
        type=int,
        help="Exclusive selected-frame index to stop after applying stride.",
    )
    input_group.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Number of trajectory frames per MDTraj iterload chunk.",
    )

    plot_group = parser.add_argument_group("Plotting")
    plot_group.add_argument(
        "--temperature-kelvin",
        type=float,
        default=300.0,
        help="Temperature used to reconstruct reweighted frame weights from delta_u.",
    )
    plot_group.add_argument(
        "--fel-bins",
        type=int,
        default=180,
        help=(
            "Number of bins per phi/psi axis when projecting --raw-frame-results "
            "and the MD trajectory."
        ),
    )
    plot_group.add_argument(
        "--fel-max-points",
        type=int,
        default=200_000,
        help="Maximum FELP frame points used for the reweighted projection.",
    )
    plot_group.add_argument(
        "--fel-cmap",
        default=DEFAULT_FEL_COLORMAP,
        help=(
            "Matplotlib colormap for the MD reference free-energy heatmap "
            "(default: %(default)s)."
        ),
    )
    plot_group.add_argument(
        "--fel-max-energy-kj-mol",
        type=float,
        default=None,
        help=(
            "Cap the MD reference free-energy color scale at this value (kJ/mol). "
            "Bins above the cap are drawn in the flat background color. "
            "Mutually exclusive with --fel-max-energy-kcal-mol."
        ),
    )
    plot_group.add_argument(
        "--fel-max-energy-kcal-mol",
        type=float,
        default=None,
        help=(
            "Same as --fel-max-energy-kj-mol, but the cap is given in kcal/mol. "
            "Mutually exclusive with --fel-max-energy-kj-mol."
        ),
    )
    plot_group.add_argument(
        "--plot-energy-unit",
        choices=("kj_mol", "kcal_mol"),
        default="kj_mol",
        help=(
            "Energy unit for the MD reference free-energy plot. "
            "Use --fel-max-energy-kcal-mol to cap the color scale in kcal/mol."
        ),
    )
    plot_group.add_argument(
        "--reweighting-weight-treatment",
        choices=("raw", "psis", "psis-smoothed"),
        default=None,
        help=(
            "Override the weight treatment used for frame-projected reweighted "
            "probabilities. By default, md-kl-plot reads the treatment recorded "
            "in the microstate summary."
        ),
    )
    plot_group.add_argument(
        "--probability-cmap",
        default=None,
        dest="probability_cmap",
        help="Matplotlib colormap for MD and reweighted probability heatmaps.",
    )
    plot_group.add_argument(
        "--divergence-cmap",
        default=None,
        dest="divergence_cmap",
        help=(
            "Matplotlib colormap for the MD-vs-reweighted KL heatmap "
            f"(default: {FELP_KL_DIVERGENCE_COLORMAP}, registered by FELP)."
        ),
    )
    plot_group.add_argument(
        "--kl-scale-by-kbt",
        action="store_true",
        help=(
            "Display the KL heatmap multiplied by kBT. This provides an "
            "energy-scale view of KL contributions; the divergence is still "
            "computed from probabilities."
        ),
    )
    plot_group.add_argument(
        "--kl-energy-unit",
        choices=("kj_mol", "kcal_mol"),
        default="kcal_mol",
        help="Energy unit used when --kl-scale-by-kbt is enabled.",
    )
    plot_group.add_argument(
        "--kl-display-threshold",
        type=float,
        default=1.0e-5,
        help=(
            "Plot per-cell signed KL values with magnitude below this threshold "
            "as zero (white at the diverging colormap center). Units match the KL "
            "colorbar (see --kl-energy-unit when --kl-scale-by-kbt is set). "
            "Default: 1e-5. Use 0 to disable masking."
        ),
    )
    plot_group.add_argument(
        "--kl-vmin",
        type=float,
        default=None,
        help=(
            "Fixed minimum for KL imbalance colorbars (display units). For "
            "symmetric maps, omit and set only --kl-vmax to use [-vmax, vmax]."
        ),
    )
    plot_group.add_argument(
        "--kl-vmax",
        type=float,
        default=None,
        help=(
            "Fixed maximum for KL imbalance colorbars (display units). For "
            "symmetric maps, omit --kl-vmin to use [-vmax, vmax]."
        ),
    )
    plot_group.add_argument(
        "--output-dir",
        required=True,
        help="Directory where standalone KL plots and summary JSON should be written.",
    )
    plot_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )


def _add_composite_figure_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for JSON-driven composite Ramachandran figures."""

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "JSON specification for the composite figure (rows, columns, style, "
            "and data paths). See data/composite_rotation_mutations.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )


def _add_ess_compare_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for comparing ESS diagnostics across FELP runs."""

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        dest="ess_inputs",
        metavar="<run-dir-or-summary-json>",
        help=(
            "FELP output directories or explicit run_summary.json files to compare. "
            "Directories are resolved to <dir>/run_summary.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where ESS comparison plots and CSV should be written.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for inputs; must match the number of --input values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    parser.add_argument(
        "--bootstrap-blocks",
        type=int,
        default=None,
        metavar="L",
        help=(
            "Block size (contiguous eligible frames) for circular block bootstrap of "
            "overlap diagnostics. Requires --bootstrap-draws and run_config.json + "
            "frame_results.csv in each run output directory."
        ),
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        metavar="N",
        help="Number of bootstrap replicates (use with --bootstrap-blocks).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="RNG seed for reproducible bootstrap resampling.",
    )
    parser.add_argument(
        "--panel-layout",
        choices=("combined", "by_reference"),
        default="by_reference",
        help=(
            "Bar chart layout: 'by_reference' uses one subplot per reference "
            "dipeptide (default); 'combined' groups reference residues "
            "side-by-side for each target on one panel."
        ),
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
    summarize_parser.add_argument(
        "--repartition",
        action="store_true",
        help=(
            "Re-assign every frame to microstates defined by the current "
            "--microstate-source / --phi-bins / --psi-bins / --auto-* options "
            "instead of using the microstate_id stored in frame_results.csv. "
            "For msm_bundle, frames are assigned to the nearest MSM feature-space "
            "centroid using stored assignment_measurements (with Angstrom→nm "
            "conversion for distance CVs). This is cheap—it reuses the expensive "
            "delta_u values already computed."
        ),
    )
    summarize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    summarize_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for by-state summarize QC/aggregation.",
    )
    _add_common_microstate_arguments(summarize_parser)
    _add_reweighting_qc_arguments(summarize_parser)
    _add_reweighting_method_arguments(summarize_parser)

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

    md_kl_plot_parser = subparsers.add_parser(
        "md-kl-plot",
        help="Generate standalone KL plots comparing MD to reweighted FELP data.",
    )
    _add_md_kl_plot_arguments(md_kl_plot_parser)

    ess_compare_parser = subparsers.add_parser(
        "ess-compare",
        help="Compare ESS and overlap diagnostics across FELP run summaries.",
    )
    _add_ess_compare_arguments(ess_compare_parser)

    composite_figure_parser = subparsers.add_parser(
        "composite-figure",
        help="Assemble multi-row Ramachandran composite figures from FELP outputs.",
    )
    _add_composite_figure_arguments(composite_figure_parser)

    return parser


def build_run_config(namespace: argparse.Namespace) -> RunConfig:
    """Construct and validate a RunConfig from the parsed CLI namespace."""

    reference_probability_mode = namespace.prior_probability_mode
    if namespace.ignore_loaded_reference_probabilities:
        reference_probability_mode = "raw_counts"

    if namespace.cpu or namespace.gpu:
        if namespace.energy_backend != "openmm":
            raise ConfigurationError(
                "--cpu and --gpu are only supported with --energy-backend openmm."
            )
    platform_name = namespace.platform_name
    if namespace.cpu:
        if namespace.platform_name is not None:
            raise ConfigurationError("Do not combine --cpu with --platform-name.")
        platform_name = "CPU"
    elif namespace.gpu:
        if namespace.platform_name is not None:
            raise ConfigurationError("Do not combine --gpu with --platform-name.")
        platform_name = resolve_openmm_gpu_platform_name()

    mutation_tokens = tuple(namespace.mutation_tokens or ())
    legacy_number = namespace.target_residue_number
    legacy_target = namespace.target_residue_name
    legacy_orig = namespace.original_residue_name
    if mutation_tokens:
        if (
            legacy_number is not None
            or legacy_target is not None
            or legacy_orig is not None
        ):
            raise ConfigurationError(
                "Use either repeatable --mutation residue:original:target or the legacy "
                "--target-residue-number/--target-residue-name options, not both."
            )
        perturbation = PerturbationConfig(
            mutations=tuple(_parse_mutation_cli_token(tok) for tok in mutation_tokens)
        )
    else:
        if legacy_number is None or legacy_target is None:
            raise ConfigurationError(
                "--target-residue-number and --target-residue-name are required unless "
                "one or more --mutation residue:original:target sites are supplied."
            )
        perturbation = PerturbationConfig.from_legacy(
            target_residue_number=legacy_number,
            target_residue_name=legacy_target,
            original_residue_name=legacy_orig,
        )
    primary_cv_residue_number = perturbation.target_residue_number

    distance_cv_specs = _parse_distance_cv_specs(namespace.distance_cv_tokens)
    cv_specs = _parse_cv_specs(
        namespace.cv_tokens,
        primary_cv_residue_number,
        include_legacy_default=not distance_cv_specs,
    )
    microstate_source = namespace.microstate_source
    if microstate_source == "grid" and distance_cv_specs and not cv_specs:
        microstate_source = "auto"
    if microstate_source == "auto":
        reference_probability_mode = "raw_counts"
    rosetta_target_backbone_restraints = tuple(
        namespace.rosetta_target_backbone_restraint_kind
    )
    rosetta_backbone_restraint_residues = tuple(
        namespace.rosetta_backbone_restraint_residue
    )
    if (
        namespace.rosetta_restraint_target_backbone
        or rosetta_backbone_restraint_residues
    ) and not rosetta_target_backbone_restraints:
        rosetta_target_backbone_restraints = ("phi", "psi")

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
        perturbation=perturbation,
        rosetta=RosettaConfig(
            init_options=namespace.pyrosetta_init_options,
            database_path=namespace.rosetta_database,
            score_function=namespace.rosetta_score_function,
            perturbation_mode=namespace.rosetta_perturbation_mode,
            relax_iterations=namespace.rosetta_relax_iterations,
            repack_trials=namespace.rosetta_repack_trials,
            cartesian=not namespace.no_cartesian_relax,
            symmetric_gly_tables=namespace.rosetta_symmetric_gly_tables,
            mute_tracers=namespace.mute_pyrosetta,
        ),
        restraints=RestraintConfig(
            mode=namespace.restraint_mode,
            force_constant=namespace.restraint_force_constant,
            rosetta_restraint_force_constant=namespace.rosetta_restraint_force_constant,
            openmm_restraint_force_constant=namespace.openmm_restraint_force_constant,
            cv_measurement_tolerance_degrees=namespace.cv_measurement_tolerance_degrees,
            distance_measurement_tolerance_angstrom=(
                namespace.distance_measurement_tolerance_angstrom
            ),
            rosetta_distance_constraint_sd_angstrom=(
                namespace.rosetta_distance_constraint_sd_angstrom
            ),
            rosetta_target_backbone_restraints=rosetta_target_backbone_restraints,
            rosetta_backbone_restraint_residue_numbers=(
                rosetta_backbone_restraint_residues
            ),
            rosetta_target_backbone_restraint_force_constant=(
                namespace.rosetta_target_backbone_restraint_force_constant
            ),
            rosetta_target_backbone_restraint_tolerance_degrees=(
                namespace.rosetta_target_backbone_restraint_tolerance_degrees
            ),
            cv_specs=cv_specs,
            distance_cv_specs=distance_cv_specs,
        ),
        mobility=MobilityConfig(
            mode=namespace.mobility_mode,
            radius_angstrom=namespace.mobility_radius_angstrom,
            sequence_window=namespace.mobility_sequence_window,
            include_backbone=namespace.mobility_include_backbone,
            include_sidechains=namespace.mobility_include_sidechains,
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
            platform_name=platform_name,
            cpu_threads=namespace.cpu_threads,
            rosetta_score_function=namespace.rosetta_score_function,
        ),
        microstates=MicrostateConfig(
            source=microstate_source,
            target_residue_number=primary_cv_residue_number,
            definition_file=namespace.microstate_file,
            phi_bins=namespace.phi_bins,
            psi_bins=namespace.psi_bins,
            msm_bundle_dir=namespace.msm_bundle_dir,
            state_level=namespace.state_level,
            allow_unassigned=namespace.allow_unassigned,
            reference_probability_mode=reference_probability_mode,
            auto_target_frames_per_state=namespace.auto_microstate_target_frames_per_state,
            auto_min_bins_per_axis=namespace.auto_microstate_min_bins_per_axis,
            auto_max_bins_per_axis=namespace.auto_microstate_max_bins_per_axis,
            auto_max_total_states=namespace.auto_microstate_max_states,
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
            debug_frame_outputs=namespace.debug_frame_outputs,
            debug_frame_indices=tuple(namespace.debug_frame_index),
            overwrite=namespace.overwrite,
            resume=namespace.resume,
        ),
        sampling=SamplingConfig(
            microstate_sample_fraction=namespace.microstate_sample_fraction,
            microstate_min_samples=namespace.microstate_min_samples,
            microstate_max_samples=namespace.microstate_max_samples,
            microstate_sampling_seed=namespace.microstate_sampling_seed,
        ),
        reweighting_qc=build_reweighting_qc_config(namespace),
        reweighting=build_reweighting_config(namespace),
    )
    config.validate()
    return config


def build_microstate_config(
    namespace: argparse.Namespace, target_residue_number: int = 2
) -> MicrostateConfig:
    """Construct a MicrostateConfig from the parsed CLI namespace."""

    reference_probability_mode = namespace.prior_probability_mode
    if namespace.ignore_loaded_reference_probabilities:
        reference_probability_mode = "raw_counts"

    return MicrostateConfig(
        source=namespace.microstate_source,
        target_residue_number=target_residue_number,
        definition_file=namespace.microstate_file,
        phi_bins=namespace.phi_bins,
        psi_bins=namespace.psi_bins,
        msm_bundle_dir=namespace.msm_bundle_dir,
        state_level=namespace.state_level,
        allow_unassigned=namespace.allow_unassigned,
        reference_probability_mode=reference_probability_mode,
        auto_target_frames_per_state=getattr(
            namespace,
            "auto_microstate_target_frames_per_state",
            100,
        ),
        auto_min_bins_per_axis=getattr(namespace, "auto_microstate_min_bins_per_axis", 3),
        auto_max_bins_per_axis=getattr(namespace, "auto_microstate_max_bins_per_axis", 64),
        auto_max_total_states=getattr(namespace, "auto_microstate_max_states", 2500),
    )


def build_reweighting_config(namespace: argparse.Namespace) -> ReweightingConfig:
    """Construct a ReweightingConfig from the parsed CLI namespace."""

    if getattr(namespace, "reweighting_methods", None):
        methods = tuple(namespace.reweighting_methods)
    else:
        methods = (getattr(namespace, "reweighting_method", "independent"),)
    return ReweightingConfig(
        methods=methods,
        neighbor_pool=NeighborPoolConfig(
            tau=getattr(namespace, "neighbor_pool_tau", None),
            tau_mode=getattr(namespace, "neighbor_pool_tau_mode", "lcurve"),
            solver=getattr(namespace, "neighbor_pool_solver", "blend"),
            min_local_weight=float(
                getattr(namespace, "neighbor_pool_min_local_weight", 0.0)
            ),
            max_neighbors=int(getattr(namespace, "neighbor_pool_max_neighbors", 4)),
            grid_connectivity=str(
                getattr(namespace, "neighbor_pool_grid_connectivity", "chebyshev3")
            ),
            chebyshev_radius=int(
                getattr(namespace, "neighbor_pool_chebyshev_radius", 3)
            ),
        ),
        laplacian=LaplacianConfig(
            lambda_kj_mol2=float(getattr(namespace, "laplacian_lambda_kj_mol2", 1.0)),
            lambda_scale_by_kbt=bool(
                getattr(namespace, "laplacian_lambda_scale_by_kbt", True)
            ),
        ),
        frame=FrameReweightingConfig(
            max_excluded_frames_global=int(
                getattr(namespace, "frame_qc_max_excluded_frames", 5)
            ),
        ),
    )


def build_reweighting_qc_config(namespace: argparse.Namespace) -> ReweightingQCConfig:
    """Construct a ReweightingQCConfig from the parsed CLI namespace."""

    return ReweightingQCConfig(
        policy=getattr(namespace, "reweighting_qc_policy", "flag"),
        method=getattr(namespace, "reweighting_qc_method", "auto"),
        weight_treatment=getattr(namespace, "reweighting_weight_treatment", "raw"),
        min_frame_count=getattr(namespace, "reweighting_qc_min_frame_count", 20),
        max_frame_weight_fraction=getattr(
            namespace,
            "reweighting_qc_max_frame_weight_fraction",
            0.5,
        ),
        min_state_ess=getattr(namespace, "reweighting_qc_min_state_ess", 5.0),
        max_leave_one_out_shift_kj_mol=getattr(
            namespace,
            "reweighting_qc_max_leave_one_out_shift_kj_mol",
            10.0,
        ),
        max_excluded_frames_per_state=getattr(
            namespace,
            "reweighting_qc_max_excluded_frames_per_state",
            5,
        ),
    )


def build_msm_fit_config(namespace: argparse.Namespace) -> MSMFitConfig:
    """Construct and validate an MSMFitConfig from the parsed CLI namespace."""

    its_lags: tuple[int, ...] | None = None
    if namespace.its_lags:
        its_lags = tuple(int(x) for x in namespace.its_lags)

    grid_bins: tuple[int, ...] | None = None
    if getattr(namespace, "grid_bins", None):
        grid_bins = tuple(int(value) for value in namespace.grid_bins)

    config = MSMFitConfig(
        topology_file=namespace.topology_file,
        trajectory_files=tuple(
            tuple(group) for group in namespace.trajectory_groups
        ),
        feature_spec_file=namespace.feature_spec_file,
        phi_psi_target_residue_number=namespace.phi_psi_residue_number,
        phi_psi_target_residue_name=namespace.phi_psi_residue_name,
        phi_psi_selection=namespace.phi_psi_selection,
        phi_psi_dihedral_representation=namespace.phi_psi_representation,
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
        grid_bins=grid_bins,
        grid_target_frames_per_state=int(namespace.grid_target_frames_per_state),
        grid_min_bins_per_axis=int(namespace.grid_min_bins_per_axis),
        grid_max_bins_per_axis=int(namespace.grid_max_bins_per_axis),
        grid_max_total_states=int(namespace.grid_max_total_states),
        msm_lag=namespace.msm_lag,
        metastable_states=namespace.metastable_states,
        hmm_states=namespace.hmm_states,
        save_pyemma_checkpoints=namespace.save_pyemma_checkpoints,
        write_plots=namespace.write_plots,
        overwrite=namespace.overwrite,
        its_max_lag=namespace.its_max_lag,
        its_lag_points=namespace.its_lag_points,
        its_lags=its_lags,
        its_processes=namespace.its_processes,
        its_n_jobs=namespace.its_n_jobs,
        ck_nsets=namespace.ck_nsets,
        ck_mlags=namespace.ck_mlags,
        ck_n_jobs=namespace.ck_n_jobs,
        skip_ck=namespace.skip_ck,
        its_errors=str(namespace.its_errors),
        plot_xlabel=getattr(namespace, "xlabel", None),
        plot_ylabel=getattr(namespace, "ylabel", None),
        validation_cluster_scatter_max_points=int(
            namespace.validation_cluster_scatter_max_points
        ),
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
        xlabel=getattr(namespace, "xlabel", None),
        ylabel=getattr(namespace, "ylabel", None),
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
        fel_bins=namespace.fel_bins,
        fel_max_points=namespace.fel_max_points,
        fel_estimator=namespace.fel_estimator,
        fel_bandwidth=namespace.fel_bandwidth,
        fel_bandwidth_rule=namespace.fel_bandwidth_rule,
        fel_bandwidth_scale=namespace.fel_bandwidth_scale,
        fel_bandwidth_max_bins=namespace.fel_bandwidth_max_bins,
        fel_bandwidth_min_bins=namespace.fel_bandwidth_min_bins,
        fel_surface_smooth_sigma_bins=namespace.fel_surface_smooth_sigma_bins,
        fel_colormap=namespace.fel_cmap,
        fel_plot_style=namespace.fel_plot_style,
        probability_colormap=getattr(namespace, "probability_cmap", None),
        divergence_colormap=getattr(namespace, "divergence_cmap", None),
        kl_scale_by_kbt=namespace.kl_scale_by_kbt,
        kl_display_threshold=namespace.kl_display_threshold,
        kl_colorbar_vmin=namespace.kl_vmin,
        kl_colorbar_vmax=namespace.kl_vmax,
        fel_max_energy_kj_mol=_resolve_fel_max_energy_kj_mol(
            namespace.fel_max_energy_kj_mol,
            namespace.fel_max_energy_kcal_mol,
        ),
        plot_energy_unit=namespace.plot_energy_unit,
        weight_treatment=namespace.reweighting_weight_treatment,
        cv_x=namespace.plot_cv_x,
        cv_y=namespace.plot_cv_y,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


def build_md_kl_plot_config(namespace: argparse.Namespace) -> MDKLPlotConfig:
    """Construct and validate an MDKLPlotConfig from the CLI namespace."""

    config = MDKLPlotConfig(
        reweighted_surface_csv=namespace.reweighted_surface,
        raw_frame_results_csv=namespace.raw_frame_results,
        felp_output_dir=namespace.felp_output_dir,
        topology_file=namespace.topology_file,
        trajectory_files=tuple(tuple(group) for group in namespace.trajectory_groups),
        output_dir=namespace.output_dir,
        temperature_kelvin=namespace.temperature_kelvin,
        phi_psi_residue_number=namespace.phi_psi_residue_number,
        phi_psi_residue_name=namespace.phi_psi_residue_name,
        phi_psi_selection=namespace.phi_psi_selection,
        plot_cv_x=namespace.plot_cv_x,
        plot_cv_y=namespace.plot_cv_y,
        distance_cv_specs=_parse_distance_cv_specs(namespace.distance_cv_tokens),
        stride=namespace.stride,
        frame_start=namespace.frame_start,
        frame_stop=namespace.frame_stop,
        chunk_size=namespace.chunk_size,
        fel_bins=namespace.fel_bins,
        fel_max_points=namespace.fel_max_points,
        fel_max_energy_kj_mol=_resolve_fel_max_energy_kj_mol(
            namespace.fel_max_energy_kj_mol,
            namespace.fel_max_energy_kcal_mol,
        ),
        plot_energy_unit=namespace.plot_energy_unit,
        fel_colormap=namespace.fel_cmap,
        weight_treatment=namespace.reweighting_weight_treatment,
        kl_scale_by_kbt=namespace.kl_scale_by_kbt,
        kl_energy_unit=namespace.kl_energy_unit,
        kl_display_threshold=namespace.kl_display_threshold,
        kl_colorbar_vmin=namespace.kl_vmin,
        kl_colorbar_vmax=namespace.kl_vmax,
        probability_colormap=namespace.probability_cmap,
        divergence_colormap=namespace.divergence_cmap,
        overwrite=namespace.overwrite,
    )
    config.validate()
    return config


def build_composite_figure_config(namespace: argparse.Namespace):
    """Load and optionally override a composite figure JSON config."""

    config = load_composite_figure_config(namespace.config)
    if namespace.overwrite:
        object.__setattr__(config, "overwrite", True)
    config.validate()
    return config


def build_ess_compare_config(namespace: argparse.Namespace) -> ESSCompareConfig:
    """Construct and validate an ESSCompareConfig from the CLI namespace."""

    labels = tuple(namespace.labels) if namespace.labels else None
    config = ESSCompareConfig(
        input_paths=tuple(namespace.ess_inputs),
        output_dir=namespace.output_dir,
        labels=labels,
        overwrite=namespace.overwrite,
        bootstrap_block_size=namespace.bootstrap_blocks,
        bootstrap_draws=namespace.bootstrap_draws,
        bootstrap_seed=namespace.bootstrap_seed,
        panel_layout=namespace.panel_layout,
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
