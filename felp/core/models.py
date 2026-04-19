"""Core data models for FELP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
import os
from typing import Any, Optional

from .exceptions import ConfigurationError

K_BOLTZMANN_KJ_PER_MOL_K = 0.00831446261815324
CANONICAL_RESIDUE_TARGET_NAMES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
COMMON_VARIANT_RESIDUE_TARGET_NAMES = frozenset(
    {"ASH", "CYM", "CYX", "GLH", "HID", "HIE", "HIP", "LYN"}
)
ALLOWED_MUTATION_TARGET_NAMES = (
    CANONICAL_RESIDUE_TARGET_NAMES | COMMON_VARIANT_RESIDUE_TARGET_NAMES
)
MUTATION_TARGET_NAME_ALIASES = {
    "CYD": "CYX",
    "CYZ": "CYM",
    "HIS_D": "HID",
    "HIS_E": "HIE",
    "HIS_P": "HIP",
    "LYP": "LYS",
}
MUTATION_TARGET_FAMILIES = {
    "ASH": "ASP",
    "ASP": "ASP",
    "CYM": "CYS",
    "CYS": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "GLU": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HIS": "HIS",
    "LYN": "LYS",
    "LYS": "LYS",
}
ROSETTA_RESIDUE_NAME_CANDIDATES = {
    "ASH": ("ASP_P2", "ASP_P1", "ASH"),
    "CYM": ("CYZ", "CYM"),
    "CYX": ("CYD", "CYX"),
    "GLH": ("GLU_P2", "GLU_P1", "GLH"),
    "HID": ("HIS_D", "HID"),
    "HIE": ("HIS_E", "HIE"),
    "HIP": ("HIS_P", "HIP"),
}
BACKBONE_DIHEDRAL_KINDS = frozenset({"phi", "psi", "omega"})


def normalize_angle_degrees(angle: float) -> float:
    """Normalize an angle to ``[-180, 180)``."""

    normalized = ((angle + 180.0) % 360.0) - 180.0
    if math.isclose(normalized, 180.0):
        return -180.0
    return normalized


def angular_width_degrees(lower: float, upper: float) -> float:
    """Return the width of a circular interval in degrees."""

    width = (upper - lower) % 360.0
    return 360.0 if math.isclose(width, 0.0) else width


def angular_midpoint_degrees(lower: float, upper: float) -> float:
    """Return the midpoint of a circular interval."""

    midpoint = lower + angular_width_degrees(lower, upper) / 2.0
    return normalize_angle_degrees(midpoint)


def angle_in_interval_degrees(angle: float, lower: float, upper: float) -> bool:
    """Check whether an angle falls within a circular interval."""

    relative = (normalize_angle_degrees(angle) - normalize_angle_degrees(lower)) % 360.0
    return relative <= angular_width_degrees(lower, upper)


def normalize_trajectory_groups(value: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize trajectory input into explicit groups of contiguous files."""

    if not value:
        return ()

    if isinstance(value, (str, os.PathLike)):
        return ((str(value),),)

    items = list(value)
    if not items:
        return ()

    first_item = items[0]
    if isinstance(first_item, (str, os.PathLike)):
        return (tuple(str(item) for item in items),)

    groups: list[tuple[str, ...]] = []
    for group in items:
        if isinstance(group, (str, os.PathLike)):
            groups.append((str(group),))
            continue
        normalized_group = tuple(str(path) for path in group)
        if not normalized_group:
            raise ConfigurationError("Trajectory groups may not be empty.")
        groups.append(normalized_group)
    return tuple(groups)


def flatten_trajectory_groups(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Flatten grouped trajectories into a single sequence of fragment files."""

    return tuple(path for group in groups for path in group)


def normalize_mutation_target_name(name: str) -> str:
    """Normalize a mutation target name or residue-identity expectation."""

    normalized = str(name).strip().upper()
    if not normalized:
        raise ConfigurationError("Residue names may not be empty.")
    return MUTATION_TARGET_NAME_ALIASES.get(normalized, normalized)


def mutation_target_family(name: str) -> str:
    """Return the shared residue-chemistry family for protonation variants."""

    normalized = normalize_mutation_target_name(name)
    return MUTATION_TARGET_FAMILIES.get(normalized, normalized)


def rosetta_residue_name_candidates(name: str) -> tuple[str, ...]:
    """Return candidate Rosetta residue type names for a CLI residue token."""

    normalized = normalize_mutation_target_name(name)
    return ROSETTA_RESIDUE_NAME_CANDIDATES.get(normalized, (normalized,))


@dataclass(frozen=True)
class InputConfig:
    """Configuration for input topologies and trajectories."""

    topology_file: str
    trajectory_files: tuple[tuple[str, ...], ...]
    stride: int = 1
    frame_start: int = 0
    frame_stop: Optional[int] = None
    atom_indices: tuple[int, ...] = ()
    radii_set: str = "mbondi3"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InputConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            topology_file=payload["topology_file"],
            trajectory_files=normalize_trajectory_groups(
                payload["trajectory_groups"]
                if "trajectory_groups" in payload
                else payload["trajectory_files"]
            ),
            stride=payload.get("stride", 1),
            frame_start=payload.get("frame_start", 0),
            frame_stop=payload.get("frame_stop"),
            atom_indices=tuple(payload.get("atom_indices", ())),
            radii_set=payload.get("radii_set", "mbondi3"),
        )

    @property
    def trajectory_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return explicit independent trajectories as grouped file fragments."""

        return self.trajectory_files

    @property
    def flattened_trajectory_files(self) -> tuple[str, ...]:
        """Return all underlying trajectory fragment files in order."""

        return flatten_trajectory_groups(self.trajectory_groups)

    @property
    def trajectory_count(self) -> int:
        """Return the number of independent trajectories."""

        return len(self.trajectory_groups)


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for the PyRosetta mutation."""

    target_residue_number: int
    target_residue_name: str
    original_residue_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target_residue_number < 1:
            raise ConfigurationError("Target residue number must be at least 1.")

        normalized_target = normalize_mutation_target_name(self.target_residue_name)
        if normalized_target not in ALLOWED_MUTATION_TARGET_NAMES:
            allowed = ", ".join(sorted(ALLOWED_MUTATION_TARGET_NAMES))
            raise ConfigurationError(
                f"Unsupported mutation target residue {self.target_residue_name!r}. "
                f"Allowed targets are: {allowed}."
            )
        object.__setattr__(self, "target_residue_name", normalized_target)

        if self.original_residue_name is not None:
            object.__setattr__(
                self,
                "original_residue_name",
                normalize_mutation_target_name(self.original_residue_name),
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerturbationConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            target_residue_number=payload["target_residue_number"],
            target_residue_name=payload["target_residue_name"],
            original_residue_name=payload.get("original_residue_name"),
        )


@dataclass(frozen=True)
class CVSpec:
    """Named backbone-dihedral CV specification."""

    kind: str
    residue_number: int
    name: str = ""

    def __post_init__(self) -> None:
        normalized_kind = self.kind.strip().lower()
        if normalized_kind not in BACKBONE_DIHEDRAL_KINDS:
            allowed = ", ".join(sorted(BACKBONE_DIHEDRAL_KINDS))
            raise ConfigurationError(
                f"Unsupported backbone CV kind {self.kind!r}. Allowed kinds: {allowed}."
            )
        if self.residue_number < 1:
            raise ConfigurationError("Backbone CV residue numbers must be at least 1.")
        normalized_name = self.name.strip() if self.name else f"{normalized_kind}_{self.residue_number}"
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "name", normalized_name)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CVSpec":
        """Rehydrate a CV specification from a serialized dictionary."""

        return cls(
            kind=payload["kind"],
            residue_number=payload["residue_number"],
            name=payload.get("name", ""),
        )


def default_backbone_cv_specs(target_residue_number: int) -> tuple[CVSpec, ...]:
    """Return the legacy phi/psi CV pair on the target residue."""

    return (
        CVSpec(kind="phi", residue_number=target_residue_number, name="phi"),
        CVSpec(kind="psi", residue_number=target_residue_number, name="psi"),
    )


@dataclass(frozen=True)
class RosettaConfig:
    """Configuration for PyRosetta initialization and relaxation."""

    init_options: str = "-ignore_unrecognized_res true -load_PDB_components false"
    database_path: Optional[str] = None
    score_function: str = "ref2015"
    relax_iterations: int = 5
    cartesian: bool = True
    mute_tracers: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosettaConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            init_options=payload.get(
                "init_options",
                "-ignore_unrecognized_res true -load_PDB_components false",
            ),
            database_path=payload.get("database_path"),
            score_function=payload.get("score_function", "ref2015"),
            relax_iterations=payload.get("relax_iterations", 5),
            cartesian=payload.get("cartesian", True),
            mute_tracers=payload.get("mute_tracers", False),
        )


@dataclass(frozen=True)
class RestraintConfig:
    """Configuration for reference-frame restraints.

    ``cv_measurement_tolerance_degrees`` gates post-relax (Rosetta) and post-minimize
    (OpenMM) backbone CV checks against restraint targets.
    """

    mode: str = "phi_psi"
    force_constant: float = 10000.0
    rosetta_restraint_force_constant: Optional[float] = None
    openmm_restraint_force_constant: Optional[float] = None
    cv_measurement_tolerance_degrees: float = 1.0
    cv_specs: tuple[CVSpec, ...] = ()

    def __post_init__(self) -> None:
        normalized_specs = tuple(self.cv_specs)
        names = [spec.name for spec in normalized_specs]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            duplicate_text = ", ".join(duplicates)
            raise ConfigurationError(
                f"Backbone CV names must be unique; duplicates: {duplicate_text}."
            )
        if self.force_constant <= 0.0:
            raise ConfigurationError("Restraint force_constant must be positive.")
        if self.cv_measurement_tolerance_degrees <= 0.0:
            raise ConfigurationError(
                "cv_measurement_tolerance_degrees must be positive."
            )
        if (
            self.rosetta_restraint_force_constant is not None
            and self.rosetta_restraint_force_constant <= 0.0
        ):
            raise ConfigurationError(
                "rosetta_restraint_force_constant must be positive when set."
            )
        if (
            self.openmm_restraint_force_constant is not None
            and self.openmm_restraint_force_constant <= 0.0
        ):
            raise ConfigurationError(
                "openmm_restraint_force_constant must be positive when set."
            )
        object.__setattr__(self, "cv_specs", normalized_specs)

    def effective_rosetta_force_constant(self) -> float:
        """Force constant for Rosetta dihedral constraints and CV protocol."""

        if self.rosetta_restraint_force_constant is not None:
            return float(self.rosetta_restraint_force_constant)
        return float(self.force_constant)

    def effective_openmm_force_constant(self) -> float:
        """Force constant for OpenMM CustomTorsionForce restraints."""

        if self.openmm_restraint_force_constant is not None:
            return float(self.openmm_restraint_force_constant)
        return float(self.force_constant)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RestraintConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            mode=payload.get("mode", "phi_psi"),
            force_constant=payload.get("force_constant", 10000.0),
            rosetta_restraint_force_constant=payload.get(
                "rosetta_restraint_force_constant"
            ),
            openmm_restraint_force_constant=payload.get(
                "openmm_restraint_force_constant"
            ),
            cv_measurement_tolerance_degrees=payload.get(
                "cv_measurement_tolerance_degrees", 1.0
            ),
            cv_specs=tuple(
                CVSpec.from_dict(spec_payload)
                for spec_payload in payload.get("cv_specs", ())
            ),
        )


@dataclass(frozen=True)
class EnergyConfig:
    """Configuration for the energy scoring backend."""

    backend: str = "openmm"
    protein_forcefield: str = "amber/protein.ff14SBonlysc.xml"
    solvent_forcefield: str = "implicit/gbn2.xml"
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0
    nonbonded_cutoff_nm: float = 1.0
    minimization_tolerance_kj_per_mol_nm: float = 1.0
    minimization_max_iterations: int = 0
    platform_name: Optional[str] = None
    cpu_threads: int = 1
    rosetta_score_function: str = "ref2015"

    def __post_init__(self) -> None:
        if int(self.cpu_threads) < 1:
            raise ConfigurationError("cpu_threads must be at least 1.")
        object.__setattr__(self, "cpu_threads", int(self.cpu_threads))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnergyConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            backend=payload.get("backend", "openmm"),
            protein_forcefield=payload.get(
                "protein_forcefield", "amber/protein.ff14SBonlysc.xml"
            ),
            solvent_forcefield=payload.get("solvent_forcefield", "implicit/gbn2.xml"),
            temperature_kelvin=payload.get("temperature_kelvin", 300.0),
            friction_per_ps=payload.get("friction_per_ps", 1.0),
            timestep_fs=payload.get("timestep_fs", 2.0),
            nonbonded_cutoff_nm=payload.get("nonbonded_cutoff_nm", 1.0),
            minimization_tolerance_kj_per_mol_nm=payload.get(
                "minimization_tolerance_kj_per_mol_nm", 1.0
            ),
            minimization_max_iterations=payload.get("minimization_max_iterations", 0),
            platform_name=payload.get("platform_name"),
            cpu_threads=payload.get("cpu_threads", 1),
            rosetta_score_function=payload.get("rosetta_score_function", "ref2015"),
        )


@dataclass(frozen=True)
class MicrostateConfig:
    """Configuration for microstate definitions and assignment sources."""

    source: str = "grid"
    target_residue_number: int = 2
    definition_file: Optional[str] = None
    phi_bins: int = 36
    psi_bins: int = 36
    msm_bundle_dir: Optional[str] = None
    state_level: str = "cluster"
    allow_unassigned: bool = False
    reference_probability_mode: str = "loaded"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MicrostateConfig":
        """Rehydrate the config from a serialized dictionary."""

        if "reference_probability_mode" in payload:
            reference_probability_mode = payload.get(
                "reference_probability_mode", "loaded"
            )
        elif "use_loaded_reference_probabilities" in payload:
            reference_probability_mode = (
                "loaded"
                if payload.get("use_loaded_reference_probabilities", True)
                else "raw_counts"
            )
        else:
            reference_probability_mode = (
                "raw_counts" if payload.get("source", "grid") == "grid" else "loaded"
            )
        return cls(
            source=payload.get("source", "grid"),
            target_residue_number=payload.get("target_residue_number", 2),
            definition_file=payload.get("definition_file"),
            phi_bins=payload.get("phi_bins", 36),
            psi_bins=payload.get("psi_bins", 36),
            msm_bundle_dir=payload.get("msm_bundle_dir"),
            state_level=payload.get("state_level", "cluster"),
            allow_unassigned=payload.get("allow_unassigned", False),
            reference_probability_mode=str(reference_probability_mode),
        )


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for multiprocessing execution."""

    workers: int = 1
    chunk_size: int = 1
    start_method: str = "spawn"
    progress_mode: str = "auto"
    fail_fast: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParallelConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            workers=payload.get("workers", 1),
            chunk_size=payload.get("chunk_size", 1),
            start_method=payload.get("start_method", "spawn"),
            progress_mode=payload.get("progress_mode", "auto"),
            fail_fast=payload.get("fail_fast", False),
        )


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output directories and file names."""

    output_dir: str
    raw_frame_results_name: str = "frame_results.csv"
    microstate_summary_name: str = "microstate_summary.csv"
    surface_results_name: str = "reweighted_surface.csv"
    config_dump_name: str = "run_config.json"
    summary_dump_name: str = "run_summary.json"
    write_pose_pdbs: bool = False
    overwrite: bool = False
    resume: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutputConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            output_dir=payload["output_dir"],
            raw_frame_results_name=payload.get(
                "raw_frame_results_name", "frame_results.csv"
            ),
            microstate_summary_name=payload.get(
                "microstate_summary_name", "microstate_summary.csv"
            ),
            surface_results_name=payload.get(
                "surface_results_name", "reweighted_surface.csv"
            ),
            config_dump_name=payload.get("config_dump_name", "run_config.json"),
            summary_dump_name=payload.get("summary_dump_name", "run_summary.json"),
            write_pose_pdbs=payload.get("write_pose_pdbs", False),
            overwrite=payload.get("overwrite", False),
            resume=payload.get("resume", False),
        )


@dataclass(frozen=True)
class RunConfig:
    """Strictly typed configuration for a single FELP run."""

    input: InputConfig
    perturbation: PerturbationConfig
    rosetta: RosettaConfig
    restraints: RestraintConfig
    energy: EnergyConfig
    microstates: MicrostateConfig
    parallel: ParallelConfig
    output: OutputConfig

    def validate(self) -> None:
        """Validate the run configuration before any heavy work begins."""

        if not self.output.output_dir:
            raise ConfigurationError("An output directory is required.")
        if not os.path.isfile(self.input.topology_file):
            raise ConfigurationError(
                f"Topology file does not exist: {self.input.topology_file}"
            )
        if not self.input.trajectory_files:
            raise ConfigurationError("At least one trajectory file is required.")
        for trajectory_group in self.input.trajectory_groups:
            if not trajectory_group:
                raise ConfigurationError("Trajectory groups may not be empty.")
            for trajectory_file in trajectory_group:
                if not os.path.isfile(trajectory_file):
                    raise ConfigurationError(
                        f"Trajectory file does not exist: {trajectory_file}"
                    )
        if self.input.atom_indices:
            raise ConfigurationError(
                "Atom-subset trajectories are not supported in the rewritten pipeline."
            )
        if self.input.stride < 1:
            raise ConfigurationError("Stride must be at least 1.")
        if self.input.frame_start < 0:
            raise ConfigurationError("frame_start must be non-negative.")
        if (
            self.input.frame_stop is not None
            and self.input.frame_stop < self.input.frame_start
        ):
            raise ConfigurationError("frame_stop must be greater than frame_start.")
        if not self.restraints.cv_specs:
            raise ConfigurationError(
                "At least one backbone CV specification is required."
            )
        if self.energy.backend not in {"openmm", "rosetta"}:
            raise ConfigurationError(
                f"Unsupported energy backend: {self.energy.backend}"
            )
        if self.microstates.source not in {"grid", "file", "msm_bundle"}:
            raise ConfigurationError(
                f"Unsupported microstate source: {self.microstates.source}"
            )
        if self.microstates.source == "file" and not self.microstates.definition_file:
            raise ConfigurationError(
                "A microstate definition file is required when source='file'."
            )
        if (
            self.microstates.source == "msm_bundle"
            and not self.microstates.msm_bundle_dir
        ):
            raise ConfigurationError(
                "An MSM bundle directory is required when source='msm_bundle'."
            )
        if self.microstates.definition_file and not os.path.isfile(
            self.microstates.definition_file
        ):
            raise ConfigurationError(
                "Microstate definition file does not exist: "
                f"{self.microstates.definition_file}"
            )
        if self.microstates.msm_bundle_dir and not os.path.isdir(
            self.microstates.msm_bundle_dir
        ):
            raise ConfigurationError(
                f"MSM bundle directory does not exist: {self.microstates.msm_bundle_dir}"
            )
        if self.microstates.state_level not in {"cluster", "metastable"}:
            raise ConfigurationError(
                f"Unsupported state level: {self.microstates.state_level}"
            )
        if self.microstates.reference_probability_mode not in {"loaded", "raw_counts"}:
            raise ConfigurationError(
                "microstates.reference_probability_mode must be one of: "
                "loaded, raw_counts."
            )
        if (
            self.microstates.source == "grid"
            and self.microstates.reference_probability_mode == "loaded"
        ):
            raise ConfigurationError(
                "Grid microstates do not embed reference probabilities. Use "
                "microstates.reference_probability_mode='raw_counts' or switch to "
                "source='file'/'msm_bundle' with loaded priors."
            )
        if self.parallel.workers < 1:
            raise ConfigurationError("Parallel worker count must be at least 1.")
        if self.parallel.chunk_size < 1:
            raise ConfigurationError("Chunk size must be at least 1.")
        if self.parallel.progress_mode not in {"auto", "on", "off"}:
            raise ConfigurationError(
                "parallel.progress_mode must be one of: auto, on, off."
            )
        if self.microstates.source in {"grid", "msm_bundle"}:
            if self.restraints.cv_specs != default_backbone_cv_specs(
                self.perturbation.target_residue_number
            ):
                raise ConfigurationError(
                    "Grid and msm_bundle microstate modes currently require the legacy "
                    "phi/psi CV pair on the target residue. Use source='file' for "
                    "custom backbone-CV state definitions."
                )

    def beta(self) -> float:
        """Return the inverse thermal energy used in Zwanzig reweighting."""

        return 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * self.energy.temperature_kelvin)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        """Rehydrate the config from a serialized dictionary."""

        perturbation = PerturbationConfig.from_dict(payload["perturbation"])
        restraints = RestraintConfig.from_dict(payload["restraints"])
        if not restraints.cv_specs:
            restraints = RestraintConfig(
                mode=restraints.mode,
                force_constant=restraints.force_constant,
                rosetta_restraint_force_constant=restraints.rosetta_restraint_force_constant,
                openmm_restraint_force_constant=restraints.openmm_restraint_force_constant,
                cv_measurement_tolerance_degrees=restraints.cv_measurement_tolerance_degrees,
                cv_specs=default_backbone_cv_specs(
                    perturbation.target_residue_number
                ),
            )
        return cls(
            input=InputConfig.from_dict(payload["input"]),
            perturbation=perturbation,
            rosetta=RosettaConfig.from_dict(payload["rosetta"]),
            restraints=restraints,
            energy=EnergyConfig.from_dict(payload["energy"]),
            microstates=MicrostateConfig.from_dict(payload["microstates"]),
            parallel=ParallelConfig.from_dict(payload["parallel"]),
            output=OutputConfig.from_dict(payload["output"]),
        )


@dataclass(frozen=True)
class SnapshotJob:
    """A single frame job emitted by the trajectory loader."""

    frame_index: int
    stride_number: int
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    trajectory_id: int = 0
    source_frame_index: Optional[int] = None
    precomputed_state_id: Optional[str] = None


@dataclass(frozen=True)
class PhiPsiMeasurement:
    """Container for a measured phi/psi angle pair."""

    phi_degrees: float
    psi_degrees: float

    def as_cv_measurement(self) -> "CVMeasurement":
        """Promote the legacy measurement to the generalized CV container."""

        return CVMeasurement({"phi": self.phi_degrees, "psi": self.psi_degrees})


@dataclass(frozen=True)
class CVMeasurement:
    """Measured values for one or more named backbone-dihedral CVs."""

    values: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            {str(name): float(value) for name, value in self.values.items()},
        )

    def as_dict(self) -> dict[str, float]:
        """Return a shallow copy of the measured CV map."""

        return dict(self.values)

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Return a measured value by CV name."""

        return self.values.get(name, default)

    @property
    def phi_degrees(self) -> Optional[float]:
        """Legacy phi accessor when present."""

        return self.values.get("phi")

    @property
    def psi_degrees(self) -> Optional[float]:
        """Legacy psi accessor when present."""

        return self.values.get("psi")

    @property
    def omega_degrees(self) -> Optional[float]:
        """Legacy omega accessor when present."""

        return self.values.get("omega")


@dataclass(frozen=True)
class DihedralRestraintSpec:
    """Specification for a single dihedral restraint to apply during relaxation."""

    label: str
    kind: str
    residue_numbers: tuple[int, int, int, int]
    atom_names: tuple[str, str, str, str]
    target_angle_degrees: float
    force_constant: float


@dataclass(frozen=True)
class MicrostateDefinition:
    """Definition of a geometric or discrete state used for FELP aggregation."""

    state_id: str
    label: str
    phi_range: Optional[tuple[float, float]] = None
    psi_range: Optional[tuple[float, float]] = None
    cv_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    reference_probability: Optional[float] = None
    plot_x: Optional[float] = None
    plot_y: Optional[float] = None
    state_level: str = "microstate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_ranges = {
            str(name): tuple(float(value) for value in bounds)
            for name, bounds in self.cv_ranges.items()
        }
        if self.phi_range is not None and "phi" not in normalized_ranges:
            normalized_ranges["phi"] = tuple(float(value) for value in self.phi_range)
        if self.psi_range is not None and "psi" not in normalized_ranges:
            normalized_ranges["psi"] = tuple(float(value) for value in self.psi_range)
        object.__setattr__(self, "cv_ranges", normalized_ranges)
        if self.phi_range is None and "phi" in normalized_ranges:
            object.__setattr__(self, "phi_range", normalized_ranges["phi"])
        if self.psi_range is None and "psi" in normalized_ranges:
            object.__setattr__(self, "psi_range", normalized_ranges["psi"])

    @property
    def phi_center(self) -> Optional[float]:
        """Return the Ramachandran phi center when it exists."""

        phi_range = self.cv_ranges.get("phi", self.phi_range)
        if phi_range is None:
            return None
        return angular_midpoint_degrees(*phi_range)

    @property
    def psi_center(self) -> Optional[float]:
        """Return the Ramachandran psi center when it exists."""

        psi_range = self.cv_ranges.get("psi", self.psi_range)
        if psi_range is None:
            return None
        return angular_midpoint_degrees(*psi_range)

    def contains(
        self,
        measurements_or_phi: CVMeasurement | Mapping[str, float] | float,
        psi_degrees: Optional[float] = None,
    ) -> bool:
        """Check whether a measurement falls inside the definition."""

        if not self.cv_ranges:
            return False
        if isinstance(measurements_or_phi, CVMeasurement):
            measurements = measurements_or_phi.values
        elif isinstance(measurements_or_phi, Mapping):
            measurements = measurements_or_phi
        elif psi_degrees is not None:
            measurements = {
                "phi": float(measurements_or_phi),
                "psi": float(psi_degrees),
            }
        else:
            raise TypeError("contains() expects a CV measurement or a phi/psi pair.")

        for name, bounds in self.cv_ranges.items():
            value = measurements.get(name)
            if value is None or not angle_in_interval_degrees(value, *bounds):
                return False
        return True


@dataclass(frozen=True)
class MicrostateAssignment:
    """Assignment of a frame to a state, plus measured geometry for provenance."""

    state_id: str
    label: str
    phi_degrees: Optional[float] = None
    psi_degrees: Optional[float] = None
    measurements: dict[str, float] = field(default_factory=dict)
    phi_center: Optional[float] = None
    psi_center: Optional[float] = None
    plot_x: Optional[float] = None
    plot_y: Optional[float] = None
    state_level: str = "microstate"

    def __post_init__(self) -> None:
        measurements = {
            str(name): float(value) for name, value in self.measurements.items()
        }
        if self.phi_degrees is not None:
            measurements.setdefault("phi", float(self.phi_degrees))
        if self.psi_degrees is not None:
            measurements.setdefault("psi", float(self.psi_degrees))
        object.__setattr__(self, "measurements", measurements)
        if self.phi_degrees is None and "phi" in measurements:
            object.__setattr__(self, "phi_degrees", measurements["phi"])
        if self.psi_degrees is None and "psi" in measurements:
            object.__setattr__(self, "psi_degrees", measurements["psi"])


@dataclass(frozen=True)
class EnergyResult:
    """Result of an energy evaluation on a structure or pose."""

    backend: str
    energy_kj_mol: float
    minimized_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameResult:
    """Complete result for a single frame, including energies and state assignment."""

    frame_index: int
    stride_number: int
    microstate_id: str
    microstate_label: str
    phi_degrees: Optional[float]
    psi_degrees: Optional[float]
    reference_energy_kj_mol: float
    perturbed_energy_kj_mol: float
    delta_u_kj_mol: float
    reference_relaxed_phi_degrees: Optional[float]
    reference_relaxed_psi_degrees: Optional[float]
    perturbed_relaxed_phi_degrees: Optional[float]
    perturbed_relaxed_psi_degrees: Optional[float]
    trajectory_id: Optional[int] = None
    source_frame_index: Optional[int] = None
    assignment_source: str = "phi_psi"
    state_level: str = "microstate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Serialize the result to a CSV-friendly row."""

        row = asdict(self)
        row["metadata"] = self.metadata
        return row


@dataclass(frozen=True)
class MicrostateResult:
    """Aggregated reweighting result for a single state."""

    state_id: str
    label: str
    phi_center: Optional[float]
    psi_center: Optional[float]
    frame_count: int
    reference_probability: float
    reweight_factor: float
    target_probability: float
    free_energy_kj_mol: Optional[float]
    effective_sample_size: float
    mean_delta_u_kj_mol: Optional[float]
    std_delta_u_kj_mol: Optional[float]
    mean_boltzmann_factor: Optional[float]
    std_boltzmann_factor: Optional[float]
    plot_x: Optional[float] = None
    plot_y: Optional[float] = None
    state_level: str = "microstate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Serialize the result to a CSV-friendly row."""

        row = asdict(self)
        row["metadata"] = self.metadata
        return row
