"""Core data models for FELP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from typing import Any, Optional

from .exceptions import ConfigurationError

K_BOLTZMANN_KJ_PER_MOL_K = 0.00831446261815324


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerturbationConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            target_residue_number=payload["target_residue_number"],
            target_residue_name=payload["target_residue_name"],
            original_residue_name=payload.get("original_residue_name"),
        )


@dataclass(frozen=True)
class RosettaConfig:
    """Configuration for PyRosetta initialization and relaxation."""

    init_options: str = "-ignore_unrecognized_res true -load_PDB_components false"
    database_path: Optional[str] = None
    score_function: str = "ref2015"
    relax_iterations: int = 5
    cartesian: bool = True

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
        )


@dataclass(frozen=True)
class RestraintConfig:
    """Configuration for reference-frame restraints."""

    mode: str = "phi_psi"
    force_constant: float = 10.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RestraintConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            mode=payload.get("mode", "phi_psi"),
            force_constant=payload.get("force_constant", 10.0),
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
    rosetta_score_function: str = "ref2015"

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
    use_loaded_reference_probabilities: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MicrostateConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            source=payload.get("source", "grid"),
            target_residue_number=payload.get("target_residue_number", 2),
            definition_file=payload.get("definition_file"),
            phi_bins=payload.get("phi_bins", 36),
            psi_bins=payload.get("psi_bins", 36),
            msm_bundle_dir=payload.get("msm_bundle_dir"),
            state_level=payload.get("state_level", "cluster"),
            allow_unassigned=payload.get("allow_unassigned", False),
            use_loaded_reference_probabilities=payload.get(
                "use_loaded_reference_probabilities", True
            ),
        )


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for multiprocessing execution."""

    workers: int = 1
    chunk_size: int = 1
    start_method: str = "spawn"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParallelConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            workers=payload.get("workers", 1),
            chunk_size=payload.get("chunk_size", 1),
            start_method=payload.get("start_method", "spawn"),
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
        if self.perturbation.target_residue_number < 2:
            raise ConfigurationError(
                "Target residue number must be at least 2 for phi/psi restraints."
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
        if self.parallel.workers < 1:
            raise ConfigurationError("Parallel worker count must be at least 1.")
        if self.parallel.chunk_size < 1:
            raise ConfigurationError("Chunk size must be at least 1.")

    def beta(self) -> float:
        """Return the inverse thermal energy used in Zwanzig reweighting."""

        return 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * self.energy.temperature_kelvin)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            input=InputConfig.from_dict(payload["input"]),
            perturbation=PerturbationConfig.from_dict(payload["perturbation"]),
            rosetta=RosettaConfig.from_dict(payload["rosetta"]),
            restraints=RestraintConfig.from_dict(payload["restraints"]),
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


@dataclass(frozen=True)
class DihedralRestraintSpec:
    """Specification for a single dihedral restraint to apply during relaxation."""

    label: str
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
    reference_probability: Optional[float] = None
    plot_x: Optional[float] = None
    plot_y: Optional[float] = None
    state_level: str = "microstate"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phi_center(self) -> Optional[float]:
        """Return the Ramachandran phi center when it exists."""

        if self.phi_range is None:
            return None
        return angular_midpoint_degrees(*self.phi_range)

    @property
    def psi_center(self) -> Optional[float]:
        """Return the Ramachandran psi center when it exists."""

        if self.psi_range is None:
            return None
        return angular_midpoint_degrees(*self.psi_range)

    def contains(self, phi_degrees: float, psi_degrees: float) -> bool:
        """Check whether a phi/psi pair falls inside the definition."""

        if self.phi_range is None or self.psi_range is None:
            return False
        return angle_in_interval_degrees(
            phi_degrees, *self.phi_range
        ) and angle_in_interval_degrees(psi_degrees, *self.psi_range)


@dataclass(frozen=True)
class MicrostateAssignment:
    """Assignment of a frame to a state, plus measured geometry for provenance."""

    state_id: str
    label: str
    phi_degrees: float
    psi_degrees: float
    phi_center: Optional[float] = None
    psi_center: Optional[float] = None
    plot_x: Optional[float] = None
    plot_y: Optional[float] = None
    state_level: str = "microstate"


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
    phi_degrees: float
    psi_degrees: float
    reference_energy_kj_mol: float
    perturbed_energy_kj_mol: float
    delta_u_kj_mol: float
    reference_relaxed_phi_degrees: float
    reference_relaxed_psi_degrees: float
    perturbed_relaxed_phi_degrees: float
    perturbed_relaxed_psi_degrees: float
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
