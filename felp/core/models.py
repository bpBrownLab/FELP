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


def cv_value_in_interval(name: str, value: float, lower: float, upper: float) -> bool:
    """Check whether a CV value falls inside the interval appropriate for that CV."""

    if name in BACKBONE_DIHEDRAL_KINDS:
        return angle_in_interval_degrees(value, lower, upper)
    return float(lower) <= float(value) <= float(upper)


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


@dataclass(frozen=True)
class AtomReference:
    """Pose-indexed atom reference used by generic CV definitions."""

    residue_number: int
    atom_name: str

    def __post_init__(self) -> None:
        if int(self.residue_number) < 1:
            raise ConfigurationError("Atom-reference residue numbers must be at least 1.")
        atom_name = self.atom_name.strip()
        if not atom_name:
            raise ConfigurationError("Atom-reference atom names may not be empty.")
        object.__setattr__(self, "residue_number", int(self.residue_number))
        object.__setattr__(self, "atom_name", atom_name)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AtomReference":
        """Rehydrate an atom reference from a serialized dictionary."""

        return cls(
            residue_number=payload["residue_number"],
            atom_name=payload["atom_name"],
        )


@dataclass(frozen=True)
class DistanceCVSpec:
    """Named distance CV between two pose-indexed atoms."""

    name: str
    atom1: AtomReference
    atom2: AtomReference

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ConfigurationError("Distance CV names may not be empty.")
        object.__setattr__(self, "name", name)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistanceCVSpec":
        """Rehydrate a distance CV specification from a serialized dictionary."""

        return cls(
            name=payload["name"],
            atom1=AtomReference.from_dict(payload["atom1"]),
            atom2=AtomReference.from_dict(payload["atom2"]),
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
    perturbation_mode: str = "relax"
    relax_iterations: int = 5
    repack_trials: int = 1
    cartesian: bool = True
    symmetric_gly_tables: bool = False
    mute_tracers: bool = False

    def __post_init__(self) -> None:
        if self.perturbation_mode not in {"relax", "repack"}:
            raise ConfigurationError(
                "rosetta.perturbation_mode must be 'relax' or 'repack'."
            )
        if int(self.relax_iterations) < 1:
            raise ConfigurationError("rosetta.relax_iterations must be at least 1.")
        if int(self.repack_trials) < 1:
            raise ConfigurationError("rosetta.repack_trials must be at least 1.")
        object.__setattr__(self, "relax_iterations", int(self.relax_iterations))
        object.__setattr__(self, "repack_trials", int(self.repack_trials))

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
            perturbation_mode=payload.get("perturbation_mode", "relax"),
            relax_iterations=payload.get("relax_iterations", 5),
            repack_trials=payload.get("repack_trials", 1),
            cartesian=payload.get("cartesian", True),
            symmetric_gly_tables=payload.get("symmetric_gly_tables", False),
            mute_tracers=payload.get("mute_tracers", False),
        )


@dataclass(frozen=True)
class RestraintConfig:
    """Configuration for reference-frame restraints.

    ``cv_measurement_tolerance_degrees`` gates post-relax (Rosetta) and post-minimize
    (OpenMM) backbone CV checks against restraint targets.

    Rosetta ``AtomPairConstraint`` + ``HarmonicFunc`` use a width parameter (σ in Å),
    not a harmonic spring constant; ``rosetta_distance_constraint_sd_angstrom`` sets
    that width during FastRelax/repack. OpenMM distance restraints use
    ``effective_openmm_force_constant()`` as the spring ``k`` (see
    ``--openmm-restraint-force-constant``).
    """

    mode: str = "phi_psi"
    force_constant: float = 10000.0
    rosetta_restraint_force_constant: Optional[float] = None
    openmm_restraint_force_constant: Optional[float] = None
    cv_measurement_tolerance_degrees: float = 1.0
    distance_measurement_tolerance_angstrom: float = 0.05
    rosetta_distance_constraint_sd_angstrom: float = 0.001
    rosetta_target_backbone_restraints: tuple[str, ...] = ()
    rosetta_backbone_restraint_residue_numbers: tuple[int, ...] = ()
    rosetta_target_backbone_restraint_force_constant: Optional[float] = None
    rosetta_target_backbone_restraint_tolerance_degrees: Optional[float] = None
    cv_specs: tuple[CVSpec, ...] = ()
    distance_cv_specs: tuple[DistanceCVSpec, ...] = ()

    def __post_init__(self) -> None:
        normalized_specs = tuple(self.cv_specs)
        normalized_distance_specs = tuple(self.distance_cv_specs)
        normalized_target_backbone_restraints = tuple(
            dict.fromkeys(
                str(kind).strip().lower()
                for kind in self.rosetta_target_backbone_restraints
                if str(kind).strip()
            )
        )
        unsupported_target_backbone_restraints = sorted(
            set(normalized_target_backbone_restraints) - BACKBONE_DIHEDRAL_KINDS
        )
        normalized_backbone_restraint_residues = tuple(
            dict.fromkeys(
                int(residue_number)
                for residue_number in self.rosetta_backbone_restraint_residue_numbers
            )
        )
        if unsupported_target_backbone_restraints:
            unsupported = ", ".join(unsupported_target_backbone_restraints)
            raise ConfigurationError(
                "rosetta_target_backbone_restraints may only contain phi, psi, "
                f"or omega; unsupported values: {unsupported}."
            )
        if any(residue_number < 1 for residue_number in normalized_backbone_restraint_residues):
            raise ConfigurationError(
                "rosetta_backbone_restraint_residue_numbers must be positive pose indices."
            )
        names = [spec.name for spec in normalized_specs] + [
            spec.name for spec in normalized_distance_specs
        ]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            duplicate_text = ", ".join(duplicates)
            raise ConfigurationError(
                f"CV names must be unique; duplicates: {duplicate_text}."
            )
        if self.force_constant <= 0.0:
            raise ConfigurationError("Restraint force_constant must be positive.")
        if self.cv_measurement_tolerance_degrees <= 0.0:
            raise ConfigurationError(
                "cv_measurement_tolerance_degrees must be positive."
            )
        if self.distance_measurement_tolerance_angstrom <= 0.0:
            raise ConfigurationError(
                "distance_measurement_tolerance_angstrom must be positive."
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
        if self.rosetta_distance_constraint_sd_angstrom <= 0.0:
            raise ConfigurationError(
                "rosetta_distance_constraint_sd_angstrom must be positive."
            )
        if (
            self.rosetta_target_backbone_restraint_force_constant is not None
            and self.rosetta_target_backbone_restraint_force_constant <= 0.0
        ):
            raise ConfigurationError(
                "rosetta_target_backbone_restraint_force_constant must be positive when set."
            )
        if (
            self.rosetta_target_backbone_restraint_tolerance_degrees is not None
            and self.rosetta_target_backbone_restraint_tolerance_degrees <= 0.0
        ):
            raise ConfigurationError(
                "rosetta_target_backbone_restraint_tolerance_degrees must be positive when set."
            )
        object.__setattr__(self, "cv_specs", normalized_specs)
        object.__setattr__(self, "distance_cv_specs", normalized_distance_specs)
        object.__setattr__(
            self,
            "rosetta_target_backbone_restraints",
            normalized_target_backbone_restraints,
        )
        object.__setattr__(
            self,
            "rosetta_backbone_restraint_residue_numbers",
            normalized_backbone_restraint_residues,
        )

    def effective_rosetta_force_constant(self) -> float:
        """Weighting for Rosetta CircularHarmonic dihedral constraints / CV protocol.

        Not used for Rosetta atom-pair distance widths; see
        ``rosetta_distance_constraint_sd_angstrom``.
        """

        if self.rosetta_restraint_force_constant is not None:
            return float(self.rosetta_restraint_force_constant)
        return float(self.force_constant)

    def effective_openmm_force_constant(self) -> float:
        """Force constant for OpenMM CustomTorsionForce restraints."""

        if self.openmm_restraint_force_constant is not None:
            return float(self.openmm_restraint_force_constant)
        return float(self.force_constant)

    def effective_rosetta_target_backbone_force_constant(self) -> float:
        """Force constant for Rosetta-only target-backbone restraints."""

        if self.rosetta_target_backbone_restraint_force_constant is not None:
            return float(self.rosetta_target_backbone_restraint_force_constant)
        return self.effective_rosetta_force_constant()

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
            distance_measurement_tolerance_angstrom=payload.get(
                "distance_measurement_tolerance_angstrom", 0.05
            ),
            rosetta_distance_constraint_sd_angstrom=payload.get(
                "rosetta_distance_constraint_sd_angstrom", 0.001
            ),
            rosetta_target_backbone_restraints=tuple(
                payload.get("rosetta_target_backbone_restraints", ())
            ),
            rosetta_backbone_restraint_residue_numbers=tuple(
                payload.get("rosetta_backbone_restraint_residue_numbers", ())
            ),
            rosetta_target_backbone_restraint_force_constant=payload.get(
                "rosetta_target_backbone_restraint_force_constant"
            ),
            rosetta_target_backbone_restraint_tolerance_degrees=payload.get(
                "rosetta_target_backbone_restraint_tolerance_degrees"
            ),
            cv_specs=tuple(
                CVSpec.from_dict(spec_payload)
                for spec_payload in payload.get("cv_specs", ())
            ),
            distance_cv_specs=tuple(
                DistanceCVSpec.from_dict(spec_payload)
                for spec_payload in payload.get("distance_cv_specs", ())
            ),
        )


@dataclass(frozen=True)
class MobilityConfig:
    """Configuration for which residues/atoms may relax after mutation."""

    mode: str = "target_only"
    radius_angstrom: float = 0.0
    sequence_window: int = 0
    include_backbone: bool = False
    include_sidechains: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"target_only", "radius", "sequence_window"}:
            raise ConfigurationError(
                "mobility.mode must be 'target_only', 'radius', or 'sequence_window'."
            )
        if float(self.radius_angstrom) < 0.0:
            raise ConfigurationError("mobility.radius_angstrom must be non-negative.")
        if int(self.sequence_window) < 0:
            raise ConfigurationError("mobility.sequence_window must be non-negative.")
        if not self.include_backbone and not self.include_sidechains:
            raise ConfigurationError(
                "At least one of mobility.include_backbone/include_sidechains must be enabled."
            )
        object.__setattr__(self, "radius_angstrom", float(self.radius_angstrom))
        object.__setattr__(self, "sequence_window", int(self.sequence_window))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MobilityConfig":
        """Rehydrate mobility configuration from a serialized dictionary."""

        return cls(
            mode=payload.get("mode", "target_only"),
            radius_angstrom=payload.get("radius_angstrom", 0.0),
            sequence_window=payload.get("sequence_window", 0),
            include_backbone=payload.get("include_backbone", False),
            include_sidechains=payload.get("include_sidechains", True),
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
    auto_target_frames_per_state: int = 100
    auto_min_bins_per_axis: int = 3
    auto_max_bins_per_axis: int = 64
    auto_max_total_states: int = 2500

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
            auto_target_frames_per_state=payload.get(
                "auto_target_frames_per_state", 100
            ),
            auto_min_bins_per_axis=payload.get("auto_min_bins_per_axis", 3),
            auto_max_bins_per_axis=payload.get("auto_max_bins_per_axis", 64),
            auto_max_total_states=payload.get("auto_max_total_states", 2500),
        )


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for microstate-stratified evaluation sampling."""

    microstate_sample_fraction: float = 1.0
    microstate_min_samples: int = 1
    microstate_max_samples: Optional[int] = None
    microstate_sampling_seed: int = 0
    sampling_manifest_name: str = "sampling_manifest.json"

    def __post_init__(self) -> None:
        fraction = float(self.microstate_sample_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError(
                "microstate_sample_fraction must be in the interval (0, 1]."
            )
        min_samples = int(self.microstate_min_samples)
        if min_samples < 0:
            raise ConfigurationError("microstate_min_samples must be non-negative.")
        max_samples = self.microstate_max_samples
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < min_samples:
                raise ConfigurationError(
                    "microstate_max_samples must be greater than or equal to "
                    "microstate_min_samples when set."
                )
        seed = int(self.microstate_sampling_seed)
        manifest_name = self.sampling_manifest_name.strip()
        if not manifest_name:
            raise ConfigurationError("sampling_manifest_name may not be empty.")
        object.__setattr__(self, "microstate_sample_fraction", fraction)
        object.__setattr__(self, "microstate_min_samples", min_samples)
        object.__setattr__(self, "microstate_max_samples", max_samples)
        object.__setattr__(self, "microstate_sampling_seed", seed)
        object.__setattr__(self, "sampling_manifest_name", manifest_name)

    @property
    def is_noop(self) -> bool:
        """Return whether sampling preserves all assigned frames."""

        return (
            self.microstate_sample_fraction == 1.0
            and self.microstate_max_samples is None
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SamplingConfig":
        """Rehydrate the config from a serialized dictionary."""

        return cls(
            microstate_sample_fraction=payload.get("microstate_sample_fraction", 1.0),
            microstate_min_samples=payload.get("microstate_min_samples", 1),
            microstate_max_samples=payload.get("microstate_max_samples"),
            microstate_sampling_seed=payload.get("microstate_sampling_seed", 0),
            sampling_manifest_name=payload.get(
                "sampling_manifest_name", "sampling_manifest.json"
            ),
        )


@dataclass(frozen=True)
class ReweightingQCConfig:
    """Configuration for by-state reweighting degeneracy checks."""

    policy: str = "flag"
    method: str = "auto"
    weight_treatment: str = "raw"
    min_frame_count: int = 20
    max_frame_weight_fraction: float = 0.5
    min_state_ess: float = 5.0
    max_leave_one_out_shift_kj_mol: float = 10.0
    max_excluded_frames_per_state: int = 5

    @staticmethod
    def normalize_weight_treatment(raw: str) -> str:
        """Normalize reweighting weight-treatment aliases."""

        weight_treatment = str(raw).strip()
        if weight_treatment == "psis":
            return "psis-smoothed"
        return weight_treatment

    def __post_init__(self) -> None:
        policy = str(self.policy).strip()
        if policy not in {"none", "flag", "exclude-frames"}:
            raise ConfigurationError(
                "reweighting QC policy must be one of: none, flag, exclude-frames."
            )
        method = str(self.method).strip()
        if method not in {"auto", "threshold"}:
            raise ConfigurationError(
                "reweighting QC method must be one of: auto, threshold."
            )
        weight_treatment = self.normalize_weight_treatment(self.weight_treatment)
        if weight_treatment not in {"raw", "psis-smoothed"}:
            raise ConfigurationError(
                "reweighting weight treatment must be one of: raw, psis-smoothed "
                "(alias: psis)."
            )
        min_frame_count = int(self.min_frame_count)
        if min_frame_count < 0:
            raise ConfigurationError("reweighting QC min_frame_count must be non-negative.")
        max_frame_weight_fraction = float(self.max_frame_weight_fraction)
        if not 0.0 < max_frame_weight_fraction <= 1.0:
            raise ConfigurationError(
                "reweighting QC max_frame_weight_fraction must be in the interval (0, 1]."
            )
        min_state_ess = float(self.min_state_ess)
        if min_state_ess < 0.0:
            raise ConfigurationError("reweighting QC min_state_ess must be non-negative.")
        max_leave_one_out_shift = float(self.max_leave_one_out_shift_kj_mol)
        if max_leave_one_out_shift < 0.0:
            raise ConfigurationError(
                "reweighting QC max_leave_one_out_shift_kj_mol must be non-negative."
            )
        max_excluded = int(self.max_excluded_frames_per_state)
        if max_excluded < 0:
            raise ConfigurationError(
                "reweighting QC max_excluded_frames_per_state must be non-negative."
            )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "weight_treatment", weight_treatment)
        object.__setattr__(self, "min_frame_count", min_frame_count)
        object.__setattr__(
            self, "max_frame_weight_fraction", max_frame_weight_fraction
        )
        object.__setattr__(self, "min_state_ess", min_state_ess)
        object.__setattr__(
            self, "max_leave_one_out_shift_kj_mol", max_leave_one_out_shift
        )
        object.__setattr__(self, "max_excluded_frames_per_state", max_excluded)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReweightingQCConfig":
        """Rehydrate reweighting QC configuration from a serialized dictionary."""

        return cls(
            policy=payload.get("policy", "flag"),
            method=payload.get("method", "auto"),
            weight_treatment=payload.get("weight_treatment", "raw"),
            min_frame_count=payload.get("min_frame_count", 20),
            max_frame_weight_fraction=payload.get(
                "max_frame_weight_fraction", 0.5
            ),
            min_state_ess=payload.get("min_state_ess", 5.0),
            max_leave_one_out_shift_kj_mol=payload.get(
                "max_leave_one_out_shift_kj_mol", 10.0
            ),
            max_excluded_frames_per_state=payload.get(
                "max_excluded_frames_per_state", 5
            ),
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
    debug_frame_outputs: bool = False
    debug_frame_indices: tuple[int, ...] = ()
    overwrite: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        normalized_indices = tuple(
            sorted({int(index) for index in self.debug_frame_indices})
        )
        if any(index < 0 for index in normalized_indices):
            raise ConfigurationError("debug_frame_indices must be non-negative.")
        object.__setattr__(self, "debug_frame_indices", normalized_indices)

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
            debug_frame_outputs=payload.get("debug_frame_outputs", False),
            debug_frame_indices=tuple(payload.get("debug_frame_indices", ())),
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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reweighting_qc: ReweightingQCConfig = field(default_factory=ReweightingQCConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)

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
        if not self.restraints.cv_specs and not self.restraints.distance_cv_specs:
            raise ConfigurationError(
                "At least one CV specification is required."
            )
        if self.energy.backend not in {"openmm", "rosetta"}:
            raise ConfigurationError(
                f"Unsupported energy backend: {self.energy.backend}"
            )
        if self.microstates.source not in {"grid", "file", "msm_bundle", "auto"}:
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
        if (
            self.microstates.source == "auto"
            and self.microstates.reference_probability_mode != "raw_counts"
        ):
            raise ConfigurationError(
                "Automatic microstate generation requires raw-count priors."
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
        if self.microstates.auto_target_frames_per_state < 1:
            raise ConfigurationError(
                "auto_target_frames_per_state must be at least 1."
            )
        if self.microstates.auto_min_bins_per_axis < 1:
            raise ConfigurationError("auto_min_bins_per_axis must be at least 1.")
        if (
            self.microstates.auto_max_bins_per_axis
            < self.microstates.auto_min_bins_per_axis
        ):
            raise ConfigurationError(
                "auto_max_bins_per_axis must be greater than or equal to "
                "auto_min_bins_per_axis."
            )
        if self.microstates.auto_max_total_states < 1:
            raise ConfigurationError("auto_max_total_states must be at least 1.")
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
                    "custom CV state definitions, including distance-CV reweighting. "
                    "If you intentionally want phi/psi microstates with additional "
                    "distance restraints, pass '--cv phi --cv psi' explicitly."
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
        if not restraints.cv_specs and not restraints.distance_cv_specs:
            restraints = RestraintConfig(
                mode=restraints.mode,
                force_constant=restraints.force_constant,
                rosetta_restraint_force_constant=restraints.rosetta_restraint_force_constant,
                openmm_restraint_force_constant=restraints.openmm_restraint_force_constant,
                cv_measurement_tolerance_degrees=restraints.cv_measurement_tolerance_degrees,
                distance_measurement_tolerance_angstrom=(
                    restraints.distance_measurement_tolerance_angstrom
                ),
                rosetta_distance_constraint_sd_angstrom=(
                    restraints.rosetta_distance_constraint_sd_angstrom
                ),
                rosetta_target_backbone_restraints=(
                    restraints.rosetta_target_backbone_restraints
                ),
                rosetta_backbone_restraint_residue_numbers=(
                    restraints.rosetta_backbone_restraint_residue_numbers
                ),
                rosetta_target_backbone_restraint_force_constant=(
                    restraints.rosetta_target_backbone_restraint_force_constant
                ),
                rosetta_target_backbone_restraint_tolerance_degrees=(
                    restraints.rosetta_target_backbone_restraint_tolerance_degrees
                ),
                cv_specs=default_backbone_cv_specs(
                    perturbation.target_residue_number
                ),
                distance_cv_specs=restraints.distance_cv_specs,
            )
        return cls(
            input=InputConfig.from_dict(payload["input"]),
            perturbation=perturbation,
            rosetta=RosettaConfig.from_dict(payload["rosetta"]),
            restraints=restraints,
            mobility=MobilityConfig.from_dict(payload.get("mobility", {})),
            energy=EnergyConfig.from_dict(payload["energy"]),
            microstates=MicrostateConfig.from_dict(payload["microstates"]),
            parallel=ParallelConfig.from_dict(payload["parallel"]),
            output=OutputConfig.from_dict(payload["output"]),
            sampling=SamplingConfig.from_dict(payload.get("sampling", {})),
            reweighting_qc=ReweightingQCConfig.from_dict(
                payload.get("reweighting_qc", {})
            ),
        )


@dataclass(frozen=True)
class SnapshotJob:
    """A single frame job emitted by the trajectory loader."""

    frame_index: int
    stride_number: int
    coordinates_angstrom: Any
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
class DistanceRestraintSpec:
    """Specification for a single distance restraint to apply during relaxation."""

    label: str
    atom1: AtomReference
    atom2: AtomReference
    target_distance_angstrom: float
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
            if value is None or not cv_value_in_interval(name, value, *bounds):
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
