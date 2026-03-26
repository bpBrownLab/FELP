"""Typed configuration for the PyEMMA-backed MSM pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Optional

from felp.core import ConfigurationError, flatten_trajectory_groups, normalize_trajectory_groups


@dataclass(frozen=True)
class MSMFitConfig:
    """Configuration for fitting a PyEMMA model and exporting an MSM bundle."""

    topology_file: str
    trajectory_files: tuple[tuple[str, ...], ...]
    feature_spec_file: str
    output_dir: str
    stride: int = 1
    dt_traj: str = "1 step"
    reducer: str = "tica"
    reducer_lag: int = 10
    variance_cutoff: float = 0.95
    clustering_method: str = "kmeans"
    k_clusters: int = 100
    k_iter: int = 10
    regspace_dmin: float = 0.5
    msm_lag: int = 1
    metastable_states: Optional[int] = None
    hmm_states: Optional[int] = None
    save_pyemma_checkpoints: bool = False
    write_plots: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        """Validate the fit configuration before importing heavy dependencies."""

        if not os.path.isfile(self.topology_file):
            raise ConfigurationError(
                f"Topology file does not exist: {self.topology_file}"
            )
        if not self.trajectory_files:
            raise ConfigurationError("At least one trajectory file is required.")
        for trajectory_group in self.trajectory_groups:
            if not trajectory_group:
                raise ConfigurationError("Trajectory groups may not be empty.")
            for trajectory_file in trajectory_group:
                if not os.path.isfile(trajectory_file):
                    raise ConfigurationError(
                        f"Trajectory file does not exist: {trajectory_file}"
                    )
        if not os.path.isfile(self.feature_spec_file):
            raise ConfigurationError(
                f"Feature spec file does not exist: {self.feature_spec_file}"
            )
        if self.stride < 1:
            raise ConfigurationError("stride must be at least 1.")
        if self.reducer not in {"none", "tica", "pca"}:
            raise ConfigurationError(f"Unsupported reducer: {self.reducer}")
        if self.reducer_lag < 1:
            raise ConfigurationError("reducer_lag must be at least 1.")
        if not 0.0 < self.variance_cutoff <= 1.0:
            raise ConfigurationError("variance_cutoff must be in the interval (0, 1].")
        if self.clustering_method not in {"kmeans", "regspace"}:
            raise ConfigurationError(
                f"Unsupported clustering method: {self.clustering_method}"
            )
        if self.k_clusters < 1:
            raise ConfigurationError("k_clusters must be at least 1.")
        if self.k_iter < 1:
            raise ConfigurationError("k_iter must be at least 1.")
        if self.regspace_dmin <= 0.0:
            raise ConfigurationError("regspace_dmin must be positive.")
        if self.msm_lag < 1:
            raise ConfigurationError("msm_lag must be at least 1.")
        if self.metastable_states is not None and self.metastable_states < 2:
            raise ConfigurationError("metastable_states must be at least 2.")
        if self.hmm_states is not None and self.hmm_states < 2:
            raise ConfigurationError("hmm_states must be at least 2.")
        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-friendly dictionary."""

        return asdict(self)

    @property
    def trajectory_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return explicit independent trajectories as grouped file fragments."""

        return normalize_trajectory_groups(self.trajectory_files)

    @property
    def flattened_trajectory_files(self) -> tuple[str, ...]:
        """Return all trajectory fragment files in order."""

        return flatten_trajectory_groups(self.trajectory_groups)


@dataclass(frozen=True)
class MSMSampleExportConfig:
    """Configuration for exporting representative state samples from a bundle."""

    bundle_dir: str
    output_dir: str
    state_level: str = "cluster"
    state_ids: tuple[str, ...] = ()
    samples_per_state: int = 10
    overwrite: bool = False

    def validate(self) -> None:
        """Validate the sampling configuration."""

        if not os.path.isdir(self.bundle_dir):
            raise ConfigurationError(f"MSM bundle directory does not exist: {self.bundle_dir}")
        if self.state_level not in {"cluster", "metastable"}:
            raise ConfigurationError(f"Unsupported state level: {self.state_level}")
        if self.samples_per_state < 1:
            raise ConfigurationError("samples_per_state must be at least 1.")
        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")
