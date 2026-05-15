"""Typed configuration for the PyEMMA-backed MSM pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Optional

from felp.core import ConfigurationError, flatten_trajectory_groups, normalize_trajectory_groups


def normalize_its_errors_kw(mode: str) -> str | None:
    """Map user mode to PyEMMA ``msm.its(..., errors=...)``.

    Returns ``None`` when error bands should be disabled (point estimates only).
    Omitting the keyword is required on recent PyEMMA/deeptime builds that
    reject ``errors=False`` with "not implemented".
    """

    key = str(mode).strip().lower()
    if key in ("none", "off", "false", "ml"):
        return None
    if key in ("bayes", "calc"):
        return key
    raise ValueError(
        f"its_errors must be one of: none, bayes, calc; got {mode!r}."
    )


@dataclass(frozen=True)
class MSMFitConfig:
    """Configuration for fitting a PyEMMA model and exporting an MSM bundle."""

    topology_file: str
    trajectory_files: tuple[tuple[str, ...], ...]
    output_dir: str
    feature_spec_file: Optional[str] = None
    phi_psi_target_residue_number: Optional[int] = None
    phi_psi_target_residue_name: Optional[str] = "GLY"
    phi_psi_selection: Optional[str] = None
    phi_psi_dihedral_representation: str = "raw"
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
    its_max_lag: int = 80
    its_lag_points: int = 35
    its_lags: Optional[tuple[int, ...]] = None
    its_processes: int = 8
    its_n_jobs: int = 1
    ck_nsets: int = 4
    ck_mlags: int = 10
    ck_n_jobs: int = 1
    skip_ck: bool = False
    its_errors: str = "none"

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
        has_file = bool(self.feature_spec_file)
        has_auto = self.phi_psi_target_residue_number is not None
        if has_file == has_auto:
            raise ConfigurationError(
                "Provide exactly one of --feature-spec-file or "
                "--phi-psi-residue-number."
            )
        if has_file and not os.path.isfile(self.feature_spec_file):
            raise ConfigurationError(
                f"Feature spec file does not exist: {self.feature_spec_file}"
            )
        if has_auto:
            if int(self.phi_psi_target_residue_number) < 1:
                raise ConfigurationError(
                    "phi_psi_target_residue_number must be at least 1."
                )
            if self.phi_psi_dihedral_representation not in {"raw", "cossin"}:
                raise ConfigurationError(
                    "phi_psi_dihedral_representation must be 'raw' or 'cossin'."
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
        if self.its_max_lag < 1:
            raise ConfigurationError("its_max_lag must be at least 1.")
        if self.its_lag_points < 3:
            raise ConfigurationError("its_lag_points must be at least 3.")
        if self.its_processes < 1:
            raise ConfigurationError("its_processes must be at least 1.")
        if self.its_n_jobs < 1:
            raise ConfigurationError("its_n_jobs must be at least 1.")
        if self.ck_nsets < 2:
            raise ConfigurationError("ck_nsets must be at least 2.")
        if self.ck_mlags < 1:
            raise ConfigurationError("ck_mlags must be at least 1.")
        if self.ck_n_jobs < 1:
            raise ConfigurationError("ck_n_jobs must be at least 1.")
        try:
            normalize_its_errors_kw(self.its_errors)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if self.its_lags is not None:
            for lag in self.its_lags:
                if int(lag) < 1:
                    raise ConfigurationError(f"ITS lags must be positive; got {lag!r}.")
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
