"""PyEMMA-backed fitting pipeline for FELP MSM bundles."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from typing import Any, Optional

import numpy as np

from felp.core import DependencyError, MicrostateDefinition

from .artifacts import MSMBundle, MSMBundleManifest, save_bundle
from .config import MSMFitConfig
from .features import FeatureSpec, build_featurizer


def _import_pyemma():
    try:
        import pyemma
    except ImportError as exc:
        raise DependencyError(
            "PyEMMA is required for msm-fit but is not installed."
        ) from exc
    return pyemma


@dataclass(frozen=True)
class PyEMMAPipelineResult:
    """Structured result returned by the PyEMMA fitting pipeline."""

    bundle: MSMBundle
    feature_spec: FeatureSpec
    reduced_trajectories: tuple[np.ndarray, ...]
    cluster_dtrajs: tuple[np.ndarray, ...]
    discrete_msm: Any
    hidden_model: Optional[Any]
    metastable_dtrajs: Optional[tuple[np.ndarray, ...]]


@dataclass
class PyEMMAPipeline:
    """Fit a discrete MSM workflow and export a FELP-readable bundle."""

    config: MSMFitConfig

    def run(self) -> PyEMMAPipelineResult:
        """Execute the configured PyEMMA workflow and export an MSM bundle."""

        self.config.validate()
        pyemma = _import_pyemma()

        spec_path = self.config.feature_spec_file
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Feature spec file not found: {spec_path}")
        # Load before optional rmtree so specs living under output_dir survive overwrite.
        feature_spec = FeatureSpec.from_file(spec_path)

        if os.path.isdir(self.config.output_dir):
            if self.config.overwrite:
                shutil.rmtree(self.config.output_dir)
            elif os.listdir(self.config.output_dir):
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {self.config.output_dir}"
                )
        os.makedirs(self.config.output_dir, exist_ok=True)
        featurizer = build_featurizer(self.config.topology_file, feature_spec)

        reader = pyemma.coordinates.source(
            [list(group) for group in self.config.trajectory_groups],
            features=featurizer,
        )
        feature_trajectories = tuple(
            np.asarray(values)
            for values in reader.get_output(stride=self.config.stride)
        )

        reducer_model, reduced_trajectories = self._reduce(pyemma, feature_trajectories)
        cluster = self._cluster(pyemma, reduced_trajectories)
        cluster_dtrajs = tuple(
            np.asarray(dtraj, dtype=int)
            for dtraj in cluster.dtrajs
        )

        discrete_msm = pyemma.msm.estimate_markov_model(
            cluster_dtrajs,
            lag=int(self.config.msm_lag),
            dt_traj=str(self.config.dt_traj),
        )

        hidden_model = None
        if self.config.hmm_states is not None:
            hidden_model = pyemma.msm.estimate_hidden_markov_model(
                cluster_dtrajs,
                int(self.config.hmm_states),
                lag=int(self.config.msm_lag),
                dt_traj=str(self.config.dt_traj),
            )

        metastable_dtrajs = self._build_metastable_assignments(
            discrete_msm=discrete_msm,
            hidden_model=hidden_model,
            cluster_dtrajs=cluster_dtrajs,
        )

        cluster_definitions = self._build_cluster_definitions(
            cluster_centers=np.asarray(cluster.clustercenters),
            discrete_msm=discrete_msm,
        )
        metastable_definitions = self._build_metastable_definitions(
            reduced_trajectories=reduced_trajectories,
            metastable_dtrajs=metastable_dtrajs,
            discrete_msm=discrete_msm,
            hidden_model=hidden_model,
        )

        state_definitions: dict[str, tuple[MicrostateDefinition, ...]] = {
            "cluster": cluster_definitions,
        }
        assignments: dict[str, tuple[np.ndarray, ...]] = {
            "cluster": cluster_dtrajs,
        }
        if metastable_definitions is not None and metastable_dtrajs is not None:
            state_definitions["metastable"] = metastable_definitions
            assignments["metastable"] = metastable_dtrajs

        frame_source_indices = tuple(
            np.arange(len(dtraj), dtype=int) * self.config.stride
            for dtraj in cluster_dtrajs
        )
        manifest = MSMBundleManifest(
            topology_file=self.config.topology_file,
            trajectory_groups=self.config.trajectory_groups,
            stride=self.config.stride,
            dt_traj=self.config.dt_traj,
            lag_time=self.config.msm_lag,
            feature_spec=feature_spec.to_dict(),
            reducer={
                "kind": self.config.reducer,
                "lag": self.config.reducer_lag,
                "variance_cutoff": self.config.variance_cutoff,
                "output_dimension": (
                    int(reduced_trajectories[0].shape[1])
                    if reduced_trajectories and reduced_trajectories[0].ndim == 2
                    else 0
                ),
            },
            clustering={
                "method": self.config.clustering_method,
                "k_clusters": self.config.k_clusters,
                "k_iter": self.config.k_iter,
                "regspace_dmin": self.config.regspace_dmin,
            },
            model={
                "discrete_msm": {
                    "active_state_fraction": float(discrete_msm.active_state_fraction),
                    "active_count_fraction": float(discrete_msm.active_count_fraction),
                },
                "metastable_states": self.config.metastable_states,
                "hmm_states": self.config.hmm_states,
            },
            state_definitions=state_definitions,
            metadata={
                "feature_count": int(feature_trajectories[0].shape[1]) if feature_trajectories else 0,
                "frame_count": int(sum(len(values) for values in cluster_dtrajs)),
            },
        )

        bundle = save_bundle(
            bundle_dir=self.config.output_dir,
            manifest=manifest,
            assignments=assignments,
            frame_source_indices=frame_source_indices,
            embedding=self._build_embedding_payload(
                reduced_trajectories=reduced_trajectories,
                cluster_centers=np.asarray(cluster.clustercenters),
            ),
        )

        if self.config.save_pyemma_checkpoints:
            self._save_checkpoints(
                reducer_model=reducer_model,
                discrete_msm=discrete_msm,
                hidden_model=hidden_model,
            )

        if self.config.write_plots:
            from .qc import write_qc_outputs

            write_qc_outputs(
                output_dir=self.config.output_dir,
                reduced_trajectories=reduced_trajectories,
                cluster_dtrajs=cluster_dtrajs,
                cluster_centers=np.asarray(cluster.clustercenters),
                discrete_msm=discrete_msm,
                hidden_model=hidden_model,
                metastable_dtrajs=metastable_dtrajs,
            )

        return PyEMMAPipelineResult(
            bundle=bundle,
            feature_spec=feature_spec,
            reduced_trajectories=reduced_trajectories,
            cluster_dtrajs=cluster_dtrajs,
            discrete_msm=discrete_msm,
            hidden_model=hidden_model,
            metastable_dtrajs=metastable_dtrajs,
        )

    def _reduce(
        self,
        pyemma,
        feature_trajectories: tuple[np.ndarray, ...],
    ) -> tuple[Optional[Any], tuple[np.ndarray, ...]]:
        """Apply the configured dimensionality reduction stage once."""

        if self.config.reducer == "none":
            return None, feature_trajectories
        if self.config.reducer == "tica":
            reducer_model = pyemma.coordinates.tica(
                feature_trajectories,
                lag=int(self.config.reducer_lag),
                var_cutoff=float(self.config.variance_cutoff),
            )
        else:
            reducer_model = pyemma.coordinates.pca(
                feature_trajectories,
                var_cutoff=float(self.config.variance_cutoff),
            )
        return reducer_model, tuple(np.asarray(values) for values in reducer_model.get_output())

    def _cluster(
        self,
        pyemma,
        reduced_trajectories: tuple[np.ndarray, ...],
    ):
        """Cluster the reduced trajectories using the configured method."""

        if self.config.clustering_method == "kmeans":
            return pyemma.coordinates.cluster_kmeans(
                reduced_trajectories,
                k=int(self.config.k_clusters),
                max_iter=int(self.config.k_iter),
            )
        return pyemma.coordinates.cluster_regspace(
            reduced_trajectories,
            dmin=float(self.config.regspace_dmin),
        )

    def _build_cluster_definitions(
        self,
        cluster_centers: np.ndarray,
        discrete_msm,
    ) -> tuple[MicrostateDefinition, ...]:
        """Build discrete cluster state definitions with stationary probabilities."""

        active_probabilities = {
            int(state_index): float(probability)
            for state_index, probability in zip(discrete_msm.active_set, discrete_msm.pi)
        }
        definitions = []
        for cluster_index, center in enumerate(cluster_centers):
            plot_x = float(center[0]) if center.size >= 1 else None
            plot_y = float(center[1]) if center.size >= 2 else None
            definitions.append(
                MicrostateDefinition(
                    state_id=str(cluster_index),
                    label=f"cluster_{cluster_index}",
                    reference_probability=active_probabilities.get(cluster_index, 0.0),
                    plot_x=plot_x,
                    plot_y=plot_y,
                    state_level="cluster",
                    metadata={
                        "cluster_index": cluster_index,
                        "active": cluster_index in active_probabilities,
                    },
                )
            )
        return tuple(definitions)

    def _build_metastable_assignments(
        self,
        discrete_msm,
        hidden_model,
        cluster_dtrajs: tuple[np.ndarray, ...],
    ) -> Optional[tuple[np.ndarray, ...]]:
        """Build per-frame metastable assignments when requested."""

        if hidden_model is not None:
            assignment_lookup = np.asarray(hidden_model.metastable_assignments, dtype=int)
            return tuple(
                np.asarray(assignment_lookup[dtraj], dtype=int)
                for dtraj in cluster_dtrajs
            )

        if self.config.metastable_states is None:
            return None

        discrete_msm.pcca(int(self.config.metastable_states))
        assignment_lookup = np.asarray(discrete_msm.metastable_assignments, dtype=int)
        return tuple(
            np.asarray(assignment_lookup[dtraj], dtype=int)
            for dtraj in cluster_dtrajs
        )

    def _build_metastable_definitions(
        self,
        reduced_trajectories: tuple[np.ndarray, ...],
        metastable_dtrajs: Optional[tuple[np.ndarray, ...]],
        discrete_msm,
        hidden_model,
    ) -> Optional[tuple[MicrostateDefinition, ...]]:
        """Build optional metastable state definitions from PCCA or an HMM."""

        if metastable_dtrajs is None:
            return None

        state_count = (
            int(hidden_model.nstates)
            if hidden_model is not None
            else int(self.config.metastable_states or 0)
        )
        flat_coordinates = np.concatenate(reduced_trajectories, axis=0)
        flat_assignments = np.concatenate(metastable_dtrajs, axis=0)
        probabilities = (
            np.asarray(hidden_model.pi, dtype=float)
            if hidden_model is not None
            else self._pcca_probabilities(discrete_msm, state_count)
        )
        source = "hidden_msm" if hidden_model is not None else "pcca"

        definitions = []
        for state_index in range(state_count):
            mask = flat_assignments == state_index
            plot_x = None
            plot_y = None
            if mask.any():
                centroid = flat_coordinates[mask].mean(axis=0)
                if centroid.size >= 1:
                    plot_x = float(centroid[0])
                if centroid.size >= 2:
                    plot_y = float(centroid[1])
            definitions.append(
                MicrostateDefinition(
                    state_id=str(state_index),
                    label=f"metastable_{state_index}",
                    reference_probability=float(probabilities[state_index]),
                    plot_x=plot_x,
                    plot_y=plot_y,
                    state_level="metastable",
                    metadata={"source": source},
                )
            )
        return tuple(definitions)

    def _pcca_probabilities(self, discrete_msm, state_count: int) -> np.ndarray:
        """Compute metastable probabilities from a discrete MSM after PCCA."""

        probabilities = np.zeros(state_count, dtype=float)
        for state_index, metastable_set in enumerate(discrete_msm.metastable_sets):
            active_mask = np.isin(discrete_msm.active_set, metastable_set)
            probabilities[state_index] = float(np.asarray(discrete_msm.pi)[active_mask].sum())
        normalization = probabilities.sum()
        if normalization > 0.0:
            probabilities /= normalization
        return probabilities

    def _build_embedding_payload(
        self,
        reduced_trajectories: tuple[np.ndarray, ...],
        cluster_centers: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build the cached embedding payload for the bundle."""

        payload: dict[str, np.ndarray] = {"cluster_centers": np.asarray(cluster_centers)}
        for trajectory_id, values in enumerate(reduced_trajectories):
            payload[f"reduced_trajectory_{trajectory_id}"] = np.asarray(values)
        return payload

    def _save_checkpoints(
        self,
        reducer_model,
        discrete_msm,
        hidden_model,
    ) -> None:
        """Persist optional PyEMMA-native checkpoints for reproducibility."""

        checkpoint_dir = os.path.join(self.config.output_dir, "pyemma_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if reducer_model is not None:
            reducer_model.save(
                os.path.join(checkpoint_dir, "reducers.h5"),
                model_name="reducer",
                overwrite=True,
            )
        discrete_msm.save(
            os.path.join(checkpoint_dir, "discrete_msm.h5"),
            model_name="discrete_msm",
            overwrite=True,
        )
        if hidden_model is not None:
            hidden_model.save(
                os.path.join(checkpoint_dir, "hidden_msm.h5"),
                model_name="hidden_msm",
                overwrite=True,
            )
