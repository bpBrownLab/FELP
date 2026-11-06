"""Orchestrate shared prep and one or more reweighting backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

from felp.core import FrameResult, K_BOLTZMANN_KJ_PER_MOL_K, ReweightingConfig, ReweightingQCConfig
from felp.io.results import ResultsWriter
from felp.reweighting.atlas import MicrostateAtlas
from felp.reweighting.engines import VALID_REWEIGHTING_METHODS, get_reweighting_engine
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import build_reweighting_prep


@dataclass
class ReweightingOrchestrator:
    """Run shared preparation once and execute one or more estimators."""

    temperature_kelvin: float
    config: ReweightingConfig
    qc_config: ReweightingQCConfig | None = None
    workers: int = 1

    def __post_init__(self) -> None:
        if self.qc_config is None:
            object.__setattr__(self, "qc_config", ReweightingQCConfig())
        for method_id in self.config.methods:
            if method_id not in VALID_REWEIGHTING_METHODS:
                from felp.core import ConfigurationError

                raise ConfigurationError(
                    f"Unknown reweighting method: {method_id}"
                )

    @property
    def beta(self) -> float:
        return 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * self.temperature_kelvin)

    def run(
        self,
        frame_results: Iterable[FrameResult],
        atlas: MicrostateAtlas,
        *,
        reference_counts: dict[str, int] | None = None,
        sampling_metadata: dict[str, object] | None = None,
        output_dir: str | None = None,
        writer: ResultsWriter | None = None,
    ) -> dict[str, ReweightingOutcome]:
        """Execute configured method(s) and optionally write outputs."""

        frame_list = list(frame_results)
        if not frame_list:
            return {}

        prep = build_reweighting_prep(
            frame_list,
            atlas,
            beta=self.beta,
            temperature_kelvin=self.temperature_kelvin,
            qc_config=self.qc_config,
            reference_counts=reference_counts,
            workers=self.workers,
            max_neighbors=self.config.neighbor_pool.max_neighbors,
            grid_connectivity=self.config.neighbor_pool.grid_connectivity,
            chebyshev_radius=self.config.neighbor_pool.chebyshev_radius,
            periodic_cv_names=self.config.neighbor_pool.periodic_cv_names,
        )

        outcomes: dict[str, ReweightingOutcome] = {}
        methods = self.config.methods
        multi_method = len(methods) > 1

        for method_id in methods:
            engine = get_reweighting_engine(method_id)
            outcome = engine.aggregate(
                prep,
                config=self.config,
                sampling_metadata=sampling_metadata,
            )
            outcomes[method_id] = outcome

            if output_dir is not None:
                method_dir = (
                    os.path.join(output_dir, f"reweighting_{method_id}")
                    if multi_method
                    else output_dir
                )
                self._write_method_outputs(
                    outcome,
                    method_dir,
                    writer=writer,
                )

        if multi_method and output_dir is not None:
            comparison_path = os.path.join(output_dir, "reweighting_comparison.json")
            with open(comparison_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "methods": list(methods),
                        "comparison": {
                            method_id: {
                                "output_dir": os.path.join(
                                    output_dir, f"reweighting_{method_id}"
                                ),
                                "overlap_coefficient": outcome.shared_metadata.get(
                                    "overlap_coefficient"
                                ),
                                "global_effective_sample_size": outcome.shared_metadata.get(
                                    "global_effective_sample_size"
                                ),
                                "state_reweighting_estimator": (
                                    outcome.state_reweighting_estimator
                                ),
                            }
                            for method_id, outcome in outcomes.items()
                        },
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )

        return outcomes

    def run_single(
        self,
        frame_results: Iterable[FrameResult],
        atlas: MicrostateAtlas,
        *,
        reference_counts: dict[str, int] | None = None,
        sampling_metadata: dict[str, object] | None = None,
    ) -> ReweightingOutcome:
        """Run exactly one method and return its outcome."""

        outcomes = self.run(
            frame_results,
            atlas,
            reference_counts=reference_counts,
            sampling_metadata=sampling_metadata,
        )
        return next(iter(outcomes.values()))

    @staticmethod
    def _write_method_outputs(
        outcome: ReweightingOutcome,
        output_dir: str,
        *,
        writer: ResultsWriter | None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if writer is not None:
            from felp.core import OutputConfig

            method_writer = ResultsWriter(
                OutputConfig(
                    output_dir=output_dir,
                    raw_frame_results_name=writer.output_config.raw_frame_results_name,
                    microstate_summary_name=writer.output_config.microstate_summary_name,
                    surface_results_name=writer.output_config.surface_results_name,
                    summary_dump_name=writer.output_config.summary_dump_name,
                )
            )
            method_writer.write_microstate_results(outcome.microstate_results)
            method_writer.write_summary(
                {
                    "reweighting_method": outcome.method_id,
                    **outcome.shared_metadata,
                }
            )
            frame_weights = outcome.auxiliary.get("frame_weights")
            if frame_weights:
                method_writer.write_frame_weights(frame_weights)
                frame_summary_path = os.path.join(
                    output_dir, "frame_reweighting_summary.json"
                )
                with open(frame_summary_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        outcome.auxiliary.get("frame_reweighting_summary", {}),
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
            return

        from felp.core import OutputConfig

        method_writer = ResultsWriter(OutputConfig(output_dir=output_dir))
        method_writer.write_microstate_results(outcome.microstate_results)
        method_writer.write_summary(
            {
                "reweighting_method": outcome.method_id,
                **outcome.shared_metadata,
            }
        )
        frame_weights = outcome.auxiliary.get("frame_weights")
        if frame_weights:
            method_writer.write_frame_weights(frame_weights)
