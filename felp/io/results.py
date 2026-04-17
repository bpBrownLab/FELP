"""Result serialization for FELP."""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Iterable

from felp.core import FrameResult, MicrostateResult, OutputConfig, RunConfig


def _parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


class ResultsWriter:
    """Owns output directory lifecycle and serialized FELP outputs."""

    def __init__(self, output_config: OutputConfig) -> None:
        self.output_config = output_config

    @property
    def output_dir(self) -> str:
        return self.output_config.output_dir

    @property
    def raw_frame_results_path(self) -> str:
        return os.path.join(self.output_dir, self.output_config.raw_frame_results_name)

    @property
    def microstate_summary_path(self) -> str:
        return os.path.join(self.output_dir, self.output_config.microstate_summary_name)

    @property
    def surface_results_path(self) -> str:
        return os.path.join(self.output_dir, self.output_config.surface_results_name)

    @property
    def config_dump_path(self) -> str:
        return os.path.join(self.output_dir, self.output_config.config_dump_name)

    @property
    def summary_dump_path(self) -> str:
        return os.path.join(self.output_dir, self.output_config.summary_dump_name)

    def prepare_output_dir(self) -> None:
        """Create the output directory, handling overwrite and resume policies."""
        if os.path.isdir(self.output_dir):
            if self.output_config.resume:
                return
            if self.output_config.overwrite:
                shutil.rmtree(self.output_dir)
            elif os.listdir(self.output_dir):
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {self.output_dir}"
                )
        os.makedirs(self.output_dir, exist_ok=True)

    def write_run_config(self, config: RunConfig) -> None:
        """Serialize the RunConfig to a JSON file."""
        with open(self.config_dump_path, "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2, sort_keys=True)

    def write_frame_results(self, frame_results: Iterable[FrameResult]) -> None:
        """Write frame-level results to a CSV file."""
        fieldnames = [
            "frame_index",
            "stride_number",
            "trajectory_id",
            "source_frame_index",
            "microstate_id",
            "microstate_label",
            "phi_degrees",
            "psi_degrees",
            "reference_energy_kj_mol",
            "perturbed_energy_kj_mol",
            "delta_u_kj_mol",
            "reference_relaxed_phi_degrees",
            "reference_relaxed_psi_degrees",
            "perturbed_relaxed_phi_degrees",
            "perturbed_relaxed_psi_degrees",
            "assignment_source",
            "state_level",
            "metadata",
        ]
        with open(self.raw_frame_results_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in sorted(frame_results, key=lambda item: item.frame_index):
                row = result.to_row()
                row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
                writer.writerow(row)

    def read_frame_results(self) -> list[FrameResult]:
        """Read previously computed frame results from a CSV file."""
        if not os.path.isfile(self.raw_frame_results_path):
            return []
        frame_results: list[FrameResult] = []
        with open(self.raw_frame_results_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame_results.append(
                    FrameResult(
                        frame_index=int(row["frame_index"]),
                        stride_number=int(row["stride_number"]),
                        trajectory_id=_parse_optional_int(row.get("trajectory_id")),
                        source_frame_index=_parse_optional_int(
                            row.get("source_frame_index")
                        ),
                        microstate_id=row["microstate_id"],
                        microstate_label=row["microstate_label"],
                        phi_degrees=_parse_optional_float(row.get("phi_degrees")),
                        psi_degrees=_parse_optional_float(row.get("psi_degrees")),
                        reference_energy_kj_mol=float(row["reference_energy_kj_mol"]),
                        perturbed_energy_kj_mol=float(row["perturbed_energy_kj_mol"]),
                        delta_u_kj_mol=float(row["delta_u_kj_mol"]),
                        reference_relaxed_phi_degrees=_parse_optional_float(
                            row.get("reference_relaxed_phi_degrees")
                        ),
                        reference_relaxed_psi_degrees=_parse_optional_float(
                            row.get("reference_relaxed_psi_degrees")
                        ),
                        perturbed_relaxed_phi_degrees=_parse_optional_float(
                            row.get("perturbed_relaxed_phi_degrees")
                        ),
                        perturbed_relaxed_psi_degrees=_parse_optional_float(
                            row.get("perturbed_relaxed_psi_degrees")
                        ),
                        assignment_source=row.get("assignment_source", "phi_psi"),
                        state_level=row.get("state_level", "microstate"),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                )
        return frame_results

    def load_completed_frame_indices(self) -> set[int]:
        """Return the set of frame indices that have already been processed."""
        return {result.frame_index for result in self.read_frame_results()}

    def write_microstate_results(self, microstate_results: Iterable[MicrostateResult]) -> None:
        """Write aggregated microstate statistics and the reweighted surface to CSV files."""
        rows = [result.to_row() for result in microstate_results]
        if not rows:
            return

        summary_fields = list(rows[0].keys())
        with open(self.microstate_summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
                writer.writerow(row)

        surface_fields = [
            "state_id",
            "label",
            "phi_center",
            "psi_center",
            "plot_x",
            "plot_y",
            "state_level",
            "reference_probability",
            "target_probability",
            "free_energy_kj_mol",
        ]
        with open(self.surface_results_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=surface_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in surface_fields})

    def write_summary(self, payload: dict) -> None:
        """Write a high-level JSON summary of the run."""
        with open(self.summary_dump_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
