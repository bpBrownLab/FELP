"""Result serialization for FELP."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from typing import Any, Callable, Iterable

from felp.core import FrameResult, MicrostateResult, OutputConfig, RunConfig


def _parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _json_default(value: Any) -> Any:
    """Normalize simple non-native JSON values before serialization."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=_json_default)


def _write_text_atomic(
    path: str,
    write_fn: Callable[[Any], None],
    *,
    newline: str | None = None,
) -> None:
    """Write a text file atomically via a temporary file and replace."""

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            write_fn(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write a JSON payload atomically."""

    def _write(handle) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)

    _write_text_atomic(path, _write)


def _load_metadata_from_row(
    row: dict[str, str],
    *,
    path: str,
    row_number: int,
) -> dict[str, Any]:
    """Deserialize metadata plus the top-level frame status columns."""

    raw_metadata = row.get("metadata")
    if raw_metadata in (None, ""):
        metadata: dict[str, Any] = {}
    else:
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid metadata JSON in {path} at row {row_number}: {exc}"
            ) from exc
    metadata = dict(metadata)
    status = row.get("status") or metadata.get("status") or "completed"
    if status != "completed" or "status" in metadata:
        metadata["status"] = status
    for key in ("error_type", "error_message", "traceback_path"):
        value = row.get(key)
        if value not in (None, ""):
            metadata[key] = value
    return metadata


def _frame_result_from_row(
    row: dict[str, str],
    *,
    path: str,
    row_number: int,
) -> FrameResult:
    """Parse one serialized `FrameResult` row with contextual errors."""

    try:
        metadata = _load_metadata_from_row(row, path=path, row_number=row_number)
        return FrameResult(
            frame_index=int(row["frame_index"]),
            stride_number=int(row["stride_number"]),
            trajectory_id=_parse_optional_int(row.get("trajectory_id")),
            source_frame_index=_parse_optional_int(row.get("source_frame_index")),
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
            metadata=metadata,
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid frame result row {row_number} in {path}: {exc}"
        ) from exc


def load_frame_results_csv(path: str) -> list[FrameResult]:
    """Read a frame-results CSV written by FELP, old or new format."""

    if not os.path.isfile(path):
        return []
    frame_results: list[FrameResult] = []
    seen_indices: set[int] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            result = _frame_result_from_row(row, path=path, row_number=row_number)
            if result.frame_index in seen_indices:
                raise ValueError(
                    f"Duplicate frame_index {result.frame_index} in {path} at row "
                    f"{row_number}."
                )
            seen_indices.add(result.frame_index)
            frame_results.append(result)
    return frame_results


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

    @property
    def frame_failure_dir(self) -> str:
        return os.path.join(self.output_dir, "frame_failures")

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

        _write_json_atomic(self.config_dump_path, config.to_dict())

    def write_frame_results(self, frame_results: Iterable[FrameResult]) -> None:
        """Write frame-level results to a CSV file."""

        sorted_results = sorted(frame_results, key=lambda item: item.frame_index)
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
            "status",
            "error_type",
            "error_message",
            "traceback_path",
            "metadata",
        ]

        if os.path.isdir(self.frame_failure_dir):
            shutil.rmtree(self.frame_failure_dir)

        rows: list[dict[str, Any]] = []
        for result in sorted_results:
            row = result.to_row()
            metadata = dict(row["metadata"])
            default_status = (
                "failed"
                if row["microstate_id"] == "failed" or row["assignment_source"] == "failed"
                else "completed"
            )
            status = str(metadata.pop("status", default_status))
            error_type = metadata.pop("error_type", "")
            error_message = metadata.pop("error_message", "")
            traceback_text = metadata.pop("traceback", "")
            traceback_path = ""
            if traceback_text:
                os.makedirs(self.frame_failure_dir, exist_ok=True)
                traceback_filename = f"frame_{result.frame_index}.traceback.txt"
                traceback_path = os.path.join("frame_failures", traceback_filename)
                _write_text_atomic(
                    os.path.join(self.output_dir, traceback_path),
                    lambda handle, text=str(traceback_text): handle.write(text),
                )
            row["status"] = status
            row["error_type"] = error_type
            row["error_message"] = error_message
            row["traceback_path"] = traceback_path
            row["metadata"] = _json_dumps(metadata)
            rows.append(row)

        def _write(handle) -> None:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        _write_text_atomic(self.raw_frame_results_path, _write, newline="")

    def read_frame_results(self) -> list[FrameResult]:
        """Read previously computed frame results from a CSV file."""

        return load_frame_results_csv(self.raw_frame_results_path)

    def load_completed_frame_indices(self) -> set[int]:
        """Return the set of frame indices that have already been processed."""

        return {result.frame_index for result in self.read_frame_results()}

    def write_microstate_results(self, microstate_results: Iterable[MicrostateResult]) -> None:
        """Write aggregated microstate statistics and the reweighted surface to CSV files."""

        rows = [result.to_row() for result in microstate_results]
        if not rows:
            return

        summary_fields = list(rows[0].keys())

        def _write_summary(handle) -> None:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                serialized["metadata"] = _json_dumps(serialized["metadata"])
                writer.writerow(serialized)

        _write_text_atomic(self.microstate_summary_path, _write_summary, newline="")

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

        def _write_surface(handle) -> None:
            writer = csv.DictWriter(handle, fieldnames=surface_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in surface_fields})

        _write_text_atomic(self.surface_results_path, _write_surface, newline="")

    def write_summary(self, payload: dict) -> None:
        """Write a high-level JSON summary of the run."""

        _write_json_atomic(self.summary_dump_path, payload)
