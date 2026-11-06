"""Result serialization for FELP."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable

csv.field_size_limit(sys.maxsize)

from felp.core import (
    FrameResult,
    MicrostateResult,
    OutputConfig,
    RunConfig,
    SidechainEnsembleMember,
)


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
            reweighting_delta_kj_mol=_parse_optional_float(
                row.get("reweighting_delta_kj_mol")
            ),
            local_free_energy_kj_mol=_parse_optional_float(
                row.get("local_free_energy_kj_mol")
            ),
            metadata=metadata,
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid frame result row {row_number} in {path}: {exc}"
        ) from exc


def _csv_row_definitely_ineligible_for_reweighting(row: dict[str, str]) -> bool:
    """Cheap row filter: True only when the row cannot be reweighting-eligible."""

    status = (row.get("status") or "").strip().lower()
    if status == "failed":
        return True
    if (row.get("microstate_id") or "").strip() == "failed":
        return True
    if (row.get("assignment_source") or "").strip() == "failed":
        return True
    try:
        ref = float(row["reference_energy_kj_mol"])
        pert = float(row["perturbed_energy_kj_mol"])
        delta_u = float(row["delta_u_kj_mol"])
    except (KeyError, ValueError, TypeError):
        return True
    if not (math.isfinite(ref) and math.isfinite(pert) and math.isfinite(delta_u)):
        return True
    return False


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


def load_reweighting_eligible_frame_results_csv(path: str) -> list[FrameResult]:
    """Like :func:`load_frame_results_csv` but keep only reweighting-eligible frames.

    Skips parsing rows that are clearly failed or have invalid energies, which
    speeds up large ``frame_results.csv`` files for block bootstrap and similar.
    """

    from felp.reweighting import is_reweighting_eligible_frame

    if not os.path.isfile(path):
        return []
    frame_results: list[FrameResult] = []
    seen_indices: set[int] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if _csv_row_definitely_ineligible_for_reweighting(row):
                continue
            result = _frame_result_from_row(row, path=path, row_number=row_number)
            if not is_reweighting_eligible_frame(result):
                continue
            if result.frame_index in seen_indices:
                raise ValueError(
                    f"Duplicate frame_index {result.frame_index} in {path} at row "
                    f"{row_number}."
                )
            seen_indices.add(result.frame_index)
            frame_results.append(result)
    return frame_results


def _parse_chi_degrees(value: str | None) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    return tuple(float(item) for item in str(value).split(",") if item != "")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def load_sidechain_ensemble_members(path: str) -> list[SidechainEnsembleMember]:
    """Read the local-ensemble sidecar CSV if it exists."""

    if not os.path.isfile(path):
        return []
    members: list[SidechainEnsembleMember] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            members.append(
                SidechainEnsembleMember(
                    frame_index=int(row["frame_index"]),
                    branch=str(row["branch"]),
                    chain_id=int(row["chain_id"]),
                    step=int(row["step"]),
                    energy_kj_mol=float(row["energy_kj_mol"]),
                    proposal_density=_parse_optional_float(row.get("proposal_density")),
                    rotamer_assignment=str(row.get("rotamer_assignment") or ""),
                    seed=int(row.get("seed") or 0),
                    chi_degrees=_parse_chi_degrees(row.get("chi_degrees")),
                    accepted=_parse_optional_bool(row.get("accepted")),
                    rosetta_score=_parse_optional_float(row.get("rosetta_score")),
                )
            )
    return members


def attach_sidechain_ensemble_members(
    frames: Iterable[FrameResult],
    members: Iterable[SidechainEnsembleMember],
) -> list[FrameResult]:
    """Attach sidecar members onto parent frame rows by ``frame_index``."""

    grouped: dict[int, list[SidechainEnsembleMember]] = {}
    for member in members:
        grouped.setdefault(int(member.frame_index), []).append(member)
    attached: list[FrameResult] = []
    for frame in frames:
        extras = tuple(grouped.get(int(frame.frame_index), ()))
        if not extras:
            attached.append(frame)
            continue
        attached.append(
            FrameResult(
                frame_index=frame.frame_index,
                stride_number=frame.stride_number,
                microstate_id=frame.microstate_id,
                microstate_label=frame.microstate_label,
                phi_degrees=frame.phi_degrees,
                psi_degrees=frame.psi_degrees,
                reference_energy_kj_mol=frame.reference_energy_kj_mol,
                perturbed_energy_kj_mol=frame.perturbed_energy_kj_mol,
                delta_u_kj_mol=frame.delta_u_kj_mol,
                reference_relaxed_phi_degrees=frame.reference_relaxed_phi_degrees,
                reference_relaxed_psi_degrees=frame.reference_relaxed_psi_degrees,
                perturbed_relaxed_phi_degrees=frame.perturbed_relaxed_phi_degrees,
                perturbed_relaxed_psi_degrees=frame.perturbed_relaxed_psi_degrees,
                trajectory_id=frame.trajectory_id,
                source_frame_index=frame.source_frame_index,
                assignment_source=frame.assignment_source,
                state_level=frame.state_level,
                reweighting_delta_kj_mol=frame.reweighting_delta_kj_mol,
                local_free_energy_kj_mol=frame.local_free_energy_kj_mol,
                ensemble_members=extras,
                metadata=frame.metadata,
            )
        )
    return attached


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

    @property
    def sidechain_ensemble_path(self) -> str:
        return os.path.join(self.output_dir, "sidechain_ensemble_results.csv")

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
            "reweighting_delta_kj_mol",
            "local_free_energy_kj_mol",
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
            if traceback_text and self.output_config.debug_frame_outputs:
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
        self.merge_sidechain_ensemble_members(
            [
                member
                for result in sorted_results
                for member in result.ensemble_members
            ]
        )

    def read_frame_results(self) -> list[FrameResult]:
        """Read previously computed frame results from a CSV file."""

        return attach_sidechain_ensemble_members(
            load_frame_results_csv(self.raw_frame_results_path),
            self.read_sidechain_ensemble_members(),
        )

    def read_sidechain_ensemble_members(self) -> list[SidechainEnsembleMember]:
        """Read the optional local-ensemble sidecar."""

        return load_sidechain_ensemble_members(self.sidechain_ensemble_path)

    def merge_sidechain_ensemble_members(
        self,
        members: Iterable[SidechainEnsembleMember],
    ) -> None:
        """Replace sidecar rows for frames present in ``members`` and keep the rest."""

        incoming = list(members)
        if not incoming and not os.path.isfile(self.sidechain_ensemble_path):
            return
        incoming_frames = {member.frame_index for member in incoming}
        retained = [
            member
            for member in self.read_sidechain_ensemble_members()
            if member.frame_index not in incoming_frames
        ]
        self.write_sidechain_ensemble_members([*retained, *incoming])

    def write_sidechain_ensemble_members(
        self,
        members: Iterable[SidechainEnsembleMember],
    ) -> None:
        """Write the dedicated local-ensemble sidecar CSV."""

        rows = [
            member.to_row()
            for member in sorted(
                members,
                key=lambda item: (item.frame_index, item.branch, item.chain_id, item.step),
            )
        ]
        fieldnames = [
            "frame_index",
            "branch",
            "chain_id",
            "step",
            "energy_kj_mol",
            "proposal_density",
            "rotamer_assignment",
            "seed",
            "chi_degrees",
            "accepted",
            "rosetta_score",
        ]

        def _write(handle) -> None:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        _write_text_atomic(self.sidechain_ensemble_path, _write, newline="")

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
            "cv_centers",
            "cv_ranges",
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
                metadata = dict(row.get("metadata", {}) or {})
                surface_row = {field: row.get(field) for field in surface_fields}
                surface_row["cv_centers"] = _json_dumps(
                    metadata.get("cv_centers", {})
                )
                surface_row["cv_ranges"] = _json_dumps(metadata.get("cv_ranges", {}))
                writer.writerow(surface_row)

        _write_text_atomic(self.surface_results_path, _write_surface, newline="")

    def write_summary(self, payload: dict) -> None:
        """Write a high-level JSON summary of the run."""

        _write_json_atomic(self.summary_dump_path, payload)

    def write_frame_weights(self, frame_weights: Iterable[dict[str, object]]) -> None:
        """Write per-frame SNIS weights for frame-level reweighting."""

        rows = list(frame_weights)
        if not rows:
            return
        fieldnames = [
            "frame_index",
            "microstate_id",
            "delta_u_kj_mol",
            "phi_degrees",
            "psi_degrees",
            "weight",
        ]

        def _write(handle) -> None:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})

        path = os.path.join(self.output_dir, "frame_weights.csv")
        _write_text_atomic(path, _write, newline="")
