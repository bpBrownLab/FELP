from __future__ import annotations

import csv
from math import isnan
from pathlib import Path

import pytest

from felp.core import FrameResult, OutputConfig
from felp.io import ResultsWriter, load_frame_results_csv


def test_results_writer_round_trip_and_resume(tmp_path: Path) -> None:
    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path)))
    writer.prepare_output_dir()

    frame_result = FrameResult(
        frame_index=4,
        stride_number=8,
        microstate_id="state",
        microstate_label="state",
        phi_degrees=-60.0,
        psi_degrees=120.0,
        reference_energy_kj_mol=-10.0,
        perturbed_energy_kj_mol=-9.5,
        delta_u_kj_mol=0.5,
        reference_relaxed_phi_degrees=-60.0,
        reference_relaxed_psi_degrees=120.0,
        perturbed_relaxed_phi_degrees=-60.0,
        perturbed_relaxed_psi_degrees=120.0,
        metadata={"source": "test"},
    )

    writer.write_frame_results([frame_result])
    loaded = writer.read_frame_results()
    assert loaded == [frame_result]
    assert writer.load_completed_frame_indices() == {4}


def test_results_writer_round_trip_with_generalized_cv_metadata(tmp_path: Path) -> None:
    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path)))
    writer.prepare_output_dir()

    frame_result = FrameResult(
        frame_index=5,
        stride_number=9,
        microstate_id="omega_locked",
        microstate_label="omega_locked",
        phi_degrees=None,
        psi_degrees=None,
        reference_energy_kj_mol=-11.0,
        perturbed_energy_kj_mol=-10.25,
        delta_u_kj_mol=0.75,
        reference_relaxed_phi_degrees=None,
        reference_relaxed_psi_degrees=None,
        perturbed_relaxed_phi_degrees=None,
        perturbed_relaxed_psi_degrees=None,
        metadata={
            "target_cv_values_degrees": {"omega": 179.0},
            "reference_cv_values_degrees": {"omega": 179.1},
        },
    )

    writer.write_frame_results([frame_result])
    loaded = writer.read_frame_results()
    assert loaded == [frame_result]


def test_results_writer_moves_tracebacks_out_of_main_csv(tmp_path: Path) -> None:
    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path)))
    writer.prepare_output_dir()

    frame_result = FrameResult(
        frame_index=9,
        stride_number=9,
        microstate_id="failed",
        microstate_label="failed",
        phi_degrees=None,
        psi_degrees=None,
        reference_energy_kj_mol=float("nan"),
        perturbed_energy_kj_mol=float("nan"),
        delta_u_kj_mol=float("nan"),
        reference_relaxed_phi_degrees=None,
        reference_relaxed_psi_degrees=None,
        perturbed_relaxed_phi_degrees=None,
        perturbed_relaxed_psi_degrees=None,
        assignment_source="failed",
        state_level="failed",
        metadata={
            "status": "failed",
            "error_type": "OpenMMException",
            "error_message": "Particle coordinate is NaN.",
            "traceback": "Traceback ...",
        },
    )

    writer.write_frame_results([frame_result])

    traceback_file = tmp_path / "frame_failures" / "frame_9.traceback.txt"
    assert traceback_file.is_file()
    assert traceback_file.read_text(encoding="utf-8") == "Traceback ..."

    with open(writer.raw_frame_results_path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "failed"
    assert row["error_type"] == "OpenMMException"
    assert row["error_message"] == "Particle coordinate is NaN."
    assert row["traceback_path"] == "frame_failures/frame_9.traceback.txt"
    assert "traceback" not in row["metadata"]

    loaded = writer.read_frame_results()
    assert len(loaded) == 1
    assert loaded[0].metadata["status"] == "failed"
    assert loaded[0].metadata["error_type"] == "OpenMMException"
    assert loaded[0].metadata["traceback_path"] == "frame_failures/frame_9.traceback.txt"
    assert isnan(loaded[0].delta_u_kj_mol)


def test_load_frame_results_csv_fails_fast_on_invalid_metadata(tmp_path: Path) -> None:
    frame_csv = tmp_path / "frame_results.csv"
    frame_csv.write_text(
        "\n".join(
            [
                (
                    "frame_index,stride_number,trajectory_id,source_frame_index,"
                    "microstate_id,microstate_label,phi_degrees,psi_degrees,"
                    "reference_energy_kj_mol,perturbed_energy_kj_mol,delta_u_kj_mol,"
                    "reference_relaxed_phi_degrees,reference_relaxed_psi_degrees,"
                    "perturbed_relaxed_phi_degrees,perturbed_relaxed_psi_degrees,"
                    "assignment_source,state_level,status,error_type,error_message,"
                    "traceback_path,metadata"
                ),
                (
                    '0,0,,,state,state,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,phi_psi,'
                    'microstate,completed,,,,"{not json}"'
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid metadata JSON"):
        load_frame_results_csv(str(frame_csv))
