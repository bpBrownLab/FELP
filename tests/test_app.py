from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from felp.app import _repartition_frame_results_to_complete_2d_grid, summarize_results
from felp.cli.parser import build_parser
from felp.core import FrameResult, MicrostateConfig, OutputConfig
from felp.io import ResultsWriter, load_frame_results_csv
from felp.reweighting import MicrostateAtlas, SamplingManifest, build_auto_microstate_payload


def _frame_result(frame_index: int, state_id: str, delta_u: float) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        stride_number=frame_index,
        microstate_id=state_id,
        microstate_label=state_id,
        phi_degrees=0.0,
        psi_degrees=0.0,
        reference_energy_kj_mol=0.0,
        perturbed_energy_kj_mol=delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=0.0,
        reference_relaxed_psi_degrees=0.0,
        perturbed_relaxed_phi_degrees=0.0,
        perturbed_relaxed_psi_degrees=0.0,
    )


def _distance_frame_result(
    frame_index: int,
    *,
    ends: float,
    turn: float,
) -> FrameResult:
    result = _frame_result(frame_index, "old", 0.0)
    return FrameResult(
        **{
            **result.to_row(),
            "phi_degrees": None,
            "psi_degrees": None,
            "metadata": {"assignment_measurements": {"ends": ends, "turn": turn}},
        }
    )


def test_summarize_results_reports_overlap_assessment(tmp_path: Path) -> None:
    state_file = tmp_path / "states.json"
    state_file.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "a",
                        "label": "a",
                        "cv_ranges": {
                            "phi": [-180.0, 0.0],
                            "psi": [-180.0, 180.0],
                        },
                    },
                    {
                        "state_id": "b",
                        "label": "b",
                        "cv_ranges": {
                            "phi": [0.0, 180.0],
                            "psi": [-180.0, 180.0],
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    raw_writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "raw")))
    raw_writer.prepare_output_dir()
    raw_writer.write_frame_results(
        [
            *(_frame_result(index, "a", 10.0) for index in range(40)),
            *(_frame_result(index, "b", 0.0) for index in range(40, 49)),
        ]
    )

    namespace = SimpleNamespace(
        output_dir=str(tmp_path / "summary"),
        raw_frame_results=raw_writer.raw_frame_results_path,
        temperature_kelvin=300.0,
        prior_probability_mode="raw_counts",
        ignore_loaded_reference_probabilities=False,
        microstate_source="file",
        microstate_file=str(state_file),
        phi_bins=72,
        psi_bins=72,
        msm_bundle_dir=None,
        state_level="microstate",
        allow_unassigned=False,
    )

    summary = summarize_results(namespace)

    assert summary["reference_probability_source"] == "raw_counts"
    assert summary["overlap_assessment"] == "poor"
    assert summary["overlap_diagnostics"]["target_probability_low_ess_mass_by_threshold"][
        "ess_lt_10"
    ] > 0.25
    assert summary["overlap_diagnostics"]["scope"].startswith(
        "These diagnostics only assess overlap"
    )
    assert any(
        "ESS < 10" in reason for reason in summary["overlap_assessment_reasons"]
    )

    written_summary = json.loads(
        (Path(namespace.output_dir) / "run_summary.json").read_text(encoding="utf-8")
    )
    assert written_summary["overlap_assessment"] == "poor"
    assert written_summary["overlap_diagnostics"]["assessment_thresholds"]["poor"][
        "global_effective_sample_fraction_lt"
    ] == 0.01


def test_summarize_results_uses_sampling_manifest_reference_counts(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "states.json"
    state_file.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "a",
                        "label": "a",
                        "cv_ranges": {
                            "phi": [-180.0, 0.0],
                            "psi": [-180.0, 180.0],
                        },
                    },
                    {
                        "state_id": "b",
                        "label": "b",
                        "cv_ranges": {
                            "phi": [0.0, 180.0],
                            "psi": [-180.0, 180.0],
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "raw")))
    raw_writer.prepare_output_dir()
    raw_writer.write_frame_results(
        [
            _frame_result(0, "a", 0.0),
            _frame_result(9, "b", 0.0),
        ]
    )
    manifest = SamplingManifest(
        version=1,
        sampling_config={
            "microstate_sample_fraction": 0.2,
            "microstate_min_samples": 1,
            "microstate_max_samples": None,
            "microstate_sampling_seed": 0,
        },
        microstate_source="file",
        state_level="microstate",
        total_assigned_frame_count=10,
        assigned_counts={"a": 9, "b": 1},
        selected_counts={"a": 1, "b": 1},
        selected_frame_indices=(0, 9),
        frame_state_ids={**{index: "a" for index in range(9)}, 9: "b"},
    )
    (Path(raw_writer.output_dir) / "sampling_manifest.json").write_text(
        json.dumps(manifest.to_dict()),
        encoding="utf-8",
    )

    namespace = SimpleNamespace(
        output_dir=str(tmp_path / "summary"),
        raw_frame_results=raw_writer.raw_frame_results_path,
        temperature_kelvin=300.0,
        prior_probability_mode="raw_counts",
        ignore_loaded_reference_probabilities=False,
        microstate_source="file",
        microstate_file=str(state_file),
        phi_bins=72,
        psi_bins=72,
        msm_bundle_dir=None,
        state_level="microstate",
        allow_unassigned=False,
    )

    summary = summarize_results(namespace)

    assert summary["reference_count_scope"] == "full_assigned_frames"
    assert summary["sampling"]["total_assigned_frame_count"] == 10
    summary_csv = Path(namespace.output_dir) / "microstate_summary.csv"
    assert "0.9" in summary_csv.read_text(encoding="utf-8")


def test_summarize_repartition_writes_matching_frame_results(tmp_path: Path) -> None:
    raw_writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "raw")))
    raw_writer.prepare_output_dir()
    raw_writer.write_frame_results(
        [
            _frame_result(0, "old_a", 0.0),
            _frame_result(1, "old_b", 1.0),
        ]
    )

    parser = build_parser()
    output_dir = tmp_path / "summary"
    namespace = parser.parse_args(
        [
            "summarize",
            "--raw-frame-results",
            raw_writer.raw_frame_results_path,
            "--output-dir",
            str(output_dir),
            "--repartition",
            "--microstate-source",
            "grid",
            "--phi-bins",
            "2",
            "--psi-bins",
            "2",
            "--prior-probability-mode",
            "raw_counts",
            "--reweighting-qc-policy",
            "none",
        ]
    )

    summarize_results(namespace)

    written_frames = load_frame_results_csv(str(output_dir / "frame_results.csv"))
    assert [result.microstate_id for result in written_frames] == ["1_1", "1_1"]
    assert [result.microstate_label for result in written_frames] == [
        "phi1_psi1",
        "phi1_psi1",
    ]


def test_fast_auto_grid_repartition_matches_atlas_assignment(tmp_path: Path) -> None:
    payload = build_auto_microstate_payload(
        cv_values={
            "ends": np.asarray([0.0, 0.5, 9.5, 10.0], dtype=float),
            "turn": np.asarray([2.0, 2.5, 7.5, 8.0], dtype=float),
        },
        cv_names=("ends", "turn"),
        min_bins_per_axis=2,
        max_bins_per_axis=2,
        max_total_states=4,
        target_frames_per_state=1,
    )
    state_file = tmp_path / "auto_microstates.json"
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(
            source="file",
            definition_file=str(state_file),
            reference_probability_mode="raw_counts",
        )
    )
    frames = [
        _distance_frame_result(0, ends=0.0, turn=2.0),
        _distance_frame_result(1, ends=5.0, turn=5.0),
        _distance_frame_result(2, ends=10.0, turn=8.0),
    ]

    repartitioned = _repartition_frame_results_to_complete_2d_grid(frames, atlas)

    assert repartitioned is not None
    expected_state_ids = [
        atlas.assign(frame.metadata["assignment_measurements"]).state_id
        for frame in frames
    ]
    assert [frame.microstate_id for frame in repartitioned] == expected_state_ids
