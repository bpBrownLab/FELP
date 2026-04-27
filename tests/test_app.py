from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from felp.app import summarize_results
from felp.core import FrameResult, OutputConfig
from felp.io import ResultsWriter
from felp.reweighting import SamplingManifest


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
