from __future__ import annotations

from pathlib import Path

from felp.core import FrameResult, OutputConfig
from felp.io import ResultsWriter


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
