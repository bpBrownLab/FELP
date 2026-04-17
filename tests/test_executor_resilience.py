from __future__ import annotations

import math

from felp.core import FrameResult, SnapshotJob
from felp.parallel import executor as executor_module


def test_safe_process_frame_job_returns_failed_row(monkeypatch) -> None:
    def boom(_job: SnapshotJob) -> FrameResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(executor_module, "process_frame_job", boom)

    job = SnapshotJob(
        frame_index=7,
        stride_number=7,
        coordinates_angstrom=((0.0, 0.0, 0.0),),
    )
    result = executor_module.safe_process_frame_job(job)

    assert result.frame_index == 7
    assert result.metadata.get("status") == "failed"
    assert result.metadata.get("error_type") == "RuntimeError"
    assert math.isnan(result.delta_u_kj_mol)
