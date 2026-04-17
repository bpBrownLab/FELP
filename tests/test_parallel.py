from __future__ import annotations

from felp.core import (
    EnergyConfig,
    FrameResult,
    InputConfig,
    MicrostateConfig,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
    SnapshotJob,
)
from felp.parallel import executor as executor_module
from felp.reweighting import MicrostateAtlas


def _config() -> RunConfig:
    return RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(target_residue_number=2, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )


def test_frame_executor_serial_path_preserves_order(monkeypatch) -> None:
    calls = {"initialized": False}

    def fake_initialize(config, atlas, assignment_policy=None) -> None:
        calls["initialized"] = True

    def fake_process(job: SnapshotJob) -> FrameResult:
        return FrameResult(
            frame_index=job.frame_index,
            stride_number=job.stride_number,
            microstate_id="state",
            microstate_label="state",
            phi_degrees=0.0,
            psi_degrees=0.0,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=0.0,
            delta_u_kj_mol=0.0,
            reference_relaxed_phi_degrees=0.0,
            reference_relaxed_psi_degrees=0.0,
            perturbed_relaxed_phi_degrees=0.0,
            perturbed_relaxed_psi_degrees=0.0,
        )

    monkeypatch.setattr(executor_module, "initialize_worker", fake_initialize)
    monkeypatch.setattr(executor_module, "safe_process_frame_job", fake_process)

    jobs = [
        SnapshotJob(frame_index=0, stride_number=0, coordinates_angstrom=((0.0, 0.0, 0.0),)),
        SnapshotJob(frame_index=1, stride_number=1, coordinates_angstrom=((1.0, 0.0, 0.0),)),
    ]
    results = executor_module.FrameExecutor(
        _config(),
        MicrostateAtlas(definitions=()),
    ).run(jobs)

    assert calls["initialized"] is True
    assert [result.frame_index for result in results] == [0, 1]
