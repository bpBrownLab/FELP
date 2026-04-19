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


def test_apply_cpu_thread_limits_sets_openmm_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENMM_CPU_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(target_residue_number=2, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(platform_name="CPU", cpu_threads=3),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    executor_module._apply_cpu_thread_limits(config)

    import os

    assert os.environ["OPENMM_CPU_THREADS"] == "3"
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"


def test_apply_cpu_thread_limits_skips_non_cpu_platform(monkeypatch) -> None:
    monkeypatch.delenv("OPENMM_CPU_THREADS", raising=False)

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(target_residue_number=2, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(platform_name="CUDA", cpu_threads=1),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    executor_module._apply_cpu_thread_limits(config)

    import os

    assert "OPENMM_CPU_THREADS" not in os.environ


def test_apply_cpu_thread_limits_respects_preexisting_user_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENMM_CPU_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(target_residue_number=2, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(platform_name="CPU", cpu_threads=2),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    executor_module._apply_cpu_thread_limits(config)

    import os

    assert os.environ["OPENMM_CPU_THREADS"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"


def test_build_pyrosetta_options_enables_opt_h_for_generic_histidine() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            target_residue_number=2,
            target_residue_name="HIS",
            original_residue_name="GLY",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    options = executor_module._build_pyrosetta_options(config)

    assert "-no_optH false" in options


def test_build_pyrosetta_options_enables_ph_mode_for_explicit_protonation_states() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            target_residue_number=2,
            target_residue_name="HIP",
            original_residue_name="HIS",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    options = executor_module._build_pyrosetta_options(config)

    assert "-keep_input_protonation_state" in options
    assert "-pH_mode" in options


def test_initialize_worker_only_runs_structure_level_protonation_checks(
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    fake_structure = object()

    class _FakeConversionService:
        def __init__(self, pyrosetta_init_options: str, radii_set: str) -> None:
            calls.append(("service_init", pyrosetta_init_options, radii_set))

        def load_topology(self, topology_file: str):
            calls.append(("load_topology", topology_file))
            return fake_structure

        def validate_rosetta_structure_residue_types(self, structure, *, context: str) -> None:
            calls.append(("validate_structure", structure, context))

        def validate_rosetta_residue_type(self, residue_name: str, *, context: str) -> None:
            calls.append(("validate_residue_type", residue_name, context))

    class _FakeEnergyEvaluator:
        def validate_topology(self, structure) -> None:
            calls.append(("validate_openmm", structure))

    config = RunConfig(
        input=InputConfig(topology_file="input.parm7", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            target_residue_number=2,
            target_residue_name="HIE",
            original_residue_name="HID",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    monkeypatch.setattr(executor_module, "_apply_cpu_thread_limits", lambda config: None)
    monkeypatch.setattr(executor_module, "StructureConversionService", _FakeConversionService)
    monkeypatch.setattr(
        executor_module,
        "ensure_perturbation_matches_structure",
        lambda structure, perturbation: calls.append(
            ("match_structure", structure, perturbation.original_residue_name)
        ),
    )
    monkeypatch.setattr(
        executor_module,
        "BackboneDihedralCVProtocol",
        lambda **kwargs: calls.append(("restraints", kwargs)) or object(),
    )
    monkeypatch.setattr(
        executor_module,
        "PyRosettaPerturbationProtocol",
        lambda **kwargs: calls.append(("protocol", kwargs)) or object(),
    )
    monkeypatch.setattr(
        executor_module,
        "build_energy_evaluator",
        lambda energy_config, conversion_service: _FakeEnergyEvaluator(),
    )

    executor_module.initialize_worker(config, MicrostateAtlas(definitions=()))

    assert ("load_topology", "input.parm7") in calls
    assert ("match_structure", fake_structure, "HID") in calls
    assert ("validate_structure", fake_structure, "the reference topology") in calls
    assert ("validate_residue_type", "HID", "reference/original state") in calls
    assert ("validate_residue_type", "HIE", "mutation target") in calls
    assert all(call[0] != "validate_openmm" for call in calls)


def test_runtime_protonation_preflight_uses_mutation_path_for_perturbed_state(
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    reference_pose = object()
    perturbed_pose = object()
    reference_structure = object()
    perturbed_structure = object()
    prepared_reference_structure = object()
    prepared_perturbed_structure = object()

    class _FakeConversionService:
        def pose_to_structure(self, pose):
            calls.append(("pose_to_structure", pose))
            if pose is reference_pose:
                return reference_structure
            if pose is perturbed_pose:
                return perturbed_structure
            raise AssertionError("unexpected pose")

        def prepare_structure_for_openmm(self, structure):
            calls.append(("prepare_for_openmm", structure))
            if structure is reference_structure:
                return prepared_reference_structure
            if structure is perturbed_structure:
                return prepared_perturbed_structure
            raise AssertionError("unexpected structure")

    class _FakeEnergyEvaluator:
        def validate_topology(self, structure) -> None:
            calls.append(("validate_openmm", structure))

    class _FakePerturbationProtocol:
        def mutate_pose(
            self,
            pose,
            target_residue_number: int,
            target_residue_name: str,
            restraint_specs=(),
        ):
            calls.append(
                (
                    "mutate_pose",
                    pose,
                    target_residue_number,
                    target_residue_name,
                    tuple(restraint_specs),
                )
            )
            assert pose is reference_pose
            return perturbed_pose

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            target_residue_number=2,
            target_residue_name="TYR",
            original_residue_name="GLY",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    monkeypatch.setattr(executor_module, "_WORKER_CONFIG", config)
    monkeypatch.setattr(
        executor_module,
        "_WORKER_CONVERSION_SERVICE",
        _FakeConversionService(),
    )
    monkeypatch.setattr(
        executor_module,
        "_WORKER_PERTURBATION_PROTOCOL",
        _FakePerturbationProtocol(),
    )
    monkeypatch.setattr(
        executor_module,
        "_WORKER_ENERGY_EVALUATOR",
        _FakeEnergyEvaluator(),
    )
    monkeypatch.setattr(executor_module, "_WORKER_RUNTIME_PROTONATION_PRECHECKED", False)
    monkeypatch.setattr(
        executor_module,
        "ensure_pose_residue_matches_expected",
        lambda pose, residue_number, expected_residue_name, *, context: calls.append(
            ("match_pose", pose, residue_number, expected_residue_name, context)
        ),
    )

    executor_module._run_runtime_protonation_preflight(reference_pose)
    executor_module._run_runtime_protonation_preflight(reference_pose)

    assert ("match_pose", reference_pose, 2, "GLY", "reference-state setup") in calls
    assert ("mutate_pose", reference_pose, 2, "TYR", ()) in calls
    assert ("match_pose", perturbed_pose, 2, "TYR", "perturbed-state setup") in calls
    assert ("validate_openmm", prepared_reference_structure) in calls
    assert ("validate_openmm", prepared_perturbed_structure) in calls
    assert calls.count(("mutate_pose", reference_pose, 2, "TYR", ())) == 1
