from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

from felp.core import (
    CVMeasurement,
    DihedralRestraintSpec,
    EnergyConfig,
    EnergyResult,
    FrameResult,
    InputConfig,
    MicrostateAssignment,
    MicrostateConfig,
    MutationSpec,
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
        perturbation=PerturbationConfig.from_legacy(target_residue_number=2, target_residue_name="ALA"),
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


def test_frame_executor_uses_configured_chunk_size(monkeypatch) -> None:
    submitted_batch_sizes: list[int] = []

    def fake_initialize(config, atlas, assignment_policy=None) -> None:
        return None

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

    class FakeProcessPoolExecutor:
        def __init__(self, *args, initializer=None, initargs=(), **kwargs) -> None:
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def submit(self, function, batch):
            submitted_batch_sizes.append(len(batch))
            future: Future = Future()
            future.set_result(function(batch))
            return future

    monkeypatch.setattr(executor_module, "initialize_worker", fake_initialize)
    monkeypatch.setattr(executor_module, "safe_process_frame_job", fake_process)
    monkeypatch.setattr(executor_module, "ProcessPoolExecutor", FakeProcessPoolExecutor)

    jobs = [
        SnapshotJob(
            frame_index=index,
            stride_number=index,
            coordinates_angstrom=((float(index), 0.0, 0.0),),
        )
        for index in range(5)
    ]
    config = replace(
        _config(),
        parallel=ParallelConfig(workers=2, chunk_size=2, progress_mode="off"),
    )

    results = executor_module.FrameExecutor(
        config,
        MicrostateAtlas(definitions=()),
    ).run(jobs)

    assert submitted_batch_sizes == [2, 2, 1]
    assert [result.frame_index for result in results] == [0, 1, 2, 3, 4]


def test_apply_cpu_thread_limits_sets_openmm_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENMM_CPU_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(target_residue_number=2, target_residue_name="ALA"),
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
        perturbation=PerturbationConfig.from_legacy(target_residue_number=2, target_residue_name="ALA"),
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
        perturbation=PerturbationConfig.from_legacy(target_residue_number=2, target_residue_name="ALA"),
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
        perturbation=PerturbationConfig.from_legacy(
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
        perturbation=PerturbationConfig.from_legacy(
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


def test_build_pyrosetta_options_enable_repack_trials_and_symmetric_gly() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(
            target_residue_number=2,
            target_residue_name="GLY",
            original_residue_name="GLY",
        ),
        rosetta=RosettaConfig(
            perturbation_mode="repack",
            repack_trials=4,
            symmetric_gly_tables=True,
        ),
        restraints=RestraintConfig(),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    options = executor_module._build_pyrosetta_options(config)

    assert "-packing:ndruns 4" in options
    assert "-symmetric_gly_tables true" in options


def test_build_pyrosetta_options_aggregates_histidine_behavior_across_all_mutations() -> None:
    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            mutations=(
                MutationSpec(
                    residue_number=2,
                    target_residue_name="HIS",
                    original_residue_name="GLY",
                ),
                MutationSpec(
                    residue_number=10,
                    target_residue_name="HIP",
                    original_residue_name="HIS",
                ),
            ),
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
    assert "-keep_input_protonation_state" in options



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
        perturbation=PerturbationConfig.from_legacy(
            target_residue_number=2,
            target_residue_name="HIE",
            original_residue_name="HID",
        ),
        rosetta=RosettaConfig(
            perturbation_mode="repack",
            repack_trials=3,
            symmetric_gly_tables=True,
        ),
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
            (
                "match_structure",
                structure,
                tuple(m.original_residue_name for m in perturbation.mutations),
            )
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
        lambda *args, **kwargs: _FakeEnergyEvaluator(),
    )

    executor_module.initialize_worker(config, MicrostateAtlas(definitions=()))

    assert ("load_topology", "input.parm7") in calls
    assert ("match_structure", fake_structure, ("HID",)) in calls
    assert ("validate_structure", fake_structure, "the reference topology") in calls
    assert ("validate_residue_type", "HID", "reference/original state") in calls
    assert ("validate_residue_type", "HIE", "mutation target") in calls
    assert all(call[0] != "validate_openmm" for call in calls)
    protocol_call = next(call for call in calls if call[0] == "protocol")
    assert protocol_call[1]["mutation_residue_numbers"] == (2,)
    assert protocol_call[1]["perturbation_mode"] == "repack"
    assert protocol_call[1]["repack_trials"] == 3


def test_initialize_worker_validates_all_mutation_roles(monkeypatch) -> None:
    calls: list[tuple] = []
    fake_structure = object()

    class _MiniConversionService:
        def __init__(self, **_kwargs) -> None:
            calls.append(("service_init",))

        def load_topology(self, topology_file: str):
            calls.append(("load_topology", topology_file))
            return fake_structure

        def validate_rosetta_structure_residue_types(self, structure, *, context: str) -> None:
            calls.append(("validate_structure", context))

        def validate_rosetta_residue_type(self, residue_name: str, *, context: str) -> None:
            calls.append(("validate_residue_type", residue_name, context))

    config = RunConfig(
        input=InputConfig(topology_file="top.pdb", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            mutations=(
                MutationSpec(
                    residue_number=2,
                    target_residue_name="ALA",
                    original_residue_name="GLY",
                ),
                MutationSpec(residue_number=6, target_residue_name="VAL"),
            ),
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )

    monkeypatch.setattr(executor_module, "_apply_cpu_thread_limits", lambda _c: None)
    monkeypatch.setattr(executor_module, "StructureConversionService", _MiniConversionService)
    monkeypatch.setattr(
        executor_module,
        "ensure_perturbation_matches_structure",
        lambda structure, perturbation: calls.append(
            ("match_structure", tuple(m.original_residue_name for m in perturbation.mutations)),
        ),
    )
    monkeypatch.setattr(executor_module, "BackboneDihedralCVProtocol", lambda **_: object())
    monkeypatch.setattr(executor_module, "PyRosettaPerturbationProtocol", lambda **_: object())

    class _MinimalEnergyEvaluator:
        def validate_topology(self, structure) -> None:
            calls.append(("validate_topology",))

    monkeypatch.setattr(
        executor_module,
        "build_energy_evaluator",
        lambda *args, **kwargs: _MinimalEnergyEvaluator(),
    )

    executor_module.initialize_worker(config, MicrostateAtlas(definitions=()))

    assert ("match_structure", ("GLY", None)) in calls
    assert ("validate_residue_type", "ALA", "mutation target") in calls
    assert ("validate_residue_type", "VAL", "mutation target") in calls
    assert ("validate_residue_type", "GLY", "reference/original state") in calls


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
            mutations,
            restraint_specs=(),
        ):
            calls.append(
                (
                    "mutate_pose",
                    pose,
                    mutations,
                    tuple(restraint_specs),
                )
            )
            assert pose is reference_pose
            return perturbed_pose

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(
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
    mutations = config.perturbation.mutations
    assert ("mutate_pose", reference_pose, mutations, ()) in calls
    assert ("match_pose", perturbed_pose, 2, "TYR", "perturbed-state setup") in calls
    assert ("validate_openmm", prepared_reference_structure) in calls
    assert ("validate_openmm", prepared_perturbed_structure) in calls
    assert calls.count(("mutate_pose", reference_pose, mutations, ())) == 1


def test_rosetta_target_backbone_restraints_skip_terminal_dihedrals(monkeypatch) -> None:
    from felp.parallel import executor as executor_module

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(
            mutations=(
                MutationSpec(residue_number=1, target_residue_name="TYR"),
                MutationSpec(residue_number=10, target_residue_name="TYR"),
            )
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(
            rosetta_target_backbone_restraints=("phi", "psi", "omega"),
            rosetta_backbone_restraint_residue_numbers=(1, 2, 9, 10),
        ),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(),
        parallel=ParallelConfig(),
        output=OutputConfig(output_dir="unused"),
    )
    monkeypatch.setattr(executor_module, "_WORKER_CONFIG", config)

    class _Pose:
        def total_residue(self) -> int:
            return 10

        def phi(self, residue_number: int) -> float:
            return float(residue_number)

        def psi(self, residue_number: int) -> float:
            return float(residue_number)

        def omega(self, residue_number: int) -> float:
            return float(residue_number)

    restraints = executor_module._rosetta_target_backbone_restraints(_Pose())

    assert [(spec.kind, spec.residue_numbers) for spec in restraints] == [
        ("psi", (1, 1, 1, 2)),
        ("omega", (1, 1, 2, 2)),
        ("phi", (1, 2, 2, 2)),
        ("psi", (2, 2, 2, 3)),
        ("omega", (2, 2, 3, 3)),
        ("phi", (8, 9, 9, 9)),
        ("psi", (9, 9, 9, 10)),
        ("omega", (9, 9, 10, 10)),
        ("phi", (9, 10, 10, 10)),
    ]


def test_openmm_frame_path_prepares_reference_before_mutation(monkeypatch) -> None:
    calls: list[tuple] = []
    template_structure = object()
    raw_structure = object()
    reference_pose = "reference-pose"
    enforced_reference_pose = "enforced-reference-pose"
    prepared_reference_pose = "prepared-reference-pose"
    perturbed_pose = "perturbed-pose"
    reference_structure = "reference-structure"
    perturbed_structure = "perturbed-structure"
    reference_openmm_structure = "reference-openmm-structure"
    perturbed_openmm_structure = "perturbed-openmm-structure"
    reference_minimized_structure = "reference-minimized-structure"
    perturbed_minimized_structure = "perturbed-minimized-structure"
    restraint_specs = (
        DihedralRestraintSpec(
            label="phi",
            kind="phi",
            residue_numbers=(1, 2, 2, 2),
            atom_names=("C", "N", "CA", "C"),
            target_angle_degrees=-60.0,
            force_constant=10000.0,
        ),
    )

    class _FakeConversionService:
        def clone_structure_with_coordinates(self, structure, coordinates):
            calls.append(("clone_structure", structure, coordinates))
            if structure is template_structure:
                return raw_structure
            if structure == reference_openmm_structure:
                return reference_minimized_structure
            if structure == perturbed_openmm_structure:
                return perturbed_minimized_structure
            raise AssertionError("unexpected clone")

        def structure_to_pose(self, structure):
            calls.append(("structure_to_pose", structure))
            assert structure is raw_structure
            return reference_pose

        def pose_to_structure(self, pose):
            calls.append(("pose_to_structure", pose))
            if pose == prepared_reference_pose:
                return reference_structure
            if pose == perturbed_pose:
                return perturbed_structure
            raise AssertionError("unexpected pose")

        def prepare_structure_for_openmm(self, structure):
            calls.append(("prepare_for_openmm", structure))
            if structure == reference_structure:
                return reference_openmm_structure
            if structure == perturbed_structure:
                return perturbed_openmm_structure
            raise AssertionError("unexpected structure")

    class _FakePerturbationProtocol:
        def enforce_residue_identity(
            self,
            pose,
            mutations,
        ):
            calls.append(("enforce", pose, mutations))
            assert pose == reference_pose
            return enforced_reference_pose

        def prepare_reference_pose(self, pose, restraints, frame_index):
            calls.append(("prepare_reference", pose, tuple(restraints), frame_index))
            assert pose == enforced_reference_pose
            return prepared_reference_pose

        def apply_perturbation(
            self,
            pose,
            mutations,
            restraints,
            frame_index,
        ):
            calls.append(
                (
                    "apply_perturbation",
                    pose,
                    mutations,
                    tuple(restraints),
                    frame_index,
                )
            )
            assert pose == prepared_reference_pose
            return perturbed_pose

    class _FakeRestraintProtocol:
        def measure_pose(self, pose):
            calls.append(("measure_pose", pose))
            return CVMeasurement({"phi": -60.0})

        def build_from_reference_pose(self, pose):
            calls.append(("build_restraints", pose))
            assert pose == enforced_reference_pose
            return list(restraint_specs)

        def measure_structure(self, structure):
            calls.append(("measure_structure", structure))
            return CVMeasurement({"phi": -60.0})

    class _FakeAssignmentPolicy:
        name = "fake"

        def assign(self, atlas, job, measurement):
            calls.append(("assign", job.frame_index, measurement.as_dict()))
            return MicrostateAssignment(
                state_id="state",
                label="state",
                measurements=measurement.as_dict(),
                state_level="microstate",
            )

    class _FakeEnergyEvaluator:
        def evaluate(
            self,
            *,
            structure=None,
            pose=None,
            restraint_specs=(),
            collect_energy_components=False,
        ):
            calls.append(("evaluate", structure, pose, tuple(restraint_specs)))
            if structure == reference_openmm_structure:
                return EnergyResult(
                    backend="openmm",
                    energy_kj_mol=1.0,
                    minimized_coordinates_angstrom=((0.0, 0.0, 0.0),),
                )
            if structure == perturbed_openmm_structure:
                return EnergyResult(
                    backend="openmm",
                    energy_kj_mol=3.0,
                    minimized_coordinates_angstrom=((1.0, 0.0, 0.0),),
                )
            raise AssertionError("unexpected energy structure")

    config = RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig.from_legacy(
            target_residue_number=2,
            target_residue_name="ALA",
            original_residue_name="GLY",
        ),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(backend="openmm"),
        microstates=MicrostateConfig(reference_probability_mode="raw_counts"),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
    )
    monkeypatch.setattr(executor_module, "_WORKER_CONFIG", config)
    monkeypatch.setattr(
        executor_module,
        "_WORKER_ATLAS",
        MicrostateAtlas(definitions=()),
    )
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
        "_WORKER_RESTRAINT_PROTOCOL",
        _FakeRestraintProtocol(),
    )
    monkeypatch.setattr(executor_module, "_WORKER_TEMPLATE_STRUCTURE", template_structure)
    monkeypatch.setattr(executor_module, "_WORKER_ASSIGNMENT_POLICY", _FakeAssignmentPolicy())
    monkeypatch.setattr(executor_module, "_WORKER_ENERGY_EVALUATOR", _FakeEnergyEvaluator())
    monkeypatch.setattr(executor_module, "_WORKER_RUNTIME_PROTONATION_PRECHECKED", True)
    monkeypatch.setattr(
        executor_module,
        "ensure_pose_residue_matches_expected",
        lambda *args, **kwargs: calls.append(("match_pose", args, kwargs)),
    )

    result = executor_module.process_frame_job(
        SnapshotJob(
            frame_index=7,
            stride_number=7,
            coordinates_angstrom=((0.0, 0.0, 0.0),),
        )
    )

    assert result.delta_u_kj_mol == 2.0
    mutations = config.perturbation.mutations
    assert calls.index(("prepare_reference", enforced_reference_pose, restraint_specs, 7)) < calls.index(
        ("apply_perturbation", prepared_reference_pose, mutations, restraint_specs, 7)
    )
    assert ("pose_to_structure", prepared_reference_pose) in calls
    protocol = result.metadata["protocol"]
    assert protocol["mobility"]["selected_residue_scope"]["residue_numbers"] == [2]
    assert protocol["rosetta_preparation"]["movemap"]["mobile_backbone_scope"] == (
        "no backbone torsions"
    )
    assert protocol["rosetta_preparation"]["task_factory"]["design_allowed"] is False
    assert protocol["openmm_final_minimization"]["mass_frozen_dofs"][
        "mobile_sidechain_scope"
    ] == "non-backbone atoms on selected residues"
    assert protocol["rosetta_preparation"]["constraints"]["runtime_restraint_specs"][
        "specs"
    ][0][
        "label"
    ] == "phi"
