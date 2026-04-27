from __future__ import annotations

from pathlib import Path

import pytest

from felp.core import (
    ConfigurationError,
    CVMeasurement,
    EnergyConfig,
    InputConfig,
    MicrostateConfig,
    MicrostateDefinition,
    OutputConfig,
    ParallelConfig,
    PerturbationConfig,
    RestraintConfig,
    RosettaConfig,
    RunConfig,
    SamplingConfig,
    SnapshotJob,
)
from felp.reweighting import MicrostateAtlas
from felp.reweighting.sampling import (
    assign_jobs_for_sampling,
    build_sampling_manifest,
    ensure_existing_results_match_manifest,
    load_or_create_sampling_manifest,
    select_microstate_stratified_frame_indices,
    selected_jobs_from_manifest,
)


def _job(frame_index: int, state_id: str | None = None, x: float = 0.0) -> SnapshotJob:
    return SnapshotJob(
        frame_index=frame_index,
        stride_number=frame_index,
        coordinates_angstrom=((x, 0.0, 0.0),),
        precomputed_state_id=state_id,
    )


def _atlas() -> MicrostateAtlas:
    return MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        ),
        reference_probability_mode="raw_counts",
    )


def _config(*, sampling: SamplingConfig, source: str = "msm_bundle") -> RunConfig:
    return RunConfig(
        input=InputConfig(topology_file="unused", trajectory_files=(("unused",),)),
        perturbation=PerturbationConfig(target_residue_number=2, target_residue_name="ALA"),
        rosetta=RosettaConfig(),
        restraints=RestraintConfig(),
        energy=EnergyConfig(),
        microstates=MicrostateConfig(
            source=source,
            msm_bundle_dir="unused" if source == "msm_bundle" else None,
            reference_probability_mode="raw_counts",
        ),
        parallel=ParallelConfig(workers=1),
        output=OutputConfig(output_dir="unused"),
        sampling=sampling,
    )


def test_stratified_selection_is_deterministic_and_per_state() -> None:
    frame_state_ids = {
        **{index: "a" for index in range(10)},
        **{index: "b" for index in range(10, 15)},
    }
    sampling = SamplingConfig(
        microstate_sample_fraction=0.4,
        microstate_min_samples=2,
        microstate_sampling_seed=123,
    )

    selected_a, assigned_counts, selected_counts = select_microstate_stratified_frame_indices(
        frame_state_ids,
        sampling,
    )
    selected_b, _, _ = select_microstate_stratified_frame_indices(
        frame_state_ids,
        sampling,
    )

    assert selected_a == selected_b
    assert assigned_counts == {"a": 10, "b": 5}
    assert selected_counts == {"a": 4, "b": 2}
    assert len(selected_a) == 6


def test_stratified_selection_respects_min_max_and_noop() -> None:
    frame_state_ids = {0: "a", 1: "a", 2: "a"}

    selected, _, selected_counts = select_microstate_stratified_frame_indices(
        frame_state_ids,
        SamplingConfig(
            microstate_sample_fraction=0.1,
            microstate_min_samples=2,
            microstate_max_samples=2,
        ),
    )
    assert len(selected) == 2
    assert selected_counts == {"a": 2}

    all_selected, _, all_selected_counts = select_microstate_stratified_frame_indices(
        frame_state_ids,
        SamplingConfig(),
    )
    assert all_selected == (0, 1, 2)
    assert all_selected_counts == {"a": 3}


def test_manifest_uses_msm_precomputed_states_and_selects_jobs() -> None:
    jobs = [
        *(_job(index, "a") for index in range(4)),
        *(_job(index, "b") for index in range(4, 8)),
    ]
    config = _config(
        sampling=SamplingConfig(
            microstate_sample_fraction=0.5,
            microstate_min_samples=1,
            microstate_sampling_seed=7,
        )
    )

    manifest = build_sampling_manifest(jobs, atlas=_atlas(), config=config)
    selected_jobs = selected_jobs_from_manifest(jobs, manifest)

    assert manifest.assigned_counts == {"a": 4, "b": 4}
    assert manifest.selected_counts == {"a": 2, "b": 2}
    assert [job.frame_index for job in selected_jobs] == list(
        manifest.selected_frame_indices
    )


def test_grid_sampling_assigns_jobs_from_cheap_cv_measurements(monkeypatch) -> None:
    class _FakeConversionService:
        def __init__(self, pyrosetta_init_options: str, radii_set: str) -> None:
            self.loaded_topology = object()

        def load_topology(self, topology_file: str):
            return self.loaded_topology

        def clone_structure_with_coordinates(self, template_structure, coordinates_angstrom):
            return {"coordinates": coordinates_angstrom}

    class _FakeCVProtocol:
        def __init__(self, **kwargs) -> None:
            pass

        def measure_structure(self, structure) -> CVMeasurement:
            x = structure["coordinates"][0][0]
            return CVMeasurement({"phi": x, "psi": 0.0})

    monkeypatch.setattr(
        "felp.reweighting.sampling.StructureConversionService",
        _FakeConversionService,
    )
    monkeypatch.setattr(
        "felp.reweighting.sampling.BackboneDihedralCVProtocol",
        _FakeCVProtocol,
    )

    jobs = [_job(0, x=-60.0), _job(1, x=60.0)]
    config = _config(sampling=SamplingConfig(), source="grid")

    assignments = assign_jobs_for_sampling(jobs, atlas=_atlas(), config=config)

    assert assignments == {0: "a", 1: "b"}


def test_load_or_create_sampling_manifest_rejects_mismatch(tmp_path: Path) -> None:
    jobs = [_job(index, "a") for index in range(5)]
    manifest_path = tmp_path / "sampling_manifest.json"
    config = _config(
        sampling=SamplingConfig(
            microstate_sample_fraction=0.5,
            microstate_sampling_seed=1,
        )
    )
    load_or_create_sampling_manifest(
        manifest_path=str(manifest_path),
        jobs=jobs,
        atlas=_atlas(),
        config=config,
    )

    changed_config = _config(
        sampling=SamplingConfig(
            microstate_sample_fraction=0.5,
            microstate_sampling_seed=2,
        )
    )
    with pytest.raises(ConfigurationError, match="sampling manifest"):
        load_or_create_sampling_manifest(
            manifest_path=str(manifest_path),
            jobs=jobs,
            atlas=_atlas(),
            config=changed_config,
        )


def test_resume_results_must_match_manifest() -> None:
    manifest = build_sampling_manifest(
        [_job(index, "a") for index in range(4)],
        atlas=_atlas(),
        config=_config(
            sampling=SamplingConfig(
                microstate_sample_fraction=0.5,
                microstate_sampling_seed=3,
            )
        ),
    )

    with pytest.raises(ConfigurationError, match="not in the current sampling manifest"):
        ensure_existing_results_match_manifest([999], manifest)
