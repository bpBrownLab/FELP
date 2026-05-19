"""Tests for ess-compare block bootstrap helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from felp.core import (
    ConfigurationError,
    FrameResult,
    MicrostateConfig,
    MicrostateDefinition,
)
from felp.plotting.ess import ESSCompareConfig
from felp.plotting.ess_bootstrap import (
    _atlas_empirical_priors_for_bootstrap,
    _frame_state_atlas_for_bootstrap,
    _resolve_microstate_paths_for_run,
    bootstrap_run_directory,
    percentile_summary,
)
from felp.reweighting import MicrostateAtlas


def test_ess_compare_config_bootstrap_requires_both(tmp_path) -> None:
    summary_a = tmp_path / "a.json"
    summary_b = tmp_path / "b.json"
    summary_a.write_text("{}", encoding="utf-8")
    summary_b.write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ConfigurationError, match="both"):
        ESSCompareConfig(
            input_paths=(str(summary_a), str(summary_b)),
            output_dir=str(out),
            bootstrap_block_size=5,
            bootstrap_draws=None,
        ).validate()


def test_percentile_summary_ignores_nan() -> None:
    med, lo, hi = percentile_summary(np.array([1.0, 2.0, 3.0, np.nan]))
    assert np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= med <= hi


def test_bootstrap_run_directory_prefers_directory_containing_summary(tmp_path) -> None:
    """Relative output_dir in summary must not override the real run folder on disk."""

    run = tmp_path / "qc_exclude_45"
    run.mkdir()
    summary_path = run / "run_summary.json"
    summary_path.write_text(
        json.dumps({"output_dir": "grid_histogram_priors/qc_exclude_45"}),
        encoding="utf-8",
    )
    (run / "run_config.json").write_text("{}", encoding="utf-8")

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    resolved = bootstrap_run_directory(summary, str(summary_path))
    assert resolved == str(run.resolve())


def test_bootstrap_run_directory_resolves_relative_output_dir_to_summary_parent(
    tmp_path,
) -> None:
    root = tmp_path / "run_root"
    root.mkdir()
    sub = root / "nested"
    sub.mkdir(parents=True)
    summary_path = root / "run_summary.json"
    summary_path.write_text(json.dumps({"output_dir": "nested"}), encoding="utf-8")
    (sub / "run_config.json").write_text("{}", encoding="utf-8")

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert bootstrap_run_directory(summary, str(summary_path)) == str(sub.resolve())


def test_bootstrap_run_directory_finds_run_config_in_ancestor(tmp_path) -> None:
    root = tmp_path / "felp_GLY_to_ALA_full.from_msm_bundle"
    nested = root / "grid_histogram_priors" / "qc_exclude_45"
    nested.mkdir(parents=True)
    (root / "run_config.json").write_text("{}", encoding="utf-8")
    summary_path = nested / "run_summary.json"
    summary_path.write_text(
        json.dumps({"output_dir": "grid_histogram_priors/qc_exclude_45"}),
        encoding="utf-8",
    )
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert bootstrap_run_directory(summary, str(summary_path)) == str(root.resolve())


def test_resolve_msm_bundle_dir_against_data_parent_not_bundle_root(tmp_path) -> None:
    """Paths like msm_bundle_X are often relative to data/, not felp_*_bundle/."""

    data = tmp_path / "data"
    gly = data / "GLY"
    bundle = gly / "felp_GLY_to_ALA_full.from_msm_bundle"
    nested = bundle / "grid_histogram_priors" / "qc_exclude_45"
    nested.mkdir(parents=True)
    msm = data / "msm_bundle_GLY"
    msm.mkdir()
    (msm / "manifest.json").write_text("{}", encoding="utf-8")

    mc = MicrostateConfig(source="msm_bundle", msm_bundle_dir="msm_bundle_GLY")
    resolved = _resolve_microstate_paths_for_run(
        mc,
        config_dir=str(bundle),
        artifact_dir=str(nested),
    )
    assert resolved.msm_bundle_dir == str(msm.resolve())


def test_atlas_empirical_priors_for_bootstrap_drops_loaded_mode() -> None:
    definition = MicrostateDefinition(
        state_id="s0",
        label="s0",
        cv_ranges={"phi": (-180.0, 0.0), "psi": (-180.0, 0.0)},
        reference_probability=0.99,
    )
    atlas = MicrostateAtlas.from_definitions(
        definitions=(definition,),
        reference_probability_mode="loaded",
    )
    boot = _atlas_empirical_priors_for_bootstrap(atlas)
    assert boot.reference_probability_mode == "raw_counts"
    assert boot.definitions[0].reference_probability == 0.99


def test_frame_state_atlas_for_bootstrap_uses_actual_frame_state_ids() -> None:
    """Nested output frame states can differ from the ancestor run_config atlas."""

    base = MicrostateAtlas.from_definitions(
        definitions=(
            MicrostateDefinition(
                state_id="msm_0",
                label="msm_0",
                reference_probability=1.0,
            ),
        ),
        reference_probability_mode="loaded",
    )
    frames = [
        FrameResult(
            frame_index=0,
            stride_number=0,
            microstate_id="phi0_psi0",
            microstate_label="phi0_psi0",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=1.0,
            delta_u_kj_mol=1.0,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            state_level="cluster",
        ),
        FrameResult(
            frame_index=1,
            stride_number=1,
            microstate_id="phi1_psi1",
            microstate_label="phi1_psi1",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=2.0,
            delta_u_kj_mol=2.0,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            state_level="cluster",
        ),
    ]

    boot = _frame_state_atlas_for_bootstrap(base.allow_unassigned, frames)
    assert boot.reference_probability_mode == "raw_counts"
    assert {d.state_id for d in boot.definitions} == {"phi0_psi0", "phi1_psi1"}


def test_bootstrap_run_directory_errors_when_missing_run_config(tmp_path) -> None:
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(json.dumps({"output_dir": "missing"}), encoding="utf-8")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    with pytest.raises(ConfigurationError, match="run_config"):
        bootstrap_run_directory(summary, str(summary_path))
