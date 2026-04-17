"""Tests for perturbation target validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from felp.core import ConversionError, PerturbationConfig
from felp.perturbation.validation import ensure_perturbation_matches_pose


def test_skips_when_original_residue_not_set() -> None:
    pose = SimpleNamespace()
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="ALA",
        original_residue_name=None,
    )
    ensure_perturbation_matches_pose(pose, perturbation)


def test_passes_when_names_match() -> None:
    residue = SimpleNamespace(name3=lambda: " GLY ")
    pose = SimpleNamespace(residue=lambda _: residue)
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="ALA",
        original_residue_name="GLY",
    )
    ensure_perturbation_matches_pose(pose, perturbation)


def test_raises_on_mismatch() -> None:
    residue = SimpleNamespace(name3=lambda: "ALA")
    pose = SimpleNamespace(residue=lambda _: residue)
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="GLY",
        original_residue_name="GLY",
    )
    with pytest.raises(ConversionError, match="target mismatch"):
        ensure_perturbation_matches_pose(pose, perturbation)
