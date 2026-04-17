"""Tests for perturbation target validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from felp.core import (
    CVMeasurement,
    ConversionError,
    DihedralRestraintSpec,
    PerturbationConfig,
    PhiPsiMeasurement,
)
from felp.perturbation.validation import (
    ensure_measurement_matches_restraints,
    ensure_perturbation_matches_pose,
    measurement_deviation_degrees,
)


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


def test_passes_when_common_variant_aliases_normalize() -> None:
    residue = SimpleNamespace(name3=lambda: " HIS_D ")
    pose = SimpleNamespace(residue=lambda _: residue)
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="hip",
        original_residue_name="hid",
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


def _restraint_specs() -> list[DihedralRestraintSpec]:
    return [
        DihedralRestraintSpec(
            label="phi",
            kind="phi",
            residue_numbers=(1, 2, 2, 2),
            atom_names=("C", "N", "CA", "C"),
            target_angle_degrees=-60.0,
            force_constant=10.0,
        ),
        DihedralRestraintSpec(
            label="psi",
            kind="psi",
            residue_numbers=(2, 2, 2, 3),
            atom_names=("N", "CA", "C", "N"),
            target_angle_degrees=140.0,
            force_constant=10.0,
        ),
    ]


def test_measurement_deviation_degrees_uses_circular_difference() -> None:
    measurement = PhiPsiMeasurement(phi_degrees=-59.5, psi_degrees=139.0)
    deviation = measurement_deviation_degrees(measurement, _restraint_specs())
    assert deviation == {"phi": 0.5, "psi": 1.0}


def test_measurement_deviation_degrees_supports_generalized_cv_maps() -> None:
    restraints = [
        DihedralRestraintSpec(
            label="phi",
            kind="phi",
            residue_numbers=(1, 2, 2, 2),
            atom_names=("C", "N", "CA", "C"),
            target_angle_degrees=-60.0,
            force_constant=10.0,
        ),
        DihedralRestraintSpec(
            label="omega",
            kind="omega",
            residue_numbers=(2, 2, 3, 3),
            atom_names=("CA", "C", "N", "CA"),
            target_angle_degrees=180.0,
            force_constant=10.0,
        ),
    ]
    measurement = CVMeasurement({"phi": -60.0, "omega": 179.5})
    deviation = measurement_deviation_degrees(measurement, restraints)
    assert deviation == {"phi": 0.0, "omega": 0.5}


def test_ensure_measurement_matches_restraints_passes_within_tolerance() -> None:
    measurement = PhiPsiMeasurement(phi_degrees=-59.5, psi_degrees=139.5)
    ensure_measurement_matches_restraints(
        measurement,
        _restraint_specs(),
        tolerance_degrees=1.0,
    )


def test_ensure_measurement_matches_restraints_raises_on_drift() -> None:
    measurement = PhiPsiMeasurement(phi_degrees=-56.0, psi_degrees=139.0)
    with pytest.raises(ConversionError, match="left the fixed backbone-CV state"):
        ensure_measurement_matches_restraints(
            measurement,
            _restraint_specs(),
            tolerance_degrees=1.0,
            context="OpenMM-minimized perturbed state",
        )
