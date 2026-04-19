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
    ensure_perturbation_matches_structure,
    ensure_pose_residue_matches_expected,
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


def test_passes_when_pose_family_matches_requested_protonation_variant() -> None:
    residue = SimpleNamespace(name3=lambda: "HIS")
    pose = SimpleNamespace(residue=lambda _: residue)
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="ALA",
        original_residue_name="HIE",
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


def test_structure_validation_passes_when_names_match() -> None:
    structure = SimpleNamespace(residues=[SimpleNamespace(name="GLY"), SimpleNamespace(name="HIE")])
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="ALA",
        original_residue_name="HIE",
    )
    ensure_perturbation_matches_structure(structure, perturbation)


def test_structure_validation_normalizes_common_variant_aliases() -> None:
    structure = SimpleNamespace(residues=[SimpleNamespace(name="CYZ")])
    perturbation = PerturbationConfig(
        target_residue_number=1,
        target_residue_name="ALA",
        original_residue_name="CYM",
    )
    ensure_perturbation_matches_structure(structure, perturbation)


def test_structure_validation_passes_when_family_matches_requested_variant() -> None:
    structure = SimpleNamespace(residues=[SimpleNamespace(name="GLU")])
    perturbation = PerturbationConfig(
        target_residue_number=1,
        target_residue_name="ALA",
        original_residue_name="GLH",
    )
    ensure_perturbation_matches_structure(structure, perturbation)


def test_structure_validation_raises_on_mismatch() -> None:
    structure = SimpleNamespace(residues=[SimpleNamespace(name="HIS"), SimpleNamespace(name="GLU")])
    perturbation = PerturbationConfig(
        target_residue_number=2,
        target_residue_name="ALA",
        original_residue_name="HIE",
    )
    with pytest.raises(ConversionError, match="topology residue index"):
        ensure_perturbation_matches_structure(structure, perturbation)


def test_pose_residue_matches_expected_allows_explicit_histidine_variant() -> None:
    residue = SimpleNamespace(name3=lambda: " HIE ")
    pose = SimpleNamespace(residue=lambda _: residue)
    ensure_pose_residue_matches_expected(
        pose,
        2,
        "HIE",
        context="Rosetta-relaxed perturbed state",
    )


def test_pose_residue_matches_expected_allows_opt_h_histidine_tautomer_choice() -> None:
    residue = SimpleNamespace(name3=lambda: "HID")
    pose = SimpleNamespace(residue=lambda _: residue)
    ensure_pose_residue_matches_expected(
        pose,
        2,
        "HIS",
        context="Rosetta-relaxed perturbed state",
    )


def test_pose_residue_matches_expected_detects_protonated_aspartate_from_atoms() -> None:
    residue = SimpleNamespace(
        name3=lambda: "ASP",
        natoms=lambda: 3,
        atom_name=lambda index: ("OD1", "OD2", "HD2")[index - 1],
    )
    pose = SimpleNamespace(residue=lambda _: residue)
    ensure_pose_residue_matches_expected(
        pose,
        2,
        "ASH",
        context="Rosetta-relaxed perturbed state",
    )


def test_pose_residue_matches_expected_rejects_histidine_tautomer_flip() -> None:
    residue = SimpleNamespace(name3=lambda: "HID")
    pose = SimpleNamespace(residue=lambda _: residue)
    with pytest.raises(ConversionError, match="changed residue identity"):
        ensure_pose_residue_matches_expected(
            pose,
            2,
            "HIE",
            context="Rosetta-relaxed perturbed state",
        )


def test_pose_residue_matches_expected_rejects_deprotonated_glutamate_for_glh() -> None:
    residue = SimpleNamespace(
        name3=lambda: "GLU",
        natoms=lambda: 2,
        atom_name=lambda index: ("OE1", "OE2")[index - 1],
    )
    pose = SimpleNamespace(residue=lambda _: residue)
    with pytest.raises(ConversionError, match="expected 'GLH'"):
        ensure_pose_residue_matches_expected(
            pose,
            2,
            "GLH",
            context="Rosetta-relaxed perturbed state",
        )


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
