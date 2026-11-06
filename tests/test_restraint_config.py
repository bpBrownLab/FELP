"""Tests for restraint configuration validation."""

from __future__ import annotations

import pytest

from felp.core import (
    AtomReference,
    ConfigurationError,
    DistanceCVSpec,
    MobilityConfig,
    RestraintConfig,
    default_backbone_cv_specs,
)


def test_restraint_config_cv_tolerance_must_be_positive() -> None:
    with pytest.raises(ConfigurationError, match="cv_measurement_tolerance"):
        RestraintConfig(
            cv_measurement_tolerance_degrees=0.0,
            cv_specs=default_backbone_cv_specs(2),
        )


def test_restraint_config_optional_force_constants_must_be_positive() -> None:
    specs = default_backbone_cv_specs(2)
    with pytest.raises(ConfigurationError, match="rosetta_restraint_force"):
        RestraintConfig(
            cv_specs=specs,
            rosetta_restraint_force_constant=0.0,
        )
    with pytest.raises(ConfigurationError, match="openmm_restraint_force"):
        RestraintConfig(
            cv_specs=specs,
            openmm_restraint_force_constant=-1.0,
        )


def test_restraint_config_rosetta_distance_sd_must_be_positive() -> None:
    specs = default_backbone_cv_specs(2)
    with pytest.raises(ConfigurationError, match="rosetta_distance_constraint_sd"):
        RestraintConfig(
            cv_specs=specs,
            rosetta_distance_constraint_sd_angstrom=0.0,
        )


def test_restraint_config_rosetta_target_backbone_restraints_are_normalized() -> None:
    config = RestraintConfig(
        rosetta_target_backbone_restraints=("PHI", "psi", "phi"),
        rosetta_target_backbone_restraint_force_constant=5000.0,
        rosetta_target_backbone_restraint_tolerance_degrees=0.5,
    )

    assert config.rosetta_target_backbone_restraints == ("phi", "psi")
    assert config.rosetta_backbone_restraint_residue_numbers == ()
    assert config.effective_rosetta_target_backbone_force_constant() == 5000.0

    window_config = RestraintConfig(
        rosetta_target_backbone_restraints=("phi",),
        rosetta_backbone_restraint_residue_numbers=(3, 4, 3, 5),
    )
    assert window_config.rosetta_backbone_restraint_residue_numbers == (3, 4, 5)


def test_restraint_config_rejects_invalid_rosetta_target_backbone_restraints() -> None:
    with pytest.raises(ConfigurationError, match="unsupported values"):
        RestraintConfig(rosetta_target_backbone_restraints=("chi1",))

    with pytest.raises(ConfigurationError, match="target_backbone_restraint_force"):
        RestraintConfig(rosetta_target_backbone_restraint_force_constant=0.0)

    with pytest.raises(ConfigurationError, match="target_backbone_restraint_tolerance"):
        RestraintConfig(rosetta_target_backbone_restraint_tolerance_degrees=0.0)

    with pytest.raises(ConfigurationError, match="positive pose indices"):
        RestraintConfig(rosetta_backbone_restraint_residue_numbers=(0,))


def test_restraint_config_accepts_distance_cv_specs() -> None:
    config = RestraintConfig(
        cv_specs=default_backbone_cv_specs(2),
        distance_cv_specs=(
            DistanceCVSpec(
                name="q61_switch",
                atom1=AtomReference(61, "CA"),
                atom2=AtomReference(35, "CA"),
            ),
        ),
    )

    assert config.distance_cv_specs[0].name == "q61_switch"


def test_mobility_config_defaults_to_target_only_sidechain() -> None:
    config = MobilityConfig()

    assert config.mode == "target_only"
    assert config.sequence_window == 0
    assert config.include_sidechains is True
    assert config.include_backbone is False


def test_mobility_config_accepts_sequence_window() -> None:
    config = MobilityConfig(mode="sequence_window", sequence_window=1)

    assert config.mode == "sequence_window"
    assert config.sequence_window == 1

    with pytest.raises(ConfigurationError, match="sequence_window"):
        MobilityConfig(mode="sequence_window", sequence_window=-1)
