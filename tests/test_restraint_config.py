"""Tests for restraint configuration validation."""

from __future__ import annotations

import pytest

from felp.core import ConfigurationError, RestraintConfig, default_backbone_cv_specs


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
