from __future__ import annotations

from felp.conversion import StructureConversionService
from felp.core import EnergyConfig
from felp.energy import OpenMMEnergyEvaluator, RosettaEnergyEvaluator, build_energy_evaluator


def test_build_energy_evaluator_openmm() -> None:
    evaluator = build_energy_evaluator(
        EnergyConfig(backend="openmm"),
        StructureConversionService(pyrosetta_init_options=""),
    )
    assert isinstance(evaluator, OpenMMEnergyEvaluator)


def test_build_energy_evaluator_rosetta() -> None:
    evaluator = build_energy_evaluator(
        EnergyConfig(backend="rosetta"),
        StructureConversionService(pyrosetta_init_options=""),
    )
    assert isinstance(evaluator, RosettaEnergyEvaluator)
