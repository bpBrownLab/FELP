"""Perturbation target and CV checks against live structures."""

from __future__ import annotations

from collections.abc import Mapping

from felp.core import (
    CVMeasurement,
    ConversionError,
    DihedralRestraintSpec,
    PerturbationConfig,
    PhiPsiMeasurement,
    normalize_angle_degrees,
    normalize_mutation_target_name,
)


def ensure_perturbation_matches_pose(pose, perturbation: PerturbationConfig) -> None:
    """Verify optional original residue identity against the pose at the Rosetta index.

    ``target_residue_number`` is the sequential Rosetta pose index (``pose.residue(i)``),
    not necessarily PDB residue numbering. When ``original_residue_name`` is set, this
    check catches common indexing mistakes before mutation.
    """

    if perturbation.original_residue_name is None:
        return
    residue = pose.residue(perturbation.target_residue_number)
    observed = normalize_mutation_target_name(residue.name3().strip().upper())
    expected = perturbation.original_residue_name
    if observed != expected:
        raise ConversionError(
            f"Perturbation target mismatch at Rosetta residue index "
            f"{perturbation.target_residue_number}: expected original residue "
            f"{expected!r}, found {observed!r}. "
            "Use the sequential pose index (not necessarily PDB numbering)."
        )


def _coerce_measurement(
    measurement: CVMeasurement | PhiPsiMeasurement | Mapping[str, float],
) -> CVMeasurement:
    """Normalize a legacy or generalized measurement payload."""

    if isinstance(measurement, CVMeasurement):
        return measurement
    if isinstance(measurement, PhiPsiMeasurement):
        return measurement.as_cv_measurement()
    return CVMeasurement(dict(measurement))


def target_cv_measurement_from_restraints(
    restraint_specs: list[DihedralRestraintSpec] | tuple[DihedralRestraintSpec, ...],
) -> CVMeasurement:
    """Extract the target named CV values represented by a restraint list."""

    return CVMeasurement(
        {
            spec.label: normalize_angle_degrees(spec.target_angle_degrees)
            for spec in restraint_specs
        }
    )


def target_phi_psi_from_restraints(
    restraint_specs: list[DihedralRestraintSpec] | tuple[DihedralRestraintSpec, ...],
) -> PhiPsiMeasurement:
    """Extract the legacy target phi/psi pair represented by a restraint list."""

    targets = target_cv_measurement_from_restraints(restraint_specs)
    try:
        return PhiPsiMeasurement(
            phi_degrees=targets.values["phi"],
            psi_degrees=targets.values["psi"],
        )
    except KeyError as exc:
        raise ConversionError(
            "Expected both phi and psi restraint specifications for CV validation."
        ) from exc


def measurement_deviation_degrees(
    measurement: CVMeasurement | PhiPsiMeasurement | Mapping[str, float],
    restraint_specs: list[DihedralRestraintSpec] | tuple[DihedralRestraintSpec, ...],
) -> dict[str, float]:
    """Return absolute per-CV deviations from the restraint targets."""

    target = target_cv_measurement_from_restraints(restraint_specs)
    observed = _coerce_measurement(measurement)
    deviations: dict[str, float] = {}
    for name, target_value in target.values.items():
        observed_value = observed.get(name)
        if observed_value is None:
            raise ConversionError(
                f"Missing measured CV {name!r} while validating fixed backbone CVs."
            )
        deviations[name] = abs(normalize_angle_degrees(observed_value - target_value))
    return deviations


def ensure_measurement_matches_restraints(
    measurement: CVMeasurement | PhiPsiMeasurement | Mapping[str, float],
    restraint_specs: list[DihedralRestraintSpec] | tuple[DihedralRestraintSpec, ...],
    *,
    tolerance_degrees: float = 1.0,
    context: str = "pose",
) -> None:
    """Fail if any fixed backbone CV drifts outside the allowed tolerance."""

    deviation = measurement_deviation_degrees(measurement, restraint_specs)
    if any(value > tolerance_degrees for value in deviation.values()):
        target = target_cv_measurement_from_restraints(restraint_specs)
        observed = _coerce_measurement(measurement)
        target_summary = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(target.values.items())
        )
        observed_summary = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(observed.values.items())
        )
        deviation_summary = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(deviation.items())
        )
        raise ConversionError(
            f"{context} left the fixed backbone-CV state: "
            f"target=({target_summary}) deg, "
            f"observed=({observed_summary}) deg, "
            f"deviation=({deviation_summary}) deg, "
            f"tolerance={tolerance_degrees:.3f} deg."
        )
