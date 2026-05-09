"""Perturbation target and CV checks against live structures."""

from __future__ import annotations

from collections.abc import Mapping

from felp.core import (
    CVMeasurement,
    ConversionError,
    DihedralRestraintSpec,
    DistanceRestraintSpec,
    PerturbationConfig,
    PhiPsiMeasurement,
    mutation_target_family,
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
    if mutation_target_family(observed) != mutation_target_family(expected):
        raise ConversionError(
            f"Perturbation target mismatch at Rosetta residue index "
            f"{perturbation.target_residue_number}: expected original residue "
            f"compatible with {expected!r}, found {observed!r}. "
            "FELP can enforce protonation/tautomer variants within the same residue "
            "family, but not a different amino acid. Use the sequential pose index "
            "(not necessarily PDB numbering)."
        )


def ensure_perturbation_matches_structure(structure, perturbation: PerturbationConfig) -> None:
    """Verify optional original residue identity against the input topology order.

    The topology residue index is the same sequential residue order later handed to
    Rosetta by ``structure_to_pose()``, not necessarily the original PDB residue
    numbering embedded in the source file.
    """

    if perturbation.original_residue_name is None:
        return
    residues = getattr(structure, "residues", ())
    if perturbation.target_residue_number < 1 or perturbation.target_residue_number > len(
        residues
    ):
        raise ConversionError(
            "Perturbation target mismatch at topology residue index "
            f"{perturbation.target_residue_number}: topology has {len(residues)} residues."
        )
    residue = residues[perturbation.target_residue_number - 1]
    observed = normalize_mutation_target_name(str(getattr(residue, "name", "")).strip().upper())
    expected = perturbation.original_residue_name
    if mutation_target_family(observed) != mutation_target_family(expected):
        raise ConversionError(
            f"Perturbation target mismatch at topology residue index "
            f"{perturbation.target_residue_number}: expected original residue "
            f"compatible with {expected!r}, found {observed!r}. "
            "FELP can enforce protonation/tautomer variants within the same residue "
            "family, but not a different amino acid. Use the sequential topology/Pose "
            "residue index (not necessarily PDB numbering)."
        )


def _pose_residue_atom_names(pose, residue_number: int) -> set[str]:
    residue = pose.residue(residue_number)
    if not hasattr(residue, "natoms") or not hasattr(residue, "atom_name"):
        return set()
    atom_names: set[str] = set()
    try:
        natoms = int(residue.natoms())
    except Exception:
        return set()
    for atom_index in range(1, natoms + 1):
        try:
            atom_name = str(residue.atom_name(atom_index)).strip().upper()
        except Exception:
            continue
        if atom_name:
            atom_names.add(atom_name)
    return atom_names


def _pose_residue_identity(pose, residue_number: int) -> str:
    residue = pose.residue(residue_number)
    raw_observed = residue.name3().strip().upper()
    observed = normalize_mutation_target_name(raw_observed)
    atom_names = _pose_residue_atom_names(pose, residue_number)
    if raw_observed.startswith("ASP_P"):
        family = "ASP"
    elif raw_observed.startswith("GLU_P"):
        family = "GLU"
    elif raw_observed.startswith("HIS_"):
        family = "HIS"
    else:
        family = mutation_target_family(observed)
    if family == "HIS":
        has_hd1 = "HD1" in atom_names
        has_he2 = "HE2" in atom_names
        if has_hd1 and has_he2:
            return "HIP"
        if has_hd1:
            return "HID"
        if has_he2:
            return "HIE"
    elif family == "ASP":
        if atom_names & {"HD1", "HD11", "HD12", "HD2", "HD21", "HD22"}:
            return "ASH"
    elif family == "GLU":
        if atom_names & {"HE1", "HE11", "HE12", "HE2", "HE21", "HE22"}:
            return "GLH"
    return observed


def ensure_pose_residue_matches_expected(
    pose,
    residue_number: int,
    expected_residue_name: str,
    *,
    context: str = "pose",
) -> None:
    """Verify that a Pose residue keeps an explicitly requested residue identity."""

    expected = normalize_mutation_target_name(expected_residue_name)
    observed = _pose_residue_identity(pose, residue_number)
    if expected == "HIS":
        if observed in {"HIS", "HID", "HIE"}:
            return
        raise ConversionError(
            f"{context} changed residue identity at Rosetta residue index "
            f"{residue_number}: expected neutral histidine chemistry compatible with "
            f"{expected!r}, found {observed!r}. FELP allows Rosetta optH to choose "
            "between HID and HIE for generic HIS, but not to change the net protonation state."
        )
    if observed != expected:
        raise ConversionError(
            f"{context} changed residue identity at Rosetta residue index "
            f"{residue_number}: expected {expected!r}, found {observed!r}. "
            "For protonation studies, use explicit residue names such as "
            "HID/HIE/HIP (rather than generic HIS) and load any required Rosetta "
            "params up front so tautomer/protonation changes fail fast."
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
    restraint_specs,
) -> CVMeasurement:
    """Extract the target named CV values represented by a restraint list."""

    values: dict[str, float] = {}
    for spec in restraint_specs:
        if isinstance(spec, DihedralRestraintSpec):
            values[spec.label] = normalize_angle_degrees(spec.target_angle_degrees)
        elif isinstance(spec, DistanceRestraintSpec):
            values[spec.label] = float(spec.target_distance_angstrom)
    return CVMeasurement(values)


def target_phi_psi_from_restraints(
    restraint_specs,
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


def measurement_deviation(
    measurement: CVMeasurement | PhiPsiMeasurement | Mapping[str, float],
    restraint_specs,
) -> dict[str, float]:
    """Return unit-aware per-CV deviations from restraint targets."""

    observed = _coerce_measurement(measurement)
    deviations: dict[str, float] = {}
    for spec in restraint_specs:
        observed_value = observed.get(spec.label)
        if observed_value is None:
            raise ConversionError(
                f"Missing measured CV {spec.label!r} while validating fixed CVs."
            )
        if isinstance(spec, DihedralRestraintSpec):
            deviations[spec.label] = abs(
                normalize_angle_degrees(observed_value - spec.target_angle_degrees)
            )
        elif isinstance(spec, DistanceRestraintSpec):
            deviations[spec.label] = abs(
                float(observed_value) - float(spec.target_distance_angstrom)
            )
    return deviations


def ensure_measurement_matches_restraints(
    measurement: CVMeasurement | PhiPsiMeasurement | Mapping[str, float],
    restraint_specs,
    *,
    tolerance_degrees: float = 1.0,
    distance_tolerance_angstrom: float = 0.05,
    context: str = "pose",
) -> None:
    """Fail if any fixed backbone CV drifts outside the allowed tolerance."""

    deviation = measurement_deviation(measurement, restraint_specs)
    failed = []
    for spec in restraint_specs:
        tolerance = (
            tolerance_degrees
            if isinstance(spec, DihedralRestraintSpec)
            else distance_tolerance_angstrom
        )
        if deviation.get(spec.label, 0.0) > tolerance:
            failed.append(spec.label)
    if failed:
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
