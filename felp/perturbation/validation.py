"""Perturbation target checks against the live PyRosetta pose."""

from __future__ import annotations

from felp.core import ConversionError, PerturbationConfig


def ensure_perturbation_matches_pose(pose, perturbation: PerturbationConfig) -> None:
    """Verify optional original residue identity against the pose at the Rosetta index.

    ``target_residue_number`` is the sequential Rosetta pose index (``pose.residue(i)``),
    not necessarily PDB residue numbering. When ``original_residue_name`` is set, this
    check catches common indexing mistakes before mutation.
    """

    if perturbation.original_residue_name is None:
        return
    residue = pose.residue(perturbation.target_residue_number)
    observed = residue.name3().strip().upper()
    expected = perturbation.original_residue_name.strip().upper()
    if observed != expected:
        raise ConversionError(
            f"Perturbation target mismatch at Rosetta residue index "
            f"{perturbation.target_residue_number}: expected original residue "
            f"{expected!r}, found {observed!r}. "
            "Use the sequential pose index (not necessarily PDB numbering)."
        )
