"""Convert sampled poses, batch-minimize, and form local free-energy deltas."""

from __future__ import annotations

from typing import Any, Sequence

from felp.core import (
    EnergyResult,
    FrameResult,
    SidechainEnsembleConfig,
    SidechainEnsembleMember,
)
from felp.sidechain_ensemble.aggregation import (
    local_free_energy_kj_mol,
    proposal_quadrature_log_z,
    proposal_weight_ess,
    rotamer_basin_counts,
)
from felp.sidechain_ensemble.sampling import SampledPose


def members_from_energy_results(
    sampled: Sequence[SampledPose],
    energies: Sequence[EnergyResult],
    *,
    frame_index: int,
) -> tuple[SidechainEnsembleMember, ...]:
    """Attach OpenMM energies to retained sampler records without deduplicating."""

    if len(sampled) != len(energies):
        raise ValueError("Each sampled pose must have exactly one OpenMM energy.")
    members: list[SidechainEnsembleMember] = []
    for sample, energy in zip(sampled, energies, strict=True):
        members.append(
            SidechainEnsembleMember(
                frame_index=int(frame_index),
                branch=sample.branch,
                chain_id=int(sample.chain_id),
                step=int(sample.step),
                energy_kj_mol=float(energy.energy_kj_mol),
                proposal_density=sample.proposal_density,
                rotamer_assignment=sample.rotamer_assignment,
                seed=int(sample.seed),
                chi_degrees=sample.chi_degrees,
                accepted=sample.accepted,
                rosetta_score=sample.rosetta_score,
            )
        )
    return tuple(members)


def _branch_members(
    members: Sequence[SidechainEnsembleMember],
    branch: str,
) -> list[SidechainEnsembleMember]:
    return [member for member in members if member.branch == branch]


def aggregate_local_ensemble(
    members: Sequence[SidechainEnsembleMember],
    *,
    config: SidechainEnsembleConfig,
    beta: float,
) -> dict[str, Any]:
    """Return local free-energy fields and diagnostics for one parent frame."""

    reference = _branch_members(members, "reference")
    perturbed = _branch_members(members, "perturbed")
    if not reference or not perturbed:
        raise ValueError(
            "Local-ensemble aggregation requires retained members on both branches."
        )
    ref_energies = [member.energy_kj_mol for member in reference]
    pert_energies = [member.energy_kj_mol for member in perturbed]
    metadata: dict[str, Any] = {
        "mode": config.mode,
        "sampler": config.sampler,
        "replaces_production_weight": config.replaces_production_weight,
        "reference_member_count": len(reference),
        "perturbed_member_count": len(perturbed),
        "reference_rotamer_counts": rotamer_basin_counts(
            member.rotamer_assignment for member in reference
        ),
        "perturbed_rotamer_counts": rotamer_basin_counts(
            member.rotamer_assignment for member in perturbed
        ),
        "deduplicated": False,
        "landscape_note": (
            "Sequence-wide additive constants cancel only in normalized "
            "conformational landscapes; this is not an absolute mutation free energy."
        ),
    }
    if config.mode == "proposal-quadrature":
        ref_log_z = proposal_quadrature_log_z(
            ref_energies,
            [member.proposal_density for member in reference],
            beta,
        )
        pert_log_z = proposal_quadrature_log_z(
            pert_energies,
            [member.proposal_density for member in perturbed],
            beta,
        )
        delta = local_free_energy_kj_mol(pert_log_z, ref_log_z, beta)
        metadata.update(
            {
                "production_weight": "proposal_quadrature_local_free_energy",
                "reference_log_z": ref_log_z,
                "perturbed_log_z": pert_log_z,
                "reference_proposal_ess": proposal_weight_ess(
                    ref_energies,
                    [member.proposal_density for member in reference],
                    beta,
                ),
                "perturbed_proposal_ess": proposal_weight_ess(
                    pert_energies,
                    [member.proposal_density for member in perturbed],
                    beta,
                ),
            }
        )
        return {
            "reweighting_delta_kj_mol": delta,
            "local_free_energy_kj_mol": delta,
            "metadata": metadata,
        }
    raise ValueError(f"Unsupported sidechain ensemble mode: {config.mode}")


def attach_ensemble_to_frame(
    frame: FrameResult,
    members: Sequence[SidechainEnsembleMember],
    *,
    config: SidechainEnsembleConfig,
    beta: float,
) -> FrameResult:
    """Return a copy of ``frame`` with ensemble members and optional local ΔA."""

    if not members:
        return frame
    aggregated = aggregate_local_ensemble(members, config=config, beta=beta)
    metadata = dict(frame.metadata)
    metadata["sidechain_ensemble"] = aggregated["metadata"]
    return FrameResult(
        frame_index=frame.frame_index,
        stride_number=frame.stride_number,
        microstate_id=frame.microstate_id,
        microstate_label=frame.microstate_label,
        phi_degrees=frame.phi_degrees,
        psi_degrees=frame.psi_degrees,
        reference_energy_kj_mol=frame.reference_energy_kj_mol,
        perturbed_energy_kj_mol=frame.perturbed_energy_kj_mol,
        delta_u_kj_mol=frame.delta_u_kj_mol,
        reference_relaxed_phi_degrees=frame.reference_relaxed_phi_degrees,
        reference_relaxed_psi_degrees=frame.reference_relaxed_psi_degrees,
        perturbed_relaxed_phi_degrees=frame.perturbed_relaxed_phi_degrees,
        perturbed_relaxed_psi_degrees=frame.perturbed_relaxed_psi_degrees,
        trajectory_id=frame.trajectory_id,
        source_frame_index=frame.source_frame_index,
        assignment_source=frame.assignment_source,
        state_level=frame.state_level,
        reweighting_delta_kj_mol=aggregated["reweighting_delta_kj_mol"],
        local_free_energy_kj_mol=aggregated["local_free_energy_kj_mol"],
        ensemble_members=tuple(members),
        metadata=metadata,
    )
