"""Small saved-score fixtures shared by the reweighting tests."""
from felp.core import FrameResult, K_BOLTZMANN_KJ_PER_MOL_K, MicrostateDefinition
from felp.reweighting import MicrostateAtlas

BETA = 1 / (K_BOLTZMANN_KJ_PER_MOL_K * 300)


def _frame(index: int, state_id: str, delta_u: float) -> FrameResult:
    return FrameResult(
        frame_index=index,
        stride_number=index,
        microstate_id=state_id,
        microstate_label=state_id,
        phi_degrees=0.0,
        psi_degrees=0.0,
        reference_energy_kj_mol=100.0,
        perturbed_energy_kj_mol=100.0 + delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=0.0,
        reference_relaxed_psi_degrees=0.0,
        perturbed_relaxed_phi_degrees=0.0,
        perturbed_relaxed_psi_degrees=0.0,
    )



def _atlas(state_ids: list[str]) -> MicrostateAtlas:
    return MicrostateAtlas.from_definitions(
        [
            MicrostateDefinition(
                state_id=state_id,
                label=state_id,
                cv_ranges={"x": (float(index), float(index) + 1.0)},
            )
            for index, state_id in enumerate(state_ids)
        ],
        reference_probability_mode="raw_counts",
    )



def _bulk_with_outliers(
    n_bulk: int = 80,
    outliers: tuple[float, ...] = (-40.0, -35.0),
) -> list[FrameResult]:
    frames = [
        _frame(index, "s0", float(index - n_bulk // 2) * 0.1) for index in range(n_bulk)
    ]
    for offset, delta_u in enumerate(outliers):
        frames.append(_frame(1000 + offset, "s0", float(delta_u)))
    return frames
