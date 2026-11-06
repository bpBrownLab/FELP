"""Independent jump-rotamer sampling for the EGFR sensitivity probe."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from felp.core import ConfigurationError, SidechainEnsembleConfig
from felp.sidechain_ensemble.aggregation import classify_rotamer_assignment


def chain_seed(base_seed: int, *, branch: str, chain_id: int) -> int:
    """Return a deterministic per-chain seed that stays in a 32-bit range."""

    branch_offset = 0 if branch == "reference" else 1
    return int(base_seed) + 1009 * int(chain_id) + 17 * branch_offset


def pose_chi_degrees(pose: Any, residue_numbers: Sequence[int]) -> tuple[float, ...]:
    """Read chi angles from a pose for the requested residues."""

    values: list[float] = []
    for residue_number in residue_numbers:
        residue = pose.residue(int(residue_number))
        nchi = int(residue.nchi()) if hasattr(residue, "nchi") else 0
        for chi_index in range(1, nchi + 1):
            values.append(float(residue.chi(chi_index)))
    return tuple(values)


def configure_sidechain_mover(mover: Any, config: SidechainEnsembleConfig) -> Any:
    """Apply detailed-balance and proposal-mixture settings to a SidechainMover."""

    if hasattr(mover, "set_preserve_detailed_balance"):
        mover.set_preserve_detailed_balance(True)
    if hasattr(mover, "set_prob_uniform"):
        mover.set_prob_uniform(float(config.prob_uniform))
    if hasattr(mover, "set_prob_withinrot"):
        mover.set_prob_withinrot(float(config.prob_withinrot))
    if hasattr(mover, "set_prob_random_pert_current"):
        mover.set_prob_random_pert_current(float(config.prob_random_pert_current))
    return mover


def apply_rosetta_seed(pyrosetta: Any, seed: int) -> None:
    """Set the PyRosetta numeric random generator when the API is available."""

    numeric = getattr(getattr(pyrosetta, "rosetta", None), "numeric", None)
    rng = getattr(getattr(numeric, "random", None), "rg", None)
    generator = rng() if callable(rng) else None
    if generator is not None and hasattr(generator, "set_seed"):
        generator.set_seed(int(seed))


@dataclass
class SampledPose:
    """One retained local-ensemble pose before OpenMM minimization."""

    branch: str
    chain_id: int
    step: int
    pose: Any
    seed: int
    proposal_density: float | None = None
    rotamer_assignment: str = ""
    chi_degrees: tuple[float, ...] = ()
    accepted: bool | None = None
    rosetta_score: float | None = None


def _score_pose(score_function: Any, pose: Any) -> float | None:
    if score_function is None:
        return None
    try:
        return float(score_function(pose))
    except Exception:
        return None


def _positive_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def residue_chi_tuple(residue: Any) -> tuple[float, ...]:
    """Return a residue's chi angles as plain floats."""

    nchi = int(residue.nchi()) if hasattr(residue, "nchi") else 0
    return tuple(float(residue.chi(index)) for index in range(1, nchi + 1))


def snapshot_residue_states(
    pyrosetta: Any,
    pose: Any,
    residue_numbers: Sequence[int],
) -> dict[int, tuple[Any, Any, tuple[float, ...]]]:
    """Capture residue type and chi angles before a proposal is applied.

    The Rosetta proposal-density API needs the pre-move residue type and chi
    vector, so they must be copied out before the mover mutates the pose.
    """

    states: dict[int, tuple[Any, Any, tuple[float, ...]]] = {}
    for number in residue_numbers:
        residue = pose.residue(int(number))
        chi_values = residue_chi_tuple(residue)
        chi_vector = pyrosetta.rosetta.utility.vector1_double()
        for value in chi_values:
            chi_vector.append(float(value))
        states[int(number)] = (residue.type(), chi_vector, chi_values)
    return states


def moved_residue_number(
    pose: Any,
    states: dict[int, tuple[Any, Any, tuple[float, ...]]],
    residue_numbers: Sequence[int],
) -> int | None:
    """Identify which sampled residue the mover changed.

    Returns ``None`` when the moved residue is ambiguous, so the caller can
    decline to report a proposal density rather than attribute one wrongly.
    """

    changed = [
        int(number)
        for number in residue_numbers
        if residue_chi_tuple(pose.residue(int(number))) != states[int(number)][2]
    ]
    if len(changed) == 1:
        return changed[0]
    if not changed and len(residue_numbers) == 1:
        return int(residue_numbers[0])
    return None


def forward_proposal_density(
    mover: Any,
    pose: Any,
    residue_number: int,
    old_residue_type: Any,
    old_chi_vector: Any,
) -> float | None:
    """Return the sampler's forward proposal density for the accepted draw.

    Rosetta exposes this as ``compute_proposal_density`` on the rotamer-jump
    movers and ``proposal_density`` on ``SidechainMover``. Builds without
    either return ``None``, which quadrature aggregation rejects.
    """

    compute = getattr(mover, "compute_proposal_density", None)
    if compute is None:
        compute = getattr(mover, "proposal_density", None)
    if compute is None:
        return None
    return _positive_finite(
        compute(
            pose.residue(int(residue_number)),
            int(residue_number),
            old_residue_type,
            old_chi_vector,
        )
    )


def run_jump_rotamer_chain(
    *,
    start_pose: Any,
    draw_fn: Callable[[Any], float | None],
    config: SidechainEnsembleConfig,
    branch: str,
    chain_id: int,
    residue_numbers: Sequence[int],
    seed: int,
    score_function: Any = None,
) -> list[SampledPose]:
    """Draw independent jump-rotamer proposals from the starting pose.

    Each draw clones the starting pose and hands the clone to ``draw_fn``,
    which applies one independent proposal in place and returns that draw's
    forward proposal density. The density and the move are produced together
    because Rosetta computes the density from pre-move residue state. Missing
    or non-positive densities are recorded as ``None`` and later rejected by
    quadrature aggregation rather than inventing weights.
    """

    members: list[SampledPose] = []
    for step in range(1, int(config.retained_samples) + 1):
        working = start_pose.clone() if hasattr(start_pose, "clone") else start_pose
        density = draw_fn(working)
        chi = pose_chi_degrees(working, residue_numbers)
        members.append(
            SampledPose(
                branch=branch,
                chain_id=int(chain_id),
                step=int(step),
                pose=working,
                seed=int(seed),
                proposal_density=_positive_finite(density),
                rotamer_assignment=classify_rotamer_assignment(chi),
                chi_degrees=chi,
                accepted=True,
                rosetta_score=_score_pose(score_function, working),
            )
        )
    return members


def resolve_sidechain_mover_class(pyrosetta: Any, name: str) -> Any:
    """Return a sidechain mover class from whichever module the build exposes.

    Current PyRosetta releases nest these movers under
    ``protocols.simple_moves.sidechain_moves``; older bindings exposed them
    directly on ``protocols.simple_moves``.
    """

    simple_moves = pyrosetta.rosetta.protocols.simple_moves
    for module in (getattr(simple_moves, "sidechain_moves", None), simple_moves):
        if module is None:
            continue
        mover_class = getattr(module, name, None)
        if mover_class is not None:
            return mover_class
    raise ConfigurationError(
        f"This PyRosetta build does not expose {name}."
    )


def _build_jump_rotamer_mover(pyrosetta: Any, task_factory: Any, config: SidechainEnsembleConfig):
    mover_class = resolve_sidechain_mover_class(pyrosetta, "JumpRotamerSidechainMover")
    mover = mover_class()
    configure_sidechain_mover(mover, config)
    if hasattr(mover, "set_task_factory"):
        mover.set_task_factory(task_factory)
    return mover


def _jump_rotamer_draw_fn(
    pyrosetta: Any,
    mover: Any,
    residue_numbers: Sequence[int],
) -> Callable[[Any], float | None]:
    """Return a callable applying one jump proposal and reporting its density."""

    def draw(pose: Any) -> float | None:
        if hasattr(mover, "init_task"):
            mover.init_task(pose)
        states = snapshot_residue_states(pyrosetta, pose, residue_numbers)
        mover.apply(pose)
        residue_number = moved_residue_number(pose, states, residue_numbers)
        if residue_number is None:
            return None
        old_residue_type, old_chi_vector, _ = states[residue_number]
        return forward_proposal_density(
            mover,
            pose,
            residue_number,
            old_residue_type,
            old_chi_vector,
        )

    return draw


def sample_branch_poses(
    protocol: Any,
    pose: Any,
    *,
    config: SidechainEnsembleConfig,
    branch: str,
    residue_numbers: Sequence[int],
    restraint_specs: Sequence[Any] = (),
) -> list[SampledPose]:
    """Sample one sequence branch using the protocol's Rosetta helpers."""

    if not config.enabled:
        return []
    protocol.conversion_service.ensure_pyrosetta_initialized()
    pyrosetta, _, _ = _import_pyrosetta()
    working, _, _, _ = protocol._prepare_working_pose(
        pose,
        restraint_specs,
        context=f"sidechain ensemble {branch}",
    )
    mobile_selector = protocol._build_ensemble_selector(
        pyrosetta,
        working,
        residue_numbers,
    )
    task_factory = protocol._build_task_factory(pyrosetta, mobile_selector)
    score_function = protocol._score_function_with_constraints(
        pyrosetta,
        protocol._build_packing_score_function(pyrosetta),
    )
    members: list[SampledPose] = []
    for chain_id in range(int(config.chains)):
        seed = chain_seed(config.seed, branch=branch, chain_id=chain_id)
        apply_rosetta_seed(pyrosetta, seed)
        if config.sampler == "jump-rotamer":
            mover = _build_jump_rotamer_mover(pyrosetta, task_factory, config)
            members.extend(
                run_jump_rotamer_chain(
                    start_pose=working,
                    draw_fn=_jump_rotamer_draw_fn(pyrosetta, mover, residue_numbers),
                    config=config,
                    branch=branch,
                    chain_id=chain_id,
                    residue_numbers=residue_numbers,
                    seed=seed,
                    score_function=score_function,
                )
            )
            continue
        raise ConfigurationError(f"Unsupported sidechain ensemble sampler: {config.sampler}")
    return members


def _import_pyrosetta():
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    return pyrosetta, AtomID, MutateResidue
