"""Block bootstrap of FELP overlap diagnostics from frame_results + run_config."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Any

import numpy as np

from felp.core import (
    ConfigurationError,
    MicrostateConfig,
    MicrostateDefinition,
    ReweightingError,
    RunConfig,
)
from felp.io import load_reweighting_eligible_frame_results_csv
from felp.reweighting import (
    MicrostateAtlas,
    ReweightingAggregator,
)

_ANCESTOR_RUN_CONFIG_MAX_STEPS = 32

METADATA_KEYS: tuple[str, ...] = (
    "global_effective_sample_fraction",
    "overlap_coefficient",
    "target_probability_unreliable_local_support",
    "max_normalized_weight",
)


def _atlas_coerce_raw_counts_if_needed(atlas: MicrostateAtlas) -> MicrostateAtlas:
    """Match summarize behavior when legacy configs claim ``loaded`` on grid states."""

    if atlas.reference_probability_mode != "loaded":
        return atlas
    if any(d.reference_probability is not None for d in atlas.definitions):
        return atlas
    return MicrostateAtlas.from_definitions(
        definitions=atlas.definitions,
        allow_unassigned=atlas.allow_unassigned,
        reference_probability_mode="raw_counts",
    )


def _atlas_empirical_priors_for_bootstrap(atlas: MicrostateAtlas) -> MicrostateAtlas:
    """Use ``raw_count`` priors from the resampled frames only.

    MSM bundles attach stationary ``reference_probability`` per state and use
    ``reference_probability_mode='loaded'``. A block resample almost always omits
    some microstates that still have positive MSM prior mass, which makes
    :meth:`ReweightingAggregator.aggregate` fail. For bootstrap diagnostics we
    match priors to the frames actually in each replicate (same as a standard
    ``raw_counts`` reweighting run).
    """

    if atlas.reference_probability_mode == "raw_counts":
        return atlas
    return MicrostateAtlas.from_definitions(
        definitions=atlas.definitions,
        allow_unassigned=atlas.allow_unassigned,
        reference_probability_mode="raw_counts",
    )


def _frame_state_atlas_for_bootstrap(
    allow_unassigned: bool,
    frame_results: list[Any],
) -> MicrostateAtlas:
    """Build a bootstrap atlas from the state IDs present in ``frame_results``.

    Nested re-analysis outputs can reuse an ancestor ``run_config.json`` whose
    microstate source does not match the nested ``frame_results.csv`` (for
    example, an MSM-bundle root with a grid-histogram-prior child output). The
    bootstrap uses empirical priors from each resample, so the correct support is
    the frame CSV itself.
    """

    state_info: dict[str, tuple[str, str]] = {}
    for frame in frame_results:
        state_id = str(frame.microstate_id)
        if state_id not in state_info:
            label = str(frame.microstate_label or state_id)
            state_level = str(frame.state_level or "microstate")
            state_info[state_id] = (label, state_level)
    definitions = tuple(
        MicrostateDefinition(
            state_id=state_id,
            label=label,
            state_level=state_level,
        )
        for state_id, (label, state_level) in sorted(state_info.items())
    )
    return MicrostateAtlas.from_definitions(
        definitions=definitions,
        allow_unassigned=allow_unassigned,
        reference_probability_mode="raw_counts",
    )


def bootstrap_run_directory(summary: dict[str, Any], summary_path: str) -> str:
    """Resolve the directory that contains ``run_config.json`` for this summary.

    Preference order:

    1. Directory containing ``run_summary.json`` (typical flat layout).
    2. Paths from ``summary['output_dir']`` (absolute, then relative to the summary
       folder, then relative to the process cwd).
    3. **Ancestors** of the summary folder (nested outputs often store
       ``run_summary.json`` under ``.../grid/...`` while ``run_config.json`` sits
       in the bundle root).
    """

    summary_dir = os.path.dirname(os.path.abspath(summary_path))
    if os.path.isfile(os.path.join(summary_dir, "run_config.json")):
        return summary_dir

    candidates: list[str] = []
    stored = summary.get("output_dir")
    if isinstance(stored, str) and stored.strip():
        raw = stored.strip()
        if os.path.isabs(raw):
            candidates.append(os.path.normpath(raw))
        else:
            candidates.append(os.path.normpath(os.path.join(summary_dir, raw)))
            candidates.append(os.path.normpath(os.path.join(os.getcwd(), raw)))

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(os.path.join(path, "run_config.json")):
            return path

    d = summary_dir
    for _ in range(_ANCESTOR_RUN_CONFIG_MAX_STEPS):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
        if d in seen:
            continue
        seen.add(d)
        if os.path.isfile(os.path.join(d, "run_config.json")):
            return d

    tried = ", ".join(repr(p) for p in [summary_dir, *candidates] if p)
    raise ConfigurationError(
        "Block bootstrap could not find run_config.json beside run_summary.json, "
        "via output_dir, or in a parent directory. Checked: "
        f"{tried} and up to {_ANCESTOR_RUN_CONFIG_MAX_STEPS} parent folders from "
        f"{summary_dir!r}. Pass a folder tree that includes run_config.json, or "
        "copy it next to run_summary.json."
    )


def _microstate_path_anchor_candidates(config_dir: str, artifact_dir: str) -> tuple[str, ...]:
    """Directories that may anchor relative ``msm_bundle_dir`` / ``definition_file`` paths.

    Relative paths in ``run_config.json`` are often written from the directory
    where ``felp`` was invoked (e.g. ``data/``) rather than the bundle output
    folder; include that tree by walking parents of the config and artifact dirs.
    """

    seen: set[str] = set()
    ordered: list[str] = []

    def push(raw: str) -> None:
        if not raw:
            return
        ap = os.path.normpath(os.path.abspath(raw))
        if ap not in seen:
            seen.add(ap)
            ordered.append(ap)

    for seed in (config_dir, artifact_dir, os.getcwd()):
        push(seed)

    for seed in (config_dir, artifact_dir):
        if not seed:
            continue
        d = os.path.normpath(os.path.abspath(seed))
        for _ in range(_ANCESTOR_RUN_CONFIG_MAX_STEPS):
            parent = os.path.dirname(d)
            if parent == d:
                break
            push(parent)
            d = parent
    return tuple(ordered)


def _msm_bundle_present(bundle_dir: str) -> bool:
    return os.path.isfile(os.path.join(bundle_dir, "manifest.json"))


def _resolve_microstate_paths_for_run(
    mc: MicrostateConfig,
    *,
    config_dir: str,
    artifact_dir: str,
) -> MicrostateConfig:
    """Resolve relative microstate paths; try bundle root, nested output, cwd, ancestors."""

    anchors = _microstate_path_anchor_candidates(config_dir, artifact_dir)

    definition_file = mc.definition_file
    if definition_file and not os.path.isabs(definition_file):
        resolved_def: str | None = None
        for base in anchors:
            candidate = os.path.normpath(os.path.join(base, definition_file))
            if os.path.isfile(candidate):
                resolved_def = candidate
                break
        if resolved_def is None:
            preview = ", ".join(repr(a) for a in anchors[:6])
            raise ConfigurationError(
                "Block bootstrap could not resolve microstate definition_file "
                f"{definition_file!r} against these search roots (and parents): "
                f"{preview}."
            )
        definition_file = resolved_def

    msm_bundle_dir = mc.msm_bundle_dir
    if msm_bundle_dir and not os.path.isabs(msm_bundle_dir):
        resolved_bundle: str | None = None
        for base in anchors:
            candidate = os.path.normpath(os.path.join(base, msm_bundle_dir))
            if _msm_bundle_present(candidate):
                resolved_bundle = candidate
                break
        if resolved_bundle is None:
            preview = ", ".join(
                repr(os.path.normpath(os.path.join(a, msm_bundle_dir))) for a in anchors[:6]
            )
            raise ConfigurationError(
                "Block bootstrap could not resolve msm_bundle_dir "
                f"{msm_bundle_dir!r} (need manifest.json). Example tried paths: "
                f"{preview}."
            )
        msm_bundle_dir = resolved_bundle

    return replace(mc, definition_file=definition_file, msm_bundle_dir=msm_bundle_dir)


@dataclass(frozen=True)
class BootstrapSetup:
    """Prepared block lists and aggregator for one run."""

    atlas: MicrostateAtlas
    aggregator: ReweightingAggregator
    blocks: list[list[Any]]
    reference_counts: dict[str, int] | None
    sampling_metadata: dict[str, Any] | None
    run_dir: str
    eligible_frame_count: int
    block_count: int
    block_size: int


def _first_existing_child_file(parents: tuple[str, ...], name: str) -> str | None:
    for base in parents:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def prepare_bootstrap_setup(summary_path: str, block_size: int) -> BootstrapSetup:
    """Load run_config, frame results, manifest, and non-overlapping blocks."""

    if block_size < 1:
        raise ConfigurationError("bootstrap block size must be at least 1.")

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ConfigurationError(f"Run summary must be a JSON object: {summary_path}")

    artifact_dir = os.path.dirname(os.path.abspath(summary_path))
    config_dir = bootstrap_run_directory(summary, summary_path)
    config_path = os.path.join(config_dir, "run_config.json")

    with open(config_path, encoding="utf-8") as handle:
        run_config = RunConfig.from_dict(json.load(handle))
    aggregator = ReweightingAggregator(
        run_config.energy.temperature_kelvin,
        qc_config=run_config.reweighting_qc,
    )

    frame_name = run_config.output.raw_frame_results_name
    frame_csv = _first_existing_child_file((artifact_dir, config_dir), frame_name)
    if not frame_csv:
        raise ConfigurationError(
            "Block bootstrap needs frame results CSV "
            f"{frame_name!r} in the directory with run_summary.json "
            f"({artifact_dir!r}) or with run_config.json ({config_dir!r})."
        )

    frame_results = load_reweighting_eligible_frame_results_csv(frame_csv)
    eligible = sorted(frame_results, key=lambda f: f.frame_index)
    if not eligible:
        raise ConfigurationError(f"No reweighting-eligible frames in {frame_csv}")

    atlas = _frame_state_atlas_for_bootstrap(
        run_config.microstates.allow_unassigned,
        eligible,
    )

    blocks = _non_overlapping_blocks(eligible, block_size)
    return BootstrapSetup(
        atlas=atlas,
        aggregator=aggregator,
        blocks=blocks,
        reference_counts=None,
        sampling_metadata=None,
        run_dir=artifact_dir,
        eligible_frame_count=len(eligible),
        block_count=len(blocks),
        block_size=block_size,
    )


def _non_overlapping_blocks(
    eligible: list[Any],
    block_size: int,
) -> list[list[Any]]:
    n = len(eligible)
    num_blocks = n // block_size
    if num_blocks < 2:
        raise ConfigurationError(
            f"Block bootstrap needs at least two full blocks of size {block_size}; "
            f"only {n} eligible frames ({num_blocks} full block(s))."
        )
    return [
        eligible[i * block_size : (i + 1) * block_size] for i in range(num_blocks)
    ]


def bootstrap_overlap_metrics(
    setup: BootstrapSetup,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Return shape (n_draws, len(METADATA_KEYS)) with NaN for failed draws."""

    if n_draws < 1:
        raise ConfigurationError("bootstrap draws must be at least 1.")
    blocks = setup.blocks
    num_blocks = len(blocks)
    metrics = np.full((n_draws, len(METADATA_KEYS)), np.nan, dtype=float)
    failed = 0
    for d in range(n_draws):
        choice = rng.integers(0, num_blocks, size=num_blocks, dtype=np.int64)
        boot: list[Any] = []
        for j in choice:
            boot.extend(blocks[int(j)])
        try:
            # Resampled frames must not be judged against manifest counts or MSM
            # stationary priors: both imply positive mass on microstates that the
            # draw may contain zero times. Use empirical priors from ``boot`` only.
            results = setup.aggregator.aggregate(
                boot,
                _atlas_empirical_priors_for_bootstrap(setup.atlas),
                reference_counts=None,
                sampling_metadata=None,
            )
            meta = results[0].metadata if results else {}
            for k, key in enumerate(METADATA_KEYS):
                v = meta.get(key)
                metrics[d, k] = float(v) if v is not None and v != "" else np.nan
        except (ReweightingError, ValueError, ZeroDivisionError):
            failed += 1
            continue
    return metrics, failed


def percentile_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Median and 95% interval; NaNs ignored."""

    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(clean)),
        float(np.percentile(clean, 2.5)),
        float(np.percentile(clean, 97.5)),
    )
