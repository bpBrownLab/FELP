"""MSM validation diagnostics (ITS, CK tests, transition matrices, relaxation timescales, cluster scatter)."""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Optional

import numpy as np

from felp.core import DependencyError

from ._compat import patch_threadpoolctl_missing_library_paths
from .config import normalize_its_errors_kw
from .plot_compat import plot_pyemma_cktest
from .pipeline import PyEMMAPipelineResult


def _import_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyError("matplotlib is required for MSM validation plots.") from exc
    try:
        patch_threadpoolctl_missing_library_paths()
        import pyemma
    except ImportError as exc:
        raise DependencyError("PyEMMA is required for MSM validation plots.") from exc
    return plt, pyemma


def _lag_grid(max_lag: int, max_points: int) -> list[int]:
    """Monotone lag list from 1 to max_lag with at most max_points values."""

    max_lag = max(1, int(max_lag))
    max_points = max(3, int(max_points))
    if max_lag <= max_points:
        return list(range(1, max_lag + 1))
    raw = np.unique(
        np.round(np.geomspace(1, max_lag, num=max_points)).astype(int)
    ).tolist()
    lags = sorted({int(x) for x in raw if 1 <= x <= max_lag})
    if 1 not in lags:
        lags.insert(0, 1)
    if max_lag not in lags:
        lags.append(max_lag)
    return sorted(set(lags))


def _normalize_lags(lags: Optional[tuple[int, ...]], max_lag: int, max_points: int) -> list[int]:
    """Return explicit ITS lags when provided, otherwise a compact default grid."""

    if lags:
        normalized = sorted({int(lag) for lag in lags})
        invalid = [lag for lag in normalized if lag < 1]
        if invalid:
            raise ValueError(f"ITS lags must be positive integers; got {invalid!r}.")
        return normalized
    return _lag_grid(max_lag, max_points)


def _feature_axis_labels(result: PyEMMAPipelineResult, ndim: int) -> list[str]:
    """Best-effort axis labels from the feature spec."""

    _LABEL_MAP = {
        "phi": r"$\phi$ (deg)",
        "psi": r"$\psi$ (deg)",
        "omega": r"$\omega$ (deg)",
    }
    labels: list[str] = []
    for operation in result.feature_spec.operations:
        raw_labels = operation.parameters.get("labels")
        if isinstance(raw_labels, list):
            labels.extend(str(label) for label in raw_labels)
    if len(labels) < ndim:
        labels.extend(f"Feature {index}" for index in range(len(labels), ndim))
    return [_LABEL_MAP.get(label, label) for label in labels[:ndim]]


def _plot_implied_timescales(pyemma: Any, its: Any, ax: Any, its_processes: int) -> None:
    """Plot only the slowest processes when the PyEMMA version supports it."""

    try:
        pyemma.plots.plot_implied_timescales(
            its,
            ylog=True,
            ax=ax,
            nits=int(its_processes),
        )
    except TypeError:
        pyemma.plots.plot_implied_timescales(its, ylog=True, ax=ax)


def _write_its_numeric_outputs(
    output_dir: str,
    its: Any,
    *,
    its_processes: int,
) -> dict[str, str]:
    """Persist ITS lags/timescales in machine-readable files."""

    outputs: dict[str, str] = {}
    try:
        lags = np.asarray(its.lags, dtype=int)
        timescales = np.asarray(its.timescales, dtype=float)
    except Exception:
        return outputs

    if timescales.ndim == 1:
        timescales = timescales[:, np.newaxis]
    max_processes = min(int(its_processes), timescales.shape[1])
    timescales = timescales[:, :max_processes]

    npz_path = os.path.join(output_dir, "implied_timescales.npz")
    np.savez(npz_path, lags=lags, timescales=timescales)
    outputs["implied_timescales_npz"] = npz_path

    csv_path = os.path.join(output_dir, "implied_timescales.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag_steps", *(f"timescale_{i + 1}_steps" for i in range(max_processes))])
        for row_index, lag in enumerate(lags):
            writer.writerow([int(lag), *[float(value) for value in timescales[row_index]]])
    outputs["implied_timescales_csv"] = csv_path
    return outputs


def _cluster_to_metastable_order(
    cluster_dtrajs: tuple[np.ndarray, ...],
    metastable_dtrajs: Optional[tuple[np.ndarray, ...]],
    state_count: int,
) -> np.ndarray | None:
    """Return cluster order grouped by metastable assignment, if available."""

    if metastable_dtrajs is None:
        return None
    cluster_flat = np.concatenate([np.asarray(values, dtype=int) for values in cluster_dtrajs])
    metastable_flat = np.concatenate([np.asarray(values, dtype=int) for values in metastable_dtrajs])
    assignment = np.full(state_count, -1, dtype=int)
    counts = np.bincount(cluster_flat[cluster_flat >= 0], minlength=state_count)
    for state_index in range(state_count):
        mask = cluster_flat == state_index
        if not mask.any():
            continue
        state_values = metastable_flat[mask]
        state_values = state_values[state_values >= 0]
        if state_values.size:
            assignment[state_index] = int(np.bincount(state_values).argmax())
    return np.lexsort((np.arange(state_count), -counts, assignment))


def write_msm_validation_report(
    result: PyEMMAPipelineResult,
    output_dir: str,
    *,
    its_max_lag: int = 80,
    its_lag_points: int = 35,
    its_lags: Optional[tuple[int, ...]] = None,
    its_errors: str = "none",
    its_processes: int = 8,
    its_n_jobs: int = 1,
    ck_nsets: int = 4,
    ck_mlags: int = 10,
    ck_n_jobs: int = 1,
    skip_ck: bool = False,
    max_tm_states: int = 48,
    cluster_scatter_max_points: int = 25000,
) -> dict[str, str]:
    """
    Write MSM diagnostic figures under ``output_dir`` (created if missing).

    Includes implied timescales, Chapman-Kolmogorov tests, stationary
    distribution, transition matrices, relaxation timescales, and a
    microstate cluster scatter (tab20 coloring) for quick visual
    diagnostics.

    ``cluster_scatter_max_points`` caps scatter point count for
    ``microstate_clusters.png``; use ``0`` for all frames.

    Publication-style FEL/probability/state projections are generated
    separately via ``generate_msm_plots`` using the bundle plotting system.
    """

    plt, pyemma = _import_plotting()
    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}
    warnings: list[str] = []

    dtrajs = list(result.cluster_dtrajs)
    lags = _normalize_lags(its_lags, its_max_lag, its_lag_points)
    errors_kw = normalize_its_errors_kw(its_errors)
    its_kwargs = {
        "lags": lags,
        "nits": int(its_processes),
        "n_jobs": int(its_n_jobs),
        "only_timescales": True,
    }
    if errors_kw is None:
        its = pyemma.msm.its(dtrajs, **its_kwargs)
    else:
        its = pyemma.msm.its(dtrajs, errors=errors_kw, **its_kwargs)
    written.update(
        _write_its_numeric_outputs(
            output_dir,
            its,
            its_processes=int(its_processes),
        )
    )
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    _plot_implied_timescales(pyemma, its, ax, its_processes)
    ax.set_title(f"Implied timescales (slowest {int(its_processes)} processes)")
    fig.tight_layout()
    path = os.path.join(output_dir, "implied_timescales.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    written["implied_timescales"] = path

    msm = result.discrete_msm
    nstates = int(msm.nstates)
    ck_nsets = max(2, min(int(ck_nsets), nstates))
    ck_mlags = max(1, int(ck_mlags))
    ck_n_jobs = max(1, int(ck_n_jobs))

    def _try_cktest(model, nsets: Optional[int], label: str) -> None:
        try:
            if nsets is None:
                ck = model.cktest(mlags=ck_mlags, n_jobs=ck_n_jobs)
            else:
                ck = model.cktest(nsets, mlags=ck_mlags, n_jobs=ck_n_jobs)
        except Exception as exc:
            warnings.append(f"CK calculation skipped for {label}: {exc}")
            return
        try:
            plot_pyemma_cktest(pyemma, ck, diag=True)
            fig = plt.gcf()
            fig.set_size_inches(7.0, 5.2, forward=True)
            # PyEMMA's CK helper already creates a figure title; adding another
            # suptitle causes the overlapping text seen in validation reports.
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            p = os.path.join(output_dir, f"ck_test_{label.replace(' ', '_')}.png")
            fig.savefig(p, dpi=300)
            plt.close(fig)
            written[f"ck_test_{label}"] = p
        except Exception as exc:
            plt.close("all")
            warnings.append(f"CK plot skipped for {label}: {exc}")

    if skip_ck:
        warnings.append("CK calculation skipped by configuration.")
    elif result.hidden_model is not None:
        _try_cktest(result.hidden_model, None, "hidden_msm")
    elif result.metastable_dtrajs is not None:
        flat = np.concatenate(result.metastable_dtrajs, axis=0)
        mcount = int(len(np.unique(flat)))
        if mcount >= 2:
            _try_cktest(msm, mcount, "metastable_sets")
        else:
            warnings.append("CK calculation skipped: fewer than 2 metastable sets.")
    else:
        _try_cktest(msm, ck_nsets, "discrete_msm")

    pi = np.asarray(msm.stationary_distribution, dtype=float)
    order = np.argsort(-pi)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.8))
    show = min(len(pi), 60)
    idx = order[:show]
    ax.bar(np.arange(show), pi[idx], color="C0")
    ax.set_xlabel(f"Microstate index (top {show} by π)")
    ax.set_ylabel("Stationary probability")
    ax.set_title("Discrete MSM stationary distribution (subset)")
    fig.tight_layout()
    path = os.path.join(output_dir, "stationary_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    written["stationary_distribution"] = path

    P = np.asarray(msm.transition_matrix, dtype=float)
    n_tm = min(P.shape[0], max_tm_states)
    sub = P[np.ix_(order[:n_tm], order[:n_tm])]
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))
    vmax = max(float(np.percentile(sub, 99.5)), 1e-6)
    im = ax.imshow(sub, aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
    ax.set_xlabel("j (π-ranked)")
    ax.set_ylabel("i (π-ranked)")
    ax.set_title(f"Transition matrix (top {n_tm} states by π)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(output_dir, "transition_matrix_top_states.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    written["transition_matrix"] = path

    metastable_order = _cluster_to_metastable_order(
        result.cluster_dtrajs,
        result.metastable_dtrajs,
        P.shape[0],
    )
    if metastable_order is not None:
        n_tm_meta = min(P.shape[0], max_tm_states)
        sub_meta = P[np.ix_(metastable_order[:n_tm_meta], metastable_order[:n_tm_meta])]
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))
        vmax = max(float(np.percentile(sub_meta, 99.5)), 1e-6)
        im = ax.imshow(sub_meta, aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_xlabel("j (metastable-grouped)")
        ax.set_ylabel("i (metastable-grouped)")
        ax.set_title(f"Transition matrix (top {n_tm_meta}, metastable grouped)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(output_dir, "transition_matrix_metastable_grouped.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written["transition_matrix_metastable_grouped"] = path

    k_ts = min(25, max(2, nstates - 1))
    try:
        ts = np.asarray(msm.timescales(k=k_ts), dtype=float)
        ts = ts[np.isfinite(ts) & (ts > 0.0)]
        if ts.size:
            indices = np.arange(1, len(ts) + 1)
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))
            ax.semilogy(
                indices, ts,
                marker="o",
                markersize=6,
                markerfacecolor="C2",
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="-",
                linewidth=1.4,
                color="C2",
            )
            ax.set_xlabel("Process index")
            ax.set_ylabel("Implied timescale (steps)")
            ax.set_title("Relaxation timescales at fitted lag")
            ax.set_xlim(0.5, len(ts) + 0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            fig.tight_layout()
            path = os.path.join(output_dir, "relaxation_timescales.png")
            fig.savefig(path, dpi=300)
            plt.close(fig)
            written["relaxation_timescales"] = path
    except Exception:
        pass

    reduced = result.reduced_trajectories
    if reduced and reduced[0].ndim == 2 and reduced[0].shape[1] == 2:
        xy = np.concatenate(reduced, axis=0)
        axis_labels = _feature_axis_labels(result, 2)
        labels = np.concatenate(
            [np.asarray(t, dtype=int) for t in result.cluster_dtrajs],
            axis=0,
        )
        n = xy.shape[0]
        if cluster_scatter_max_points <= 0:
            cap = n
            pick = np.arange(n, dtype=int)
        else:
            cap = min(n, int(cluster_scatter_max_points))
            pick = np.random.default_rng(0).choice(n, size=cap, replace=False)
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.8))
        sc = ax.scatter(
            xy[pick, 0],
            xy[pick, 1],
            c=labels[pick],
            cmap="tab20b",
            s=4,
            alpha=0.45,
            edgecolors="black",
            linewidths=0.1,
            rasterized=True,
        )
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(f"Microstate clusters ({cap:,} frames)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Cluster")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = os.path.join(output_dir, "microstate_clusters.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written["microstate_clusters"] = path

    summary_path = os.path.join(output_dir, "validation_summary.txt")
    pyemma_errors_note = repr(errors_kw) if errors_kw is not None else "kw omitted"
    lines = [
        "MSM validation summary",
        "======================",
        f"Discrete states (active): {nstates}",
        f"MSM lag (steps): {int(msm.lag)}",
        f"ITS lags (steps): {', '.join(str(lag) for lag in lags)}",
        f"ITS error estimate: {its_errors} (PyEMMA errors={pyemma_errors_note})",
        f"ITS plotted processes: {int(its_processes)}",
        f"ITS worker jobs: {int(its_n_jobs)}",
        f"CK skipped: {bool(skip_ck)}",
        f"CK lag multiples: {int(ck_mlags)}",
        f"CK worker jobs: {int(ck_n_jobs)}",
        f"Active state fraction: {float(msm.active_state_fraction):.4f}",
        f"Active count fraction: {float(msm.active_count_fraction):.4f}",
    ]
    try:
        ts0 = msm.timescales(1)
        if np.size(ts0) and math.isfinite(float(ts0[0])):
            lines.append(f"Slowest implied timescale (steps): {float(ts0[0]):.4g}")
    except Exception:
        pass
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    written["validation_summary"] = summary_path
    if warnings:
        warnings_path = os.path.join(output_dir, "validation_plot_warnings.txt")
        with open(warnings_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(warnings) + "\n")
        written["validation_plot_warnings"] = warnings_path
    return written

