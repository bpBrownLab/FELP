"""Extended MSM validation plots (ITS, CK tests, spectra, Ramachandran views)."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np

from felp.core import DependencyError

from .pipeline import PyEMMAPipelineResult


def _import_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyError("matplotlib is required for MSM validation plots.") from exc
    try:
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


def write_msm_validation_report(
    result: PyEMMAPipelineResult,
    output_dir: str,
    *,
    its_max_lag: int = 80,
    its_lag_points: int = 35,
    ck_nsets: int = 4,
    max_tm_states: int = 48,
    rama_scatter_max_points: int = 25000,
) -> dict[str, str]:
    """
    Write validation figures under ``output_dir`` (created if missing).

    Includes implied timescales, Chapman–Kolmogorov tests (discrete and/or
    metastable when available), stationary distribution, transition matrix,
    implied relaxation timescales, and Ramachandran-style views when the
    reduced feature dimension is two (typical phi/psi, no reducer).
    """

    plt, pyemma = _import_plotting()
    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}

    dtrajs = list(result.cluster_dtrajs)
    lags = _lag_grid(its_max_lag, its_lag_points)
    its = pyemma.msm.its(dtrajs, lags=lags, errors="bayes")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    pyemma.plots.plot_implied_timescales(its, ylog=True, ax=ax)
    ax.set_title("Implied timescales (Bayesian error bars)")
    fig.tight_layout()
    path = os.path.join(output_dir, "implied_timescales.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    written["implied_timescales"] = path

    msm = result.discrete_msm
    nstates = int(msm.nstates)
    ck_nsets = max(2, min(int(ck_nsets), nstates))

    def _try_cktest(model, nsets: Optional[int], label: str) -> None:
        try:
            if nsets is None:
                ck = model.cktest()
            else:
                ck = model.cktest(nsets)
        except Exception:
            return
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.0))
        pyemma.plots.plot_cktest(ck, ax=axes, diag=True)
        fig.suptitle(f"Chapman–Kolmogorov test ({label})")
        fig.tight_layout()
        p = os.path.join(output_dir, f"ck_test_{label.replace(' ', '_')}.png")
        fig.savefig(p, dpi=300)
        plt.close(fig)
        written[f"ck_test_{label}"] = p

    _try_cktest(msm, ck_nsets, "discrete_msm")

    if result.hidden_model is not None:
        _try_cktest(result.hidden_model, None, "hidden_msm")
    elif result.metastable_dtrajs is not None:
        flat = np.concatenate(result.metastable_dtrajs, axis=0)
        mcount = int(len(np.unique(flat)))
        if mcount >= 2:
            _try_cktest(msm, mcount, "metastable_sets")

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

    k_ts = min(25, max(2, nstates - 1))
    try:
        ts = np.asarray(msm.timescales(k=k_ts), dtype=float)
        ts = ts[np.isfinite(ts) & (ts > 0.0)]
        if ts.size:
            fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.8))
            ax.semilogy(np.arange(1, len(ts) + 1), ts, "o-", color="C2")
            ax.set_xlabel("Process index")
            ax.set_ylabel("Implied timescale (steps)")
            ax.set_title("Relaxation timescales at fitted lag")
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
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
        pyemma.plots.plot_free_energy(xy[:, 0], xy[:, 1], ax=ax, legacy=False)
        ax.set_xlabel("Feature 0 (e.g. φ / deg)")
        ax.set_ylabel("Feature 1 (e.g. ψ / deg)")
        ax.set_title("Projected free energy (feature space)")
        fig.tight_layout()
        path = os.path.join(output_dir, "rama_feature_free_energy.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written["rama_feature_free_energy"] = path

        labels = np.concatenate(
            [np.asarray(t, dtype=int) for t in result.cluster_dtrajs],
            axis=0,
        )
        rng = np.random.default_rng(0)
        n = xy.shape[0]
        cap = min(n, rama_scatter_max_points)
        pick = rng.choice(n, size=cap, replace=False)
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
        sc = ax.scatter(
            xy[pick, 0],
            xy[pick, 1],
            c=labels[pick],
            cmap="tab20",
            s=2,
            alpha=0.35,
            rasterized=True,
        )
        ax.set_xlabel("Feature 0 (e.g. φ / deg)")
        ax.set_ylabel("Feature 1 (e.g. ψ / deg)")
        ax.set_title(f"K-means clusters (n={cap} subsampled frames)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="cluster")
        fig.tight_layout()
        path = os.path.join(output_dir, "rama_feature_clusters.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written["rama_feature_clusters"] = path

    summary_path = os.path.join(output_dir, "validation_summary.txt")
    lines = [
        "MSM validation summary",
        "======================",
        f"Discrete states (active): {nstates}",
        f"MSM lag (steps): {int(msm.lag)}",
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
    return written

