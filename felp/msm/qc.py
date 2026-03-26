"""Optional QC plotting helpers for the PyEMMA pipeline."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from felp.core import DependencyError


def _import_plot_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyError(
            "matplotlib is required to write MSM QC plots."
        ) from exc

    try:
        import pyemma
    except ImportError as exc:
        raise DependencyError(
            "PyEMMA is required to write MSM QC plots."
        ) from exc

    return plt, pyemma


def write_qc_outputs(
    output_dir: str,
    reduced_trajectories: tuple[np.ndarray, ...],
    cluster_dtrajs: tuple[np.ndarray, ...],
    cluster_centers: np.ndarray,
    discrete_msm,
    hidden_model,
    metastable_dtrajs: Optional[tuple[np.ndarray, ...]],
) -> None:
    """Write a compact set of diagnostic plots for an MSM bundle."""

    plt, pyemma = _import_plot_dependencies()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    lags = [max(1, int(discrete_msm.lag)), max(2, int(discrete_msm.lag) * 2), max(3, int(discrete_msm.lag) * 4)]
    its = pyemma.msm.its(cluster_dtrajs, lags=lags, errors="bayes")
    fig, ax = plt.subplots(1, 1)
    pyemma.plots.plot_implied_timescales(its, ylog=True, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "implied_timescales.png"), dpi=300)
    plt.close(fig)

    if reduced_trajectories and reduced_trajectories[0].ndim == 2 and reduced_trajectories[0].shape[1] >= 2:
        concatenated = np.concatenate(reduced_trajectories, axis=0)

        fig, ax = plt.subplots(1, 1)
        pyemma.plots.plot_free_energy(*concatenated.T[0:2], ax=ax, legacy=False)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "free_energy.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        pyemma.plots.plot_density(*concatenated.T[0:2], ax=ax, cbar=False, alpha=0.15)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=4, c="C1")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "cluster_centers.png"), dpi=300)
        plt.close(fig)

        if metastable_dtrajs is not None:
            metastable_traj = np.concatenate(metastable_dtrajs, axis=0)
            fig, ax = plt.subplots(1, 1)
            pyemma.plots.plot_state_map(*concatenated.T[0:2], metastable_traj, ax=ax, legacy=False)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "metastable_state_map.png"), dpi=300)
            plt.close(fig)

    if metastable_dtrajs is not None:
        state_count = int(len(np.unique(np.concatenate(metastable_dtrajs, axis=0))))
        if state_count >= 2:
            model_for_ck = hidden_model if hidden_model is not None else discrete_msm
            fig, ax = plt.subplots(1, 1)
            if hidden_model is not None:
                ck_result = model_for_ck.cktest()
            else:
                ck_result = model_for_ck.cktest(state_count)
            pyemma.plots.plot_cktest(ck_result, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "ck_test.png"), dpi=300)
            plt.close(fig)
