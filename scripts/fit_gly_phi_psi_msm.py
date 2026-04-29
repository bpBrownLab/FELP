#!/usr/bin/env python3
"""
Fit a discrete MSM on GLY (or any) reference trajectories using backbone φ/ψ
for a single residue, export a FELP ``msm_bundle``, and write validation plots.

This wraps :class:`felp.msm.PyEMMAPipeline` with opinionated defaults for
two-dimensional Ramachandran features (``reducer=none``) and a rich report under
``<output-dir>/validation/``.

Example::

    python scripts/fit_gly_phi_psi_msm.py \\
      --topology-file ref.pdb \\
      --trajectory run1.xtc run2.xtc \\
      --trajectory replica3.xtc \\
      --output-dir data/GLY/msm_phi_psi_bundle \\
      --target-residue-number 2 \\
      --msm-lag 20 \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit PyEMMA MSM on φ/ψ (single residue), export FELP msm_bundle, "
            "and write ITS / CK / Ramachandran validation plots."
        )
    )
    p.add_argument(
        "--topology-file",
        required=True,
        help="Topology path (same as FELP/PyEMMA, e.g. PDB or PRMTOP).",
    )
    p.add_argument(
        "--trajectory",
        action="append",
        nargs="+",
        required=True,
        metavar="FRAG",
        dest="trajectory_groups",
        help=(
            "Repeat once per independent trajectory; within one flag, list "
            "contiguous chunks that form one logical trajectory (same as "
            "``felp.py msm-fit --trajectory``)."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="MSM bundle root (manifest.json, assignments.npz, …).",
    )
    p.add_argument(
        "--target-residue-number",
        type=int,
        default=2,
        help=(
            "Target residue number used to disambiguate the residue when possible. "
            "Match FELP ``--target-residue-number``."
        ),
    )
    p.add_argument(
        "--target-residue-name",
        default="GLY",
        help=(
            "Expected residue name for the phi/psi residue. Defaults to GLY so "
            "terminal caps such as ACE/NME are not selected by residue number."
        ),
    )
    p.add_argument(
        "--selection",
        help=(
            "Optional MDTraj atom selection that identifies exactly one residue. "
            "The script then builds explicit phi/psi dihedral atom indices for "
            "that residue. By default it matches --target-residue-number against "
            "resSeq first, then 1-based topology residue index."
        ),
    )
    p.add_argument("--stride", type=int, default=1, help="Frame stride (match future FELP runs).")
    p.add_argument(
        "--dt-traj",
        default="1 step",
        help="Time spacing label passed to PyEMMA (e.g. ``20 ps``).",
    )
    p.add_argument("--msm-lag", type=int, default=20, help="MSM lag in frames (after stride).")
    p.add_argument(
        "--reducer",
        choices=("none", "tica", "pca"),
        default="none",
        help="Dimensionality reduction before clustering (``none`` for pure φ/ψ).",
    )
    p.add_argument("--reducer-lag", type=int, default=10, help="TICA lag when reducer=tica.")
    p.add_argument(
        "--variance-cutoff",
        type=float,
        default=0.95,
        help="Variance cutoff for PCA/TICA when reducer is not ``none``.",
    )
    p.add_argument(
        "--clustering-method",
        choices=("kmeans", "regspace"),
        default="kmeans",
    )
    p.add_argument("--k-clusters", type=int, default=100)
    p.add_argument("--k-iter", type=int, default=100)
    p.add_argument("--regspace-dmin", type=float, default=0.5)
    p.add_argument(
        "--metastable-states",
        type=int,
        default=None,
        help="If set, run PCCA and export metastable level in the bundle.",
    )
    p.add_argument(
        "--hmm-states",
        type=int,
        default=None,
        help="Optional hidden MSM state count (alternative to PCCA).",
    )
    p.add_argument(
        "--write-pipeline-qc",
        action="store_true",
        help="Also write compact plots under <output-dir>/plots (msm-fit style).",
    )
    p.add_argument(
        "--skip-validation-report",
        action="store_true",
        help="Skip extended validation under <output-dir>/validation/.",
    )
    p.add_argument(
        "--skip-bundle-style-plots",
        action="store_true",
        help="Skip felp.msm.generate_msm_plots publication figures.",
    )
    p.add_argument("--its-max-lag", type=int, default=120)
    p.add_argument("--its-lag-points", type=int, default=40)
    p.add_argument(
        "--ck-nsets",
        type=int,
        default=4,
        help="Number of coarse sets for discrete MSM Chapman–Kolmogorov test.",
    )
    p.add_argument("--save-pyemma-checkpoints", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _load_topology(topology_file: str):
    try:
        import mdtraj as md
    except ImportError as exc:
        raise RuntimeError("mdtraj is required to resolve phi/psi atom indices.") from exc

    suffix = Path(topology_file).suffix.lower()
    if suffix in {".parm7", ".prmtop", ".top"} and hasattr(md, "load_prmtop"):
        return md.load_prmtop(topology_file)
    return md.load_topology(topology_file)


def _atom_index(residue, atom_name: str) -> int:
    for atom in residue.atoms:
        if atom.name == atom_name:
            return int(atom.index)
    raise ValueError(
        f"Residue {residue} is missing required backbone atom {atom_name!r}."
    )


def _normalized_residue_name(residue) -> str:
    return str(getattr(residue, "name", "")).upper()


def _is_internal_residue(residue, residues: list) -> bool:
    position = residues.index(residue)
    return 0 < position < len(residues) - 1


def _residue_matches_name(residue, expected_name: str | None) -> bool:
    if not expected_name:
        return True
    return _normalized_residue_name(residue) == expected_name.upper()


def _residue_table(residues: list) -> list[str]:
    rows = []
    for residue in residues:
        rows.append(
            f"{residue}: index={int(residue.index)}, "
            f"one_based={int(residue.index) + 1}, "
            f"resSeq={getattr(residue, 'resSeq', None)}, "
            f"internal={_is_internal_residue(residue, residues)}"
        )
    return rows


def _resolve_target_residue(
    topology_file: str,
    target_residue_number: int,
    target_residue_name: str | None,
    selection: str | None,
):
    topology = _load_topology(topology_file)
    residues = list(topology.residues)
    if selection:
        atom_indices = topology.select(selection)
        residue_indices = sorted({int(topology.atom(index).residue.index) for index in atom_indices})
        if len(residue_indices) != 1:
            raise ValueError(
                f"--selection must identify exactly one residue; got {len(residue_indices)} "
                f"residues from selection {selection!r}."
            )
        residue = residues[residue_indices[0]]
        if not _is_internal_residue(residue, residues):
            raise ValueError(
                f"--selection resolved terminal residue {residue}; phi/psi needs an internal residue."
            )
        return topology, residues, residue

    def usable(matches: list) -> list:
        return [
            residue
            for residue in matches
            if _is_internal_residue(residue, residues)
            and _residue_matches_name(residue, target_residue_name)
        ]

    candidate_sets = [
        usable(
            [
                residue
                for residue in residues
                if getattr(residue, "resSeq", None) == target_residue_number
            ]
        ),
        usable(
            [
                residue
                for residue in residues
                if int(residue.index) + 1 == target_residue_number
            ]
        ),
        usable(
            [
                residue
                for residue in residues
                if int(residue.index) == target_residue_number
            ]
        ),
        usable(list(residues)),
    ]
    for matches in candidate_sets:
        if len(matches) == 1:
            return topology, residues, matches[0]

    raise ValueError(
        "Could not resolve an internal target residue "
        f"number={target_residue_number}, name={target_residue_name!r}. "
        "Pass --selection to identify the residue explicitly. Residues:\n  "
        + "\n  ".join(_residue_table(residues))
    )


def _target_phi_psi_indices(
    topology_file: str,
    target_residue_number: int,
    target_residue_name: str | None,
    selection: str | None,
) -> tuple[list[int], list[int], str]:
    _topology, residues, residue = _resolve_target_residue(
        topology_file,
        target_residue_number,
        target_residue_name,
        selection,
    )
    position = residues.index(residue)
    if position == 0 or position == len(residues) - 1:
        raise ValueError(
            f"Residue {residue} must have both previous and next residues to define phi/psi."
        )
    previous_residue = residues[position - 1]
    next_residue = residues[position + 1]

    phi = [
        _atom_index(previous_residue, "C"),
        _atom_index(residue, "N"),
        _atom_index(residue, "CA"),
        _atom_index(residue, "C"),
    ]
    psi = [
        _atom_index(residue, "N"),
        _atom_index(residue, "CA"),
        _atom_index(residue, "C"),
        _atom_index(next_residue, "N"),
    ]
    label = (
        f"{residue.name}(topology_index={int(residue.index)}, "
        f"resSeq={getattr(residue, 'resSeq', None)})"
    )
    return phi, psi, label


def _write_feature_spec(path: Path, *, phi: list[int], psi: list[int], residue_label: str) -> None:
    spec = {
        "use_degrees": True,
        "use_cossin": False,
        "periodic": False,
        "metadata": {
            "description": "Explicit target-residue Ramachandran phi/psi dihedrals.",
            "residue": residue_label,
            "columns": ["phi", "psi"],
        },
        "operations": [
            {
                "kind": "dihedrals",
                "indices": [phi, psi],
                "labels": ["phi", "psi"],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def _readme_felp(output_dir: Path, selection: str, stride: int) -> str:
    return f"""FELP MSM bundle (phi/psi / backbone torsions)
=============================================

Selection used for features: {selection!r}

Artifacts
---------
- manifest.json, assignments.npz, embedding.npz — load with FELP ``msm_bundle`` mode.
- inputs/phi_psi_feature_spec.json — feature JSON used for this fit.
- validation/ — implied timescales, CK tests, stationary π, transition matrix, Ramachandran views.

Run FELP reweighting (cluster microstates)
------------------------------------------
Use the **same** topology, trajectory grouping, stride, and target residue as this fit.

::

  python felp.py run \\
    --topology-file <same as MSM> \\
    --trajectory ... \\
    --stride {int(stride)} \\
    --microstate-source msm_bundle \\
    --msm-bundle-dir {output_dir} \\
    --state-level cluster \\
    ...

If you use ``--state-level metastable``, you must have fit PCCA or HMM here
(``--metastable-states`` / ``--hmm-states``).

Inspect::

  python felp.py msm-inspect --msm-bundle-dir {output_dir}
"""


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from felp.msm import MSMFitConfig, PyEMMAPipeline
    from felp.msm.validation_plots import write_msm_validation_report
    from felp.plotting import MSMPlotConfig, generate_msm_plots

    out = Path(args.output_dir).resolve()
    top = str(Path(args.topology_file).resolve())
    phi_indices, psi_indices, residue_label = _target_phi_psi_indices(
        top,
        int(args.target_residue_number),
        args.target_residue_name,
        args.selection,
    )
    print(f"Resolved target residue for phi/psi: {residue_label}")
    print(f"  phi atom indices: {phi_indices}")
    print(f"  psi atom indices: {psi_indices}")
    inputs_dir = out / "inputs"
    feature_path = inputs_dir / "phi_psi_feature_spec.json"
    _write_feature_spec(
        feature_path,
        phi=phi_indices,
        psi=psi_indices,
        residue_label=residue_label,
    )

    traj_groups = tuple(tuple(str(Path(p).resolve()) for p in group) for group in args.trajectory_groups)

    config = MSMFitConfig(
        topology_file=top,
        trajectory_files=traj_groups,
        feature_spec_file=str(feature_path),
        output_dir=str(out),
        stride=int(args.stride),
        dt_traj=str(args.dt_traj),
        reducer=str(args.reducer),
        reducer_lag=int(args.reducer_lag),
        variance_cutoff=float(args.variance_cutoff),
        clustering_method=str(args.clustering_method),
        k_clusters=int(args.k_clusters),
        k_iter=int(args.k_iter),
        regspace_dmin=float(args.regspace_dmin),
        msm_lag=int(args.msm_lag),
        metastable_states=args.metastable_states,
        hmm_states=args.hmm_states,
        save_pyemma_checkpoints=bool(args.save_pyemma_checkpoints),
        write_plots=bool(args.write_pipeline_qc),
        overwrite=bool(args.overwrite),
    )

    result = PyEMMAPipeline(config).run()
    # Pipeline may rm output_dir on --overwrite; restore feature JSON next to bundle.
    _write_feature_spec(
        feature_path,
        phi=phi_indices,
        psi=psi_indices,
        residue_label=residue_label,
    )

    if not args.skip_validation_report:
        val_dir = out / "validation"
        try:
            written = write_msm_validation_report(
                result,
                str(val_dir),
                its_max_lag=int(args.its_max_lag),
                its_lag_points=int(args.its_lag_points),
                ck_nsets=int(args.ck_nsets),
            )
        except Exception as exc:
            val_dir.mkdir(parents=True, exist_ok=True)
            warning_path = val_dir / "validation_plot_warnings.txt"
            warning_path.write_text(
                f"Validation plotting failed after MSM bundle export: {exc}\n",
                encoding="utf-8",
            )
            print(f"WARNING: validation plotting failed; bundle was still written. See {warning_path}")
        else:
            print("Wrote validation report:")
            for key, path in sorted(written.items()):
                print(f"  {key}: {path}")

    if not args.skip_bundle_style_plots:
        style_root = out / "validation" / "bundle_style_plots"
        for level in ("cluster", "metastable"):
            if level == "metastable" and "metastable" not in result.bundle.assignments:
                continue
            sub = style_root / level
            try:
                generate_msm_plots(
                    MSMPlotConfig(
                        bundle_dir=str(out),
                        output_dir=str(sub),
                        state_level=level,
                        overwrite=True,
                    )
                )
            except Exception as exc:
                sub.mkdir(parents=True, exist_ok=True)
                warning_path = sub / "plot_warnings.txt"
                warning_path.write_text(
                    f"Bundle-style plotting failed: {exc}\n",
                    encoding="utf-8",
                )
                print(f"WARNING: bundle-style plots failed for {level}; see {warning_path}")
            else:
                print(f"Wrote bundle-style plots: {sub}")

    readme = out / "FELP_MSM_BUNDLE_README.txt"
    readme.write_text(
        _readme_felp(out, residue_label, int(args.stride)),
        encoding="utf-8",
    )
    print(f"Wrote {readme}")

    summary = result.bundle.summary()
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
