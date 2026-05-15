"""Build explicit phi/psi feature specifications for MSM featurization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from felp.core import ConfigurationError


def _load_topology(topology_file: str):
    try:
        import mdtraj as md
    except ImportError as exc:
        raise ConfigurationError(
            "MDTraj is required to resolve phi/psi atom indices for --phi-psi-residue-number."
        ) from exc

    suffix = Path(topology_file).suffix.lower()
    if suffix in {".parm7", ".prmtop", ".top"} and hasattr(md, "load_prmtop"):
        return md.load_prmtop(topology_file)
    return md.load_topology(topology_file)


def _atom_index(residue, atom_name: str) -> int:
    for atom in residue.atoms:
        if atom.name == atom_name:
            return int(atom.index)
    raise ConfigurationError(
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
        residue_indices = sorted(
            {int(topology.atom(index).residue.index) for index in atom_indices}
        )
        if len(residue_indices) != 1:
            raise ConfigurationError(
                f"--phi-psi-selection must identify exactly one residue; got "
                f"{len(residue_indices)} from selection {selection!r}."
            )
        residue = residues[residue_indices[0]]
        if not _is_internal_residue(residue, residues):
            raise ConfigurationError(
                f"--phi-psi-selection resolved terminal residue {residue}; "
                "phi/psi requires an internal residue."
            )
        return topology, residues, residue

    def usable(matches: list):
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

    rows = [
        f"{residue}: index={int(residue.index)}, "
        f"one_based={int(residue.index) + 1}, "
        f"resSeq={getattr(residue, 'resSeq', None)}"
        for residue in residues
    ]
    raise ConfigurationError(
        "Could not resolve a unique internal residue for phi/psi with "
        f"phi_psi_target_residue_number={target_residue_number}, "
        f"phi_psi_target_residue_name={target_residue_name!r}. "
        "Pass --phi-psi-selection to pick the residue explicitly.\n  "
        + "\n  ".join(rows)
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
        raise ConfigurationError(
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


def write_phi_psi_feature_spec_json(
    topology_file: str,
    output_path: str,
    *,
    target_residue_number: int,
    target_residue_name: str | None = "GLY",
    selection: str | None = None,
    dihedral_representation: str = "raw",
) -> str:
    """Write a featurizer JSON with explicit phi/psi dihedral atom indices."""

    if dihedral_representation not in {"raw", "cossin"}:
        raise ConfigurationError(
            f"phi_psi_dihedral_representation must be 'raw' or 'cossin'; "
            f"got {dihedral_representation!r}."
        )
    phi, psi, residue_label = _target_phi_psi_indices(
        topology_file,
        target_residue_number,
        target_residue_name,
        selection,
    )
    use_cossin = dihedral_representation == "cossin"
    spec = {
        "use_degrees": True,
        "use_cossin": use_cossin,
        "periodic": False,
        "metadata": {
            "description": "Explicit target-residue Ramachandran phi/psi dihedrals.",
            "residue": residue_label,
            "dihedral_representation": dihedral_representation,
            "columns": (
                ["cos(phi)", "sin(phi)", "cos(psi)", "sin(psi)"]
                if use_cossin
                else ["phi", "psi"]
            ),
        },
        "operations": [
            {
                "kind": "dihedrals",
                "indices": [phi, psi],
                "labels": ["phi", "psi"],
            }
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return str(path.resolve())
