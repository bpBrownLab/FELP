"""Structured summaries of FELP preparation, minimization, and restraints."""

from __future__ import annotations

from typing import Iterable

from felp.core import DihedralRestraintSpec, DistanceRestraintSpec, RunConfig


_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O", "OXT")


def _mutation_anchor_residues_sorted(config: RunConfig) -> list[int]:
    return sorted(set(int(r) for r in config.perturbation.mutation_residue_numbers))


def _sequence_window_union_residue_numbers(config: RunConfig) -> list[int]:
    residues: set[int] = set()
    window = config.mobility.sequence_window
    for anchor in config.perturbation.mutation_residue_numbers:
        first_residue = max(1, int(anchor) - int(window))
        last_residue = int(anchor) + int(window)
        residues.update(range(first_residue, last_residue + 1))
    return sorted(residues)


def _selected_residue_scope(config: RunConfig) -> dict[str, object]:
    anchor_list = _mutation_anchor_residues_sorted(config)
    anchor_text = ", ".join(str(r) for r in anchor_list)
    if config.mobility.mode == "target_only":
        return {
            "mode": "target_only",
            "residue_numbers": anchor_list,
            "mutation_anchor_residue_numbers": anchor_list,
            "description": (
                "All configured mutation residues are selected (sequential pose indices: "
                f"{anchor_text})."
            ),
        }
    if config.mobility.mode == "sequence_window":
        residues = _sequence_window_union_residue_numbers(config)
        return {
            "mode": "sequence_window",
            "mutation_anchor_residue_numbers": anchor_list,
            "sequence_window": config.mobility.sequence_window,
            "residue_numbers": residues,
            "description": (
                "Union of residue-centered sequence windows around all mutation anchors "
                f"({anchor_text}) given the configured window."
            ),
        }
    if config.mobility.mode == "radius_and_sequence":
        residues = _sequence_window_union_residue_numbers(config)
        return {
            "mode": "radius_and_sequence",
            "mutation_anchor_residue_numbers": anchor_list,
            "sequence_window": config.mobility.sequence_window,
            "radius_angstrom": config.mobility.radius_angstrom,
            "always_included_sequence_residue_numbers": residues,
            "spatial_residue_numbers": "frame-dependent",
            "description": (
                "Union of the sequence window around all mutation anchors "
                f"({anchor_text}) with residues that have atoms within the "
                "configured radius of each mutation site. Exact spatial neighbors "
                "are frame-dependent; the sequence-window residues listed here "
                "are always included."
            ),
        }
    return {
        "mode": "radius",
        "mutation_anchor_residue_numbers": anchor_list,
        "radius_angstrom": config.mobility.radius_angstrom,
        "description": (
            "Union over all mutation residues of residues with atoms within the "
            f"configured radius around each mutation site (anchors {anchor_text}); "
            "exact residues are frame-dependent."
        ),
    }


def _mobile_scope(config: RunConfig, *, component: str) -> dict[str, object]:
    selected_scope = _selected_residue_scope(config)
    backbone_enabled = bool(config.mobility.include_backbone)
    sidechains_enabled = bool(config.mobility.include_sidechains)
    if component == "openmm":
        backbone_name = "backbone atoms"
        sidechain_name = "non-backbone atoms"
    else:
        backbone_name = "backbone torsions"
        sidechain_name = "sidechain chi torsions"
    return {
        "selected_residue_scope": selected_scope,
        "backbone_enabled": backbone_enabled,
        "sidechains_enabled": sidechains_enabled,
        "mobile_backbone_scope": (
            f"{backbone_name} on selected residues"
            if backbone_enabled
            else f"no {backbone_name}"
        ),
        "mobile_sidechain_scope": (
            f"{sidechain_name} on selected residues"
            if sidechains_enabled
            else f"no {sidechain_name}"
        ),
    }


def restraint_spec_to_dict(spec) -> dict[str, object]:
    """Serialize a runtime restraint specification for logs and JSON metadata."""

    if isinstance(spec, DihedralRestraintSpec):
        return {
            "type": "dihedral",
            "label": spec.label,
            "kind": spec.kind,
            "residue_numbers": list(spec.residue_numbers),
            "atom_names": list(spec.atom_names),
            "target_angle_degrees": spec.target_angle_degrees,
            "force_constant": spec.force_constant,
        }
    if isinstance(spec, DistanceRestraintSpec):
        return {
            "type": "distance",
            "label": spec.label,
            "atom1": {
                "residue_number": spec.atom1.residue_number,
                "atom_name": spec.atom1.atom_name,
            },
            "atom2": {
                "residue_number": spec.atom2.residue_number,
                "atom_name": spec.atom2.atom_name,
            },
            "target_distance_angstrom": spec.target_distance_angstrom,
            "force_constant": spec.force_constant,
        }
    return {"type": type(spec).__name__, "repr": repr(spec)}


def _restraint_specs_to_list(restraint_specs: Iterable[object]) -> list[dict[str, object]]:
    return [restraint_spec_to_dict(spec) for spec in restraint_specs]


def _runtime_restraints_payload(
    restraint_specs: Iterable[object],
    *,
    unavailable_note: str,
) -> dict[str, object]:
    specs = _restraint_specs_to_list(restraint_specs)
    return {
        "available": bool(specs),
        "count": len(specs),
        "specs": specs,
        "note": None if specs else unavailable_note,
    }


def _distance_cv_templates(config: RunConfig) -> list[dict[str, object]]:
    return [
        {
            "label": spec.name,
            "atom1": {
                "residue_number": spec.atom1.residue_number,
                "atom_name": spec.atom1.atom_name,
            },
            "atom2": {
                "residue_number": spec.atom2.residue_number,
                "atom_name": spec.atom2.atom_name,
            },
            "target_source": "measured from each reference frame",
        }
        for spec in config.restraints.distance_cv_specs
    ]


def _target_backbone_templates(config: RunConfig) -> list[dict[str, object]]:
    residue_numbers = (
        config.restraints.rosetta_backbone_restraint_residue_numbers
        or config.perturbation.mutation_residue_numbers
    )
    return [
        {
            "label": f"rosetta_residue_{residue_number}_{kind}",
            "kind": kind,
            "residue_number": residue_number,
            "force_constant": (
                config.restraints.effective_rosetta_target_backbone_force_constant()
            ),
            "target_source": "measured from each reference frame",
        }
        for residue_number in residue_numbers
        for kind in config.restraints.rosetta_target_backbone_restraints
    ]


def build_protocol_summary(
    config: RunConfig,
    *,
    restraint_specs: Iterable[object] = (),
    rosetta_restraint_specs: Iterable[object] = (),
    openmm_restraint_specs: Iterable[object] = (),
) -> dict[str, object]:
    """Return an explicit JSON-ready summary of DOFs and constraints."""

    selected_scope = _selected_residue_scope(config)
    rosetta_restraints = tuple(rosetta_restraint_specs)
    openmm_restraints = tuple(openmm_restraint_specs)
    runtime_restraints = tuple(restraint_specs)
    per_frame_note = (
        "Targets are frame-specific and are not available in the config-level run "
        "summary. See frame_results.csv metadata.protocol or frame_debug "
        "energy_components.json for actual runtime specs."
    )
    return {
        "symmetry": {
            "reference_state": (
                "reference pose is Rosetta-prepared before final energy evaluation"
            ),
            "perturbed_state": (
                "perturbed pose is mutated from the prepared reference pose and then "
                "Rosetta-prepared before final energy evaluation"
            ),
            "openmm_backend": (
                "both states are converted to OpenMM after Rosetta preparation and "
                "receive the same OpenMM minimization/scoring protocol"
            ),
        },
        "mobility": {
            "mode": config.mobility.mode,
            "target_residue_number": config.perturbation.target_residue_number,
            "mutation_residue_numbers": list(config.perturbation.mutation_residue_numbers),
            "radius_angstrom": config.mobility.radius_angstrom,
            "sequence_window": config.mobility.sequence_window,
            "include_backbone": config.mobility.include_backbone,
            "include_sidechains": config.mobility.include_sidechains,
            "selected_residue_scope": selected_scope,
        },
        "rosetta_preparation": {
            "perturbation_mode": config.rosetta.perturbation_mode,
            "score_function": config.rosetta.score_function,
            "relax_iterations": config.rosetta.relax_iterations,
            "repack_trials": config.rosetta.repack_trials,
            "cartesian": config.rosetta.cartesian,
            "fastrelax_min_type": (
                "lbfgs_armijo_nonmonotone" if config.rosetta.cartesian else None
            ),
            "movemap": {
                **_mobile_scope(config, component="rosetta"),
                "jump_dofs": "disabled",
                "scope_note": (
                    "MoveMap starts with all backbone, chi, and jump DOFs disabled; "
                    "only enabled DOFs on selected residues are turned on."
                ),
            },
            "task_factory": {
                "design_allowed": False,
                "packing_allowed": config.mobility.include_sidechains,
                "packable_residue_scope": (
                    selected_scope if config.mobility.include_sidechains else "none"
                ),
                "operations": [
                    "InitializeFromCommandline",
                    "IncludeCurrent",
                    "RestrictToRepacking",
                    (
                        "OperateOnResidueSubset(PreventRepackingRLT, "
                        "NotResidueSelector(mobile_selector))"
                        if config.mobility.include_sidechains
                        else "PreventRepacking"
                    ),
                ],
            },
            "constraints": {
                "distance_atom_pair_constraints": {
                    "enabled": bool(config.restraints.distance_cv_specs),
                    "source": "configured distance CVs",
                    "harmonic_width_sd_angstrom": (
                        config.restraints.rosetta_distance_constraint_sd_angstrom
                    ),
                    "score_weight": 1.0,
                    "configured_templates": _distance_cv_templates(config),
                },
                "dihedral_constraints": {
                    "enabled": bool(
                        config.restraints.cv_specs
                        or config.restraints.rosetta_target_backbone_restraints
                    ),
                    "felp_cv_dihedrals": [spec.name for spec in config.restraints.cv_specs],
                    "rosetta_only_target_backbone_dihedrals": list(
                        config.restraints.rosetta_target_backbone_restraints
                    ),
                    "rosetta_only_backbone_restraint_residue_numbers": list(
                        config.restraints.rosetta_backbone_restraint_residue_numbers
                        or config.perturbation.mutation_residue_numbers,
                    )
                    if config.restraints.rosetta_target_backbone_restraints
                    else [],
                    "target_backbone_force_constant": (
                        config.restraints.effective_rosetta_target_backbone_force_constant()
                    ),
                    "score_weight": 1.0,
                    "validation_tolerance_degrees": (
                        config.restraints.rosetta_target_backbone_restraint_tolerance_degrees
                        or config.restraints.cv_measurement_tolerance_degrees
                    ),
                    "configured_rosetta_only_target_backbone_templates": (
                        _target_backbone_templates(config)
                    ),
                },
                "runtime_restraint_specs": _runtime_restraints_payload(
                    rosetta_restraints,
                    unavailable_note=per_frame_note,
                ),
            },
        },
        "openmm_final_minimization": {
            "enabled": config.energy.backend == "openmm",
            "backend": config.energy.backend,
            "protein_forcefield": config.energy.protein_forcefield,
            "solvent_forcefield": config.energy.solvent_forcefield,
            "platform_name": config.energy.platform_name,
            "cpu_threads": config.energy.cpu_threads,
            "use_nonbonded_cutoff": config.energy.use_nonbonded_cutoff,
            "nonbonded_method": (
                "CutoffNonPeriodic" if config.energy.use_nonbonded_cutoff else "NoCutoff"
            ),
            "nonbonded_cutoff_nm": config.energy.nonbonded_cutoff_nm,
            "minimization_tolerance_kj_per_mol_nm": (
                config.energy.minimization_tolerance_kj_per_mol_nm
            ),
            "minimization_max_iterations": config.energy.minimization_max_iterations,
            "mass_frozen_dofs": {
                **_mobile_scope(config, component="openmm"),
                "backbone_atom_names": list(_BACKBONE_ATOM_NAMES),
                "frozen_atom_scope": (
                    "All atoms not included by the selected residue scope and "
                    "backbone/sidechain switches are assigned zero mass."
                ),
            },
            "restraints": {
                "force_constant": config.restraints.effective_openmm_force_constant(),
                "configured_distance_templates": _distance_cv_templates(config),
                "runtime_restraint_specs": _runtime_restraints_payload(
                    openmm_restraints,
                    unavailable_note=per_frame_note,
                ),
            },
        },
        "analysis_restraints": {
            "configured_cv_names": [spec.name for spec in config.restraints.cv_specs],
            "configured_distance_cv_names": [
                spec.name for spec in config.restraints.distance_cv_specs
            ],
            "configured_distance_templates": _distance_cv_templates(config),
            "runtime_restraint_specs": _runtime_restraints_payload(
                runtime_restraints,
                unavailable_note=per_frame_note,
            ),
            "per_frame_targets": (
                "Runtime restraint targets are measured from each reference frame before "
                "Rosetta/OpenMM preparation."
            ),
        },
        "sidechain_ensemble": {
            "mode": config.sidechain_ensemble.mode,
            "enabled": config.sidechain_ensemble.enabled,
            "sampler": config.sidechain_ensemble.sampler,
            "residue_scope": config.sidechain_ensemble.residue_scope,
            "retained_samples": config.sidechain_ensemble.retained_samples,
            "chains": config.sidechain_ensemble.chains,
            "seed": config.sidechain_ensemble.seed,
            "prob_uniform": config.sidechain_ensemble.prob_uniform,
            "prob_withinrot": config.sidechain_ensemble.prob_withinrot,
            "prob_random_pert_current": (
                config.sidechain_ensemble.prob_random_pert_current
            ),
            "replaces_production_weight": (
                config.sidechain_ensemble.replaces_production_weight
            ),
            "note": (
                "EGFR sensitivity probe at fixed backbone/CVs. Each proposal is "
                "OpenMM-minimized and combined with its proposal-density correction. "
                "The paper's production landscapes use single endpoints."
            ),
        },
        "reweighting": {
            "methods": list(config.reweighting.methods),
            "default_method": config.reweighting.methods[0],
            "available_methods": (
                "independent",
                "neighbor_pool",
            ),
            "neighbor_pool_solver": config.reweighting.neighbor_pool.solver,
            "neighbor_pool_tau": config.reweighting.neighbor_pool.tau,
            "neighbor_pool_tau_mode": config.reweighting.neighbor_pool.tau_mode,
            "neighbor_pool_grid_connectivity": (
                config.reweighting.neighbor_pool.grid_connectivity
            ),
            "neighbor_pool_chebyshev_radius": (
                config.reweighting.neighbor_pool.chebyshev_radius
            ),
            "neighbor_pool_periodic_cv_names": (
                list(config.reweighting.neighbor_pool.periodic_cv_names)
            ),
            "qc_policy": config.reweighting_qc.policy,
            "qc_exclusion_scope": config.reweighting_qc.exclusion_scope,
            "qc_exclusion_strategy": config.reweighting_qc.exclusion_strategy,
            "weight_treatment": config.reweighting_qc.weight_treatment,
        },
    }
