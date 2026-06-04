"""Declarative feature specifications for the PyEMMA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Optional

import numpy as np

from felp.core import DependencyError

from ._compat import patch_numpy_legacy_aliases, patch_threadpoolctl_missing_library_paths


def _import_pyemma():
    patch_numpy_legacy_aliases()
    patch_threadpoolctl_missing_library_paths()
    try:
        import pyemma
    except ImportError as exc:
        raise DependencyError(
            "PyEMMA is required for MSM feature generation but is not installed."
        ) from exc
    return pyemma


def _load_serialized_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise DependencyError(
                    "PyYAML is required to load YAML feature specs."
                ) from exc
            return yaml.safe_load(handle)
        return json.load(handle)


@dataclass(frozen=True)
class FeatureOperation:
    """A single featurization operation to apply to a PyEMMA featurizer."""

    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the operation to a JSON-friendly dictionary."""

        return {"kind": self.kind, **self.parameters}


@dataclass(frozen=True)
class FeatureSpec:
    """Structured feature specification for a PyEMMA coordinates pipeline."""

    use_degrees: bool = True
    use_cossin: bool = False
    periodic: bool = False
    operations: tuple[FeatureOperation, ...] = ()

    @classmethod
    def from_file(cls, path: str) -> "FeatureSpec":
        """Load a feature specification from JSON or YAML."""

        payload = _load_serialized_payload(path)
        operations = tuple(
            FeatureOperation(
                kind=str(raw_operation["kind"]),
                parameters={
                    key: value
                    for key, value in raw_operation.items()
                    if key != "kind"
                },
            )
            for raw_operation in payload.get("operations", [])
        )
        return cls(
            use_degrees=payload.get("use_degrees", True),
            use_cossin=payload.get("use_cossin", False),
            periodic=payload.get("periodic", False),
            operations=operations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature spec to a JSON-friendly dictionary."""

        return {
            "use_degrees": self.use_degrees,
            "use_cossin": self.use_cossin,
            "periodic": self.periodic,
            "operations": [operation.to_dict() for operation in self.operations],
        }


def build_featurizer(topology_file: str, spec: FeatureSpec):
    """Construct and populate a PyEMMA featurizer from a feature specification."""

    pyemma = _import_pyemma()
    featurizer = pyemma.coordinates.featurizer(topology_file)
    for operation in spec.operations:
        _apply_operation(featurizer, operation, spec)
    return featurizer


def _apply_operation(featurizer, operation: FeatureOperation, spec: FeatureSpec) -> None:
    """Apply a single declarative feature operation to a PyEMMA featurizer."""

    params = dict(operation.parameters)
    degrees = bool(spec.use_degrees)
    periodic = bool(spec.periodic)
    cossin = bool(spec.use_cossin)

    if operation.kind == "residue_mindist":
        pairs = params["pairs"]
        scheme = params.get("scheme", "sidechain-heavy")
        threshold = params.get("threshold")
        for pair in pairs:
            featurizer.add_residue_mindist(
                [pair],
                scheme=scheme,
                threshold=threshold,
                periodic=periodic,
            )
        return

    if operation.kind == "dihedrals":
        for indices in params["indices"]:
            featurizer.add_dihedrals(
                [indices],
                deg=degrees,
                cossin=cossin,
                periodic=periodic,
            )
        return

    if operation.kind == "angles":
        for indices in params["indices"]:
            featurizer.add_angles(
                [indices],
                deg=degrees,
                cossin=cossin,
                periodic=periodic,
            )
        return

    if operation.kind == "distances":
        for pair in params["pairs"]:
            featurizer.add_distances([pair], periodic=periodic)
        return

    if operation.kind == "iterable_distances":
        indices1 = params["indices1"]
        indices2 = params.get("indices2")
        if indices2 is None:
            featurizer.add_distances(indices1, periodic=periodic)
        else:
            featurizer.add_distances(
                indices1,
                indices2=indices2,
                periodic=periodic,
            )
        return

    if operation.kind == "contacts":
        threshold = params.get("threshold", 0.35)
        for pair in params["pairs"]:
            featurizer.add_contacts([pair], threshold=threshold, periodic=periodic)
        return

    if operation.kind == "group_mindist":
        group_pairs = np.asarray(params["group_pairs"], dtype=int)
        featurizer.add_group_mindist(
            params["atom_groups"],
            group_pairs=group_pairs,
            threshold=params.get("threshold"),
            periodic=periodic,
        )
        return

    if operation.kind == "backbone_torsions":
        selection = params.get("selection")
        if selection:
            featurizer.add_backbone_torsions(
                selstr=str(selection),
                deg=degrees,
                cossin=cossin,
                periodic=periodic,
            )
        else:
            featurizer.add_backbone_torsions(
                deg=degrees,
                cossin=cossin,
                periodic=periodic,
            )
        return

    if operation.kind == "backbone_ca_distances":
        featurizer.add_distances_ca(
            periodic=periodic,
            excluded_neighbors=params.get("excluded_neighbors", 2),
        )
        return

    if operation.kind == "sidechain_torsions":
        selection = params.get("selection")
        kwargs: dict[str, Any] = {
            "deg": degrees,
            "cossin": cossin,
            "which": params.get("which", "all"),
            "periodic": periodic,
        }
        if selection:
            kwargs["selstr"] = str(selection)
        featurizer.add_sidechain_torsions(**kwargs)
        return

    raise ValueError(f"Unsupported feature operation kind: {operation.kind}")
