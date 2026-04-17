"""Microstate definition and assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

from felp.core import (
    CVMeasurement,
    DependencyError,
    MicrostateAssignment,
    MicrostateAssignmentError,
    MicrostateConfig,
    MicrostateDefinition,
)
from felp.msm import load_bundle


def _load_serialized_payload(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise DependencyError(
                    "PyYAML is required to load YAML microstate definitions."
                ) from exc
            return yaml.safe_load(handle)
        return json.load(handle)


@dataclass(frozen=True)
class MicrostateAtlas:
    """Collection of explicit state definitions used for assignment and aggregation."""

    definitions: tuple[MicrostateDefinition, ...]
    allow_unassigned: bool = False
    use_loaded_reference_probabilities: bool = True

    @classmethod
    def from_config(cls, config: MicrostateConfig) -> "MicrostateAtlas":
        """Build an atlas from the provided configuration."""

        if config.source == "file":
            definitions = tuple(load_microstate_definitions(config.definition_file))
        elif config.source == "msm_bundle":
            bundle = load_bundle(config.msm_bundle_dir)
            definitions = bundle.definitions_for_level(config.state_level)
        else:
            definitions = tuple(generate_phi_psi_grid(config.phi_bins, config.psi_bins))
        return cls.from_definitions(
            definitions=definitions,
            allow_unassigned=config.allow_unassigned,
            use_loaded_reference_probabilities=config.use_loaded_reference_probabilities,
        )

    @classmethod
    def from_definitions(
        cls,
        definitions: Iterable[MicrostateDefinition],
        allow_unassigned: bool = False,
        use_loaded_reference_probabilities: bool = True,
    ) -> "MicrostateAtlas":
        """Build an atlas directly from an iterable of definitions."""

        return cls(
            definitions=tuple(definitions),
            allow_unassigned=allow_unassigned,
            use_loaded_reference_probabilities=use_loaded_reference_probabilities,
        )

    def assign(
        self,
        measurements_or_phi: CVMeasurement | dict[str, float] | float,
        psi_degrees: float | None = None,
    ) -> MicrostateAssignment:
        """Assign a CV measurement to a known geometric state."""

        measurements = _coerce_measurements(measurements_or_phi, psi_degrees)
        for definition in self.definitions:
            if definition.contains(measurements):
                return self._assignment_from_definition(definition, measurements)
        if self.allow_unassigned:
            return MicrostateAssignment(
                state_id="unassigned",
                label="unassigned",
                measurements=measurements,
            )
        measurement_summary = ", ".join(
            f"{name}={value:.3f}" for name, value in sorted(measurements.items())
        )
        raise MicrostateAssignmentError(
            f"No microstate matched {measurement_summary}."
        )

    def assign_precomputed(
        self,
        state_id: str,
        measurements_or_phi: CVMeasurement | dict[str, float] | float,
        psi_degrees: float | None = None,
    ) -> MicrostateAssignment:
        """Assign a frame using a precomputed discrete state identifier."""

        measurements = _coerce_measurements(measurements_or_phi, psi_degrees)
        definition = self.by_id().get(state_id)
        if definition is None:
            if self.allow_unassigned and state_id == "unassigned":
                return MicrostateAssignment(
                    state_id="unassigned",
                    label="unassigned",
                    measurements=measurements,
                )
            raise MicrostateAssignmentError(f"Unknown precomputed state id: {state_id}")
        return self._assignment_from_definition(definition, measurements)

    def _assignment_from_definition(
        self,
        definition: MicrostateDefinition,
        measurements: dict[str, float],
    ) -> MicrostateAssignment:
        return MicrostateAssignment(
            state_id=definition.state_id,
            label=definition.label,
            measurements=measurements,
            phi_center=definition.phi_center,
            psi_center=definition.psi_center,
            plot_x=definition.plot_x,
            plot_y=definition.plot_y,
            state_level=definition.state_level,
        )

    def by_id(self) -> dict[str, MicrostateDefinition]:
        """Return definitions keyed by state id."""

        return {definition.state_id: definition for definition in self.definitions}

    def summary(self) -> dict[str, object]:
        """Return a high-level summary of the atlas."""

        state_levels = sorted({definition.state_level for definition in self.definitions})
        return {
            "definition_count": len(self.definitions),
            "allow_unassigned": self.allow_unassigned,
            "state_ids": [definition.state_id for definition in self.definitions[:10]],
            "state_levels": state_levels,
            "cv_names": sorted(
                {
                    cv_name
                    for definition in self.definitions
                    for cv_name in definition.cv_ranges
                }
            ),
            "has_plot_coordinates": any(
                definition.plot_x is not None or definition.plot_y is not None
                for definition in self.definitions
            ),
        }


def _coerce_measurements(
    measurements_or_phi: CVMeasurement | dict[str, float] | float,
    psi_degrees: float | None = None,
) -> dict[str, float]:
    """Normalize assignment inputs into a measurement dictionary."""

    if isinstance(measurements_or_phi, CVMeasurement):
        return measurements_or_phi.as_dict()
    if isinstance(measurements_or_phi, dict):
        return {str(name): float(value) for name, value in measurements_or_phi.items()}
    if psi_degrees is None:
        raise TypeError("A psi angle is required when assigning by phi/psi scalars.")
    return {"phi": float(measurements_or_phi), "psi": float(psi_degrees)}


def load_microstate_definitions(path: str) -> list[MicrostateDefinition]:
    """Load explicit microstate definitions from a JSON or YAML file."""

    if not path:
        raise ValueError("A microstate definition file is required.")
    payload = _load_serialized_payload(path)
    raw_states = payload.get("states", [])
    definitions = []
    for index, raw_state in enumerate(raw_states):
        cv_ranges = {
            str(name): tuple(bounds)
            for name, bounds in dict(raw_state.get("cv_ranges", {})).items()
        }
        if raw_state.get("phi_range") is not None:
            cv_ranges.setdefault("phi", tuple(raw_state["phi_range"]))
        if raw_state.get("psi_range") is not None:
            cv_ranges.setdefault("psi", tuple(raw_state["psi_range"]))
        definitions.append(
            MicrostateDefinition(
                state_id=str(raw_state.get("state_id", index)),
                label=str(raw_state.get("label", raw_state.get("state_id", index))),
                phi_range=(
                    tuple(raw_state["phi_range"])
                    if raw_state.get("phi_range") is not None
                    else None
                ),
                psi_range=(
                    tuple(raw_state["psi_range"])
                    if raw_state.get("psi_range") is not None
                    else None
                ),
                cv_ranges=cv_ranges,
                reference_probability=raw_state.get("reference_probability"),
                plot_x=raw_state.get("plot_x"),
                plot_y=raw_state.get("plot_y"),
                state_level=str(raw_state.get("state_level", "microstate")),
                metadata=dict(raw_state.get("metadata", {})),
            )
        )
    return definitions


def generate_phi_psi_grid(phi_bins: int, psi_bins: int) -> Iterable[MicrostateDefinition]:
    """Generate a uniform grid of phi/psi microstates."""

    phi_width = 360.0 / phi_bins
    psi_width = 360.0 / psi_bins
    state_index = 0
    for phi_bin in range(phi_bins):
        phi_lower = -180.0 + phi_bin * phi_width
        phi_upper = phi_lower + phi_width
        for psi_bin in range(psi_bins):
            psi_lower = -180.0 + psi_bin * psi_width
            psi_upper = psi_lower + psi_width
            definition = MicrostateDefinition(
                state_id=f"{phi_bin}_{psi_bin}",
                label=f"phi{phi_bin}_psi{psi_bin}",
                phi_range=(phi_lower, phi_upper),
                psi_range=(psi_lower, psi_upper),
                plot_x=phi_lower + phi_width / 2.0,
                plot_y=psi_lower + psi_width / 2.0,
                state_level="microstate",
                metadata={"grid_index": state_index},
            )
            state_index += 1
            yield definition
