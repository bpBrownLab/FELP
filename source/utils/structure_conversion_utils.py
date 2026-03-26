"""Compatibility wrappers for the rewritten conversion layer."""

from felp.conversion import (
    PMD_TO_ROSETTA_1LETTER,
    POSE_ATOM_NAME_CONVERSION,
    StructureConversionService,
)


class StructureToPose:
    """Legacy wrapper around :class:`StructureConversionService`."""

    @staticmethod
    def from_structure(structure):
        service = StructureConversionService(
            pyrosetta_init_options="-ignore_unrecognized_res true -load_PDB_components false"
        )
        return service.structure_to_pose(structure)


__all__ = [
    "PMD_TO_ROSETTA_1LETTER",
    "POSE_ATOM_NAME_CONVERSION",
    "StructureConversionService",
    "StructureToPose",
]
