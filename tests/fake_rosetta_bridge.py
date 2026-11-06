from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


class FakeAtomID:
    def __init__(self, atomno: int, rsd: int) -> None:
        self._atomno = atomno
        self._rsd = rsd

    def atomno(self) -> int:
        return self._atomno

    def rsd(self) -> int:
        return self._rsd


class FakeAtomCoordinate:
    def __init__(self, xyz: tuple[float, float, float]) -> None:
        self._xyz = tuple(float(value) for value in xyz)

    def xyz(self) -> tuple[float, float, float]:
        return self._xyz


class FakeAtomType:
    def __init__(
        self,
        *,
        element: str,
        atom_type_name: str,
        lj_radius: float,
        lj_wdepth: float,
        is_virtual: bool = False,
    ) -> None:
        self._element = element
        self._atom_type_name = atom_type_name
        self._lj_radius = lj_radius
        self._lj_wdepth = lj_wdepth
        self._is_virtual = is_virtual

    def element(self) -> str:
        return self._element

    def atom_type_name(self) -> str:
        return self._atom_type_name

    def lj_radius(self) -> float:
        return self._lj_radius

    def lj_wdepth(self) -> float:
        return self._lj_wdepth

    def is_virtual(self) -> bool:
        return self._is_virtual


@dataclass
class FakeResidueType:
    unique_name: str
    name3_value: str
    name1_value: str
    atom_order: tuple[str, ...]
    atom_types: dict[str, FakeAtomType]
    charges: dict[str, float] = field(default_factory=dict)
    mm_names: dict[str, str] = field(default_factory=dict)
    atom_alias_map: dict[str, str] = field(default_factory=dict)
    canonical_alias_map: dict[str, str] = field(default_factory=dict)
    polymer: bool = True

    def name(self) -> str:
        return self.unique_name

    def base_name(self) -> str:
        return self.unique_name.split(":", 1)[0]

    def name3(self) -> str:
        return self.name3_value

    def name1(self) -> str:
        return self.name1_value

    def mm_name(self, atomno: int) -> str:
        atom_name = self.atom_order[atomno - 1]
        return self.mm_names.get(atom_name, "")

    def atom_aliases(self) -> dict[str, str]:
        return dict(self.atom_alias_map)

    def canonical_atom_aliases(self) -> dict[str, str]:
        return dict(self.canonical_alias_map)

    def canonical_atom_alias(self, name: str) -> str:
        return self.canonical_alias_map.get(name, "")


class FakeResidue:
    def __init__(
        self,
        residue_type: FakeResidueType,
        *,
        coordinates: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self._type = residue_type
        self._coordinates = [
            tuple(float(value) for value in (coordinates or {}).get(atom_name, (0.0, 0.0, 0.0)))
            for atom_name in residue_type.atom_order
        ]
        self._connected_lower = 0
        self._connected_upper = 0
        self._lower_terminus = "NtermProteinFull" in residue_type.unique_name
        self._upper_terminus = "CtermProteinFull" in residue_type.unique_name

    def atoms(self):
        return [FakeAtomCoordinate(xyz) for xyz in self._coordinates]

    def natoms(self) -> int:
        return len(self._type.atom_order)

    def atom_name(self, atomno: int) -> str:
        return self._type.atom_order[atomno - 1]

    def atom_type(self, atomno: int) -> FakeAtomType:
        return self._type.atom_types[self.atom_name(atomno)]

    def atomic_charge(self, atomno: int) -> float:
        return self._type.charges.get(self.atom_name(atomno), 0.0)

    def type(self) -> FakeResidueType:
        return self._type

    def name3(self) -> str:
        return self._type.name3()

    def is_polymer(self) -> bool:
        return self._type.polymer

    def is_lower_terminus(self) -> bool:
        return self._lower_terminus

    def is_upper_terminus(self) -> bool:
        return self._upper_terminus

    def is_virtual_residue(self) -> bool:
        return False

    def connected_residue_at_lower(self) -> int:
        return self._connected_lower

    def connected_residue_at_upper(self) -> int:
        return self._connected_upper

    def set_xyz(self, atom_index: int, xyz_in) -> None:
        self._coordinates[atom_index - 1] = tuple(float(value) for value in xyz_in)

    def atom_index(self, atom_name: str) -> int:
        return self._type.atom_order.index(atom_name) + 1


class FakeConformation:
    def __init__(self, pose) -> None:
        self.pose = pose

    def bonded_neighbor_all_res(self, atom_id: FakeAtomID):
        return list(self.pose._bonded_neighbors.get((atom_id.rsd(), atom_id.atomno()), ()))

    def detect_disulfides(self) -> None:
        self.pose.detect_disulfides_called += 1


class FakePDBInfo:
    def __init__(self, pose_or_size) -> None:
        size = pose_or_size.total_residue() if hasattr(pose_or_size, "total_residue") else int(pose_or_size)
        self._residue_records = {
            index: {"chain": "A", "number": index, "icode": "", "labels": []}
            for index in range(1, size + 1)
        }
        self._atom_records = {}

    def _residue_record(self, res: int) -> dict:
        return self._residue_records.setdefault(res, {"chain": "A", "number": res, "icode": "", "labels": []})

    def _atom_record(self, res: int, atom_index: int) -> dict:
        return self._atom_records.setdefault(
            (res, atom_index),
            {"altloc": "", "occupancy": 0.0, "bfactor": 0.0, "is_het": False},
        )

    def chain(self, res: int, chain_id: str | None = None):
        record = self._residue_record(res)
        if chain_id is None:
            return record["chain"]
        record["chain"] = chain_id

    def number(self, res: int, pdb_res: int | None = None):
        record = self._residue_record(res)
        if pdb_res is None:
            return record["number"]
        record["number"] = pdb_res

    def icode(self, res: int, ins_code: str | None = None):
        record = self._residue_record(res)
        if ins_code is None:
            return record["icode"]
        record["icode"] = ins_code

    def add_reslabel(self, res: int, label: str) -> None:
        self._residue_record(res)["labels"].append(label)

    def get_reslabels(self, res: int):
        return list(self._residue_record(res)["labels"])

    def alt_loc(self, res: int, atom_index: int, loc: str | None = None):
        record = self._atom_record(res, atom_index)
        if loc is None:
            return record["altloc"]
        record["altloc"] = loc

    def occupancy(self, res: int, atom_index: int, value: float | None = None):
        record = self._atom_record(res, atom_index)
        if value is None:
            return record["occupancy"]
        record["occupancy"] = float(value)

    def bfactor(self, res: int, atom_index: int, value: float | None = None):
        record = self._atom_record(res, atom_index)
        if value is None:
            return record["bfactor"]
        record["bfactor"] = float(value)

    def is_het(self, res: int, atom_index: int, value: bool | None = None):
        record = self._atom_record(res, atom_index)
        if value is None:
            return record["is_het"]
        record["is_het"] = bool(value)


class FakePose:
    def __init__(
        self,
        residues: list[FakeResidue] | None = None,
        *,
        chain_indices: list[int] | None = None,
        bonds: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        pdb_info: FakePDBInfo | None = None,
    ) -> None:
        self._residues = list(residues or [])
        self._chain_indices = list(chain_indices or ([1] * len(self._residues)))
        self._bonded_neighbors: dict[tuple[int, int], list[FakeAtomID]] = {}
        self.append_operations: list[tuple[str, bool]] = []
        self.disulfides: list[tuple[int, int]] = []
        self.detect_disulfides_called = 0
        self._pdb_info = pdb_info if pdb_info is not None else FakePDBInfo(len(self._residues))
        for bond in bonds or ():
            self._add_bond(*bond)
        self._refresh_polymer_connections()

    def _add_bond(self, atom_a: tuple[int, int], atom_b: tuple[int, int]) -> None:
        bond_a = FakeAtomID(*atom_a[::-1])
        bond_b = FakeAtomID(*atom_b[::-1])
        key_a = (atom_a[0], atom_a[1])
        key_b = (atom_b[0], atom_b[1])
        self._bonded_neighbors.setdefault(key_a, []).append(bond_b)
        self._bonded_neighbors.setdefault(key_b, []).append(bond_a)

    def _refresh_polymer_connections(self) -> None:
        for residue in self._residues:
            residue._connected_lower = 0
            residue._connected_upper = 0
            residue._lower_terminus = False
            residue._upper_terminus = False
        for index, residue in enumerate(self._residues, start=1):
            chain_index = self._chain_indices[index - 1]
            previous_same_chain = index > 1 and self._chain_indices[index - 2] == chain_index
            next_same_chain = index < len(self._residues) and self._chain_indices[index] == chain_index
            residue._lower_terminus = not previous_same_chain
            residue._upper_terminus = not next_same_chain
            if residue.is_polymer() and previous_same_chain:
                residue._connected_lower = index - 1
            if residue.is_polymer() and next_same_chain:
                residue._connected_upper = index + 1

    def total_residue(self) -> int:
        return len(self._residues)

    def size(self) -> int:
        return self.total_residue()

    def residue(self, index: int) -> FakeResidue:
        return self._residues[index - 1]

    def conformation(self) -> FakeConformation:
        return FakeConformation(self)

    def chain(self, seqpos: int) -> int:
        return self._chain_indices[seqpos - 1]

    def num_chains(self) -> int:
        return len(set(self._chain_indices))

    def chain_begin(self, chain_num: int) -> int:
        for index, chain in enumerate(self._chain_indices, start=1):
            if chain == chain_num:
                return index
        raise KeyError(chain_num)

    def chain_end(self, chain_num: int) -> int:
        matching = [index for index, chain in enumerate(self._chain_indices, start=1) if chain == chain_num]
        if not matching:
            raise KeyError(chain_num)
        return matching[-1]

    def append_residue_by_bond(self, new_rsd: FakeResidue, build_ideal_geometry: bool = False, *args) -> None:
        start_new_chain = bool(args[4]) if len(args) >= 5 else False
        self.append_operations.append(("bond", start_new_chain))
        self._residues.append(new_rsd)
        if len(self._chain_indices) == 0:
            self._chain_indices.append(1)
        else:
            previous_chain = self._chain_indices[-1]
            self._chain_indices.append(previous_chain + 1 if start_new_chain else previous_chain)
            if not start_new_chain and len(self._residues) > 1:
                previous_residue = self._residues[-2]
                if previous_residue.is_polymer() and new_rsd.is_polymer():
                    self._add_bond(
                        (len(self._residues) - 1, previous_residue.atom_index("C") if "C" in previous_residue._type.atom_order else previous_residue.natoms()),
                        (len(self._residues), new_rsd.atom_index("N") if "N" in new_rsd._type.atom_order else 1),
                    )
        self._refresh_polymer_connections()

    def append_residue_by_jump(self, new_rsd: FakeResidue, jump_anchor_residue: int, *args) -> None:
        start_new_chain = bool(args[2]) if len(args) >= 3 else False
        self.append_operations.append(("jump", start_new_chain))
        self._residues.append(new_rsd)
        if len(self._chain_indices) == 0:
            self._chain_indices.append(1)
        else:
            previous_chain = self._chain_indices[-1]
            self._chain_indices.append(previous_chain + 1 if start_new_chain else previous_chain)
        self._refresh_polymer_connections()

    def set_xyz(self, atom_id: FakeAtomID, point) -> None:
        self._residues[atom_id.rsd() - 1].set_xyz(atom_id.atomno(), point)

    def pdb_info(self, new_info: FakePDBInfo | None = None):
        if new_info is None:
            return self._pdb_info
        self._pdb_info = new_info
        return self._pdb_info


class FakeResidueFactory:
    @staticmethod
    def create_residue(residue_type: FakeResidueType) -> FakeResidue:
        return FakeResidue(residue_type)


class FakeResidueTypeSet:
    def __init__(self, residue_types: list[FakeResidueType]) -> None:
        self._by_name = {residue_type.name(): residue_type for residue_type in residue_types}
        self._by_name3 = {}
        self._by_base_name = {}
        for residue_type in residue_types:
            self._by_name3.setdefault(residue_type.name3(), residue_type)
            self._by_base_name.setdefault(residue_type.base_name(), residue_type)

    def has_name(self, name: str) -> bool:
        return name in self._by_name

    def name_map(self, name: str) -> FakeResidueType:
        return self._by_name[name]

    def has_name3(self, name3: str) -> bool:
        return name3 in self._by_name3

    def get_representative_type_name3(self, name3: str) -> FakeResidueType:
        return self._by_name3[name3]

    def get_representative_type_base_name(self, base_name: str) -> FakeResidueType:
        return self._by_base_name[base_name]


class FakeChemicalManager:
    _instance = None

    def __init__(self, residue_type_set: FakeResidueTypeSet) -> None:
        self._residue_type_set = residue_type_set

    def residue_type_set(self, tag: str):
        return self._residue_type_set

    @classmethod
    def install(cls, residue_type_set: FakeResidueTypeSet) -> None:
        cls._instance = cls(residue_type_set)

    @classmethod
    def get_instance(cls):
        return cls._instance


def build_fake_import(residue_types: list[FakeResidueType]):
    residue_type_set = FakeResidueTypeSet(residue_types)
    FakeChemicalManager.install(residue_type_set)
    init_calls = []

    def _form_disulfide(conformation: FakeConformation, lower_residue: int, upper_residue: int) -> None:
        conformation.pose.disulfides.append((lower_residue, upper_residue))
        lower = conformation.pose.residue(lower_residue)
        upper = conformation.pose.residue(upper_residue)
        if "SG" in lower._type.atom_order and "SG" in upper._type.atom_order:
            conformation.pose._add_bond(
                (lower_residue, lower.atom_index("SG")),
                (upper_residue, upper.atom_index("SG")),
            )

    fake_pyrosetta = SimpleNamespace(
        init=lambda options="": init_calls.append(options),
        init_calls=init_calls,
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                pose=SimpleNamespace(PDBInfo=FakePDBInfo),
                chemical=SimpleNamespace(ChemicalManager=FakeChemicalManager),
                conformation=SimpleNamespace(form_disulfide=_form_disulfide),
            )
        ),
    )
    def xyz_vector(x, y, z):
        return (float(x), float(y), float(z))

    return fake_pyrosetta, FakePose, FakeAtomID, FakeResidueFactory, FakeChemicalManager, xyz_vector
