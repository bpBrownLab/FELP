import pyrosetta
import numpy as np
import parmed
from parmed import load_file, Structure, load_rosetta
from pyrosetta import Pose
from pyrosetta import pose_from_sequence
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue 
from parmed.exceptions import RosettaError
from parmed.periodic_table import AtomicNum, Mass
from parmed.topologyobjects import Atom, ExtraPoint, Bond

# Imports needed for the revised StructureToPose1
from pyrosetta.rosetta.core.chemical import ChemicalManager, ResidueTypeSet
from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue, remove_upper_terminus_type_from_pose_residue
from pyrosetta.rosetta.core.conformation import Residue # Keep Residue for type hints maybe, but factory not needed here
# Removed ResidueFactory, LOWER_TERMINUS, UPPER_TERMINUS, add_variant_type_to_pose_residue
from pyrosetta.rosetta.numeric import xyzVector_double_t # For setting coordinates

# Basic mapping from 3-letter codes (common in PDB/ParmEd) to 1-letter codes (for pose_from_sequence)
# Needs to be extended for DNA/RNA or other standard residues if needed.
# Assumes standard protonation (e.g., HIS maps to H, not differentiating HID/HIE/HIP)
PMD_TO_ROSETTA_1LETTER = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Handle common PDB variants for HIS if necessary, mapping all to H for simplicity here.
    'HID': 'H', 'HIE': 'H', 'HIP': 'H',
    # Handle CYX/CYM if they appear, mapping to C (disulfide state handled later)
    'CYX': 'C', 'CYM': 'C',
    #Handle ace and nme if they appear
    'ACE': 'A[ACE]', 'NME': 'N[NME]',
    # Add DNA/RNA if needed, e.g.:
    # 'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T', # DNA
    # 'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',     # RNA (Need context to differentiate from protein)
}

POSE_ATOM_NAME_CONVERSION = {
    '1HH3': 'H1', '2HH3': 'H2', '3HH3': 'H3', '1HH2': 'H1', '2HH2': 'H2', '3HH2': 'H3',
    '1HA': 'HA2', '2HA': 'HA3', 'HN2' : 'H',
}

class StructureToPose:

    @staticmethod
    def from_structure(structure):
        """
        Convert a ParmEd Structure object containing standard residues
        to a PyRosetta Pose object using pose_from_sequence.

        Limitations:
        - Primarily designed for standard amino acids (and potentially DNA/RNA if mapping is added).
        - Will fail if the structure contains non-canonical residues not in the mapping.
        - Assumes a single polymer chain. Does not handle multiple chains or complex topologies well.
        - Relies on standard protonation states defined in Rosetta by default.

        Parameters
        ----------
        structure : :class:`Structure`
            ParmEd Structure object to convert.
        """
        if not isinstance(structure, Structure):
            raise TypeError('Object is not a ParmEd Structure object.')
        if not structure.residues:
            raise ValueError("Input ParmEd Structure has no residues.")

        # Initialize PyRosetta (consider initializing only once outside the function if called multiple times)
        pyrosetta.init("-ignore_unrecognized_res true -load_PDB_components false")

        # --- 1. Generate 1-Letter Sequence ---
        pose = Pose()
        #save the original structure to a pdb file
        #structure.save(f"/original_structure.pdb")
        sequence = ""
        structure_atom_coords = []
        pose_atom_coords = []
        coord_differences = []

        print("Generating 1-letter sequence from ParmEd structure...")
        for i, pmd_res in enumerate(structure.residues):
            #strip blank spaces from the residue name
            res_name_upper = pmd_res.name.upper() # Use upper case for map lookup
            one_letter_code = PMD_TO_ROSETTA_1LETTER.get(res_name_upper)
            if one_letter_code:
                sequence += one_letter_code
            else:
                # If residue name not in map, this method cannot proceed.
                raise ValueError(
                    f"Residue {i+1} ('{pmd_res.name}') has no defined 1-letter mapping "
                    f"in PMD_TO_ROSETTA_1LETTER. Cannot use pose_from_sequence. "
                    f"Consider using the append_residue_by_res approach for structures "
                    f"with non-canonical residues."
                )
        print(f"Generated sequence: {sequence}")

        # --- 2. Build Pose from Sequence ---
        print(f"\nBuilding Pose from sequence using pose_from_sequence...")
        try:
            # Create pose using the 1-letter sequence.
            # auto_termini=True (default) should handle standard N/C termini.
            # Use 'fa_standard' residue type set by default
            #trim blank spaces from the sequence
            sequence = sequence.replace(' ', '')
            print(f"Sequence: {sequence}")
            pose = pose_from_sequence(sequence)
            print(f"Successfully created Pose. Number of residues: {pose.total_residue()}")
        except Exception as e:
            raise RuntimeError(f"Failed to create Pose from sequence '{sequence}': {e}") from e

        # # Sanity check residue count, ignore ACE and NME
        # if pose.total_residue()+2 != len(structure.residues):
        #      # This might happen if sequence contains characters pose_from_sequence doesn't recognize
        #      raise RuntimeError(
        #          f"Mismatch in residue count after pose_from_sequence: "
        #          f"ParmEd ({len(structure.residues)}) vs Pose ({pose.total_residue()}). "
        #          f"Sequence used: '{sequence}'"
        #     )

        # --- 3. Set Atom Coordinates ---
        # (This section remains largely the same as before)
        print("\n--- Proceeding to set coordinates ---")
        pmd_atom_maps = []
        for pmd_res in structure.residues:
            print(f"Processing residue {pmd_res.name}")
            atom_map = {atom.name.strip(): atom for atom in pmd_res.atoms}
            pmd_atom_maps.append(atom_map)
        #print(f"Atom maps: {pmd_atom_maps}")

        for rsd_idx in range(1, pose.total_residue() + 1): # Use 1-based indexing
            pose_res = pose.residue(rsd_idx)
            pmd_res_map = pmd_atom_maps[rsd_idx - 1]
            pmd_res_ref = structure.residues[rsd_idx-1]

            # Check residue name consistency (Rosetta might use different names, e.g., HIE vs HIS)
            # This check is less critical here as the sequence mapping already happened, but good for debug
            if pose_res.name3() != pmd_res_ref.name:
                 # Allow for known auto-conversions by pose_from_sequence (like HIS->HIE based on context)
                 # or our explicit mapping variants (HID->H, etc.)
                 rosetta_name_in_map = PMD_TO_ROSETTA_1LETTER.get(pmd_res_ref.name.upper())
                 # Simple check: if they map to the same 1-letter code, likely okay.
                 if not (rosetta_name_in_map and pose_res.name1() == rosetta_name_in_map):
                    print(f"Notice: Residue name mismatch at index {rsd_idx}: Pose='{pose_res.name3()}', ParmEd='{pmd_res_ref.name}'")

            # --- 3. Set Atom Coordinates ---
            for atom_idx in range(1, pose_res.natoms() + 1): # Use 1-based indexing
                pose_atom_name = pose_res.atom_name(atom_idx).strip()
                pmd_atom = pmd_res_map.get(pose_atom_name)
                #print(f"Pose atom name: {pose_atom_name}")
                #print(f"ParmEd atom: {pmd_atom}")
                if pmd_atom:
                    atom_id = AtomID(atom_idx, rsd_idx)
                    coords = xyzVector_double_t(float(pmd_atom.xx), float(pmd_atom.xy), float(pmd_atom.xz))
                    structure_atom_coords.append(coords)
                    try:
                        pose.set_xyz(atom_id, coords)
                        pose_atom_coords.append(coords)
                        #subtract the structure_atom_coords from the pose_atom_coords
                        coord_differences.append(np.subtract(structure_atom_coords, pose_atom_coords))
                        #print(f"Pose atom differences: {coord_differences}")
                    except Exception as e:
                        print(f"ERROR setting coordinates for AtomID {atom_id} (Pose Atom: '{pose_atom_name}', Pose Res: {rsd_idx} '{pose_res.name3()}', ParmEd Atom: '{pmd_atom.name}', ParmEd Res: {rsd_idx-1} '{pmd_res_ref.name}')")
                        print(f" Coordinates: {coords.x}, {coords.y}, {coords.z}")
                        raise RuntimeError(f"Failed to set coordinates for atom {atom_id}: {e}") from e
                elif pose_atom_name in POSE_ATOM_NAME_CONVERSION:
                    #print(f"Converting atom name {pose_atom_name} to {POSE_ATOM_NAME_CONVERSION[pose_atom_name]}")
                    pose_atom_name = POSE_ATOM_NAME_CONVERSION[pose_atom_name]
                    pmd_atom = pmd_res_map.get(pose_atom_name)
                    coords = xyzVector_double_t(float(pmd_atom.xx), float(pmd_atom.xy), float(pmd_atom.xz))
                    atom_id = AtomID(atom_idx, rsd_idx)
                    structure_atom_coords.append(coords)
                    try:
                        pose.set_xyz(atom_id, coords)
                        pose_atom_coords.append(coords)
                        #subtract the structure_atom_coords from the pose_atom_coords
                        coord_differences.append(np.subtract(structure_atom_coords, pose_atom_coords))
                        #print(f"Pose atom differences: {coord_differences}")
                    except Exception as e:
                        print(f"ERROR setting coordinates for AtomID {atom_id} (Pose Atom: '{pose_atom_name}', Pose Res: {rsd_idx} '{pose_res.name3()}', ParmEd Atom: '{pmd_atom.name}', ParmEd Res: {rsd_idx-1} '{pmd_res_ref.name}')")
                        print(f" Coordinates: {coords.x}, {coords.y}, {coords.z}")
                        raise RuntimeError(f"Failed to set coordinates for atom {atom_id}: {e}") from e
                    #print(f"ParmEd atom: {pmd_atom}")
                else:
                    print(f"Warning: Atom '{pose_atom_name}' in Pose residue {rsd_idx} ('{pose_res.name3()}') not found in corresponding ParmEd residue {rsd_idx-1} ('{pmd_res_ref.name}'). Check naming conventions, termini (OXT), or missing atoms (e.g., hydrogens).")
                    #if not pose_res.atom_is_virtual(atom_idx):
                         # Atom name matching might still fail if Rosetta uses different conventions
                         # (e.g. HA vs HA1/HA2, O vs OXT at C-term) or adds hydrogens.
                         #print(f"Warning: Atom '{pose_atom_name}' in Pose residue {rsd_idx} ('{pose_res.name3()}') not found in corresponding ParmEd residue {rsd_idx-1} ('{pmd_res_ref.name}'). Check naming conventions, termini (OXT), or missing atoms (e.g., hydrogens).")
                
        # --- 4. Final Checks / Bonding (Optional) ---
        # Standard backbone bonds are handled by pose_from_sequence.
        # Termini variants are typically handled by pose_from_sequence(auto_termini=True)
        # Disulfide bonds might need explicit detection/creation after coordinates are set:
        try:
            print("\nAttempting to detect disulfide bonds based on CYS proximity...")
            pose.conformation().detect_disulfides()
            print("Disulfide detection complete.")
        except Exception as e:
            print(f"Warning: Error during disulfide detection: {e}")

        # Other non-standard bonds (e.g., ligands) would require manual intervention if the pose was built differently.

        print("\nCoordinate setting finished.")
        return pose
