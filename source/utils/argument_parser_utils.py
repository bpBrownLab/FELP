import argparse


class MasterArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Master argument parser for FELP."
        )

        # Add global arguments
        self.parser.add_argument(
            "--output_dir",
            required=True,
            help="Path to the output directory."
        )
        self.parser.add_argument(
            "--radii",
            required=False,
            type=str,
            default="mbondi3",
            help="Radii to use for the trajectory file."
        )
        # Add groups for MD Trajectories and Rosetta
        md_group = self.parser.add_argument_group("MD Trajectory")
        md_group.add_argument(
            "--topology_file",
            required=False,
            help="Path to the topology file (e.g., .pdb, .parm7, etc.)."
        )
        md_group.add_argument(
            "--trajectory_file",
            required=False,
            nargs="+",
            help="Path to the trajectory file (e.g., .dcd, .xtc, .trr, .nc, etc.)."
        )
        md_group.add_argument(
            "--stride",
            required=False,
            type=int,
            default=1,
            help="Stride for the trajectory file."
        )
        md_group.add_argument(
            "--atom_indices",
            required=False,
            nargs="+",
            help="List of atom indices to use for the trajectory file."
        )
        
        #add groups for OpenMM
        omm_group = self.parser.add_argument_group("OpenMM")
        omm_group.add_argument(
            "--protein_forcefield",
            required=False,
            type=str,
            default="amber/protein.ff14SBonlysc.xml",
            help="Path to the protein forcefield file."
        )
        omm_group.add_argument(
            "--solvent_forcefield",
            required=False,
            type=str,
            default="implicit/gbn2.xml",
            help="Path to the solvent forcefield file."
        )
        
        # Add groups for Rosetta
        rosetta_group = self.parser.add_argument_group("Rosetta")
        rosetta_group.add_argument(
            "--pdbs",
            required=False,
            nargs="+",
            help="List of PDB files to use for Rosetta energy calculations."
        )
        rosetta_group.add_argument(
            "--pdbs_dir",
            required=False,
            help="Path to the directory containing the PDB files."
        )
        rosetta_group.add_argument(
            "--score_function",
            required=False,
            type=str,
            default="ref2015",
            help="Score function to use for Rosetta energy calculations."
        )
        rosetta_group.add_argument(
            "--pyrosetta_init_options",
            required=False,
            help="PyRosetta initialization options."
        )

        # Add groups for Rosetta Chemical Conversion
        rosetta_chemical_conversion_group = self.parser.add_argument_group("Rosetta Chemical Conversion")
        rosetta_chemical_conversion_group.add_argument(
            "--chemical_conversion",
            required=False,
            help="Rosetta chemical conversion options."
        )
        rosetta_chemical_conversion_group.add_argument(
            "--target_residue_name",
            required=False,
            help="Target residue to mutate to.Format: 3 letter residue name."
        )
        rosetta_chemical_conversion_group.add_argument(
            "--original_residue_name",
            required=False,
            help="Original residue to mutate from.Format: 3 letter residue name."
        )
        rosetta_chemical_conversion_group.add_argument(
            "--target_residue_number",
            required=False,
            type=int,
            help="Target residue number to mutate to."
        )

        # Add groups for data structure interconversion
        conversion_group = self.parser.add_argument_group("Data Structure Interconversion")
        conversion_group.add_argument(
            "--direction",
            choices=["StructureToPose", "pose2struct"],
            default="StructureToPose",
            help="Conversion direction: Structure to Pose or Pose to Structure."
        )



        

    def parse_args(self, args=None):
        return self.parser.parse_args(args)
