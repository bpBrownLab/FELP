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

        # Add groups for MD Trajectories and Rosetta
        md_group = self.parser.add_argument_group("MD Trajectory")
        md_group.add_argument(
            "--topology_file",
            required=False,
            help="Path to the topology file (e.g., PDB)."
        )
        md_group.add_argument(
            "--trajectory_file",
            required=False,
            nargs="+",
            help="Path to the trajectory file (e.g., .dcd, .xtc, .trr, .nc, etc.)."
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

        # Add groups for data structure interconversion
        conversion_group = self.parser.add_argument_group("Data Structure Interconversion")
        conversion_group.add_argument(
            "--direction",
            choices=["struct2pose", "pose2struct"],
            default="struct2pose",
            help="Conversion direction: Structure to Pose or Pose to Structure."
        )

    def parse_args(self, args=None):
        return self.parser.parse_args(args)
