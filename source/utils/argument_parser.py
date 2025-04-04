import argparse

def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert between ParmEd Structure and PyRosetta Pose objects."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input file (e.g., PDB)."
    )
    parser.add_argument(
        "--output-file",
        help="Path to save the output (optional)."
    )
    parser.add_argument(
        "--direction",
        choices=["struct2pose", "pose2struct"],
        default="struct2pose",
        help="Conversion direction: Structure to Pose or Pose to Structure."
    )
    return parser
