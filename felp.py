import argparse
import os
import sys

# Add the source directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, 'source', 'utils')
sys.path.insert(0, source_dir)

import parmed
from parmed import load_file
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdb

# Import custom modules after adjusting sys.path
from argument_parser import setup_arg_parser
from ParmED_Rosetta import StructureToPose, PoseToStructure

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Ensure input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        if args.direction == "struct2pose":
            print(f"Loading ParmEd Structure from: {args.input_file}")
            structure = load_file(args.input_file)
            print("Converting Structure to Pose...")
            pose = StructureToPose.from_structure(structure)
            print("Conversion successful.")

            if args.output_file:
                print(f"Saving PyRosetta Pose to: {args.output_file}")
                pose.dump_pdb(args.output_file)
                print("Pose saved.")
            else:
                print("\nResulting PyRosetta Pose (first 10 lines):")
                # Saving to a string buffer to print head
                import io
                s = io.StringIO()
                pose.dump_pdb(s)
                pdb_content = s.getvalue()
                s.close()
                print("\n".join(pdb_content.splitlines()[:10]))
                print("...")

        elif args.direction == "pose2struct":
            print(f"Loading PyRosetta Pose from: {args.input_file}")
            # Assuming input for pose is PDB format
            pose = pose_from_pdb(args.input_file)
            print("Converting Pose to Structure...")
            structure = PoseToStructure.to_structure(pose)
            print("Conversion successful.")

            if args.output_file:
                print(f"Saving ParmEd Structure to: {args.output_file}")
                # Infer format from extension, default to PDB
                structure.save(args.output_file, overwrite=True)
                print("Structure saved.")
            else:
                print("\nResulting ParmEd Structure:")
                print(structure)

    except ImportError as e:
        print(f"Error: Failed to import required libraries. Make sure ParmEd and PyRosetta are installed. Details: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
         print(f"Error: File operation failed. {e}")
         sys.exit(1)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
