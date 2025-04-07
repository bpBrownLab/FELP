#! /usr/bin/env python

# Common imports
import argparse
import os
import sys

# MD simulation imports
import mdtraj as md
import parmed

# Rosetta imports
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdb
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.id import AtomID
from typing import List

# Project imports
from source.utils.argument_parser_utils import MasterArgumentParser
from source.utils.structure_conversion_utils import StructureToPose, PMD_TO_ROSETTA_1LETTER


# (1) Read in a topology file and a trajectory file
# (2) Convert the trajectory frames to PyRosetta poses
# (3) Compute Rosetta energy for each pose
# (4) Save the poses and energies to a directory

def main():

    # Parse arguments
    parser = MasterArgumentParser()
    args = parser.parse_args()

    # Load trajectory into memory with MDTraj
    traj = md.load(args.input_file)



if __name__ == "__main__":
    main()


