# Common imports
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
# MD simulation imports
import mdtraj as md
import parmed

#ParmEd imports
from parmed import load_rosetta
from parmed.tools.changeradii import mbondi3, ChRad
from parmed.topologyobjects import Atom

# Rosetta imports
import pyrosetta
from pyrosetta import Pose, pose_from_pdb
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.id import AtomID
from typing import List
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue 

#openmm
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import openmm.unit as unit


# Project imports
from source.utils.argument_parser_utils import MasterArgumentParser
from source.utils.structure_conversion_utils import StructureToPose, PMD_TO_ROSETTA_1LETTER

def init():
    # Parse arguments
    parser = MasterArgumentParser()
    args = parser.parse_args()

    # Use os to mkdir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def mutate_pose(pose, target_residue_number, target_residue_name, output_prefix):
    print(f"Pose residue: {pose.residue(target_residue_number)}")
    pyrosetta.init(options="-database /dors/bpbrownlab/shared/apps/rocky9/rosetta/public/rosetta/database/ -overwrite")
    current_phi = pose.phi(target_residue_number)
    current_psi = pose.psi(target_residue_number)
    current_residue = pose.residue(target_residue_number)
    #print(f"Current phi: {current_phi}")
    #print(f"Current psi: {current_psi}")
    #print(f"Current residue: {current_residue}")
    mutate_mover = MutateResidue(target_residue_number, target_residue_name)
    mutate_mover.apply(pose)
    #pose.dump_pdb(f"{output_prefix}_mutant.pdb")
    pose.set_phi(target_residue_number, current_phi)
    pose.set_psi(target_residue_number, current_psi)
    #print(f"Mutated residue: {pose.residue(target_residue_number)}")
    #print(f"Mutated residue phi: {pose.phi(target_residue_number)}")
    #print(f"Mutated residue psi: {pose.psi(target_residue_number)}")
    #add constraint weights
    # Apply restraints to fix phi/psi values
    #scorefxn = pyrosetta.create_score_function('ref2015')  # Use a modern energy function
    # Add constraint weights
    scorefxn = pyrosetta.ScoreFunction()
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 10.0)
    #scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 10.0)    

    constraint_set = pyrosetta.rosetta.core.scoring.constraints.ConstraintSet()

    # Convert phi/psi to radians (-pi to pi) from -180 to 180
    phi_rad = np.radians(current_phi)
    psi_rad = np.radians(current_psi)

    # Define torsion angle constraints
    _phi_harmonic_func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(phi_rad, 0.01)  # Small deviation
    _psi_harmonic_func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(psi_rad, 0.01)  # Small deviation
    #create AtomID Objects
    # Create phi constraint for residue 2
    phi_constraint = pyrosetta.rosetta.core.scoring.constraints.DihedralConstraint(
        pyrosetta.rosetta.core.id.AtomID(pose.residue(1).atom_index("C"), 1),  # C from previous residue
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("N"), 2),  # N from current residue 
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("CA"), 2), # CA from current residue
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("C"), 2),  # C from current residue
        _phi_harmonic_func)  # Target phi value

    # Create psi constraint for residue 2 
    psi_constraint = pyrosetta.rosetta.core.scoring.constraints.DihedralConstraint(
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("N"), 2),   # N from current residue
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("CA"), 2),  # CA from current residue
        pyrosetta.rosetta.core.id.AtomID(pose.residue(2).atom_index("C"), 2),   # C from current residue
        pyrosetta.rosetta.core.id.AtomID(pose.residue(3).atom_index("N"), 3),   # N from next residue
        _psi_harmonic_func)  # Target psi value

    # Add constraints to the constraint set
    constraint_set.add_constraint(phi_constraint)
    constraint_set.add_constraint(psi_constraint)

    # Apply constraints to the pose
    pose.constraint_set(constraint_set)

    # Minimize/relax

    relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn, 5)  # 5 iterations
    relax.apply(pose)
    #pose.dump_pdb(f"{output_prefix}.relaxed.pdb")
    print(f"Relaxed residue: {pose.residue(target_residue_number)}")
    print(f"Relaxed residue phi: {pose.phi(target_residue_number)}")
    print(f"Relaxed residue psi: {pose.psi(target_residue_number)}")
    return pose

def omm_system_energy(structure):
    args = init()
    # Create the OpenMM system
    print('Creating OpenMM System')
    ff = ForceField(args.protein_forcefield, args.solvent_forcefield)
    #ff = ForceField('amber/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
    mod = Modeller(structure.topology, structure.positions)
    # Create OpenMM System
    #system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff, implicitSolvent=GBn2)
    system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff,
                             nonbondedCutoff=1*nanometers, 
                             #implicitSolvent=GBn2
                             )
    integrator = LangevinIntegrator(300*kelvin, 1/picoseconds,2*femtoseconds)
    simulation = Simulation(mod.topology, system, integrator)
    simulation.context.setPositions(mod.positions)
    # Minimize System
    simulation.minimizeEnergy(maxIterations=0, tolerance=1*unit.kilojoules_per_mole/unit.nanometer)
    #get the energy
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    #clear the simulation
    del simulation
    return energy
   


def main():
    # Initialize
    args = init()
    #pyrosetta.init(options=args.pyrosetta_init_options)
    #pyrosetta.init(options="-database /dors/bpbrownlab/shared/apps/rocky9/rosetta/public/rosetta/database/")

    # Load trajectory into memory with MDTraj
    traj = md.load(args.trajectory_file, top=args.topology_file, stride=args.stride, atom_indices=args.atom_indices)

    # Convert trajectory frames to ParmEd Structures
    structures = []
    structure_coords = []
    mutated_structures = []
    poses = []
    omm_energies = []
    mutated_energies = []
    energy_diffs = []
    phi_values = []
    psi_values = []
    mutated_phi_values = []
    mutated_psi_values = []
    stride_values = []
    #energy_function = pyrosetta.create_score_function("ref2015")

    # Load the topology file as a ParmEd structure
    topology = parmed.load_file(args.topology_file)

    #set radii to mbondi3
    radii_set = str(args.radii)
    ChRad(topology, radii_set)

    # Iterate through each frame in the trajectory
    for frame_idx, frame in enumerate(tqdm(traj, desc="Processing frames", total=traj.n_frames)):
        # Create a copy of the topology structure
        structure = topology # .copy()
        stride_number = frame_idx * args.stride

        # HACK
        # if stride_number not in [0,58355,71745]:
        #     continue

        stride_values.append(stride_number)
        # Update the coordinates from the current frame
        # MDTraj uses nm, ParmEd uses Angstroms, so convert by multiplying by 10
        structure.coordinates = frame.xyz[0] * 10.0
        os.makedirs(args.output_dir, exist_ok=True)
        # Add the structure and coordinates to our lists
        structures.append(structure)
        structure_coords.append(frame.xyz[0])
        #calculate and save the OMM energy with parmed energy function
        omm_energy = omm_system_energy(structure)
        omm_energies.append(omm_energy)      

        #save structure to a pdb file
        #structure.save(f"{args.output_dir}/structure_{frame_idx}.pdb")
        pose = StructureToPose.from_structure(structure)
        #sanity check if needed
        #pose.dump_pdb(f"{args.output_dir}/pose_{frame_idx}.pdb")
        poses.append(pose)
        phi_values.append(pose.phi(args.target_residue_number))
        psi_values.append(pose.psi(args.target_residue_number))
        output_prefix = f"{args.output_dir}/mutated_pose_{frame_idx}"
        mutated_pose = mutate_pose(pose, args.target_residue_number, args.target_residue_name, output_prefix)
        mutated_phi_values.append(mutated_pose.phi(args.target_residue_number))
        mutated_psi_values.append(mutated_pose.psi(args.target_residue_number))
        mutated_structure = load_rosetta(mutated_pose)
        #mutated_structure.save(f"{output_prefix}_mutant_structure.pdb")
        mutated_structures.append(mutated_structure)
        structure = mutated_structure
        mutated_energy = omm_system_energy(structure)
        mutated_energies.append(mutated_energy)
        #with open(f"{args.output_dir}/mutated_energy24_{frame_idx}.txt", "w") as f:
        #    f.write(f"Energy: {mutated_energy}\n")
        energy_diff = mutated_energy - omm_energy
        energy_diffs.append(energy_diff)
        # if args.verbose:
        print(f"Processed frame {frame_idx + 1}/{traj.n_frames}")
    #calculate the difference between the mutated and original phi and psi values
    phi_diff = [mutated_phi_values[i] - phi_values[i] for i in range(len(phi_values))]
    psi_diff = [mutated_psi_values[i] - psi_values[i] for i in range(len(psi_values))]

    #prepare data for csv file
    omm_energies = [float(energy.value_in_unit(kilojoules_per_mole)) for energy in omm_energies]
    mutated_energies = [float(energy.value_in_unit(kilojoules_per_mole)) for energy in mutated_energies]
    energy_diffs = [float(energy.value_in_unit(kilojoules_per_mole)) for energy in energy_diffs]

    #Save all information to a csv file
    with open(f"{args.output_dir}/omm_energies.csv", "w") as f:
        f.write("stride, omm_energy (kJ/mol), mutated_energy (kJ/mol), energy_diff (kJ/mol), phi, psi, phi_diff, psi_diff, \n")
        for i in range(len(omm_energies)):
            f.write(f"{stride_values[i]}, {omm_energies[i]}, {mutated_energies[i]}, {energy_diffs[i]}, {phi_values[i]}, {psi_values[i]}, {phi_diff[i]}, {psi_diff[i]}\n")

    #save a log of the arguments used to run the script
    with open(f"{args.output_dir}/args.log", "w") as f:
        f.write(f"{args}\n")
    




if __name__ == "__main__":
    main()



