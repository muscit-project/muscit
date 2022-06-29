import argparse
import logging
import os
import pickle
import sys
import json

import numpy as np
from scipy.sparse import coo_matrix

from trajec_io import readwrite
from lmc.prepare_lmc import Settings, AnalysisHelper, boolean_string
from lmc.jump_rates_and_fit import oldformat_to_fermi, json_to_fermi, all_distances_in_frame, all_distances_angle_criterium_in_frame, fermi

def main():
        #print("command line:")
        #print(sys.argv)
        #readwrite.get_git_version()
        readwrite.start_logging()
        logging.info(sys.argv)
        parser = argparse.ArgumentParser(description='This executable calculates a jump probability matrix (pickle_jump_mat_proc.p). The ij-th  matrix element correpond to probability for a transfer/jump of an ion betweent lattice sites i and j.  This scripts requires the output of the script evaluate_jumps_on_grid and the fermi_param file created by the script  adv_calc_jump_rate using the option --fit. fermi_param file contains only the three parameters for a fermi function.')
        parser.add_argument("path1", help="path to xyz trajec")
        parser.add_argument("pbc_path", help="path to pbc numpy mat")
        parser.add_argument("com", help="remove center of mass movement" , type=boolean_string)
        parser.add_argument("wrap", help="wrap trajectory, unwrap trajectory or do nothing", choices=["wrap", "unwrap", "nowrap"])

        parser.add_argument("lattice_types", help="specify atom type for grid", nargs='+')
        # might want to remove this -> this script would necessarily need output from jumprates_and_fit to work; currently those outputs can be generated on-the-fly
        parser.add_argument("--jumping_ion_type", help="which atoms are transfered?", default = "H")
        parser.add_argument("--speed", help="every i-th step from trajec is used for neighbor mat", type = int, default=1)
        parser.add_argument("--angle_atoms", help="atoms for angle definiton", nargs = '+')
        parser.add_argument("--normal_occupation", help="usual number of ions attached to each lattice point", nargs = '+', default = None)
        parser.add_argument("--asymmetric", action="store_true", help = "calculate jump probabilities asymmetrically")
        parser.add_argument("--custom", action="store_true", help = "use custom_lmc.py for custom function prepare_trajectory")

        args = parser.parse_args()

        # Load pbc, trajectory and fixed lattice coordinates if supplied
        pbc_mat = np.loadtxt(args.pbc_path)
        traj = readwrite.easy_read(args.path1, pbc_mat, args.com, args.wrap)
        # put passed arguments into Settings object
        settings = Settings(args.jumping_ion_type, args.lattice_types, args.angle_atoms, args.normal_occupation, args.speed, None)

        # apply custom function to trajectory if supplied
        if args.custom:
            sys.path.append(os.getcwd())
            print(f"Loading function prepare_trajectory from {os.getcwd()}custom_lmc.py")
            from custom_lmc import prepare_trajectory
            prepare_trajectory(traj)

        # construct AnalysisHelper and calculate everything else from it
        analysishelper = AnalysisHelper(traj, settings)
        #jump_info = find_jumps( analysishelper, fixed_lattice )
        #find_jumpprobs_and_fit(analysishelper, neighborfile = "jumping_ion_neighbors.npy", jumpfile = "jump_info.npy", fermi_file = "fermi_param")
        create_jumpprob_matrix(analysishelper, fermi_file = "fermi_param", distances_file = "pickle_dist_mat_angle.p", out_jumpprobs = "pickle_jump_mat_proc.p")

def create_propagation_matrix(coord, atom, pbc_mat, lattice_atoms, angle_atoms = None, fermi_file = "fermi.json", asymmetric = False, normal_occupation = None):
    traj = readwrite.Trajectory(coord, atom, pbc_mat)
    settings = Settings(None, lattice_atoms, angle_types = angle_atoms, normal_occupation = normal_occupation, speed = 1, fixed_lattice = None)
    helper = AnalysisHelper(traj, settings)
    return create_jumpprob_matrix(helper, fermi_file = fermi_file, distances_file = None, out_jumpprobs = "pickle_jump_mat_proc.p")


def create_jumpprob_matrix(analysishelper, fermi_file = "fermi_param", distances_file = "pickle_dist_mat_angle.p", out_jumpprobs = "pickle_jump_mat_proc.p"):
    try:
        with open(distances_file, "r") as f:
            all_distances_angle = pickle.loads(f)
    except (TypeError, FileNotFoundError):
        all_distances_angle = None

    if fermi_file is not None:
        if fermi_file.endswith('.json'):
            fermi_params = json_to_fermi(fermi_file)
        else:
            fermi_params = oldformat_to_fermi(analysishelper, fermi_file)
    else:
        fermi_params = find_jumpprobs_and_fit(analysishelper)

    all_jumpprobs = []
    for i in range(analysishelper.frame_no):
        if i % 100 == 0:
            print(f"{i} / {analysishelper.frame_no}", end = '\r')
        if all_distances_angle is not None:
            frame_distances_angle = all_distances_angle[i]
            sources, destinations, distances = frame_distances_angle.row, frame_distances_angle.col, frame_distances_angle.data
        else:
            # Calculate jump distances, apply angle criterium and attach results to all_distances and all_distances_angle
            frame_distances = all_distances_in_frame(analysishelper.lattice_traj, i)
            frame_distances[ frame_distances > 3 ] = 0
            frame_distances[ frame_distances < 0.5 ] = 0
            frame_distances_angle = all_distances_angle_criterium_in_frame(analysishelper, i, frame_distances)
            sources, destinations = np.where(frame_distances_angle)
            distances = frame_distances_angle[sources, destinations]

        for pair in analysishelper.pair_generator():
            pair_mask = analysishelper.pair_identifiers[sources, destinations] == pair
            distances[pair_mask] = fermi( distances[pair_mask], *fermi_params[pair] )

        #all_jumpprobs.append( coo_matrix( (distances, (sources, destinations)), shape = (analysishelper.lattice_no, analysishelper.lattice_no)) )
        all_jumpprobs.append( coo_matrix( (distances, (destinations, sources)), shape = (analysishelper.lattice_no, analysishelper.lattice_no)) )

    if out_jumpprobs is not None:
        with open( out_jumpprobs, "wb" ) as f:
            pickle.dump(all_jumpprobs, f)

    return all_jumpprobs
