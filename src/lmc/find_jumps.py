import argparse
import logging
import os
import pickle
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix

from trajec_io import readwrite
from lmc.prepare_lmc import Settings, AnalysisHelper, boolean_string

def main():
#        logging.basicConfig(filename='misp.log', level=logging.DEBUG)
#        logging.debug('command line:')
        readwrite.start_logging()
        logging.info(sys.argv)
        #logging.debug(readwrite.get_git_version())
        #logging.debug(readwrite.log_git_version())
        #print("command line:")
        #print(sys.argv)
        #readwrite.get_git_version()
        parser = argparse.ArgumentParser()
        parser.add_argument("path1", help="path to xyz trajec")
        parser.add_argument("pbc_path", help="path to pbc numpy mat")
        parser.add_argument("com", help="remove center of mass movement" , type=boolean_string)
        parser.add_argument("wrap", help="wrap trajectory, unwrap trajectory or do nothing", choices=["wrap", "unwrap", "nowrap"])

        parser.add_argument("speed", help="every i-th step from trajec is used for neighbor mat", type = int)
        parser.add_argument("jumping_ion_type", help="which atoms are transfered?")
        parser.add_argument("lattice_types", help="specify atom type for grid", nargs='+')
        parser.add_argument("--dynamic",  action="store_true", help="update grid from trajectory in every step")
        #parser.add_argument("--lattice_coords", help="path to lattice coordinates; otherwise first frame of trajec will be used", const="first_frame", nargs='?')
        parser.add_argument("--lattice_coords", help="path to lattice coordinates; otherwise first frame of trajec will be used", default="first_frame")
        parser.add_argument("--speed", help="every i-th step from trajec is used for neighbor mat", type = int, default = 1)
        parser.add_argument("--custom", action="store_true", help = "use custom_lmc.py for custom function prepare_trajectory")


        args = parser.parse_args()
        # Load pbc, trajectory and fixed lattice coordinates if supplied
        pbc_mat = np.loadtxt(args.pbc_path)
        traj = readwrite.easy_read(args.path1, pbc_mat, args.com, args.wrap)
        if args.dynamic:
            fixed_lattice = None
        else:
            if args.lattice_coords == "first_frame":
                fixed_lattice = traj.coords[ 0, np.isin( traj.atomlabels, args.lattice_types ), : ]
                #np.save("lattice1.npy", fixed_lattice)
                np.save("lattice1.npy", np.squeeze(fixed_lattice))
#            elif args.lattice_coords is not None:
            else:
                fixed_lattice = readwrite.easy_read( args.lattice_coords, pbc_mat, True, "nowrap" ).coords
                #np.save("lattice1.npy", fixed_lattice)
                np.save("lattice1.npy", np.squeeze(fixed_lattice))
        # put passed arguments into Settings object
        angle_atoms = None
        normal_occupation = None
        settings = Settings(args.jumping_ion_type, args.lattice_types, angle_atoms, normal_occupation, args.speed, fixed_lattice)

        # apply custom function to trajectory if supplied
        if args.custom:
            sys.path.append(os.getcwd())
            print(f"Loading function prepare_trajectory from {os.getcwd()}/custom_lmc.py")
            from custom_lmc import prepare_trajectory
            prepare_trajectory(traj)

        # construct AnalysisHelper and calculate everything else from it
        analysishelper = AnalysisHelper(traj, settings)
        jump_info = find_jumps( analysishelper, fixed_lattice )

def find_jumps(helper, fixed_lattice = None, neighbors = None, neighbor_file = "jumping_ion_neighbors.npy", output_file = "jump_info.npy"):
    """
    Find all jumps based on the information contained in the AnalysisHelper.

    helper: AnalysisHelper
        AnalysisHelper constructed from the trajectory and settings.
    fixed_lattice: ndarray or None
        2D Array containing fixed lattice points to use instead of atom positions from the trajectory.
    neighbor_file: string
        indices of the lattice point neighbors of each jumping ion over the course of the trajectory (will be created, if it doesn't exist, unless None is passed)
    output_file: string or None
        path into which the jump information will be written (unless None)

    Returns
    -------
    jump_info: ndarray
        All jumps found in the trajectory as a sparse matrix of shape (jump_no, 3). Entries are of format (timestep, source_lattice_site, destination_lattice_site).
    """
    if not isinstance(neighbors, np.ndarray):
        try:
            neighbors = np.load(neighbor_file)
        except TypeError:
            neighbors = neighbors_in_traj(helper, fixed_lattice)
        except FileNotFoundError:
            neighbors = neighbors_in_traj(helper, fixed_lattice)
            np.save(neighbor_file, neighbors)

    print(f"Found {neighbors.shape[0]} timesteps.")
    jump_timesteps, jump_ions = np.where(neighbors[1:] != neighbors[:-1]) # timestep and ion for which the lattice site changed
    lattice_start = neighbors[jump_timesteps, jump_ions]   # lattice site before jump
    lattice_end = neighbors[jump_timesteps + 1, jump_ions] # lattice site after jump
    jump_info = np.column_stack((jump_timesteps, lattice_start, lattice_end))
    print(f"Found {len(jump_info)} jumps.")
    if output_file is not None:
        np.save(output_file, jump_info)
        #final_list = convert_jump_info_to_pickle_coo_list(jump_info, helper)
        #pickle.dump(final_list,  open( "pickle_" + output_file + ".p", "wb" ))
        #pickle.dump(final_list,  open( "pickle_" + "jump_info" + ".p", "wb" ))
    return jump_info

# find indices of lattice point neighbors among lattice atoms for all jumping ions in a frame
def neighbors_in_frame(helper, timestep):
    neighbor_indices, _ = helper.ion_traj.next_neighbor2( helper.ion_traj.coords[timestep], helper.lattice_traj.coords[timestep])
    return neighbor_indices

# find indices of lattice point neighbors for all jumping ions in whole trajectory
def neighbors_in_traj(helper, fixed_lattice = None):
    neighbors = np.zeros((helper.frame_no, helper.ion_no), dtype = np.int64)
    if fixed_lattice is not None:
        for i in range(helper.frame_no):
            neighbors[i] = neighbors_in_frame_fixed(helper, fixed_lattice, i)
    else:
        for i in range(helper.frame_no):
            neighbors[i] = neighbors_in_frame(helper, i)
    return neighbors

# find indices of lattice point neighbors among fixed lattice coordinates for all jumping ions in a frame
def neighbors_in_frame_fixed(helper, fixed_lattice, timestep):
    neighbor_indices, _ = helper.ion_traj.next_neighbor2(helper.ion_traj.coords[timestep], fixed_lattice)
    return neighbor_indices

# turn jump information from custom sparse format into list of scipy.sparse.coo_matrix
def convert_jump_info_to_pickle_coo_list(jump_info, helper):
    final_list = []
    #for i in range(max(jump_info[:, 0])):
    for i in range(helper.frame_no):
            row = jump_info[:, 2][jump_info[:, 0] == i]
            col = jump_info[:, 1][jump_info[:, 0] == i]
            data = np.ones(col.shape[0])
            final_list.append(coo_matrix((data, (row, col)), shape=(helper.lattice_no, helper.lattice_no)))
    return final_list
