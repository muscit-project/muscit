import os
import sys


import argparse
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
import json
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit

from trajec_io import readwrite

#import lmc.find_jumps
#import lmc.jump_rates_and_fit
#import lmc.propagation_matrix


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

def prepare_trajectory(traj):
    return

# settings passed to the preperation, most likely on the command line
class Settings:
    def __init__(self, jumping_ion_type, lattice_types, angle_types = None, normal_occupation = None, speed = 1, fixed_lattice = None):
        self.jumping_ion_type = jumping_ion_type
        self.lattice_types = lattice_types
        # turn angle criteria and normal occupations into dictionaries mapping lattice_type to the appropriate angle type/occupation
        self.angle_criteria = {lattice: angle for lattice, angle in zip(lattice_types, angle_types)} if angle_types is not None else None
        self.normal_occupation = {lattice: occupation for lattice, occupation in zip(lattice_types, normal_occupation)} if normal_occupation is not None else None
        self.speed = speed
        self.fixed_lattice = fixed_lattice

# the best thing since sliced bread (if you're trying to write lmc code that is)
class AnalysisHelper:
    def __init__(self, traj, settings):
        # jumping ion
        self.jumping_ion_type = settings.jumping_ion_type
        self.ion_traj = traj[::settings.speed, traj.atomlabels == settings.jumping_ion_type]
        self.ion_no = np.count_nonzero(traj.atomlabels == settings.jumping_ion_type)
        # lattice
        self.lattice_types = settings.lattice_types
        self.lattice_no = np.count_nonzero(np.isin(traj.atomlabels, settings.lattice_types))
        self.lattice_traj = traj[::settings.speed, np.isin(traj.atomlabels, settings.lattice_types)]
        # pair identifiers as strings "{type1}->{type2}" in an array of shape (lattice_no, lattice_no)
        self.pair_identifiers = np.array( [[source + '->' + destination for destination in self.lattice_traj.atomlabels] for source in self.lattice_traj.atomlabels] )
        # array of strings used to identify pairs; for n lattice_types there will be n^2 of these
        self.pair_strings = np.unique( self.pair_identifiers )
        # boolean arrays that can be used to extract data for one pair from an array of shape (lattice_no, lattice_no)
        self.pair_masks = { pair: self.pair_identifiers == pair for pair in self.pair_strings }
        # normal protonation for each lattice atom, e.g. 2 for a water oxygen
        if settings.normal_occupation is None:
            self.normal_protonation = np.zeros( self.lattice_no, dtype = int )
        else:
            self.normal_protonation = np.array( [settings.normal_occupation[lattice_label] for lattice_label in self.lattice_traj.atomlabels] )
        # atoms for angle criterium
        if settings.angle_criteria is None:
            self.angle_types = ["None"]
            self.angle_labels = np.full( self.lattice_no, "None" )
            self.angle_trajs = {}
        else:
            self.angle_types = settings.angle_criteria.values()
            self.angle_labels = np.array([settings.angle_criteria[lattice_label] for lattice_label in self.lattice_traj.atomlabels])
            self.angle_trajs = {angle_type: traj[::settings.speed, traj.atomlabels == angle_type] for angle_type in self.angle_labels}
        # nice-to-haves
        self.frame_no = self.ion_traj.coords.shape[0]

    # return iterator containing every possible combination of lattice types in tuples
    def pair_generator(self):
        return iter( [l1 + '->' + l2 for l1 in self.lattice_types for l2 in self.lattice_types] )
#        return itertools.product(self.lattice_types, self.lattice_types)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to xyz trajec")
    parser.add_argument("pbc_path", help="path to pbc numpy mat")
    parser.add_argument("com", help="remove center of mass movement" , type=boolean_string)
    parser.add_argument("wrap", help="wrap trajectory, unwrap trajectory or do nothing", choices=["wrap", "unwrap", "nowrap"])

    parser.add_argument("speed", help="every i-th step from trajec is used for neighbor mat", type = int)
    parser.add_argument("jumping_ion_type", help="which atoms are transfered?")
    parser.add_argument("lattice_types", help="specify atom type for grid", nargs='+')
    parser.add_argument("--angle_atoms", help="atoms for angle definiton", nargs = '+')
    parser.add_argument("--normal_occupation", help="usual number of ions attached to each lattice point", nargs = '+', default = [0])
#    parser.add_argument("--dynamic",  action="store_true", help="update grid from trajectory in every step")
    parser.add_argument("--lattice_coords", help="path to lattice coordinates; otherwise first frame of trajec will be used", default = None, const = "first_frame", nargs = '?')
    parser.add_argument("--asymmetric", action="store_true", help = "calculate jump probabilities asymmetrically")
    parser.add_argument("--custom", action="store_true", help = "use custom_lmc.py for custom function prepare_trajectory")

    args = parser.parse_args()

    # Load pbc, trajectory and fixed lattice coordinates if supplied
    pbc_mat = np.loadtxt(args.pbc_path)
    traj = readwrite.easy_read(args.path1, pbc_mat, args.com, args.wrap)
    if args.lattice_coords == "first_frame":
        fixed_lattice = traj.coords[ 0, np.isin( traj.atomlabels, args.lattice_type ), : ]
        np.save("lattice1.npy", np.squeeze(fixed_lattice))
    elif args.lattice_coords is not None:
        #fixed_lattice = np.load( args.lattice_coords )
        fixed_coords, fixed_atoms = readwrite.easy_read( args.lattice_coords, pbc_mat, True, "nowrap" )
        fixed_coords = np.squeeze(fixed_coords)
        fixed_lattice = fixed_coords[np.isin( fixed_atoms, args.lattice_types ),:]
        np.save("lattice1.npy", np.squeeze(fixed_lattice))
    else:
        fixed_lattice = None
    # put passed arguments into Settings object
    settings = Settings(args.jumping_ion_type, args.lattice_types, args.angle_atoms, args.normal_occupation, args.speed, fixed_lattice)

    # apply custom function to trajectory if supplied
    if args.custom:
        sys.path.append(os.getcwd())
        print(f"Loading function prepare_trajectory from {os.getcwd()}custom_lmc.py")
        from custom_lmc import prepare_trajectory
        prepare_trajectory(traj)

    # construct AnalysisHelper and calculate everything else from it
    analysishelper = AnalysisHelper(traj, settings)
#    lmc.find_jumpsfind_jumps( analysishelper, fixed_lattice )
#    lmc.jump_rates_and_fit.find_jumpprobs_and_fit(analysishelper, neighborfile = "jumping_ion_neighbors.npy", jumpfile = "jump_info.npy", fermi_file = "fermi_param")
#    lmc.propagation_matrix.create_jumpprob_matrix(analysishelper, fermi_file = "fermi_param", distances_file = "pickle_dist_mat_angle.p", out_jumpprobs = "pickle_jump_mat_proc.p")



def test_snphosphat():
    pbc = np.loadtxt("sn_phosphat.pbc")
    traj = readwrite.easy_read("MD_600_sn_phosphat-pos-1.xyz", pbc, com = False, wrapchoice = "nowrap")
    frame = traj.coords[0, :, :]
    frameO = frame[traj.atomlabels == "O", :]
    frameP = frame[traj.atomlabels == "P", :]
    OP_distances = traj.next_neighbor2(frameO, frameP)[1]
    is_phosphate_O = OP_distances < 1.7
    O_indices = np.where(traj.atomlabels == "O")[0]
    traj.atomlabels[O_indices[is_phosphate_O]] = "Op"
    traj.atomlabels[O_indices[np.logical_not(is_phosphate_O)]] = "Ow"

    helper = Settings("H", ["Op", "Ow"], ["P", "None"], [0, 2], 1)
    return traj, helper

