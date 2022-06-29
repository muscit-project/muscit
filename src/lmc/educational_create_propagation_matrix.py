import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix

from trajec_io import readwrite


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def angle1(v1, v2, acute):
    # v1 is your first vector
    # v2 is your second vector
    angle = np.arccos(
        np.sum(v1 * v2, axis=1)
        / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    )
    if acute == True:
        return angle
    else:
        return 2 * np.pi - angle


def fermi(x, a, b, c):
    return a / (1 + np.exp((x - b) * c))


def main():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser(
        description="This executable calculates a jump probability matrix (pickle_jump_mat_proc.p). The ij-th  matrix element correpond to probability for a transfer/jump of an ion betweent lattice sites i and j.  This scripts requires the output of the script evaluate_jumps_on_grid and the fermi_param file created by the script  adv_calc_jump_rate using the option --fit. fermi_param file contains only the three parameters for a fermi function.")
    parser.add_argument("path1", help="path to xyz trajec")
    parser.add_argument("pbc", help="path to pbc numpy mat")
    parser.add_argument("com", help="remove center of mass movement", type=boolean_string)
    parser.add_argument("wrap", help="wrap trajec", type=boolean_string)
    parser.add_argument("lattice_atoms", help="lattice atoms")
    parser.add_argument("--angle_atoms", help="atoms for angle definiton")
    args = parser.parse_args()
    pbc_mat = np.loadtxt(args.pbc)
    execute_create_propagation_matrix(args.path1, pbc_mat, args.com, args.wrap, args.lattice_atoms, args.angle_atoms)




def execute_create_propagation_matrix(path1, pbc_mat, com, wrap, lattice_atoms, angle_atoms = False, fermi_path = "fermi_param"): 
    coord, atom = readwrite.easy_read(path1, pbc_mat, com, wrap)
    coord_o = coord[:, atom == lattice_atoms, :]
    final = []
    for i in range(coord_o.shape[0]):
        test1 = readwrite.pbc_dist2(coord_o[i, :, :], coord_o[i, :, :], pbc_mat)
        test2 = np.linalg.norm(test1, axis=2)
        test2[test2 > 3.0] = 0
        test2[test2 < 0.5] = 0
        ind = np.where(test2 > 0.5)
        row = ind[0]
        col = ind[1]
        data = test2[ind]
        final.append(
            coo_matrix((data, (row, col)), shape=(coord_o.shape[1], coord_o.shape[1]))
        )
    if angle_atoms:    
        coord_p = coord[:, atom == angle_atoms, :]
        next_p_atoms, tmp2 = readwrite.next_neighbor2(
        coord_o[0, :, :], coord_p[0, :, :], pbc_mat
    )

        final_dist = []
        for i in range(len(final)):
            pind = next_p_atoms[final[i].row]
            vec1 = coord_o[i, final[i].row, :] - coord_p[i, pind, :]
            vec2 = coord_o[i, final[i].col, :] - coord_o[i, final[i].row, :]
            cc = 180 / np.pi * angle1(vec1, vec2, True)
            dd = cc > 89
            row = final[i].row[dd]
            col = final[i].col[dd]
            data = final[i].row[dd]
            final_dist.append(
                coo_matrix((data, (row, col)), shape=(coord_o.shape[1], coord_o.shape[1]))
                )
    else: 
        final_dist = final
    # riesen fehler:
    # pickle.dump(final,  open( "pickle_dist_mat_angle.p", "wb" ))
    pickle.dump(final_dist, open("pickle_dist_mat_angle.p", "wb"))
    dist_mat_angle_pickle = pickle.load(open("pickle_dist_mat_angle.p", "rb"))
    a, b, c = np.loadtxt(fermi_path)
    for i in range(len(dist_mat_angle_pickle)):
        dist_mat_angle_pickle[i].data = fermi(dist_mat_angle_pickle[i].data, a, b, c)
    pickle.dump(dist_mat_angle_pickle, open("pickle_jump_mat_proc.p", "wb"))
