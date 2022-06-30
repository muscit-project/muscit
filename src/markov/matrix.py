import numpy as np
from trajec_io import readwrite
import argparse
import os
import pickle
import sys
import time
from scipy import sparse
import warnings
from scipy.sparse import coo_matrix



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def scale_transition_matrix(markov_mat, factor):
    markov_mat *= factor
    #for i in range(markov_mat.shape[0]):
    #    markov_mat[i,i] = 0
    np.fill_diagonal(markov_mat, 0)
    sum_mat = np.sum(markov_mat, axis = 0)
    new_diag = 1 - sum_mat
    markov_mat += np.diag(new_diag)
    return markov_mat


def get_msd_from_markov(mat_av, lattice1, dt_markov, msd_steps, pbc):
    square_mat = np.zeros(mat_av.shape)
    for i in range(square_mat.shape[0]):
        for j in range(square_mat.shape[0]):
            square_mat[i, j] = np.linalg.norm(readwrite.pbc_dist(lattice1[i],lattice1[j], pbc))**2
    time1 = []
    msd = []
    tmp1 = np.eye(mat_av.shape[0])
    for i in range(msd_steps):
        dist_act = square_mat * tmp1
        msd.append(dist_act.sum())
        time1.append(i * dt_markov)
        tmp1 = mat_av @ tmp1
    final = np.zeros((msd_steps, 2))
    msd = np.array(msd) / mat_av.shape[0]
    time1 = np.array(time1)
    final[:, 0] = time1
    final[:, 1] = msd
    np.savetxt("msd_from_markov", final)




def script_msd():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("path_jump_mat", help="path to jump mat")
    parser.add_argument("path_pbc", help="path to file with pbc")
    parser.add_argument("path_to_lattice", help="path to lattice")
    parser.add_argument("msd_steps", type=int, help="length of markov chain")
    parser.add_argument("time_step", type=float, help="time intervall described by  markov matrix")
    args = parser.parse_args()
    
    pbc = np.loadtxt(args.path_pbc)
    if args.path_jump_mat[-2:] == ".p":
        warnings.warn("A list of sparse matrices is assumed as pickle object. This converts only the first element of the list into a dense metrix.")
        pre_jump_mat = pickle.load(open(args.path_jump_mat, "rb"))
        jump_mat = pre_jump_mat[0].todense()
    elif args.path_jump_mat[-4:] == ".txt":
        jump_mat = np.loadtxt(args.path_jump_mat)
    elif args.path_jump_mat[-4:] == ".npy":
        jump_mat = np.load(args.path_jump_mat)
    else:
        raise Exception("error: jump matrix is not in .p or .txt or .npy format")
    print(jump_mat[:5,:5])
    
    if args.path_to_lattice[-4:] == ".txt":
        lattice = np.loadtxt(args.path_to_lattice)
    elif args.path_to_lattice[-4:] == ".npy":
        lattice = np.load(args.path_to_lattice)
    else:
        raise Exception("error:  lattice is not in .txt or .npy format")
    get_msd_from_markov(jump_mat, lattice, args.time_step, args.msd_steps, pbc)


def markov_matrix_from_dynamic_approach(jump_mat, steps):
    dummy_mat = jump_mat[0].todense()
    markov_mat = np.eye(dummy_mat.shape[0])
    for i in range(steps):
        current_jump_mat = np.copy(jump_mat[i].todense())
        for j in range(current_jump_mat.shape[1]):
            current_jump_mat[j,j] = 1 - np.sum(current_jump_mat[:,j])
        markov_mat = current_jump_mat @ markov_mat

    return markov_mat

def script_markov_matrix_from_dynamic_approach():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("path_jump_mat", help="path to jump mat")
    parser.add_argument("--path_to_trajec", help="path to trajectory, if specified lattice1.npy is created")
    parser.add_argument("--lattice_atoms", help="lattice atoms for creation  of lattice1.npy")
    parser.add_argument("--intervall_length_in_steps", type=int, help="how many elementary transition matrices will be assembled?")
    parser.add_argument("--pbc", help="path to pbc numpy mat")
    parser.add_argument("--com", help="remove center of mass movement" , type=boolean_string)
    parser.add_argument("--wrap", help="wrap trajec", type=boolean_string)
    args = parser.parse_args()
    
    if args.path_jump_mat[-2:] == ".p":
        warnings.warn("A list of sparse matrices is assumed as pickle object.")
        jump_mat = pickle.load(open(args.path_jump_mat, "rb"))
    else:
        raise Exception("error: jump matrix is not in .p  format")


 
    if not args.intervall_length_in_steps:
        intervall_length_in_steps = len(jump_mat)
    else:
        intervall_length_in_steps = args.intervall_length_in_steps
    
    if args.path_to_trajec:
        pbc_mat = np.loadtxt(args.pbc)
        coord, atom = readwrite.easy_read(args.path_to_trajec, pbc_mat, args.com, args.wrap)     
        lattice =  coord[intervall_length_in_steps-1, atom == args.lattice_atoms, :]
        np.save("lattice1", lattice)

    markov_matrix = markov_matrix_from_dynamic_approach(jump_mat, intervall_length_in_steps) 
    np.savetxt("markov_matrix_" + str(intervall_length_in_steps) + "_.txt", markov_matrix)
        



# turn jump information from custom sparse format into list of scipy.sparse.coo_matrix
def convert_jump_info_to_pickle_coo_list(jump_info):
    final_list = []
    lattice = np.load("lattice1.npy")
    neighbors = np.load("jumping_ion_neighbors.npy")
    frame_no = neighbors.shape[0]
    lattice_no = lattice.shape[0]
    #for i in range(max(jump_info[:, 0])):
    for i in range(frame_no):
            row = jump_info[:, 2][jump_info[:, 0] == i]
            col = jump_info[:, 1][jump_info[:, 0] == i]
            data = np.ones(col.shape[0])
            final_list.append(coo_matrix((data, (row, col)), shape=(lattice_no, lattice_no)))
    return final_list        

def script_markov_matrix_from_jump_info():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("path_jump_mat", help="path to jump mat")
    parser.add_argument("intervall_length_in_steps", type=int, help="how many elementary transition matrices will be assembled?")
    parser.add_argument("number_of_jumping_ions",  type=int, help="number of jumpin ions in MD simulation.")
    args = parser.parse_args()


    warnings.warn("path_jump_mat is not used to obtain the matrix jump_mat. path_jump_mat is nor used, dummy variable for backward compatibility. jump is obtained from jump_info.npy")


    jump_info = np.load("jump_info.npy")
    jump_mat =  convert_jump_info_to_pickle_coo_list(jump_info)


    markov_matrix = markov_matrix_from_jump_info(jump_mat, args.intervall_length_in_steps, args.number_of_jumping_ions)
    np.savetxt("markov_matrix_" + str(args.intervall_length_in_steps) + "_.txt", markov_matrix)

def markov_matrix_from_jump_info(jump_mat, steps, noji):
    final_list = []
    for i in range(0,len(jump_mat)-steps):
        dummy_mat = jump_mat[0].todense()
        markov_mat = np.eye(dummy_mat.shape[0])
        for j in range(steps):
            current_jump_mat = np.copy(jump_mat[i+j].todense())
            for k in range(current_jump_mat.shape[1]):
                if np.sum(current_jump_mat[:,k]) == 0:
                    current_jump_mat[k,k] = 1
                if np.sum(current_jump_mat[:,k]) > 1:
                    current_jump_mat[:,k] = current_jump_mat[:,k]/np.sum(current_jump_mat[:,k])
            markov_mat = markov_mat @ current_jump_mat
        final_list.append(markov_mat)
    markov_mat = np.mean(np.array(final_list), axis = 0)
    markov_mat = scale_transition_matrix(markov_mat, markov_mat.shape[0]/noji)
    return markov_mat


def script_markov_matrix_from_neigh_mat():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("path_neigh_mat", help="path to neighbor matrix")
    parser.add_argument("intervall_length_in_steps", type=int, help="how many elementary transition matrices will be assembled?")
    parser.add_argument("lattice", help="lattice")
    parser.add_argument("number_of_jumping_ions",  type=int, help="number of jumpin ions in MD simulation.")
    args = parser.parse_args()
 

    neigh_mat = np.load(args.path_neigh_mat)
    lattice = np.load(args.lattice)
    
    markov_matrix = markov_matrix_from_neigh_mat(neigh_mat, args.intervall_length_in_steps, lattice, args.number_of_jumping_ions)
    np.savetxt("markov_matrix_" + str(args.intervall_length_in_steps) + "_.txt", markov_matrix)


def markov_matrix_from_neigh_mat(neigh_mat, steps, lattice, noji):
    final_list = []
    for i in range(0, neigh_mat.shape[0]-steps):
        dummy_mat = np.zeros((lattice.shape[0], lattice.shape[0]))
        for j in range(neigh_mat.shape[1]):
            dummy_mat[neigh_mat[i+steps,j] ,neigh_mat[i,j]] += 1
        for j in range(lattice.shape[0]):
            if np.sum(dummy_mat[:,j]) == 0:
                dummy_mat[j,j] =  1
        final_list.append(dummy_mat)
    markov_mat = np.mean(np.array(final_list), axis = 0)
    markov_mat = scale_transition_matrix(markov_mat, markov_mat.shape[0]/noji)
    return markov_mat




########### deprecated functions

#def markov_main(jump_mat, start1, end1, duration, delay1):
#    mat_list = []
#    for i in range(start1, end1 - delay1, delay1):
#        tmp_mat = np.eye((jump_mat.shape[1]))
#        for j in range(duration):
#            tmp_mat = jump_mat[i + j] @ tmp_mat
#        mat_list.append(tmp_mat)
#    mat_av = np.array(mat_list)
#    mat_av = mat_av.sum(axis=0)
#    print(mat_av.shape)
#    for i in range(mat_av.shape[0]):
#        mat_av[:, i] /= mat_av[:, i].sum()
#    np.save("markov_matrix_interval_" + str(duration), mat_av)
#    return mat_av
#def markov_mat_from_count(jump_mat, duration, delay):
#    mat_list = []
#    for i in range(0, len(jump_mat) - duration, delay):
#        tmp1 = sparse.coo_matrix(jump_mat[0].shape)
#        for j in range(duration):
#            tmp1 += jump_mat[i + j]
#        tmp1 /= duration
#        mat_list.append(tmp1)
#    markov_mat = np.array(mat_list).sum(axis=0) / len(mat_list)
#    return markov_mat
#
#def spielerei_markov_matrix_from_jump_info(jump_mat, steps):
#    current_jump_mat = jump_mat[0].todense()
#    for i in range(1,len(jump_mat)):
#        current_jump_mat += np.copy(jump_mat[i].todense())    
#    return current_jump_mat/len(jump_mat)
