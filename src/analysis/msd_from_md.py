import numpy as np
from trajec_io import readwrite
import timeit


import numpy as np
from ana_tools import basic
from trajec_io import readwrite
import argparse
import os
import pickle
import sys
import time
from scipy import sparse
from markov import neighbor
import warnings



#start = timeit.default_timer()

#Your statements here

#stop = timeit.default_timer()

#print('Time: ', stop - start)  







def pbc_dist_c(pos1, pos2, pbc_mat):
    dist_diff = pos1 - pos2
    inv = np.linalg.inv(pbc_mat)
    pbc_dist  = inv.T @ dist_diff.T
    #return pbc_dist
    pbc_dist -= np.fix(pbc_dist + np.sign(pbc_dist) * 0.5)
    new_dist = pbc_mat.T @ pbc_dist
    return new_dist.T

def msd_multiinteral_fast(coord_h, pbc_mat, tau_steps, delay, max_length = None):
    tmp1= []
    if max_length:
        limit = max_length
    else:
        limit = coord_h.shape[0]
    for key, i in enumerate(range(0, limit, tau_steps)):
       tmp1.append([])
       if i % 1 == 0:
           print(i)
       #coord_roll = np.roll(coord_h[:,:,:], i, axis = 0)    
       #dist_ar = np.linalg.norm(readwrite.pbc_dist(coord_h[:,:,:], coord_roll, pbc_mat), axis = 1)**2 
       if i == 0:
           dist_ar = np.linalg.norm(readwrite.pbc_dist(coord_h[0,:,:], coord_h[0,:,:], pbc_mat), axis = 1)**2
       else:
           dist_ar = np.linalg.norm(readwrite.pbc_dist(coord_h[i:,:,:], coord_h[:-i,:,:], pbc_mat), axis = 2)**2 
       dist_ar = dist_ar.flatten()
       tmp1[key] = tmp1[key] + dist_ar.tolist()
       tmp1[key] = np.mean(np.array(tmp1[key]))
    return np.array(tmp1)




def msd_multiinteral_slow(coord_h, pbc_mat, tau_steps, delay, max_length = None):
    tmp1= []
    if max_length:
        limit = max_length
    else:
        limit = coord_h.shape[0]
    #for key, i in enumerate(range(0, coord_h.shape[0], tau_steps)):
    for key, i in enumerate(range(0, limit, tau_steps)):
       #breakpoint() 
       #
       tmp1.append([])
       #if i % 500 == 0:
       if i % 1 == 0:
           print(i)
       #tmp1[key]
       for k in range(0, coord_h.shape[0]-i, delay):
       #for k in range(0, limit-i, delay):
            #for j in range(coord_h.shape[1]):
                #tmp1[key].append(np.linalg.norm(pbc_dist_c(coord_h[k+i,:,:], coord_h[k,:,:], pbc_mat), axis = 1)**2)
                #tmp1[key].append(list(np.linalg.norm(readwrite.pbc_dist(coord_h[k+i,:,:], coord_h[k,:,:], pbc_mat), axis = 1)**2))
                #breakpoint()
                #tmp1[key] + list(np.linalg.norm(readwrite.pbc_dist(coord_h[k+i,:,:], coord_h[k,:,:], pbc_mat), axis = 1)**2)
                tmp1[key] = tmp1[key] + (np.linalg.norm(readwrite.pbc_dist(coord_h[k+i,:,:], coord_h[k,:,:], pbc_mat), axis = 1)**2).tolist()
       #breakpoint()
       tmp1[key] = np.mean(np.array(tmp1[key]))
       #tmp1[key] = np.mean(tmp1[key])
    #np.savetxt("msd_multi_interval_one_loop_reduced", tmp1[:int(len(tmp1)/2)])
    return np.array(tmp1)


def msd_multiinteral_slow_from_neigh(lattice, neigh_mat, pbc_mat, tau_steps, delay, max_length = None):
    #df
    #tmp1= []
    if max_length:
        limit = max_length
    else:
        limit = neigh_mat.shape[0]
    tmp1= []
    #for key, i in enumerate(range(0, neigh_mat.shape[0], tau_steps)):
    for key, i in enumerate(range(0, limit, tau_steps)):
       tmp1.append([])
       if i % 500 == 0:
           print(i)
       for k in range(0, neigh_mat.shape[0]-i, delay):
            #for j in range(coord_h.shape[1]):
                tmp1[key].append(np.linalg.norm(pbc_dist_c(lattice[neigh_mat[k+i,:],:], lattice[neigh_mat[k,:],:], pbc_mat), axis = 1)**2)
       tmp1[key] = np.mean(tmp1[key])
    #np.savetxt("msd_multi_interval_one_loop_reduced", tmp1[:int(len(tmp1)/2)])
    return np.array(tmp1)




def script_msd_from_md():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    #parser.add_argument("path_jump_mat", help="path to jump mat")
    parser.add_argument("path_to_trajec", help=" Path to xyz-file of the trajectory.")
    #parser.add_argument("--lattice_atoms", help="lattice atoms for creation  of lattice1.npy")
    #parser.add_argument("--intervall_length_in_steps", type=int, help="how many elementary transition matrices will be assembled?")
    #parser.add_argument("--angle_atoms", help="atoms for angle definiton")
    #parser.add_argument("msd_steps", type=int, help="length of markov chain")
    parser.add_argument("pbc", help="Periodic boundary condition given as three vectors stacked on top of each other (shape: 3x3).")
    parser.add_argument("com", help="remove center of mass movement" , type= neighbor.boolean_string)
    parser.add_argument("wrap", help="Whether to wrap the trajectory, unwrap the trajectory or do neither. {wrap, unwrap, ...}")
    parser.add_argument("atom_type",  help="msd will be calculated for this atom type")
    parser.add_argument("time_step", help="time intervall for trajectory frame", type= float)
    parser.add_argument("tau_increment", help="variation of timeintervall for calculation of the msd (bins of x axis)", type= int)
    parser.add_argument("delay", help="delay betweent two time intervalls for calclation of msd" , type=  int)
    parser.add_argument("--max_length", type=int, help="maximum number of steps for msd")
    #path1: string
    #    Path to xyz-file of the trajectory.
    #pbc: ndarray
    #    Periodic boundary condition given as three vectors stacked on top of each other (shape: 3x3).
    #com: boolean
    #    If true, the center of mass will be set to (0, 0, 0) for every frame.
    #wrap: string {wrap, unwrap, ...}
    #    Whether to wrap the trajectory, unwrap the trajectory or do neither.
    args = parser.parse_args()


    pbc_mat = np.loadtxt(args.pbc)
    coord, atom = readwrite.easy_read(args.path_to_trajec, pbc_mat, args.com, args.wrap)
    coord_h = coord[:, atom == args.atom_type, :]
    
    start = timeit.default_timer()
    msd = msd_multiinteral_fast(coord_h, pbc_mat, args.tau_increment, args.delay, args.max_length)
    #msd = msd_multiinteral_slow(coord_h, pbc_mat, args.tau_increment, args.delay, args.max_length)
    stop = timeit.default_timer()
    print(f"duration: {stop-start}")
    tau = np.arange(0, msd.shape[0])
    tau *= args.tau_increment
    tau = tau * args.time_step
    
    data = np.vstack((tau, msd)).T
    #np.savetxt("msd_multi_interval_one_loop_reduced", tmp1[:int(len(tmp1)/2)])
    np.savetxt("msd_multi_interval_one_loop_reduced", data[:int(data.shape[0]/2)])


#path = "../csh2po4-pos-1_out.xyz"
#pbc_mat = np.loadtxt("../pbc")
#com = True
#unwrap = False
#coord, atom = readwrite.easy_read(path, pbc_mat, com, unwrap)
#coord_h = coord[:, atom == "H", :]
#tau_steps  = 10
#delay = 20
#
#msd_multiinteral_slow(coord_h, tau_steps, delay)
#stop = timeit.default_timer()
#print('Time: ', stop - start)

def script_msd_from_neigh():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("path_to_neighbor_mat", help=" Path to neighbor matrix.")
    parser.add_argument("path_to_lattice", help=" Path to neighbor matrix.")
    #parser.add_argument("--lattice_atioms", help="lattice atoms for creation  of lattice1.npy")
    #parser.add_argument("--intervall_length_in_steps", type=int, help="how many elementary transition matrices will be assembled?")
    #parser.add_argument("--angle_atoms", help="atoms for angle definiton")
    #parser.add_argument("msd_steps", type=int, help="length of markov chain")
    parser.add_argument("pbc", help="Periodic boundary condition given as three vectors stacked on top of each other (shape: 3x3).")
    #parser.add_argument("com", help="remove center of mass movement" , type= neighbor.boolean_string)
    #parser.add_argument("wrap", help="Whether to wrap the trajectory, unwrap the trajectory or do neither. {wrap, unwrap, ...}")
    #parser.add_argument("atom_type",  help="msd will be calculated for this atom type")
    parser.add_argument("time_step", help="time intervall for trajectory frame", type= float)
    parser.add_argument("tau_increment", help="variation of timeintervall for calculation of the msd (bins of x axis)", type= int)
    parser.add_argument("delay", help="delay betweent two time intervalls for calclation of msd" , type=  int)
    parser.add_argument("--max_length", type=int, help="maximum number of steps for msd")   
    args = parser.parse_args()
    
    neigh_mat = np.load(args.path_to_neighbor_mat)
    
    if args.path_to_lattice[-4:] == ".txt":
        lattice = np.loadtxt(args.path_to_lattice)
    elif args.path_to_lattice[-4:] == ".npy":
        lattice = np.load(args.path_to_lattice)
    else:
        raise Exception("error:  lattice is not in .txt or .npy format")



    pbc_mat = np.loadtxt(args.pbc)
    #coord, atom = readwrite.easy_read(args.path_to_trajec, pbc_mat, args.com, args.wrap)
    #coord_h = coord[:, atom == args.atom_type, :]

    start = timeit.default_timer()
    #msd = msd_multiinteral_slow(coord_h, pbc_mat, args.tau_increment, args.delay)
    #msd = msd_multiinteral_slow_from_neigh(lattice, neigh_mat,  pbc_mat, args.tau_increment, args.delay)
    msd = msd_multiinteral_slow_from_neigh(lattice, neigh_mat,  pbc_mat, args.tau_increment, args.delay, args.max_length)
    stop = timeit.default_timer()

    tau = np.arange(0, msd.shape[0])
    tau *= args.tau_increment
    tau = tau * args.time_step

    data = np.vstack((tau, msd)).T
    #np.savetxt("msd_multi_interval_one_loop_reduced", tmp1[:int(len(tmp1)/2)])
    np.savetxt("msd_multi_intevall_neigh_mat", data[:int(data.shape[0]/2)])
