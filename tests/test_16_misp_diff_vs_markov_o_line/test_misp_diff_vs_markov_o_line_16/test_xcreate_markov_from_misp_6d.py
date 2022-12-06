# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit
from trajec_io import readwrite
from analysis import msd




def test_recalc_diff1():
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '16'
    #create_propagation_matrix o_400.xyz  pbc False False O
    os.chdir("single_prot")
    pbc = np.loadtxt("pbc")
    #subprocess.call(['create_propagation_matrix', 'o_line.xyz', 'pbc', 'False', 'nowrap', 'O'])
    #subprocess.call(['misp',  'o_line.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])

    os.chdir("create_markov_diffusion_model")
    #evaluate_jumps_on_grid  lmc_final.xyz  pbc False nowrap 1 H O
    #markov_mat_from_neigh_mat  jumping_ion_neighbors.npy 1 lattice.npy 1
    #msd_from_markov  markov_matrix_1_.txt pbc lattice.npy  5 1
    #msd_from_md lmc_final.xyz pbc True wrap H 1 1 1 --max_length 10
    subprocess.call(['evaluate_jumps_on_grid',  '../lmc_final.xyz',   '../pbc',  'False', 'nowrap', '1', 'H', 'O'])
    
    subprocess.call(['markov_mat_from_neigh_mat', 'jumping_ion_neighbors.npy', '1', 'lattice1.npy', '1'])
    subprocess.call(['msd_from_markov',  'markov_matrix_1_.txt', '../pbc', 'lattice1.npy',  '5', '1'])
    subprocess.call(['msd_from_md', '../lmc_final.xyz', '../pbc', 'True', 'wrap', 'H', '1', '1', '1', '--max_length', '10'])
    markov = np.loadtxt("msd_from_markov") 
    ref = np.loadtxt("msd_multi_interval_one_loop_reduced")  
    unwrapped  = np.loadtxt("../msd_unwrap.out")
    assert markov[1,1] == pytest.approx(ref[1,1], 0.01)
    assert markov[1,1] == pytest.approx(unwrapped[1], 0.01)
    print(markov[1,1], ref[1,1], unwrapped[1])

    os.chdir("..")
    os.chdir("..")


def test_recalc_diff2():
    os.chdir("two_prot")
    pbc = np.loadtxt("pbc")
    os.chdir("create_markov_diffusion_model")

#evaluate_jumps_on_grid  special_coord.xyz  pbc False nowrap 1 H O
    subprocess.call(['evaluate_jumps_on_grid',  '../special_coord.xyz',   '../pbc',  'False', 'nowrap', '1', 'H', 'O'])
    #markov_mat_from_neigh_mat  jumping_ion_neighbors.npy 1 lattice.npy 1
    #msd_from_markov  markov_matrix_1_.txt pbc lattice.npy  5 1
    #msd_from_md special_coord.xyz pbc True wrap H 1 1 1 --max_length 10
    subprocess.call(['markov_mat_from_neigh_mat', 'jumping_ion_neighbors.npy', '1', 'lattice1.npy', '1'])
    subprocess.call(['msd_from_markov',  'markov_matrix_1_.txt', '../pbc', 'lattice1.npy',  '5', '1'])
    #subprocess.call(['msd_from_md', '../lmc_final.xyz', '../pbc', 'True', 'wrap', 'H', '1', '1', '1', '--max_length', '10'])
    subprocess.call(['msd_from_md', '../special_coord.xyz', '../pbc', 'True', 'wrap', 'H', '1', '1', '1', '--max_length', '10'])
    markov = np.loadtxt("msd_from_markov")
    ref = np.loadtxt("msd_multi_interval_one_loop_reduced")
    unwrapped  = np.loadtxt("../msd_hole.out")
    assert markov[1,1] == pytest.approx(ref[1,1], 0.01)
    assert markov[1,1] == pytest.approx(unwrapped[1], 0.01)
    print(markov[1,1], ref[1,1], unwrapped[1])
    os.chdir("..")
    os.chdir("..")


if __name__ == "__main__":
    test_recalc_diff2()

    test_recalc_diff1()


