# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit

def test_markov_5():
    start_dir = os.getcwd()
    subprocess.call(['./aimd_automtic_create']) 
    subprocess.call(['./aimd_from_neighbor_automtic_create']) 
    subprocess.call(['./msd_from_jump_info_automtic_create']) 
    subprocess.call(['./msd_from_neigh_automtic_create']) 
    number =  1000
    for i in range(4):
        proton = i+1
        aimd = np.loadtxt(f"{number}/{proton}/aimd/msd_multi_interval_one_loop_reduced")
        #print(aimd[1,1])
        aimd_neigh = np.loadtxt(f"{number}/{proton}/aimd_from_neigh_mat/msd_multi_intevall_neigh_mat")
        msd_neigh = np.loadtxt(f"{number}/{proton}/msd_from_neigh/msd_from_markov")
        msd_jump = np.loadtxt(f"{number}/{proton}/msd_from_jump_info/msd_from_markov")
        print(aimd[1,1], aimd_neigh[1,1], msd_neigh[1,1], msd_jump[1,1])      
        print(aimd[249,1], aimd_neigh[249,1], msd_neigh[249,1], msd_jump[249,1])      
        assert  msd_neigh[1,1] == pytest.approx(aimd[1,1] , rel=0.001, abs=0.0001 )
        assert  msd_neigh[1,1] == pytest.approx(aimd_neigh[1,1] , rel=0.001, abs=0.0001 )
        assert  msd_neigh[249,1] == pytest.approx(2.0, rel=0.001, abs=0.0001 )
  
    os.chdir(start_dir)


if __name__ == "__main__":
    test_markov_5()
