# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit



def test_markov_9():
    cur_dir = os.getcwd()
    assert cur_dir[-1:]  == '9'
    subprocess.call(['create_propagation_matrix', 'o_n_sixteen_binary.xyz', 'pbc', 'False', 'nowrap', 'O', 'N', '--asymmetric'])
    
    subprocess.call(['misp', 'o_n_sixteen_binary.xyz', 'pbc', 'False', 'nowrap', 'O', 'N', '1', '1000', '100', '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose'])
     
    msd = np.loadtxt("msd_from_lmc.out")
    print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n 
    popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    print(popt, pcov)
    assert popt == pytest.approx(np.array([ 0.03228293, -0.00642909]),0.01 )

#def test_markov_8_proton_1_hole():
#    #os.chdir("/home/dressler/projects/chrisbase/tests/test1")
#    start_dir = os.getcwd()
#    os.chdir("test_misp_diff_coef_o_nine_6")
#    cur_dir = os.getcwd()
#    assert cur_dir[-1:]  == '6'
#    #create_propagation_matrix o_400.xyz  pbc False False O
#    subprocess.call(['create_propagation_matrix', 'o_nine.xyz', 'pbc', 'False', 'False', 'O'])
#
#    subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '8', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose'])
#    #subprocess.call(['msd_from_markov', 'jump_mat_added_trace.txt', 'pbc', 'lattice.txt', '10', '1'])
#
#    msd = np.loadtxt("msd_from_lmc.out")
#    print(msd)
#    def diff_coef(t, D, n):
#        return 6* D *t + n
#    popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
#    print(popt, pcov)
#    
#    #assert popt == pytest.approx(np.array([ 5.73166667e-02, -2.18116972e-12]),0.01 )
#    #assert popt == pytest.approx(np.array([0.05762717, 0.00437636]),0.01 )
#    #assert np.array([popt[0]*8, popt[1]]) == pytest.approx(np.array([0.05762717, 0.00437636]),0.01 )
#    assert np.array([popt[0]*8, popt[1]]) == pytest.approx(np.array([0.05762717, 0.00057182]),0.01 )
#    #assert mat1 == pytest.approx(ref_mat[0].todense(), 0.01)
#    os.chdir(start_dir)
#
#
#
if __name__ == "__main__":
    test_markov_9()
