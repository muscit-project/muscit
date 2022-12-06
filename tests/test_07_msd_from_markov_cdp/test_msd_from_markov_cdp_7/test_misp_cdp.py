# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit

def test_misp_msd_cdp_7():
    start_dir = os.getcwd()
    p = os.getcwd()
    print( "My Path: ", p )
    assert p[-1] == "7"
    #cur_dir = os.getcwd()
    #assert cur_dir[-1]  == '7'
    #create_propagation_matrix o_400.xyz  pbc False False O
    #create_propagation_matrix csh2po4-pos-1_out.xyz cell True True O --angle_atoms P
    #subprocess.call(['create_propagation_matrix', 'csh2po4-pos-1_out.xyz', 'cell', 'True', 'True', 'O', '--angle_atoms', 'P'])
    subprocess.call(['create_propagation_matrix', 'csh2po4-pos-1_out.xyz', 'cell', 'True', 'nowrap', 'O', '--angle_atoms', 'P'])
    #misp csh2po4-pos-1_out.xyz cell True nowrap O 1 50 1000 1000000 lmc.out 10000 100 True
    subprocess.call(['misp',  'csh2po4-pos-1_out.xyz',  'cell', 'True', 'nowrap', 'O', '1', '50', '1000',  '1000000', 'lmc.out', '10000', '100', 'True', '--seed', '12341234', '--verbose'])
    #subprocess.call(['msd_from_markov', 'jump_mat_added_trace.txt', 'pbc', 'lattice.txt', '10', '1'])

    msd = np.loadtxt("msd_from_lmc.out")
    print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n
    popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    print(popt, pcov)
    #assert popt == pytest.approx(np.array([3.46400737e-03, 5.14881123e+00]),0.01 )
    #assert popt == pytest.approx(np.array([-9.50766437e-05,  2.23054282e+00]),0.01 )
    assert popt == pytest.approx(np.array([0.00801737, 1.72526928]),0.01 )

if __name__ == "__main__":
    test_misp_msd_cdp_7()
