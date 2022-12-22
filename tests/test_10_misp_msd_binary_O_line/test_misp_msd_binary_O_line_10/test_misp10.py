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
    assert cur_dir[-2:]  == '10'
    subprocess.call(['create_propagation_matrix', 'o_line.xyz', 'pbc', 'False', 'nowrap', 'O'])
    subprocess.call(['misp',  'o_line.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose'])
    msd = np.loadtxt("msd_from_lmc.out")
    print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n 
    popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    print(popt, pcov)
    assert popt == pytest.approx(np.array([0.03199121, 0.00094727]),0.01 )


if __name__ == "__main__":
    test_markov_9()
