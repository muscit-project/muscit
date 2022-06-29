# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit

def test_markov_5():
    cur_dir = os.getcwd()
    assert cur_dir[-1:]  == '5'
    subprocess.call(['create_propagation_matrix', 'o_400.xyz', 'pbc', 'False', 'nowrap', 'O'])

    #create special jump mat (add trace)
    mat = pickle.load(open("pickle_jump_mat_proc.p", "rb"))
    tmp1 = mat[0].todense()
    import numpy as np
    tmp2 = np.copy(tmp1)
    for i in range(tmp1.shape[0]):
        tmp2[i,i] = 1- tmp1[i,:].sum()
    np.savetxt("jump_mat_added_trace.txt", tmp2)
    
    #create lattice
    xyz_tmp = np.load("o_400.xyz.npz")
    np.savetxt("lattice.txt", xyz_tmp['arr_0'][0])
    

    subprocess.call(['msd_from_markov', 'jump_mat_added_trace.txt', 'pbc', 'lattice.txt', '10', '1'])
     
    msd = np.loadtxt("msd_from_markov")
    def diff_coef(t, D, n):
        return 6* D *t + n 
    popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    assert popt == pytest.approx(np.array([ 5.73166667e-02, -2.18116972e-12]),0.000001 )
    #assert mat1 == pytest.approx(ref_mat[0].todense(), 0.01)

if __name__ == "__main__":
    test_markov_5()
