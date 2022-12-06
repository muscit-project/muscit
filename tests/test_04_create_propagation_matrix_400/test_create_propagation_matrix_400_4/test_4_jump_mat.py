# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest

def test_4_jump_mat_400():
    subprocess.call(['create_propagation_matrix', 'o_400.xyz', 'pbc', 'False', 'nowrap', 'O'])
    #ref = pickle.load(open("pickle_jump_mat_o_nine.p", "rb"))
    ref_mat = pickle.load(open("ref_pickle_jump_mat_o_400.p","rb"))
    #own = pickle.load(open("pickle_jump_mat_proc.p", "rb"))
    jump_mat = pickle.load(open("pickle_jump_mat_proc.p", "rb"))
    #mat1 = ref[0].todense()
    #np.fill_diagonal(mat1, 0)
    #assert_array_almost_equal(own[0].todense(), mat1)
    for i in range(len(jump_mat)):
        mat1 = np.copy(jump_mat[i].todense())
        np.fill_diagonal(mat1,np.array(jump_mat[i].todense().shape[0]*[0.6]))
        assert mat1 == pytest.approx(ref_mat[0].todense(), 0.01)

if __name__ == "__main__":
    test_4_jump_mat_400()
