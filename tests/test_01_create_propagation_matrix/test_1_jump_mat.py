# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest

def test_1_jump_mat_nine():
    p = os.getcwd()
    print( "My Path: ", p )
    assert p[-1] == "1"
    subprocess.call(['create_propagation_matrix', '../data_o_nine/o_nine.xyz', '../data_o_nine/cell', 'False', 'nowrap', 'O'])
    ref_mat = pickle.load(open("../data_o_nine/ref_pickle_jump_mat_o_nine.p","rb"))
    jump_mat = pickle.load(open("pickle_jump_mat_proc.p", "rb"))
    for i in range(len(jump_mat)):
        mat1 = np.copy(jump_mat[i].todense())
        np.fill_diagonal(mat1,np.array(jump_mat[i].todense().shape[0]*[0.6]))
        assert mat1 == pytest.approx(ref_mat[0].todense(), 0.01)

if __name__ == "__main__":
    test_1_jump_mat_nine()
