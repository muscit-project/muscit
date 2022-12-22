# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest

def test_create_jump_mat_14():
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '14'
    os.chdir("no_neigh")
    #o_line_4_non_eq_dist.xyz
    subprocess.call(['create_propagation_matrix', 'o_line_4_non_eq_dist.xyz', 'pbc', 'False', 'nowrap', 'O'])
    os.chdir("..")
    #os.chdir("p_neigh")
    #subprocess.call(['create_propagation_matrix', 'o_line_2p.xyz', 'pbc', 'False', 'nowrap', 'O', '--angle_atoms', 'P'])
    #os.chdir("..")
    os.chdir("p4_neigh")
    subprocess.call(['create_propagation_matrix', 'o_line_4p.xyz', 'pbc', 'False', 'nowrap', 'O', '--angle_atoms', 'P'])
    os.chdir("..")
    os.chdir("c_p_neigh")
    subprocess.call(['create_propagation_matrix', 'o_line_n_p_c.xyz', 'pbc', 'False', 'nowrap', 'O', 'N', '--angle_atoms', 'P', 'C', '--asymmetric'])
    os.chdir("..")
    ref_mat = pickle.load(open("no_neigh/pickle_jump_mat_proc.p","rb"))
    #jump_mat = pickle.load(open("p_neigh/pickle_jump_mat_proc.p", "rb"))
    jump_mat = pickle.load(open("p4_neigh/pickle_jump_mat_proc.p", "rb"))
    jump_mat3 = pickle.load(open("c_p_neigh/pickle_jump_mat_proc.p", "rb"))
    #for i in range(len(jump_mat)):
    #    mat1 = np.copy(jump_mat[i].todense())
    #    np.fill_diagonal(mat1,np.array(jump_mat[i].todense().shape[0]*[0.6]))
    assert jump_mat[0].todense() == pytest.approx(ref_mat[0].todense(), 0.01)


if __name__ == "__main__":
    test_create_jump_mat_14()
