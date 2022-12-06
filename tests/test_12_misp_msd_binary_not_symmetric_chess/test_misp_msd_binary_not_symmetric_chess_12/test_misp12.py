# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit
from trajec_io import readwrite


def test_misp_12():
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '12'
    subprocess.call(['create_propagation_matrix', 'o_n_sixteen_binary_chess.xyz', 'pbc', 'False', 'nowrap', 'O', 'N', '--asymmetric'])
    subprocess.call(['misp', 'o_n_sixteen_binary_chess.xyz', 'pbc', 'False', 'nowrap', 'O', 'N', '1', '1000', '100', '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])
     
    pbc_mat = np.loadtxt("pbc")
    coord, atom = readwrite.easy_read("lmc_final.xyz", pbc_mat, False, False)
    coord_n_o = coord[:,np.isin(atom, ["N","O"] ),:]
    atom_n_o = atom[np.isin(atom, ["N","O"] )]
    coord_h = coord[:,np.isin(atom, ["H"] ),:]
    atom_h = atom[np.isin(atom, ["H"] )]
    acceptor_list = ["N", "O"]
    count= {}
    for i in acceptor_list:
        count[i] = 0
    
    for i in range(coord_h.shape[0]):
        for j in range(coord_h.shape[1]):
            tmp = np.linalg.norm(readwrite.pbc_dist(coord_h[i,j,:], coord_n_o[i,:,:], pbc_mat), axis=1)
            atom_type = atom_n_o[tmp < 0.00001]
            #if atom_type == "N":
            for k in acceptor_list:
                if k == atom_type:
                    count[k] += 1
    
    print(count)
    for key in count.keys():
        for jey in count.keys():
            print(f"{key} durch {jey}: ", count[key]/count[jey])

    #assert (np.array([count['N'], count['O']]) == np.array([10332,89668])).all()
    #assert count['N']/count['O'] == pytest.approx(0.11522505241557747, 0.001 )
    assert (np.array([count['N'], count['O']]) == np.array([10448, 89552])).all()
    assert count['N']/count['O'] == pytest.approx(0.1166696444523852, 0.001 )




if __name__ == "__main__":
    test_misp_12()
