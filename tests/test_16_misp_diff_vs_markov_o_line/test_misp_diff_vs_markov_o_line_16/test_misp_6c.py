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




def test_markov_3_o():
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '16'
    #create_propagation_matrix o_400.xyz  pbc False False O
    os.chdir("single_prot")
    pbc = np.loadtxt("pbc")
    subprocess.call(['create_propagation_matrix', 'o_line.xyz', 'pbc', 'False', 'nowrap', 'O'])
    subprocess.call(['misp',  'o_line.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])

    #coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "unwrap")
    coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "unwrap")
    readwrite.easy_write( coords[:,atoms == "H",:] ,  np.array(["H"]), "lmc_final_unwrap.xyz")

    coords =  readwrite.unwrap_trajectory(coords, pbc)
    tau, msd1 = msd.msd_for_unwrap(coords, atoms, "H", 1, 1, 100, 100)
    np.savetxt("msd_unwrap.out", msd1)

    def diff_coef(t, D, n):
        return 6* D *t + n 
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    popt, pcov = curve_fit(diff_coef, tau[:10], msd1[:10])
    print(popt, pcov)
    #assert popt == pytest.approx(np.array([0.03177856, -0.00096471]),  abs=0.001 )
    assert popt == pytest.approx(np.array([0.03194621, -0.00217708]), 0.001)
    os.chdir("..")

def test_markov_3_o_v2():
    #G>os.chdir("/home/dressler/projects/chrisbase/tests/test1")
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '16'

    os.chdir("two_prot")

    subprocess.call(['create_propagation_matrix', 'o_line.xyz', 'pbc', 'False', 'nowrap', 'O'])

    subprocess.call(['misp',  'o_line.xyz',  'pbc', 'False', 'nowrap', 'O', '2', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])
    #subprocess.call(['misp',  'o_line.xyz',  'pbc', 'False', 'nowrap', 'O', '2', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose'])
    #subprocess.call(['msd_from_markov', 'jump_mat_added_trace.txt', 'pbc', 'lattice.txt', '10', '1'])


    pbc = np.loadtxt("pbc")
    coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "nowrap")
    #coords, atoms = readwrite.easy_read("lmc.out", pbc, False, "nowrap")
    hole_coord = coords[:, atoms == "O", :].sum(axis = 1) - coords[:, atoms == "H", :].sum(axis = 1)
    readwrite.easy_write(hole_coord[:, np.newaxis, :],  np.array(["H"]), "hole_coord.xyz")
    
    special_coord = np.zeros((coords.shape[0], 4, coords.shape[2]))
    special_coord[:,:3,:] = coords[:, atoms == "O", :]
    special_coord[:,3,:] =  hole_coord
    special_atom = np.array(["O", "O","O", "H"])
    readwrite.easy_write(special_coord, special_atom,  "special_coord.xyz")



    coords, atoms = readwrite.easy_read("hole_coord.xyz", pbc, False, "unwrap")
    readwrite.easy_write( coords ,  np.array(["H"]), "hole_coord_unwrap.xyz")

    special_coord =  readwrite.unwrap_trajectory(special_coord, pbc)
   
    coords =  readwrite.unwrap_trajectory(coords, pbc)
    print("first ", coords.shape, atoms.shape)
    #tau, msd1 = msd.msd_for_unwrap(np.array([coords]), atoms, "H", 1, 1, 100, 100)
    #tau, msd1 = msd.msd_for_unwrap(coords, atoms, "H", 1, 1, 100, 100)
    tau, msd1 = msd.msd_for_unwrap(coords, np.array([atoms]), "H", 1, 1, 100, 100)
    #tau, msd1 = msd.msd_for_unwrap(special_coord, special_atom, "H", 1, 1, 100, 100)
    print("second ", special_coord.shape, special_atom.shape)
    #breakpoint()
    #np.savetxt(msd, "msd_hole.out")  
    np.savetxt("msd_hole.out", msd1)
  
     #print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n 
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
#    popt, pcov = curve_fit(diff_coef, tau, msd1)
    popt, pcov = curve_fit(diff_coef, tau[:10], msd1[:10])

    print(popt, pcov)
    #assert popt == pytest.approx(np.array([0.03177856, -0.00096471]), abs=0.001 )
    assert popt == pytest.approx(np.array([0.03177856, -0.00096471]), 0.001 )
    #assert popt == pytest.approx(np.array([0.03194621, -0.00217708]), 0.001)
    #assert popt == pytest.approx(np.array([0.05762717, 0.00437636]),0.01 )
    os.chdir("..")

def check_for_hole_creation():
    os.chdir("two_prot")
    pbc = np.loadtxt("pbc")
    coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "nowrap")
    h_coords = coords[:,  atoms == "H", :]
    o_coords = coords[:,  atoms == "O", :]
    dist = readwrite.pbc_dist2(h_coords, o_coords, pbc)
    dist = np.linalg.norm(dist, axis = -1)
    foo = np.count_nonzero(dist < 0.1, axis = 1)
    new_coord = []
    for i in range(foo.shape[0]):
        new_coord.append(o_coords[i,foo[i] < 0.1,:])
    readwrite.easy_write(np.array(new_coord), np.array(["H"]), "alternative_hole_trajec.xyz")

    coords2, atoms2 = readwrite.easy_read("hole_coord.xyz", pbc, False, "nowrap")
    assert (coords2 == new_coord).all()
    print("hole trajectory was correctly  created ")
    os.chdir("..")

#test_markov_5()
#test_markov_8_proton_1_hole()
#test_msd_for_unwrap()

if __name__ == "__main__":
    test_markov_3_o()
    test_markov_3_o_v2()
    check_for_hole_creation()
