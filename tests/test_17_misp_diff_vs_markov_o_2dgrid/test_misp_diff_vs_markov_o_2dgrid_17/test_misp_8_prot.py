# coding: utf-8
#import glob
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit
from trajec_io import readwrite
from analysis import msd




def test_misp_2d():
    #os.chdir("/home/dressler/projects/chrisbase/tests/test1")
    #start_dir = os.getcwd()
    #os.chdir("test_misp_diff_vs_markov_o_2dgrid_17")
    #cur_dir = os.getcwd()
    #assert cur_dir[-2:]  == '17'
    #create_propagation_matrix o_400.xyz  pbc False False O
    os.chdir("single_prot")
#    files_to_be_deleted = glob.glob("*.npz")
#    for f in files_to_be_deleted:
#        os.remove(f)

    pbc = np.loadtxt("pbc")
    subprocess.call(['create_propagation_matrix', 'o_nine.xyz', 'pbc', 'False', 'nowrap', 'O'])
    subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '1000000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])

    coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "unwrap")
    readwrite.easy_write( coords[:,atoms == "H",:] ,  atoms[atoms == "H"], "lmc_final_unwrap.xyz")
    #readwrite.easy_write( coords[:,atoms == "H",:] ,  np.array(["H"]), "lmc_final_unwrap.xyz")

    coords =  readwrite.unwrap_trajectory(coords, pbc)
    tau, msd1 = msd.msd_for_unwrap(coords, atoms, "H", 1, 1, 100, 10)
    
    #np.savetxt(msd, "msd_hole.out")
    np.savetxt("msd_unwrap.out", msd1)
    
     #print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n 
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    popt, pcov = curve_fit(diff_coef, tau, msd1)
    print(popt, pcov)
    assert popt[0] == pytest.approx( 5.73E-2 , rel = 1E-2)
#    assert popt == pytest.approx(np.array([0.05831035, -0.00903418]),0.01 )



    os.chdir("..")
    #os.chdir(start_dir)

def test_misp_2d_direct_msd():
    #os.chdir("/home/dressler/projects/chrisbase/tests/test1")
    #start_dir = os.getcwd()
    #os.chdir("test_misp_diff_vs_markov_o_2dgrid_17")
    #cur_dir = os.getcwd()
    #assert cur_dir[-2:]  == '17'
    #create_propagation_matrix o_400.xyz  pbc False False O
    os.chdir("single_prot_direct_msd")
#    files_to_be_deleted = glob.glob("*.npz")
#    for f in files_to_be_deleted:
#        os.remove(f)

    pbc = np.loadtxt("pbc")
    subprocess.call(['create_propagation_matrix', 'o_nine.xyz', 'pbc', 'False', 'nowrap', 'O'])
#    subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '10000000', 'lmc.out', '100', '1', 'True', '--seed', '12341234', '--verbose'])
    subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '1', '1000', '100',  '1000000', 'lmc.out', '10', '1', 'True', '--verbose'])

     #print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n
    msd_output = np.loadtxt("msd_from_lmc.out")
    tau, msd1 = msd_output[:, 0], msd_output[:, 1]
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    popt, pcov = curve_fit(diff_coef, tau, msd1)
    print(popt, pcov)
    assert popt[0] == pytest.approx( 5.73E-2 , rel = 1E-2)
    os.chdir("..")

def test_misp_8prot_2d():
    #G>os.chdir("/home/dressler/projects/chrisbase/tests/test1")
    #start_dir = os.getcwd()
    #os.chdir("test_misp_diff_vs_markov_o_2dgrid_17")
    #cur_dir = os.getcwd()
    #assert cur_dir[-2:]  == '17'

    os.chdir("two_prot")
#    files_to_be_deleted = glob.glob("*.npz")
#    for f in files_to_be_deleted:
#        os.remove(f)

    subprocess.call(['create_propagation_matrix', 'o_nine.xyz', 'pbc', 'False', 'nowrap', 'O'])

    subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '8', '1000', '100',  '1000000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose', '--write_xyz'])
    #subprocess.call(['misp',  'o_nine.xyz',  'pbc', 'False', 'nowrap', 'O', '8', '1000', '100',  '100000', 'lmc.out', '10', '1', 'True', '--seed', '12341234', '--verbose'])
    #subprocess.call(['msd_from_markov', 'jump_mat_added_trace.txt', 'pbc', 'lattice.txt', '10', '1'])


    pbc = np.loadtxt("pbc")
    coords, atoms = readwrite.easy_read("lmc_final.xyz", pbc, False, "nowrap")
    hole_coord = coords[:, atoms == "O", :].sum(axis = 1) - coords[:, atoms == "H", :].sum(axis = 1)
    readwrite.easy_write(hole_coord[:, np.newaxis, :],  np.array(["H"]), "hole_coord.xyz")

    coords, atoms = readwrite.easy_read("hole_coord.xyz", pbc, False, "unwrap")
    readwrite.easy_write( coords ,  np.array(["H"]), "hole_coord_unwrap.xyz")

    coords =  readwrite.unwrap_trajectory(coords, pbc)
    tau, msd1 = msd.msd_for_unwrap(coords, np.array([atoms]), "H", 1, 1, 100, 10)
    #breakpoint()
    #np.savetxt(msd, "msd_hole.out")  
    np.savetxt("msd_hole.out", msd1)
  
     #print(msd)
    def diff_coef(t, D, n):
        return 6* D *t + n 
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    popt, pcov = curve_fit(diff_coef, tau, msd1)
    print(popt, pcov)

     #[0.05745485 0.00175143]
    assert popt[0] == pytest.approx( 5.73E-2 , rel = 1E-2)
    #assert popt == pytest.approx(np.array([0.05745485, 0.00175143]),0.01 )
    os.chdir("..")


    #print(msd)
    #def diff_coef(t, D, n):
    #    return 6* D *t + n 
    ##popt, pcov = curve_fit(diff_coef, msd[:,0], msd[:,1])
    #popt, pcov = curve_fit(diff_coef, tau, msd)
    #print(popt, pcov)
    #assert popt == pytest.approx(np.array([0.05762717, 0.00437636]),0.01 )
    #os.chdir(start_dir)

#test_markov_5()
#test_markov_8_proton_1_hole()
#test_msd_for_unwriap()
if __name__ == "__main__":
    test_misp_2d_direct_msd()
    #test_misp_8prot_2d()
    #test_misp_2d()
