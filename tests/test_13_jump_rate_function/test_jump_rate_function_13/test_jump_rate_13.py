# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest
from scipy.optimize import curve_fit
from trajec_io import readwrite


def test_jumprate_13():
    cur_dir = os.getcwd()
    assert cur_dir[-2:]  == '13'
    subprocess.call(['evaluate_jumps_on_grid', 'multiple_dimers.xyz',  'pbc', 'True', 'nowrap' ,'1', 'H', 'O',  'N',  '--dynamic'])
    #subprocess.call(['adv_calc_jump_rate',  'multiple_dimers.xyz',  'pbc',  'True',  'nowrap', 'O',  'N', '--nonsymmetric',  '--neighfile', 'multiple_dimers.xyz_True_nowrap_speed_1.npz.npy'])
    subprocess.call(['adv_calc_jump_rate',  'multiple_dimers.xyz',  'pbc',  'True',  'nowrap', 'O',  'N', '--neighfile', 'jumping_ion_neighbors.npy'])

    OO = np.loadtxt("jumpprob_histogram_OO")
    NN = np.loadtxt("jumpprob_histogram_NN")
    ON = np.loadtxt("jumpprob_histogram_ON")
    NO = np.loadtxt("jumpprob_histogram_NO")
    assert 0.5 == OO[50][1]
    assert 0.2 == NN[50][1]
    assert 0.333333 == NO[50][1]
    assert 0.5 == ON[50][1]

if __name__ == "main":
    test_jumprate_13()
