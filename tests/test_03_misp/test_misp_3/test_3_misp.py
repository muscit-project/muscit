# coding: utf-8
import pickle
import numpy as np
import subprocess
from numpy.testing import assert_array_almost_equal
import os
import pytest

def test_3_jump_mat_nine():
    cur_dir = os.getcwd()
    assert cur_dir[-1:]  == '3'
    subprocess.call(['misp', '../data_o_nine/o_nine.xyz', '../data_o_nine/cell', 'False', 'nowrap', 'O', '1', '1', '10', '10000', 'lmc.out', '100', '1', 'True'])
    overall_jumps = np.loadtxt("overall_number_of_jumps")
    assert (overall_jumps > 3200) and (overall_jumps < 3600)

if __name__ == "main":
    test_3_jump_mat_nine()
