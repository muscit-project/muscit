import subprocess

import numpy as np
import pytest
from trajec_io import readwrite


def test_easy_read():
    path = "trajec/li13si4_timestep_50fs.xyz"
    pbc_mat = np.loadtxt("trajec/pbc_li13si4")
    com = True  # remove center of mass movement?
    unwrap = True  # wrap/unwrap trejectory?
    auxiliary_file = path + "_" + str(com) + "_" + str(unwrap) + "_" + ".npz"
    list_files = subprocess.run(["rm", auxiliary_file])
    coord, atom = readwrite.easy_read(path, pbc_mat, com, unwrap)
    ref = np.load("ref_data_trajec/li13si4_timestep_50fs.xyz_True_True_.npz")
    assert ref["arr_0"] == pytest.approx(coord.astype("float64"), rel=0.001)
