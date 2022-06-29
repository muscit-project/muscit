import subprocess
import sys

import numpy as np
from trajec_io import readwrite

import design_func


def test_design_wrap_pbc():
    path_list = [
        "fix_trajec.xyz",
        "periodic_long_diff_wrap.xyz",
        "periodic_long_diff.xyz",
        "periodic_pbc_jumps.xyz",
        "periodic.xyz",
    ]
    pbc_mat = np.loadtxt("design_trajec/pbc")
    # com = False #remove center of mass movement?
    # unwrap = False #wrap/unwrap trejectory?
    list1 = [True, False]
    for com in list1:
        for unwrap in list1:
            coord_dict = {}
            for path in path_list:
                auxiliary_file = (
                    "design_trajec/"
                    + path
                    + "_"
                    + str(com)
                    + "_"
                    + str(unwrap)
                    + "_"
                    + ".npz"
                )
                list_files = subprocess.run(["rm", auxiliary_file])
                coord_dict[path[:-4]], atom = readwrite.easy_read(
                    "design_trajec/" + path, pbc_mat, com, unwrap
                )
            for key in coord_dict.keys():
                auxiliary_file = (
                    "design_trajec/rewrite"
                    + key
                    + "_"
                    + str(com)
                    + "_"
                    + str(unwrap)
                    + "_"
                    + ".xyz"
                )
                readwrite.easy_write(
                    coord_dict[key],
                    atom,
                    "design_trajec/rewrite"
                    + key
                    + "_"
                    + str(com)
                    + "_"
                    + str(unwrap)
                    + "_"
                    + ".xyz",
                )

            if (not com) and unwrap and (key == "periodic_long_diff_wrap"):
                print(com, unwrap, key)
                func1 = getattr(design_func, "periodic_long_diff")
                tmp1 = func1()
                assert (tmp1 == coord_dict[key]).all()

