import subprocess
import sys

import numpy as np
from trajec_io import readwrite

import design_func


def periodic():
    h2o1 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    h2o2 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    atom1 = ["O", "H", "H"]
    h2o2 = h2o2 + np.array([6.0, 0, 0])
    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
    trajec1 = []
    for i in range(10):
        a = h2o1
        if i % 2 == 0:
            b = h2o2 + np.array([3.0, 0, 0])
            c = h2o3 + np.array([3.0, 0, 0])
        else:
            b = h2o2
            c = h2o3
        eins = a.tolist()
        zwei = b.tolist()
        drei = c.tolist()
        tmp = eins + zwei + drei
        trajec1.append(tmp)

    return np.array(trajec1)


def fix_trajec():
    h2o1 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    h2o2 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    atom1 = ["O", "H", "H"]
    h2o2 = h2o2 + np.array([6.0, 0, 0])
    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
    trajec1 = []
    for i in range(10):
        eins = h2o1.tolist()
        zwei = h2o2.tolist()
        drei = h2o3.tolist()
        tmp = eins + zwei + drei
        trajec1.append(tmp)

    return np.array(trajec1)


def periodic_long_diff():
    h2o1 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    h2o2 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    atom1 = ["O", "H", "H"]
    h2o2 = h2o2 + np.array([6.0, 0, 0])
    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
    trajec1 = []
    for i in range(20):
        a = h2o1
        if i % 2 == 0:
            b = h2o2 + np.array([3.0, 0, 0])
            # c = h2o3 + np.array([3.0, 0, 0 ])
        else:
            b = h2o2
            # c = h2o3
        c = h2o3 + np.array([3.0 * i, 0, 0])
        # if (i+1)%3 == 0:
        # c = c + np.array([20.0, 0, 0 ])
        eins = a.tolist()
        zwei = b.tolist()
        drei = c.tolist()
        tmp = eins + zwei + drei
        trajec1.append(tmp)

    return np.array(trajec1)


def periodic_long_diff_wrap():
    h2o1 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    h2o2 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    atom1 = ["O", "H", "H"]
    h2o2 = h2o2 + np.array([6.0, 0, 0])
    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
    trajec1 = []
    for i in range(20):
        a = h2o1
        if i % 2 == 0:
            b = h2o2 + np.array([3.0, 0, 0])
        else:
            b = h2o2
        c = h2o3 + np.array([3.0 * i % 20, 0, 0])
        c[:, 0] = c[:, 0] % 20
        eins = a.tolist()
        zwei = b.tolist()
        drei = c.tolist()
        tmp = eins + zwei + drei
        trajec1.append(tmp)
    return np.array(trajec1)


def periodic_pbc_jumps():
    h2o1 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    h2o2 = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    atom1 = ["O", "H", "H"]
    h2o2 = h2o2 + np.array([6.0, 0, 0])
    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
    trajec1 = []
    for i in range(10):
        a = h2o1
        if i % 2 == 0:
            b = h2o2 + np.array([3.0, 0, 0])
            c = h2o3 + np.array([3.0, 0, 0])
        else:
            b = h2o2
            c = h2o3
        if (i + 1) % 3 == 0:
            c = c + np.array([20.0, 0, 0])
        eins = a.tolist()
        zwei = b.tolist()
        drei = c.tolist()
        tmp = eins + zwei + drei
        trajec1.append(tmp)
    return np.array(trajec1)


def test_design_read():
    path_list = [
        "fix_trajec.xyz",
        "periodic_long_diff_wrap.xyz",
        "periodic_long_diff.xyz",
        "periodic_pbc_jumps.xyz",
        "periodic.xyz",
    ]
    pbc_mat = np.loadtxt("design_trajec/pbc")
    com = False  # remove center of mass movement?
    unwrap = False  # wrap/unwrap trejectory?
    coord_dict = {}

    for path in path_list:
        auxiliary_file = (
            "design_trajec/" + path + "_" + str(com) + "_" + str(unwrap) + "_" + ".npz"
        )
        list_files = subprocess.run(["rm", auxiliary_file])
        coord_dict[path[:-4]], atom = readwrite.easy_read(
            "design_trajec/" + path, pbc_mat, com, unwrap
        )
    # print(coord_dict.keys())

    for key in coord_dict.keys():
        auxiliary_file = "design_trajec/rewrite" + key + ".xyz"
        readwrite.easy_write(
            coord_dict[key], atom, "design_trajec/rewrite" + key + ".xyz"
        )

    for key in coord_dict.keys():
        func1 = getattr(design_func, key)
        tmp1 = func1()
        assert (tmp1 == coord_dict[key]).all()
