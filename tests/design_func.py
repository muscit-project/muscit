import copy
import pickle
import subprocess
import sys

import numpy as np
from scipy.sparse import coo_matrix
from trajec_io import readwrite


def o_nine_prop_mat():
    pbc = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    coord, atom = o_nine()
    dist_mat = readwrite.pbc_dist2(coord[0], coord[0], pbc)
    dist_mat = np.linalg.norm(dist_mat, axis=2)
    print(dist_mat)
    ar1 = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            if dist_mat[i, j] < 0.01:
                ar1[i, j] = 0.6
            if dist_mat[i, j] == 1.0:
                ar1[i, j] = 0.1
    print(ar1)
    print(ar1.sum(axis=0))
    print(ar1.sum(axis=1))
    jump_mat = []
    final = []
    for i in range(10):
        jump_mat.append(ar1)
        ind = np.where(ar1 > 0.0001)
        row = ind[0]
        col = ind[1]
        data = ar1[ind]
        final.append(coo_matrix((data, (row, col)), shape=(ar1.shape[0], ar1.shape[0])))

    pickle.dump(final, open("pickle_jump_mat_o_nine.p", "wb"))
    jump_mat = np.array(jump_mat)
    return jump_mat



def o_multiple_dimers_fram1():
    coord_raw = np.array(
        [
            [0, 2.5, 0.01],
            [4.0, 2.5, 0.01],
            [8.0, 2.5, 0.01],
            [0, 0, 0],
            [0, 2.5, 0],
            [4.0, 0, 0],
            [4.0, 2.5, 0],
            [8, 0, 0],
            [8.0, 2.5, 0],
            [12.0, 0.0, 0],
            [12.0, 2.8, 0],
            [16.0, 0.0, 0],
            [16.0, 2.8, 0],
            [20.0, 0.0, 0],
            [20.0, 2.8, 0]
        ]
    )
    atom1 = ["H", "H", "H", "O", "O", "N", "N", "O", "N" ,  "O", "O", "N", "N", "O", "N" ]
    trajec1 = []
    for i in range(40):
        trajec1.append(coord_raw)
    trajec1 = np.array(trajec1)
    for i in range(40):
        if i % 4 == 1:
            trajec1[i,0,:] -= np.array([0, 2.5, 0])
        if i % 4 == 2:
            trajec1[i,0,:] -= np.array([0, 2.5, 0])
        if i % 10  <  3 or i % 10 > 6:
            trajec1[i,1,:] -= np.array([0, 2.5, 0])
        #if i % 10 == 6:
        #    trajec1[i,1,:] += np.array([0, 2.5, 0])
        if i % 5 == 1 or i % 5 == 2:
            trajec1[i,2,:] -= np.array([0, 2.5, 0])
        #if i % 5 == 3:
        #    trajec1[i,2,:] += np.array([0, 2.5, 0])
    return trajec1, np.array(atom1)



def o_nine():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [0, 1.0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [0, 2.0, 0],
            [1.0, 1.0, 0],
            [2.0, 1.0, 0],
            [2.0, 2.0, 0],
            [1.0, 2.0, 0],
        ]
    )
    atom1 = ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)

def o_double():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
        ]
    )
    atom1 = ["O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0]
        ]
    )
    atom1 = ["O", "O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_line_five():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [3.0, 0, 0],
            [4.0, 0, 0]
        ]
    )
    atom1 = ["O", "O", "O", "O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)



def o_line_plus_h():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [0,0,0]
        ]
    )
    atom1 = ["O", "O", "O", "H"]
    trajec1 = []
    coord_raw = np.array(coord_raw)
    for i in range(300):
        x_val = i % 3
        #print(x_val)
        #coord_raw[3,:] = np.array([x_val,0,0])
        coord_raw[3,0] = x_val
        #print(x_val, coord_raw[3,0])
        #print(coord_raw)
        #print()
        trajec1.append(np.copy(coord_raw))
        #print(trajec1[i])
    #print(trajec1)
    return np.array(trajec1), np.array(atom1)



def o_4_non_eq_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [3.0, 0, 0],
            [4.0, 0, 0],
            [7.0, 0, 0]
        ]
    )
    atom1 = ["O", "O", "O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)

def o_4_p_2_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [3.0, 0, 0],
            [0.9, 0, 0],
            [2.1, 0, 0]
        ]
    )
    atom1 = ["O", "O", "O", "O","P","P"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_4_p_4_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [3.0, 0, 0],
            [0.9, 0, 0],
            [2.1, 0, 0],
            [-0.1, 0, 0],
            [3.1, 0, 0],
        ]
    )
    atom1 = ["O", "O", "O", "O","P","P","P","P"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_4_p_c_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [3.0, 0, 0],
            [0.9, 0, 0],
            [2.1, 0, 0]
        ]
    )
    atom1 = ["O", "O", "O", "O","P","C"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_3_n_p_c_line():
    coord_raw = np.array(
        [
            [0, 0, 0],
            [1.0, 0, 0],
            [2.0, 0, 0],
            [3.0, 0, 0],
            [0.9, 0, 0],
            [2.1, 0, 0]
        ]
    )
    atom1 = ["O", "O", "N", "O","P","C"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_fourhundred():
    coord_raw=[]
    for i in range(20):
        for j in range(20):
            coord_raw.append([i,j,0.0])
    coord_raw = np.array(coord_raw)
    atom1 = ["O"] * 400
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_hundred():
    coord_raw=[]
    for i in range(100):
        #for j in range(20):
            coord_raw.append([i,0.0,0.0])
    coord_raw = np.array(coord_raw)
    atom1 = ["O"] * 100
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)




def o_twentyfive():
    coord_raw=[]
    for i in range(5):
        for j in range(5):
            coord_raw.append([i,j,0.0])
    coord_raw = np.array(coord_raw)
    atom1 = ["O"] * 25
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)




def o_sixteen_pure():
    coord_raw=[]
    atom1 = []
    for i in range(4):
        for j in range(4):
            coord_raw.append([i,j,0.0])
            if i % 2 == 0:
                atom1.append("O")
            else:
                atom1.append("O")
    coord_raw = np.array(coord_raw)
    #atom1 = ["O"] * 400
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)

def o_sixteen():
    coord_raw=[]
    atom1 = []
    for i in range(4):
        for j in range(4):
            coord_raw.append([i,j,0.0])
            if i % 2 == 0: 
                atom1.append("O")
            else:
                atom1.append("N")
    coord_raw = np.array(coord_raw)
    #atom1 = ["O"] * 400
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_sixteen_alter():
    coord_raw=[]
    atom1 = []
    for i in range(4):
        for j in range(4):
            coord_raw.append([i,j,0.0])
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                atom1.append("O")
            else:
                atom1.append("N")
    coord_raw = np.array(coord_raw)
    #atom1 = ["O"] * 400
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)



def dist_to_prob(dist:float):
    if dist < 0.01:
        prob = 0.6
    elif (dist > 0.9) and (dist < 1.1):
        prob = 0.1
    else:
        prob = 0.0
    return prob

#np.ndarray
def prob_o_fourhundred(trajec:np.array, atom):
    prob_ar = np.zeros((trajec.shape[0], trajec.shape[1], trajec.shape[1]))
    pbc = np.eye(3)*20
    print(pbc)
    for k in range(trajec.shape[0]):
        dist_ar = readwrite.pbc_dist2(trajec[k], trajec[k], pbc)
        for i in range(trajec.shape[1]):
            for j in range(trajec.shape[1]):
                prob_ar[k,i,j] = dist_to_prob(np.linalg.norm(dist_ar[i,j]))
    final = []
    for i in range(10):
        #jump_mat.append(ar1)
        ind = np.where(prob_ar[i] > 0.0001)
        row = ind[0]
        col = ind[1]
        data = prob_ar[i][ind]
        final.append(coo_matrix((data, (row, col)), shape=(prob_ar.shape[1], prob_ar.shape[1])))
    pickle.dump(final, open("pickle_jump_mat_o_400.p", "wb"))
    return prob_ar



def o_square_prop_mat():
    ar1 = np.array(
        [
            [0.6, 0.2, 0, 0.2],
            [0.2, 0.6, 0.2, 0.0],
            [0, 0.2, 0.6, 0.2],
            [0.2, 0.0, 0.2, 0.6],
        ]
    )
    jump_mat = []
    final = []
    for i in range(10):
        jump_mat.append(ar1)
        ind = np.where(ar1 > 0.1)
        row = ind[0]
        col = ind[1]
        data = ar1[ind]
        final.append(coo_matrix((data, (row, col)), shape=(ar1.shape[0], ar1.shape[0])))

    pickle.dump(final, open("pickle_jump_mat_o_square.p", "wb"))
    jump_mat = np.array(jump_mat)
    return jump_mat


def o_square():
    coord_raw = np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0, 1.0, 0]])
    atom1 = ["O", "O", "O", "O"]
    trajec1 = []
    for i in range(10):
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_square_multiple_iterations_four_prot():
    coord_raw_orig = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [10.0, 0, 0],
            [10.0, 0, 1],
            [10.0, 0, -1],
            [10.0, 10.0, 0],
            [10.0, 10.0, 1],
            [10.0, 10.0, -1],
            [0, 10.0, 0],
            [0, 10.0, 1],
            [0, 10.0, -1],
        ]
    )
    atom1 = ["O", "H", "C", "O", "H", "C", "O", "H", "C", "O", "H", "C"]
    trajec1 = []
    for i in range(11):
        coord_raw = copy.copy(coord_raw_orig)
        coord_raw[0:3] += np.array([1.0, 0, 0]) * i
        coord_raw[3:6] += np.array([0.0, 1.0, 0]) * i
        coord_raw[6:9] += np.array([-1.0, 0, 0]) * i
        coord_raw[9:12] += np.array([0.0, -1.0, 0]) * i
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


def o_square_multiple_iterations_two_prot():
    coord_raw_orig = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -2],
            [10.0, 0, 0],
            [10.0, 0, -2],
            [10.0, 10.0, 0],
            [10.0, 10.0, 1],
            [10.0, 10.0, -2],
            [0, 10.0, 0],
            [0, 10.0, -2],
        ]
    )
    atom1 = ["O", "H", "C", "O", "C", "O", "H", "C", "O", "C"]
    trajec1 = []
    for i in range(11):
        coord_raw = copy.copy(coord_raw_orig)
        coord_raw[0:3] += np.array([1.0, 0, 0]) * i
        coord_raw[3:5] += np.array([0.0, 1.0, 0]) * i
        coord_raw[5:8] += np.array([-1.0, 0, 0]) * i
        coord_raw[8:11] += np.array([0.0, -1.0, 0]) * i
        trajec1.append(coord_raw)
    return np.array(trajec1), np.array(atom1)


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
        else:
            b = h2o2
        c = h2o3 + np.array([3.0 * i, 0, 0])
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
