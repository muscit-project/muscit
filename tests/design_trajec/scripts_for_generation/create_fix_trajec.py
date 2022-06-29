# coding: utf-8
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(10):
    eins = h2o1.tolist()
    zwei = h2o2.tolist()
    drei = h2o3.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "fix_trajec.xyz")
