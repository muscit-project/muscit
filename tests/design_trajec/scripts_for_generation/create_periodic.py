# coding: utf-8
import numpy as np
from trajec_io import readwrite

from tests import design_func

trajec1 = design_func.periodic()

atom1 = ["O", "H", "H"]
atom_final = atom1 * 3
atom_final = np.array(atom_final)

readwrite.easy_write(trajec1, atom_final, "periodic.xyz")



#h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
#h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
#atom1 = ["O", "H", "H"]
#h2o2 = h2o2 + np.array([6.0, 0, 0])
#h2o3 = h2o1 + np.array([7.0, 2.0, 0])
#trajec1 = []
#for i in range(10):
#    a = h2o1
#    if i%2 == 0:
#        b = h2o2 + np.array([3.0, 0, 0 ])
#        c = h2o3 + np.array([3.0, 0, 0 ]) 
#    else:
#        b = h2o2
#        c = h2o3
#    eins = a.tolist()
#    zwei = b.tolist()
#    drei = c.tolist()
#    tmp = eins +  zwei + drei 
#    trajec1.append(tmp)
#
#trajec1 = np.array(trajec1)
#atom_final = atom1 * 3
#atom_final = np.array(atom_final)
#readwrite.easy_write(trajec1, atom_final, "periodic.xyz")
#
#
#
#import design_func
#
#
#
#
#def create_periodic():
#    h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
#    h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
#    atom1 = ["O", "H", "H"]
#    h2o2 = h2o2 + np.array([6.0, 0, 0])
#    h2o3 = h2o1 + np.array([7.0, 2.0, 0])
#    trajec1 = []
#    for i in range(10):
#        a = h2o1
#        if i%2 == 0:
#            b = h2o2 + np.array([3.0, 0, 0 ])
#            c = h2o3 + np.array([3.0, 0, 0 ])
#        else:
#            b = h2o2
#            c = h2o3
#        eins = a.tolist()
#        zwei = b.tolist()
#        drei = c.tolist()
#        tmp = eins +  zwei + drei
#        trajec1.append(tmp)
#    
#    return  np.array(trajec1)
