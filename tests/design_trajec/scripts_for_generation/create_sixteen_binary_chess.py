# coding: utf-8
import numpy as np
from trajec_io import readwrite
import pickle
from test_dir import design_func

trajec, atom = design_func.o_sixteen_alter()
print(trajec.shape)
print(atom.shape)
    
#atom1 = ["O", "H", "H"]
#atom_final = atom1 * 3
##atom_final = np.array(atom_final)

#readwrite.easy_write(trajec, atom, "o_square.xyz")
readwrite.easy_write(trajec, atom, "o_n_sixteen_binary_chess.xyz")

#design_func.prob_o_fourhundred(trajec, atom)

#from tests import design_func 
#test = design_func.o_nine_prop_mat()
#xx =pickle.load( open( "pickle_jump_mat_o_nine.p", "rb" ) )
#
#for i in range(len(xx)):
#     print((xx[i].todense()  == test[i]).all())
#     #print(i)
#     ##print(xx[i].todense())
#     #print(test[i])
