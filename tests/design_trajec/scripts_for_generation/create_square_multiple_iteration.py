# coding: utf-8
import numpy as np
from trajec_io import readwrite
import pickle
from tests import design_func

trajec, atom = design_func.o_square_multiple_iterations_two_prot()

#atom1 = ["O", "H", "H"]
#atom_final = atom1 * 3
##atom_final = np.array(atom_final)

readwrite.easy_write(trajec, atom, "o_square_mutliple_iterations.xyz")


#from tests import design_func 
#test = design_func.o_square_prop_mat()
#xx =pickle.load( open( "pickle_jump_mat_o_square_mutliple_iterations.p", "rb" ) )

#for i in range(len(xx)):
#    print((xx[i].todense()  == test[i]).all())
