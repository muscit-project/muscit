# coding: utf-8
import numpy as np
from trajec_io import readwrite
import pickle
from test_dir import design_func

trajec, atom = design_func.o_double()

#atom1 = ["O", "H", "H"]
#atom_final = atom1 * 3
##atom_final = np.array(atom_final)

#readwrite.easy_write(trajec, atom, "o_square.xyz")
readwrite.easy_write(trajec, atom, "o_double.xyz")
