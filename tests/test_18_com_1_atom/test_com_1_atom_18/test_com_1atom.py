import numpy as np
from trajec_io import readwrite
import os
def test_com1():
    pbc = np.loadtxt("pbc")
    try:
       os.remove("hole_coord.xyz.npz")
    except OSError:
        pass

    coords, atoms = readwrite.easy_read("hole_coord.xyz", pbc, False, "nowrap")
    #readwrite.remove_com(coords, atoms) 
    #possible solution:
    readwrite.remove_com(coords, np.array([atoms]))    

if __name__ == "__main__":  
    test_com1()    
