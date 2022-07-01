# coding: utf-8
import numpy as np
from trajec_io import readwrite
path = "trajec/li13si4_timestep_50fs.xyz"
pbc_mat = np.loadtxt("trajec/pbc_li13si4")
com = True #remove center of mass movement?
unwrap = True #wrap/unwrap trejectory?
coord, atom = readwrite.easy_read(path, pbc_mat, com, unwrap)
readwrite.get_com(coord[0,:,:], atom, pbc_mat) #calculate center of mass of atoms from first frame of the trajectory
coord_li = coord[:, atom == "Li", :]
readwrite.pbc_dist(coord_li[2000,42,:], coord_li[0,42,:], pbc_mat)
coord_li[2000,42,:] - coord_li[0,42,:]
coord_li[2000,42,:] - coord_li[0,42,:] #calculate simple distance between coordinates of the 42-th Li atom at the t=0  and t=2000
readwrite.pbc_dist(coord_li[2000,42,:], coord_li[0,42,:], pbc_mat)#calculate distance between two points with respect to the minimum image convention (PBC)
