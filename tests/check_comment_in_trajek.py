# coding: utf-8
import numpy as np
from trajec_io import readwrite

pbc_mat = np.loadtxt("comment_trajek/pbc")
path = "comment_trajek/comment_tra.xyz"
com = True  # remove center of mass movement?
unwrap = True  # wrap/unwrap trejectory?
coord, atom = readwrite.easy_read(path, pbc_mat, com, unwrap)
