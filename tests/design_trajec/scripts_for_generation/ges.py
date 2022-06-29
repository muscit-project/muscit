# coding: utf-8
import numpy as np
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
h2o1
h2o2
atom1 = ["O", "H", "H"]
h2o1[2,:]
h2o1[1,:]
h2o1[0,:]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o2
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
h2o3
get_ipython().run_line_magic('ls', '')
trajec1 = []
trajec1 = [ ]
for i in range(10):
    eins = h2o1.tolist()
    zwei = h2o2.tolist()
    drei = h2o3.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)
    
tmp[0]
h2o1.tolist() 
eins + zwei
eins + zwei + drei
trajec1
trajec1[0]
trajec1 = np.array(trajec1)
atom1
atom_final = atom1 * 3
atom_final
atom_final = np.array(atom_final)
import chrisbase.trajec_io.readwrite
import trajec_io.readwrite
readwrite.easy_write(trajec1, atom_final, "fix_trajec")
from chrisbase.trajec_io import readwrite
from trajec_io import readwrite
get_ipython().run_line_magic('ls', '')
readwrite.easy_write(trajec1, atom_final, "fix_trajec")
readwrite.easy_write(trajec1, atom_final, "fix_trajec.xyz")
get_ipython().run_line_magic('save', '  "creare_fix_trajec.py" 1-36')
get_ipython().run_line_magic('save', '  "create_fix_trajec.py" 1-36')
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
    
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "fix_trajec.xyz")
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
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(10):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        c = h2o3
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic.xyz")
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(10):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        c = h2o3
    if (i+1)%3 = 0:
        c = c + np.array([20.0, 0, 0 ])
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic_pbc_jumps.xyz")
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(10):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        c = h2o3
    if (i+1)%3 == 0:
        c = c + np.array([20.0, 0, 0 ])
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic_pbc_jumps.xyz")
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(20):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        #c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        #c = h2o3
    c = h2o3 + np.array([3.0 * i, 0, 0 ])    
    #if (i+1)%3 == 0:
        #c = c + np.array([20.0, 0, 0 ])
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic_long_diff.xyz")
4 % -20
0&-10
0%-10
3 % 5
5 & 10
5 & 12
5 % 12
5 % 10
10  % 5
12  % 5
0  % -20
-10  % -20
-4  % -20
3  % -20
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(20):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        #c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        #c = h2o3
    c = h2o3 + np.array([3.0 * i % 20 , 0, 0 ])    
    #if (i+1)%3 == 0:
        #c = c + np.array([20.0, 0, 0 ])
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic_long_diff_wrap.xyz")
np.array([20,30,34])%10
import numpy as np
from trajec_io import readwrite
h2o1 = np.array([[0,0,0],[1.0, 0,0],[0,0, 1.0]])
h2o2 = np.array([[0,0,0],[1.0, 0,0],[0,1.0, 0]])
atom1 = ["O", "H", "H"]
h2o2 = h2o2 + np.array([6.0, 0, 0])
h2o3 = h2o1 + np.array([7.0, 2.0, 0])
trajec1 = []
for i in range(20):
    a = h2o1
    if i%2 == 0:
        b = h2o2 + np.array([3.0, 0, 0 ])
        #c = h2o3 + np.array([3.0, 0, 0 ]) 
    else:
        b = h2o2
        #c = h2o3
    c = h2o3 + np.array([3.0 * i % 20 , 0, 0 ])    
    c[:,0] = c[:,0] % 20
    #if (i+1)%3 == 0:
        #c = c + np.array([20.0, 0, 0 ])
    eins = a.tolist()
    zwei = b.tolist()
    drei = c.tolist()
    tmp = eins +  zwei + drei 
    trajec1.append(tmp)

trajec1 = np.array(trajec1)
atom_final = atom1 * 3
atom_final = np.array(atom_final)
readwrite.easy_write(trajec1, atom_final, "periodic_long_diff_wrap.xyz")
get_ipython().run_line_magic('save', '  "create_periodic_long_diff_wrap.py" 60')
get_ipython().run_line_magic('save', '  "create_periodic_long_diff.py" 43')
get_ipython().run_line_magic('save', '  "create_periodic_pbc_jumps.py" 42')
get_ipython().run_line_magic('save', '  "create_periodic.py" 40')
