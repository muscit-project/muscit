import logging
import os
import time
import pandas as pd
import itertools
import subprocess

import numpy as np
import numpy.typing as npt



# currently int mass, might become float at some point
def atomic_mass(chemicalsymbol : str) -> int:
    """Atomic mass of most common isotope of a given element.

    Parameters
    ----------
    chemicalsymbol : str
        
    Returns
    -------
    atomic mass : float

    Warning
    -------
    Only has the most important elements (as in those we used so far).

    Todo
    ----
    Add missing elements.
    """
    mass_dict = dict()
    # Atom masses in u
    mass_dict["Si"] = 28
    mass_dict["Li"] = 0
    mass_dict["O"] = 16
    mass_dict["H"] = 1
    mass_dict["C"] = 12
    mass_dict["N"] = 1
    mass_dict["Na"] = 23
    mass_dict["F"] = 1
    mass_dict["S"] = 32
    mass_dict["Cs"] = 133
    mass_dict["P"] = 31
    mass_dict["Pb"] = 207
    mass_dict["Sn"] = 119
    mass_dict["W"] = 184

    return mass_dict[chemicalsymbol]

class Trajectory:
    """Coordinates and periodic boundary conditions of a trajectory.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of n frames containing m atoms in a 3D-array of shape (n, m, 3). 1D and 2D arrays will be expanded to a 3D array of 1 frame.
    atom : np.ndarray
        Element labels in a 1D-array of shape (m,).
    pbc : np.ndarray or None
        Periodic boundary conditions as a 2D-array of shape (3, 3).

    Attributes
    ----------
    coords : np.ndarray
    atomlabels : np.ndarray
    pbc : np.ndarray
    inv_pbc : np.ndarray
        precomputed inverse of pbc via np.linalg.inv(pbc)


    Notes
    -----
    For legacy reasons, a Trajectory object `traj` can be iterated equal to `(traj.coords, traj.atom)`. As a result, they can be unpacked::

        coords, atom = traj
    """
    def __init__(self, coords : np.ndarray, atom : np.ndarray, pbc : np.ndarray) -> None:
        # Try to reshape coords and atom into a 3D and 1D array respectively
        while coords.ndim < 3:
            coords = np.expand_dims(coords, 0)
        atom = np.squeeze(atom)
        # Make sure that atom numbers are equal in both
        assert(coords.shape[2] == atom.shape[1])
        # Set attributes
        self.coords = coords
        self.atomlabels = atom
        self.frame_no, self.atom_no, _ = self.coords.shape
        # With periodic boundary conditions
        if (pbc is not None) and np.any(pbc):
            self.pbc = pbc
            self.inv_pbc = np.linalg.inv(pbc)
        # Without periodic boundary conditions
        else:
            # for non-periodic systems
            self.pbc = None
            self.inv_pbc = None
            # without pbc, calculate direct distance
            self.pbc_dist = lambda p1, p2 : p1 - p2
            self.pbc_dist2 = lambda p1, p2 : p1[..., np.newaxis, :] - p2[..., np.newaxis, :, :]
            def next_neighbor2_direct(group1, group2):
                dist_list = group1[..., np.newaxis, :] - group2[..., np.newaxis, :, :]
                dist_list = np.linalg.norm(dist_list, axis=-1)
                listx = np.argmin(dist_list, axis=-1)
                min_list = np.take_along_axis(dist_list, listx[..., None], axis = -1)[..., 0]
                return listx, min_list
            self.next_neighbor2 = next_neighbor2_direct

    # for backwards compability this allows one to assign two variables to Trajectory, similar to the way it was in the old easy_read version
    # coords, atom = easy_read(...) -> easy_read returns a Trajectory, which gets turned into its coords and atomlabels
    def __iter__(self):
        return iter([self.coords, self.atomlabels])

    # index trajectory using numpy-like slicing via trajectory[timesteps, atoms, dimensions]
    # returns trajectory containing only those coords and atomlabels specified
    def __getitem__(self, sl):
        """Slice out frames from the trajectory, just like an array.
        """
        if isinstance(sl, slice) or isinstance(sl, int) or isinstance(sl, list) or sl is Ellipsis or sl is None:
            # if slicing out multiple timesteps, then slice in coords and leave atomlabels unchanged
            subtraj = Trajectory(self.coords[sl], self.atomlabels, self.pbc)
        elif isinstance(sl, tuple):
            # if slicing happens along at least two axes, then apply whole slice to coords and only slice atomlabels along axis 1 of slice
            # make sure that coords and atomlabels retain their dimensions
            subtraj = Trajectory(self.coords[sl], self.atomlabels[sl[1]], self.pbc)
        else:
            subtraj = Trajectory(self.coords[sl], self.atomlabels, self.pbc)
        return subtraj

    def __repr__(self):
        return f"Trajectory( np.{repr(self.coords)}, {repr(self.atomlabels)}, np.{repr(self.pbc)} )"

    def __str__(self):
        atom_no = len(self.atomlabels)
        frame_no = self.coords.shape[0]
        atom_string = f"{atom_no} atoms" if atom_no != 1 else f"{atom_no} atom"
        frame_string = f"{frame_no} frames" if frame_no != 1 else f"{frame_no} frame"
        return "Trajectory of " + atom_string + " in " + frame_string

    # Wrapper for distance and neighbor functions
    def pbc_dist(self, point1, point2):
        """Calculates distance between two points in a periodic box according to the minimum-image convention.

        See Also
        --------
        pbc_dist
        """
        return pbc_dist(point1, point2, self.pbc, self.inv_pbc)
    def pbc_dist2(self, group1, group2):
        """Calculates all pair-wise distance between two lists of points in a periodic box according to the minimum-image convention.

            See Also
            --------
            pbc_dist2
        """
        return pbc_dist2(group1, group2, self.pbc, self.inv_pbc)
    def next_neighbor2(self, group1, group2):
        """Find closest neighbors to the first group of coordinates from among the second group.

        See Also
        --------
        next_neighbor2
        """
        return next_neighbor2(group1, group2, self.pbc, self.inv_pbc)

def start_logging():
    logging.basicConfig(filename='misp.log', level=logging.INFO)
    log_git_version()

def return_git_version():
    command = "git log -n 1 --format=%H%n%s%n%ad"
    commit_hash, commit_message, commit_date = subprocess.check_output(command.split(), cwd=os.path.dirname(os.path.abspath(__file__))).strip().split(b"\n")
    return commit_hash.decode(), commit_message.decode(), commit_date.decode()

def get_git_version():
    commit_hash, commit_message, commit_date = return_git_version()

    print("# Hello. I am from commit {}".format(commit_hash))
    print("# Commit Date: {}".format(commit_date))
    print("# Commit Message: {}".format(commit_message))


def log_git_version():
    commit_hash, commit_message, commit_date = return_git_version()

    logging.info("# Hello. I am from commit {}".format(commit_hash))
    logging.info("# Commit Date: {}".format(commit_date))
    logging.info("# Commit Message: {}".format(commit_message))

def log_input(arg_list):
    logging.info(" ".join(arg_list))
    with open("misp_commands.txt", "a+") as mc:
        mc.write(" ".join(arg_list) + "\n")

def relative_pbc(coord, pbc_mat):
    inv_pbc_mat = np.linalg.inv(pbc_mat)
    rel_coord = np.squeeze(np.matmul(inv_pbc_mat.T, coord[:,:,:,np.newaxis]))
    return rel_coord


# Unwraps trajectory so that msd can be calculated directly with FFT
# currently only supported for orthogonal cells
def unwrap_trajectory(coord, pbc_mat):
    pbc_ortho = np.diag(pbc_mat)
    dist1 = np.diff(coord, axis=0)
    wrap_matrix = (dist1 > pbc_ortho / 2).astype(int) - (
        dist1 < -pbc_ortho / 2
    )  # marks all instances where an atom has been wrapped, sign marks direction of movement
    wrap_matrix = np.cumsum(
        wrap_matrix, axis=0
    )  # calculates sum of previous movements by wrapping for each timestep
    coord[1:] -= wrap_matrix * pbc_ortho  # subtracts movement used to wrap atoms
    return coord


# Wraps trajectory into pbc_mat box
def wrap_trajectory(coord, pbc_mat):
    """wraps all atoms to unit cell, so that their relative coordinates fall between [0 ,0 ,0] and [1, 1, 1]

    Parameters
    ----------
    coord: ndarray
         3-dimensional array of coordinates. Shape (timesteps, atom_no, 3).
    pbc_mat: ndarray
         2-dimensional matrix with pbcs.

    Returns
    -------
    coord_wrapped: ndarray
         new coordinates with same dimensions as input array
    """
    inv_pbc_mat = np.linalg.inv(pbc_mat)
    rel_coord = np.matmul(inv_pbc_mat.T, coord[:,:,:,np.newaxis])
    rel_coord -= np.floor(rel_coord)
    coord_wrapped = np.squeeze(np.matmul(pbc_mat.T, rel_coord))
    return coord_wrapped


# Returns center of mass as array of dimensionality (timesteps, 3 spatial dimensions)
def get_com(coord, atoms, pbc_mat):
    # Get atom masses from mass_dict and calculate weights for the com as the weighted average of all coordinates
    mass_list = []
    for atom_type in atoms:
        mass_list.append(atomic_mass(atom_type))
    mges = sum(mass_list)
    mass_weights = np.array(mass_list) / mges

    # Print warning if one of the elements has a mass of zero
    for i in np.unique(atoms):
        if np.all(mass_list[atoms == i] == 0):
            print("WARNING mass of " + i + " is equal to zero.")

    # Calculate com for each frame
    com = np.multiply(coord, mass_weights[np.newaxis, :, np.newaxis])
    com = np.sum(com, axis=1)

    return com

def limit_for_pbc_dist(pbc):
    cell_volume = np.linalg.det(pbc)
    a, b, c = pbc
    width_1 = cell_volume / np.linalg.norm(np.cross(a, b))
    width_2 = cell_volume / np.linalg.norm(np.cross(b, c))
    width_3 = cell_volume / np.linalg.norm(np.cross(c, a))
    return min(width_1, width_2, width_3)/2

def pbc_dist(pos1, pos2, pbc, inv_pbc = None):
    """Calculates distance between two points in a periodic box according to the minimum-image convention.

    Parameters
    ----------
    pos1: ndarray
        1D array of shape (3,). Starting point of final distance vector.
    pos2: ndarray
        1D array of shape (3,). End point of final distance vector.
    pbc_mat: ndarray
        2D array of shape (3, 3). Periodic boundary conditions with the three cell vectors as its rows.
    inv : ndarray
        2D array of shape (3, 3). Inverse of the periodic boudnary conditions.

    Returns
    -------
    new_dist : ndarray
        1D array of shape (3,). Distance vector connecting the correct images of the two points.

    Warning
    -------
    For non-orthorhombic boxes (if pbc is not diagonal), this is only accurate up to a certain limit, which can be calculated using limit_for_pbc_dist(pbc).
    Above that limit, the distance will be overestimated.

    Notes
    -----
    Can be applied to arrays of arbitrary shapes (..., 3). The result will follow the same broadcasting rules as the subtraction pos1 - pos2.

    See also
    --------
    limit_for_pbc_dist
    pbc_dist2
    """
    if inv_pbc is None:
        inv_pbc = np.linalg.inv(pbc)
    dist_diff = pos1 - pos2
    # equal to inv.T @ dist_diff
    rel_dist = np.dot(dist_diff, inv_pbc)
    rel_dist -= np.rint(rel_dist)
    #equal to pbc_mat.T @ pbc_dist
    new_dist = np.dot(rel_dist, pbc)
    return new_dist

def pbc_dist2_old(pos1, pos2, pbc_mat):
    # transpose pbc matrix to fix triclinic case
    pbc_mat = pbc_mat.T

    dist_diff = pos1[:, np.newaxis,:] - pos2[np.newaxis,:,:]
    dist_diff = np.moveaxis(dist_diff, 2,1)
    inv = np.linalg.inv(pbc_mat)
    pbc_dist = inv @ dist_diff
    pbc_dist -= pbc_dist.astype(int)
    pbc_dist[abs(pbc_dist) > 1.5] = pbc_dist[abs(pbc_dist) > 1.5] - 2 * np.sign(
        pbc_dist[abs(pbc_dist) > 1.5]
    )
    pbc_dist[abs(pbc_dist) > 0.5] = pbc_dist[abs(pbc_dist) > 0.5] - np.sign(
        pbc_dist[abs(pbc_dist) > 0.5]
    )
    new_dist = pbc_mat @ pbc_dist
    return np.moveaxis(new_dist, 2,1) 

def pbc_dist2_nice(pos1, pos2, pbc_mat, inv = None):
    # same algorithm as pbc_dist, but using matrix multiplication instead of einsum and without memory magic
    # takes roughly twice as long as a result
    if inv is None:
        inv = np.linalg.inv(pbc_mat)
    rel1 = pos1 @ inv
    rel2 = pos2 @ inv
    pbc_dist = rel1[np.newaxis, :, :] - rel2[:, np.newaxis, :]
    pbc_dist -= np.around(pbc_dist)
    new_dist = pbc_dist @ pbc_mat
    return np.swapaxes(new_dist, 0, 1)

def pbc_dist2(pos1, pos2, pbc, inv_pbc = None):
    """Calculates all pair-wise distance between two lists of points in a periodic box according to the minimum-image convention.

    Parameters
    ----------
    pos1: ndarray
        2D array of shape (i, 3). Each row is one point.
    pos2: ndarray
        1D array of shape (j, 3). Each row is one point.
    pbc: ndarray
        2D array of shape (3, 3). Periodic boundary conditions with the three cell vectors as its rows.
    inv_pbc : ndarray
        2D array of shape (3, 3). Inverse of the periodic boudnary conditions.

    Returns
    -------
    new_dist : ndarray
        3D array of shape (i, j, 3). Distance vectors connecting all pair-wise combinations of points in pos1 and pos2.

    Warning
    -------
    For non-orthorhombic boxes (if pbc is not diagonal), this is only accurate up to a certain limit, which can be calculated using limit_for_pbc_dist(pbc).
    Above that limit, the distance will be overestimated.

    Notes
    -----
    The first array pos1 can also be 1D of shape (3,). The output will be exactly the same as if it had the shape (1,3).
    Can also be applied to higher-dimensional arrays according to some broadcasting rules. For example for two arrays of shape (100, 6, 3) and (100, 12, 3), the output array has the shape (100, 6, 12, 3) and contains all pair-wise distance of points within the same first index.

    See also
    --------
    limit_for_pbc_dist

    """
    # pbc_mat contains the three cell vectors a, b and c as rows, so that pbc_mat = [ [ax, ay, az], [bx, by, bz], [cx, cy, cz] ]
    if inv_pbc is None:
        inv_pbc = np.linalg.inv(pbc)
    # transform coordinates into relative coordinates
    # equal to matrix multiplication pos1 @ inv
    # which in turn is equal to row-wise matrix multiplication inv @ pos1[i, :]
    rel1 = np.matmul(pos1, inv_pbc, order = 'F')
    rel2 = np.matmul(pos2, inv_pbc, order = 'F')
    # all pair-wise distances in relative coordinates
    rel_dist = rel1[..., np.newaxis, :] - rel2[..., np.newaxis, :, :]
    rel_dist -= np.rint(rel_dist)
    # transform relative coordinates back into real coordinates
    # equal to matrix multiplication pbc_dist @ pbc_mat
    # which in turn is equal to row-wise matrix multiplication pbc_mat @ pbc_dist[i, j, :]
    pbc = np.asfortranarray(pbc)
    new_dist = np.einsum('...j,jk', rel_dist, pbc)
    return new_dist

def pbc_dist_point_point_triclinic(point1, point2, pbc_mat):
    """
    Calculates the distance between two points in a periodic box applying the minimum image conversion.

    Parameters
    ----------
    point1, point2: ndarray
        Absolute coordinates of the two points. Shape (3,).
    pbc_mat: ndarray
        Periodic boundary conditions given as a stack of three vectors. Shape (3, 3).

    Returns
    -------
    min_difference: float
        Distance between them.

    See also
    --------
    pbc_dist: Distance between two points. Only tested for orthorhombic box.
    pbc_dist2: All pair-wise distances between two groups of points. Only tested for orthorhombic box.
    pbc_dist_group_group_triclinic: All pair-wise distances between two groups of points.
    """
    # array containing all combinations of -1, 0, 1: [[-1, -1, -1], [-1, -1, 0], [-1, -1, 0], ..., [1, 1, 0], [1, 1, 1]]
    # represents all possible combinations of the three pbc vectors, which describe the neighboring cells
    all_shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    # by adding all_shift_coordinates to coordinates of shape (3), all coordinates in the neighboring periodic cells is generated
    all_shift_coordinates = np.squeeze(all_shifts[:,None,:] @ pbc_mat)
    # calculate difference between point_1 and all "copies" of point_2 in the neighboring cells
    all_shifted_differences = point1 - point2 + all_shift_coordinates
    # per minimum-image convention only the difference with the smallest distance is accepted
    min_difference = all_shifted_differences[np.argmin(np.linalg.norm(all_shifted_differences, axis = 1))]
    return min_difference

def pbc_dist_group_group_triclinic(group1: np.ndarray, group2: np.ndarray, pbc_mat: np.ndarray) -> np.ndarray:
    """
    Calculates all pair-wise distances between the points in group1 and the points in group2 in accordance with the periodic boundary condition.

    Parameters
    ----------
    group1 : ndarray
        coordinate points of first group; will be represented as axis 0; shape (atom_no_1, 3)
    group2 : ndarray
        coordinate points of second group; will be represented as axis 1; shape (atom_no_2, 3)
    pbc : ndarray
        periodic boundary conditions as dictated by three vectors; shape (3, 3)

    Returns
    -------
    correct_distances : ndarray
        all pair-wise distances between the points in group1 and group2 as float values; shape (atom_no_1, atom_no_2)

    Notes
    -----
    This should work for all periodic boundary conditions. Since it checks all directly neighboring cells, the trajectory must be wrapped.
    For orthorhombic cells pbc_dist2 should be used, as it's faster and not dependent on a wrapped trajectory.

    See also
    --------
    pbc_dist: Distance between two points. Only tested for orthorhombic box.
    pbc_dist2: All pair-wise distances between two groups of points. Only tested for orthorhombic box.
    pbc_dist_point_point_triclinic: Distance between two points.
    """
    # array containing all combinations of -1, 0, 1: [[-1, -1, -1], [-1, -1, 0], [-1, -1, 0], ..., [1, 1, 0], [1, 1, 1]]
    # represents all possible combinations of the three pbc vectors, which describe the neighboring cells
    all_shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    # by adding all_shift_coordinates to coordinates of shape (3), all coordinates in the neighboring periodic cells is generated
    all_shift_coordinates = np.squeeze(all_shifts[:,None,:] @ pbc_mat)
    # calculate difference between point_1 and all "copies" of point_2 in the neighboring cells
    naive_differences = group1[:, np.newaxis, :] - group2[np.newaxis, :, :]
    all_shift_differences = naive_differences[:, :, np.newaxis, :] - all_shift_coordinates
    all_shift_distances = np.linalg.norm(all_shift_differences, axis = -1)
    correct_distances = np.amin(all_shift_distances, axis = -1)

    return correct_distances

def next_neighbor(atom, xyz_list, pbc_mat):
    dist_list = pbc_dist2(atom_1[np.newaxis, :], atom_2, pbc_mat)
    dist_list = np.linalg.norm(dist_list, axis=2)
    listx = np.argmin(dist_list, axis=1)
    min_list = dist_list[np.arange(len(atom_1)), listx]
    return listx, min_list


def next_neighbor2(atom_1, atom_2, pbc_mat, inv_pbc = None):
    if inv_pbc is None:
        inv_pbc = np.linalg.inv(pbc_mat)
    # atom_1, atom_2: list of coordinates of reference atoms and observed atoms -> (atom_no_1/2,3)
    # listx/min_list: list of indices of/distance to closest neighbor to reference atom in "atom_1" among observed atoms in "atom_2"
    dist_list = pbc_dist2(atom_1, atom_2, pbc_mat, inv_pbc)
    dist_list = np.linalg.norm(dist_list, axis=-1)
    listx = np.argmin(dist_list, axis=-1)
    min_list = np.take_along_axis(dist_list, listx[..., None], axis = -1)[..., 0]
    return listx, min_list

def next_neighbor2_triclinic(atom_1, atom_2, pbc_mat):
    #atom_1, atom_2: list of coordinates of reference atoms and observed atoms -> (atom_no_1/2,3)
    #listx/min_list: list of indices of/distance to closest neighbor to reference atom in "atom_1" among observed atoms in "atom_2"
    dist_list = pbc_dist_group_group_triclinic(atom_1, atom_2, pbc_mat)
    #dist_list = np.linalg.norm(dist_list, axis = 2)
    listx=np.argmin(dist_list,axis=1)
    min_list = dist_list[np.arange(len(atom_1)),listx]
    return listx, min_list

def remove_com(coord, atoms, zero = True):
    # Get atom masses from mass_dict and calculate weights for the com as the weighted average of all coordinates
    mass_list = []
    for atom_type in atoms:
        mass_list.append(atomic_mass(atom_type))
    mges = sum(mass_list)
    mass_weights = np.array(mass_list) / mges

    # Print warning if one of the elements has a mass of zero
    for i in np.unique(atoms):
        if np.all(mass_list[atoms == i] == 0):
            print("WARNING mass of " + i + " is equal to zero.")

    # Calculate com for each frame
    com = np.multiply(coord, mass_weights[np.newaxis, :, np.newaxis])
    com = np.sum(com, axis=1)

    # Set com to (0, 0, 0) if zero is true, otherwise keep com of first frame
    if zero == True:
        com_init = np.array([0.0, 0.0, 0.0])
    else:
        com_init = com[0]

    # Subtract com from coord and return result
    coord_com = coord - com[:, np.newaxis, :]
    coord_com += com_init
    return coord_com


def traj_to_ar(path):
    # Read number of atoms from first line
    with open(path) as f:
        first_line = f.readline()
    noa = int(first_line.strip())
    # Read atom types from first column
    atom = np.genfromtxt(path, usecols=(0), dtype="str", skip_header=2, max_rows=noa)
    # Read atom coordinates skipping line containing atom number and comment line; converted to flattened numpy array and reshaped
    trajectory = pd.read_csv(
        path,
        header=None,
        usecols=[1, 2, 3],
        sep=r"\s+",
        skiprows=lambda x: x % (noa + 2) in [0, 1],
    ).to_numpy()
    timesteps = int(trajectory.shape[0] / noa)
    trajectory = np.reshape(trajectory, (timesteps, noa, 3))

    return trajectory, atom

def easy_read(path1: str, pbc: np.ndarray, com: bool = True, wrapchoice: str = "nowrap") -> (np.ndarray, np.ndarray):
    """reads trajectory and checks if fast npz exist, if not npz file is created

    Parameters
    ----------
    path1: string
        Path to xyz-file of the trajectory.
    pbc: ndarray
        Periodic boundary condition given as three vectors stacked on top of each other (shape: 3x3).
    com: boolean
        If true, the center of mass will be set to (0, 0, 0) for every frame.
    wrap: string {wrap, unwrap, ...}
        Whether to wrap the trajectory, unwrap the trajectory or do neither.

    Returns
    -------
    coord: ndarray
        Coordinates of all atoms in the trajectory. Shape (timesteps, atom_no, 3).
    atom: ndarray
        Labels of all atoms. Shape (atom_no).
    """
    start_time = time.time()
    auxiliary_file = path1 + ".npz"
#    auxiliary_file = path1 + "_"+ str(com) + "_" +str(wrap) + "_"+ ".npz"

    # If npz-file doesn't exist read trajectory from xyz-file, otherwise read npz-file
    if os.path.exists(auxiliary_file) == False:
        print("auxiliary file " + auxiliary_file  +" does not exist")
        # Read trajectory from file
        coord, atom = traj_to_ar(path1)
        nof = coord.shape[0]
        print("number of frames: " + str(nof))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        np.savez(auxiliary_file, coord, atom)
    else:
        print("auxiliary file " + auxiliary_file + "  exists")
        ar2 = np.load(auxiliary_file)
        coord = ar2["arr_0"]
        atom = ar2["arr_1"]
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    # Remove center of mass movement
    if com:
        print("remove com")
        coord = remove_com(coord, atom)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    # Check wrap and wrap/unwrap/do nothing
    try:
        wrapchoice = wrapchoice.tolower()
    except AttributeError:
        pass
    if wrapchoice == "unwrap":
        print("unwrapping trajectory")
        coord = unwrap_trajectory(coord, pbc)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    elif wrapchoice == "wrap":
        print("wrapping trajectory")
        coord = wrap_trajectory(coord, pbc)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    else:
        print("Trajectory will be neither wrapped nor unwrapped.")

    return Trajectory(coord, atom, pbc)


def easy_write(xyz, atoms, filename, append = False):
  if append:
    open_option = 'a'
  else:
    open_option = 'w'
  with open(filename, open_option) as the_file:
      for i in range(xyz.shape[0]):
          the_file.write(str(xyz.shape[1]) + '\n')
          the_file.write('\n')
          for j in range(atoms.shape[0]):
              the_file.write('%s  %f   %f   %f \n' % (atoms[j], xyz[i,j,0], xyz[i,j,1],xyz[i,j,2]) )
           #the_file.write('\n')




###########################################################################
#deprecated functions

#transponieren hier vergessen!
#def pbc_dist(pos1, pos2, pbc_mat):
#    dist_diff = pos1 - pos2
#    inv = np.linalg.inv(pbc_mat)
#    pbc_dist = inv @ dist_diff
#    pbc_dist -= pbc_dist.astype(int)
#    for i in range(3):
#        if abs(pbc_dist[i]) > 1.5:
#            pbc_dist[i] -= 2 * np.sign(pbc_dist[i])
#        if abs(pbc_dist[i]) > 0.5:
#            pbc_dist[i] -= np.sign(pbc_dist[i])
#    new_dist = pbc_mat @ pbc_dist
#    return new_dist
#
