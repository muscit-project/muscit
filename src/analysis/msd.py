import numpy as np
from scipy import stats

import argparse
import os
import sys
import time

import numpy as np
from trajec_io import readwrite

from markov import matrix

def msd_fft(trajectory, atom, pbc_mat, atom_type, fit_range = (0.2, 0.4)):
    """Calculates the mean square displacement for a specified atom type.

    From a given trajectory (passed as coordinates and atom labels) the mean square displacement of a certain atom type is calculated.
    The result will be averaged over all atoms of the specified type and all shifted time windows.

    Parameters
    ----------
    trajectory : ndarray
        Coordinates of the trajectory.
    atom: array-like
        Atom labels.
    pbc_mat : pbc_mat
        Periodic boundary conditions as a 3x3 array.
    atom_type : str
        Atom type, for which the MSD is calculated. Should exactly match at least one entry in `atom`.
    fit_range : tuple( float, float )
        Range of orrelation times to be used for the linear regression, given as fractions of the total trajectory length.

    Returns
    -------
    msd : ndarray
        Correlation time in ps and MSD in pm^2.
    D : float
        Correlation coefficient as found by linear regression of the second half of `msd`.
    R2 : float
        Coefficient of determination of the linear regression used to calculate D.

    """
    def autocorrFFT(x):
        N=len(x)
        F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res= (res[:N]).real   #now we have the autocorrelation in convention B
        n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res/n #this is the autocorrelation in convention A
    print(f"Calculating MSD (FFT) of {atom_type}...", end="    ", flush=True)
    traj_msd = trajectory[:, atom == atom_type, :]
    N=len(traj_msd)
    msd = np.zeros((N))

    # Iterate through atoms, calculate individual MSDs and them up
    for i in range(traj_msd.shape[1]):
        traj_atom = traj_msd[:,i,:]
        D=np.square(traj_atom).sum(axis=1)
        D=np.append(D,0)
        S2=sum([autocorrFFT(traj_atom[:, j]) for j in range(traj_atom.shape[1])])
        Q=2*D.sum()
        S1=np.zeros(N)
        for m in range(N):
            Q=Q-D[m-1]-D[N-m]
            S1[m]=Q/(N-m)
        msd += S1 - 2 * S2
    msd /= traj_msd.shape[1]

    # Add time axis to msd array
    msd_time = np.arange(N)*0.5
    msd = np.column_stack((msd_time/1E3, msd*1E4))

    # Apply linear fit and truncate msd array to fit_range
    slope, intercept, r_value, p_value, std_err = stats.linregress(msd[round(N*fit_range[0]):round(N*fit_range[1])])
    msd = msd[:round(N*fit_range[1])]
    return msd, slope/6, r_value**2


def msd_for_unwrap(trajectory,atom, atom_type, timestep_md, tau_steps, verbosity = 1,  max_length = None):
    msd= []
    tau = []
    msd.append(0.0)
    tau.append(0.0)
    trajectory = trajectory[:, atom == atom_type, :] 
    if max_length:
        limit = max_length
    else:
        limit = trajectory.shape[0]
    for  i in range(0, limit, tau_steps):
       if i %  verbosity == 0:
           print(i)
       if i == 0:
           continue
       dist_ar = np.linalg.norm(trajectory[i:,:,:] -  trajectory[:-i,:,:], axis = 2)**2
       #dist_ar = dist_ar.flatten()
       msd.append(np.mean(dist_ar))
       tau.append( i*timestep_md )
    return np.array(tau), np.array(msd)


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def script_msd_unwrap():
    print("command line:")
    print(sys.argv)
    readwrite.get_git_version()
    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="path to trajek")
    parser.add_argument("pbc", help="path to pbc numpy mat")
    parser.add_argument(
        "com", help="remove center of mass movement", type=boolean_string
    )
    parser.add_argument("wrap", help="wrap trajec", choices = ["wrap", "nowrap", "unwrap"])
    parser.add_argument(
        "every",
        help="every i-th step from trajec is used for new reduced trajectory",
        type=int,
    )
    parser.add_argument("atom_type_for_msd", help="atom type for msd calculation")
    parser.add_argument("timestep_md", type = float, help= "timestep of underlying md simulation")
    parser.add_argument("tau", type = int, help="resolution of msd, delay between two intervalls is set to one step of trajectory = no delay ")
    parser.add_argument("--verbosity", default = 1, type = int,  help="frequency of progress report")
    parser.add_argument("--max_length", default = None, type = int, help="maximum interval for msd")  


    args = parser.parse_args()
    pbc_mat = np.loadtxt(args.pbc)
    print(args.com, args.wrap)
    
    coords, atoms = readwrite.easy_read(args.path, pbc_mat, com=args.com, wrapchoice=args.wrap)
    coords = readwrite.unwrap_trajectory(coords, pbc_mat)
    
    #if args.verbosity:
    #    verbosity =  args.verbosity
    #else:
    #    verbosity = 1

    #tau, msd = msd.msd_for_unwrap(coords, atoms ,  args.atom_type_for_msd, args.timestep_md, args.tau, verbosity, max_length)
    tau, msd = msd_for_unwrap(coords, atoms , args.atom_type_for_msd, args.timestep_md, args.tau, args.verbosity, args.max_length)
    np.savetxt('std_msd_unwrap_from_md.out', np.column_stack((tau, msd)), header="t / ps\tMSD / A^2")

    #readwrite.easy_write(coord[:: args.every, :, :], atom, args.new_filename)






def msd_std(trajectory, atom, pbc_mat, atom_type, resolution = 100, time_range = 0.4):
    """Calculates the mean square displacement for a specified atom type.

    From a given trajectory (passed as coordinates and atom labels) the mean square displacement of a certain atom type is calculated.
    The result will be averaged over all atoms of the specified type and all shifted time windows.

    Parameters
    ----------
    trajectory : ndarray
        Coordinates of the trajectory.
    atom: array-like
        Atom labels.
    pbc_mat : pbc_mat
        Periodic boundary conditions as a 3x3 array.
    atom_type : str
        Atom type, for which the MSD is calculated. Should exactly match at least one entry in `atom`.
    resolution : int
        Number of correlation times to sample.
    time_range : float
        Largest correlation time to be sampled as a fraction of the total trajectory length.

    Returns
    -------
    msd : ndarray
        Correlation time in ps and MSD in pm^2.
    D : float
        Correlation coefficient as found by linear regression of the second half of `msd`.
    R2 : float
        Coefficient of determination of the linear regression used to calculate D.

    """
    timesteps = len(trajectory)
    #returns MSD sampling {resolution} tau values in the interval {0, time_range*len(trajectory)} and Diffusion coefficient
    #trajectory: (timesteps,atom_no,3 dimensions), resolution: int, time_range: float
    trajectory = trajectory[:, atom == atom_type, :]
    diff1 = np.zeros((resolution))
    corr_time_list = np.linspace(0, timesteps * time_range, resolution + 1, dtype = int)[1:]    #corr_time_list: list of all correlation times sampled -> len(corr_time_list) = resolution

    # Calculate MSD for each correlation time i
    for i in range(resolution):
        corr_time = corr_time_list[i]
        print_progress(f"Calculating MSD (slow) of {atom_type}...", i, resolution)
        traj_start = trajectory[:-corr_time]  #coord at time frame beginning
        traj_end = trajectory[corr_time:]     #coord at time frame end
#        diff_tmp = pbc_dist3(traj_start,traj_end,pbc_mat)  #using pbc_mat is slower and unnecessry when using unwrapped trajectory
        diff_tmp = traj_start - traj_end
        diff_tmp = np.linalg.norm(diff_tmp, axis=2) ** 2
        diff1[i] = diff_tmp.mean(axis=(0,1))

    #Convert to pm^2/ps and create output array
    msd = diff1*1E4   #msd in pm^2
    msd_time = corr_time_list*0.5*1E-3     #timesteps in ps
    msd = np.column_stack((msd_time,msd))

    # Apply linear fit to determine D
    slope, intercept, r_value, p_value, std_err = stats.linregress(msd[round(len(msd)*0.5):])
    return msd, slope/6, r_value**2

