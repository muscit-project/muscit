import argparse
import logging
import os
import pickle
import sys
import time
import random
from dataclasses import dataclass, asdict

import numpy as np
from scipy import sparse
from trajec_io import readwrite

from transonic import boost

@dataclass
class SettingsManager:
    noji : int
    nols : int
    #md_timestep : float
    sweeps : int
    equilibration_sweeps : int
    verbose : bool
    xyz_output : bool
    print_freq : int
    reset_freq : int
    #output_path : str
    seed : int

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def prepare_trajectory(coord, atoms, pbc):
    return

def initialize_oxygen_lattice(oxygen_number, proton_number, rng):
    """Creates an integer array of length <oxygen_number> filled with randomly
    distributed numbers from 1 to <proton_number>

    Parameters
    ----------
    oxygen_number: int
                   The number of oxygen sites
    proton_number: int
                   The number of protons"""
    oxygen_lattice = np.zeros(oxygen_number, np.uint8)
    oxygen_lattice[:proton_number] = range(1, proton_number + 1)
    rng.shuffle(oxygen_lattice)
    return oxygen_lattice

def prepare_custom_prob(helper):
    return None

class Helper:
    noji: int
    nols: int
    jumps: int
    jump_mat_recalc: "int[][]"
    sources: "int[] list"
    destinations: "int[] list"
    probs: "float[] list"
    prob_frame: "float[][]"

    def __init__(self, jump_mat, noji, nols):
#        self.jump_mat = jump_mat
        self.noji = noji
        self.nols = nols
        self.jumps = 0
        self.jump_mat_recalc = np.zeros((nols, nols), dtype = int)
        self.sources, self.destinations, self.probs = self.unpack_scipy_matrices( jump_mat )
        self.prob_frame = np.zeros( (nols, nols) )

    def unpack_scipy_matrices(self, jump_mat):
        sources, destinations, probs = [], [], []
        for j in jump_mat:
#            sources.append( np.ascontiguousarray(j.row.copy()) )
#            destinations.append( np.ascontiguousarray(j.col.copy()) )
            sources.append( np.ascontiguousarray(j.col.copy()) )
            destinations.append( np.ascontiguousarray(j.row.copy()) )
            probs.append( np.ascontiguousarray(j.data.copy()) )
        return sources, destinations, probs

@boost
def a_sweep_step(lattice : "uint8[]", sources : "int32[]", destinations : "int32[]", probs : "float[]", jump_mat_recalc : "int[][]"):
    jumps = 0
    start_protonation = np.copy(lattice)
    no_of_pairs = len(sources)
    for i in range(no_of_pairs):
        index = np.random.randint(0, no_of_pairs)
        s = sources[index]
        d = destinations[index]
        p = probs[index]
#        if lattice[s] != 0 and lattice[d] == 0 and start_protonation[s] != 0 and start_protonation[d] == 0 and np.random.uniform(0, 1) < p:
        if lattice[s] != 0 and lattice[d] == 0 and np.random.uniform(0, 1) < p:
            lattice[d] = lattice[s]
            lattice[s] = 0
            jumps += 1
            jump_mat_recalc[d , s] += 1
    return jumps

@boost
def sweep_step(lattice : "uint8[]", sources : "int32[]", destinations : "int32[]", probs : "float[]", jump_mat_recalc : "int[][]"):
    rand_mat1 = np.random.uniform(0,1,len(probs))
    #probs[rand_mat1 >  probs] = 0    # prevent jumps based on their probability
    #probs[lattice[destinations] != 0] = 0       # prevent jumps if destination site is occupied at the beginning of sweep
    #probs[lattice[sources] == 0] = 0       # prevent jumps if souce site is empty at the beginning of sweep
    allowed_probs = (rand_mat1 <  probs) & (lattice[destinations] == 0) & (lattice[sources] != 0)

    probs, destinations, sources = probs[allowed_probs], destinations[allowed_probs], sources[allowed_probs]
    destination = list(destinations)
    source = list(sources)
    banned_destination =[]
    banned_source=[]
    jumps = 0
    while source:
        i = random.randrange(len(source)) # get random index
        source[i], source[-1] = source[-1], source[i]
        destination[i], destination[-1] = destination[-1], destination[i]
        final_source = source.pop()
        final_destination = destination.pop()
        if (final_source not in banned_source) and (final_destination not in banned_destination):
            banned_destination.append(final_destination)
            banned_source.append(final_source)
            lattice[final_destination] = lattice[final_source]
            lattice[final_source] = 0
            jumps += 1
            jump_mat_recalc[final_destination , final_source ] += 1
    return jumps

def cmd_lmc_run(oxygen_trajectory, pbc, oxygen_lattice, helper, settings, md_timestep, output):
    oxygen_trajectory = np.ascontiguousarray( oxygen_trajectory )
    pbc = np.ascontiguousarray( pbc )
    msd_lmc, autocorr_lmc, jump_mat_recalc = lmc_goes_brrr(oxygen_trajectory, pbc, np.linalg.inv(pbc), oxygen_lattice, helper.sources, helper.destinations, helper.probs, asdict(settings), md_timestep, output )
    np.savetxt("jump_mat_sampled_from_lmc_run", jump_mat_recalc)
    np.savetxt("overall_number_of_jumps", np.array([jump_mat_recalc.sum()]))
    print("overall_number_of_jumps: " + str(jump_mat_recalc.sum()))
    return msd_lmc, autocorr_lmc, jump_mat_recalc

@boost
def lmc_one_reset(lattice : "uint8[]", sources : "int32[] list", destinations : "int32[] list", probs : "float[] list", reset_freq : "int", print_freq : "int"):
    no_of_prints = int( reset_freq / print_freq )
    lattice_over_time = np.zeros( ( no_of_prints, len(lattice) ), dtype = np.uint8 )
    jumps = 0
    jumps_over_time = np.zeros(no_of_prints, dtype = int)
    jump_mat_recalc = np.zeros( (len(lattice), len(lattice)), dtype = int )
    for sweep in range(reset_freq):
        frame = sweep % len(sources)
        jumps += sweep_step(lattice, sources[frame], destinations[frame], probs[frame], jump_mat_recalc)
        if sweep % print_freq == 0:
            print_no = int(sweep/print_freq)
            jumps_over_time[print_no] = jumps
            lattice_over_time[print_no, :] = lattice
    return lattice_over_time, jumps_over_time, jump_mat_recalc

@boost
def write_xyz_format( oxygen_trajectory : "float[][][]", proton_trajectory: "float[][][]" , outpath : "str"):
    f = open( outpath, "a+" )
#    output_string = ""
    no_of_atoms = oxygen_trajectory.shape[1] + proton_trajectory.shape[1]
    for oxygen_frame, proton_frame in zip( oxygen_trajectory, proton_trajectory ):
        f.write( f"{no_of_atoms:3d}\n\n" )
        for O_position in oxygen_frame:
            f.write( f"  O\t{O_position[0]:8f}\t{O_position[1]:8f}\t{O_position[2]:8f}\n" )
        for H_position in proton_frame:
            f.write( f"  H\t{H_position[0]:8f}\t{H_position[1]:8f}\t{H_position[2]:8f}\n" )
    f.close()
    return


@boost
def lmc_goes_brrr(oxygen_trajectory : "float[][][]", pbc : "float[][]", inv_pbc : "float[][]", oxygen_lattice : "uint8[]", sources : "int32[] list", destinations : "int32[] list", probs : "float[] list", settings : "Dict[str, int]", md_timestep : "float", outpath : "str"):
    # unpack variables
    equilibration_sweeps = settings["equilibration_sweeps"]
    sweeps = settings["sweeps"]
    reset_freq = settings["reset_freq"]
    print_freq = settings["print_freq"]
    xyz_print = settings["xyz_output"]

    jumps = 0
    jump_mat_recalc = np.zeros( ( len( oxygen_lattice ), len( oxygen_lattice ) ), dtype = int )
    eq_start = time.time()
    for sweep in range(equilibration_sweeps):
        if sweep % round(equilibration_sweeps / 10) == 0:
            print(f"Equlibration sweep {sweep:d} / {equilibration_sweeps:d}, {jumps:d} so far")
        frame = sweep % oxygen_trajectory.shape[0]
        jumps += sweep_step(oxygen_lattice, sources[frame], destinations[frame], probs[frame], jump_mat_recalc)
    eq_end = time.time()
    eq_time = eq_end - eq_start
    print(f"Equilibrated over {equilibration_sweeps:d} sweeps with {jumps:d} jumps in {eq_time:f} s")

    ## timer
    no_of_prints = int(reset_freq / print_freq)
    time_lmc = np.arange( no_of_prints ) * md_timestep * print_freq
    msd_xyz_averaged = np.zeros( (no_of_prints, 3) )
    autocorr_averaged = np.zeros( no_of_prints )
    start_time = time.time()
    no_of_resets = int(sweeps / reset_freq)
    jump_mat_recalc = np.zeros( ( len( oxygen_lattice ), len( oxygen_lattice ) ), dtype = int )
    for i in range(no_of_resets):
        # do sweep and gather information on lattice and jumps
        lattice_over_time, jumps_over_time, jump_mat_reset = lmc_one_reset(oxygen_lattice, sources, destinations, probs, reset_freq, print_freq)
        jump_mat_recalc += jump_mat_reset
        proton_trajectory = lattice_to_proton_positions_over_time( oxygen_trajectory, lattice_over_time, pbc, inv_pbc)
        # create output, either by appending xyz-file or writing MSD to output file
        if settings["xyz_output"]:
            write_xyz_format( oxygen_trajectory, proton_trajectory, outpath ) # + f"_reset_{i:d}.xyz" )
        else:
            # Calculate observables (msd, autocorrelation)
            msd = calculate_msd_averaged( proton_trajectory )
            auto_correlation = calculate_auto_correlation( lattice_over_time )
            # Add them to the overall average
            msd_xyz_averaged += msd / no_of_resets
            autocorr_averaged += auto_correlation / no_of_resets
            # Create output string and write it to file
            output_string = print_observables( i * reset_freq, sweeps, print_freq, md_timestep, msd, jumps_over_time, auto_correlation)
            f = open( outpath, "a+")
            f.write(output_string)
            f.close()
        # print performance information
        resets_to_give_info = set( round( j * no_of_resets / 20 ) for j in range(20) )
        if i in resets_to_give_info:
            seconds_per_reset = (time.time() - start_time) / (i+1)
            time_left = seconds_per_reset * (no_of_resets - i)
            jumps = np.sum( jump_mat_recalc )
            print(f"Reset {i+1:d} / {no_of_resets:d}, {jumps:d} jumps so far.\tETA: {time_left:.3f} s")

    sweep_time = time.time() - start_time
    print(f"Finished {sweeps:d} sweeps in {sweep_time:f} s")
    return msd_xyz_averaged, autocorr_averaged, jump_mat_recalc

# transonic def lattice_to_proton_positions( float[][], uint8[] )
@boost
def lattice_to_proton_positions( oxygen_frame, lattice):
    out = np.zeros( ( np.count_nonzero(lattice), 3 ) )
    for o_ind, h_ind in enumerate(lattice):
        if h_ind > 0:
            out[h_ind-1] = oxygen_frame[o_ind]
    return out

# transonic def lattice_to_proton_positions_over_time( float[][][], uint8[][], float[][], float[][] )
@boost
def lattice_to_proton_positions_over_time( oxygen_trajectory, lattice_over_time, pbc, inv_pbc):
    no_of_protons = np.count_nonzero(lattice_over_time[0, :])
    out = np.zeros( ( len(lattice_over_time), no_of_protons, 3 ) )
    start_positions = lattice_to_proton_positions( oxygen_trajectory[0, :, :], lattice_over_time[0, :] )
    out[0, :, :] = start_positions
    for frame in range( 1, len(lattice_over_time) ):
        traj_frame = frame % len(oxygen_trajectory)
        new_positions = lattice_to_proton_positions( oxygen_trajectory[traj_frame, :, :], lattice_over_time[frame] )
        displacements = cc_dist( new_positions, start_positions, pbc, inv_pbc )
        start_positions += displacements
        out[ frame, :, :] = start_positions
    return out

# transonic def cc_dist(float64[][], float64[][], float64[][], float64[][])
@boost
def cc_dist( p1, p2, pbc_view, inv_pbc_view):
    n1 = p1.shape[0]
    n2 = p2.shape[0]
    out = np.zeros( (n1, 3) )
    for i in range(n1):
        if p1[i, 0] is np.nan or p1[i, 1] is np.nan or p1[i, 2] is np.nan or p2[i, 0] is np.nan or p2[i, 1] is np.nan or p2[i, 2] is np.nan:
            out[i, 0] = np.nan
            out[i, 1] = np.nan
            out[i, 2] = np.nan
        else:
            d1 = p1[i, 0] - p2[i, 0]
            d2 = p1[i, 1] - p2[i, 1]
            d3 = p1[i, 2] - p2[i, 2]

            offset1 = round( inv_pbc_view[0, 0] * d1 + inv_pbc_view[1, 0] * d2 + inv_pbc_view[2, 0] * d3 )
            offset2 = round( inv_pbc_view[0, 1] * d1 + inv_pbc_view[1, 1] * d2 + inv_pbc_view[2, 1] * d3 )
            offset3 = round( inv_pbc_view[0, 2] * d1 + inv_pbc_view[1, 2] * d2 + inv_pbc_view[2, 2] * d3 )

            out[i, 0] = d1 - pbc_view[0, 0] * offset1 - pbc_view[1, 0] * offset2 - pbc_view[2, 0] * offset3
            out[i, 1] = d2 - pbc_view[0, 1] * offset1 - pbc_view[1, 1] * offset2 - pbc_view[2, 1] * offset3
            out[i, 2] = d3 - pbc_view[0, 2] * offset1 - pbc_view[1, 2] * offset2 - pbc_view[2, 2] * offset3
    return out

@boost
def calculate_msd( proton_positions : "float[][][]" ):
    msd_xyz = np.mean( np.square( proton_positions - proton_positions[0, :, :] ) , axis = 1 )
    return msd_xyz

@boost
def calculate_msd_averaged( proton_positions : "float[][][]" ):
    l = proton_positions.shape[0]
    msd_xyz = np.zeros( ( l, 3 ) )
    for i in range(1, l):
        start_positions = proton_positions[ :-i ]
        end_positions = proton_positions[ i: ]
        diff = end_positions - start_positions
        squared_displacements = np.square( diff )
        msd_xyz[i] = np.mean( np.mean( squared_displacements, axis = 0), axis = 0 )
    return msd_xyz

@boost
def calculate_auto_correlation( lattice_over_time : "uint8[][]" ):
    original_bonds = ( lattice_over_time == lattice_over_time[0] ) & ( lattice_over_time[0] > 0 )
    return np.sum( original_bonds , axis = 1)

@boost
def print_observables(sweep : "int", sweeps : "int", print_freq : "int", md_timestep : "float", mean_square_displacement : "float[][]", jumps_over_time : "int[]", auto_correlation : "int[]"):
    if mean_square_displacement.shape == (4, 3):
        msd2, msd3, msd4 = mean_square_displacement[1:]
        msd_higher = "{msd2.sum():18.8f} {msd3.sum():18.8f} {msd4.sum():18.8f}"
    else:
        msd_higher = ""

    no_of_prints = auto_correlation.shape[0]
    output_string = ""
    for i in range(no_of_prints):
        msd_x, msd_y, msd_z = mean_square_displacement[i, :]
        autocorr = auto_correlation[i]
        jumps = jumps_over_time[i]
        current_sweep = sweep + i * print_freq #- no_of_prints * print_freq + i * print_freq
        simulation_time = current_sweep * md_timestep
        output_string += f"{current_sweep:10d} " +\
                        f"{simulation_time:10f} " +\
                        f"{msd_x:18.8f} " +\
                        f"{msd_y:18.8f} " +\
                        f"{msd_z:18.8f} " +\
                        msd_higher +" "+\
                        f"{autocorr:8d} " +\
                        f"{jumps:10d} " +\
                        f"{no_of_prints:d}"+\
                        "\n"
    return output_string

def diff_coef(t, D, n):
    return 6 * D * t + n


def process_lmc_results(sweeps, reset_freq, print_freq, md_timestep_fs, msd_xyz, autocorrelation):
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # path_lmc = "lmc2.out"
    number_of_intervals = int(sweeps / reset_freq)
    length_of_interval = int(reset_freq / print_freq)
    dt_lmc = md_timestep_fs/1000
    
    time_for_msd = np.arange( length_of_interval ) * print_freq * md_timestep_fs / 1000 # time written in lmc.out converted from fs to ps
    msd_mean = np.sum(msd_xyz, axis = 1)
    np.savetxt('msd_from_lmc.out', np.column_stack((time_for_msd, msd_mean)), header="t / ps\tMSD / A^2")
    np.savetxt('msd_xyz_from_lmc.out', np.column_stack((time_for_msd, msd_xyz)), header="t / ps\tMSD / A^2")

    # Create times for the MSD values and fit linear
    x_lmc = time_for_msd
    y_lmc = np.sum(msd_xyz, axis = 1)
    print("diff coeff lmc in A**2/ps:")
    len1 = x_lmc.shape[0]
    fit_x_lmc = x_lmc[int(len1 * 0.2) : int(len1 * 0.7)]
    fit_y_lmc = y_lmc[int(len1 * 0.2) : int(len1 * 0.7)]
    popt_lmc, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc)
    print(popt_lmc[0])
    
    # Plot MSD and linear fit
    fig, ax = plt.subplots()
    ax.plot(x_lmc, y_lmc, label='cMD/LMC')
    ax.plot(x_lmc, diff_coef(x_lmc, popt_lmc[0], popt_lmc[1]), label='fit cMD/LMC')
    ax.set(xlabel='time [ps]', ylabel='MSD  [$\AA^2$]',
           title='Comparison of the MSDs from AIMD and cMD/LMC')
    legend = ax.legend(loc = 'best')
    ax.grid()
    fig.savefig("msd_lmc.png")

    # Slice autocorrelation and save it as well
    np.savetxt('autocorr_from_lmc.out', np.column_stack((time_for_msd, autocorrelation)), header="t / ps\tautocorrelation")

    # For experiment's sake write D to file
    with open("D_random", "a+") as f:
        f.write(f"{popt_lmc[0]}\n")

    fig.savefig("msd_lmc.png")

def post_process_lmc(sweeps, reset_freq, print_freq, md_timestep_fs, path_lmc):
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # path_lmc = "lmc2.out"
    number_of_intervals = int(sweeps / reset_freq)
    length_of_interval = int(reset_freq / print_freq)
    dt_lmc = md_timestep_fs/1000
    
    # Read lmc output file
    #raw_lmc_data = np.loadtxt(path_lmc, skiprows=0, usecols=(0,1,2,3,4,5))
    raw_lmc_data = np.loadtxt(path_lmc, skiprows=1, usecols=(0,1,2,3,4,5))

    # Slice out MSD_x, MSD_y and MSD_z
    tmp1 = np.reshape(raw_lmc_data[:,2:5], (int(number_of_intervals),int(length_of_interval),3))
    tmp1 = np.sum(tmp1, axis=2)
    msd_mean = np.mean(tmp1, axis=0)
    time_for_msd = raw_lmc_data[:length_of_interval, 1] / 1000 # time written in lmc.out converted from fs to ps
    np.savetxt('msd_from_lmc.out', np.column_stack((time_for_msd, msd_mean)), header="t / ps\tMSD / A^2")

    # Create times for the MSD values and fit linear
    x_lmc = raw_lmc_data[:length_of_interval,0]*dt_lmc
    y_lmc = msd_mean
    print("diff coeff lmc in A**2/ps:")
    len1 = x_lmc.shape[0]
    fit_x_lmc = x_lmc[int(len1 * 0.2) : int(len1 * 0.7)]
    fit_y_lmc = y_lmc[int(len1 * 0.2) : int(len1 * 0.7)]
    popt_lmc, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc)
    print(popt_lmc[0])
    
    # Plot MSD and linear fit
    fig, ax = plt.subplots()
    ax.plot(x_lmc, y_lmc, label='cMD/LMC')
    ax.plot(x_lmc, diff_coef(x_lmc, popt_lmc[0], popt_lmc[1]), label='fit cMD/LMC')
    ax.set(xlabel='time [ps]', ylabel='MSD  [$\AA^2$]',
           title='Comparison of the MSDs from AIMD and cMD/LMC')
    legend = ax.legend(loc = 'best')
    ax.grid()
    fig.savefig("msd_lmc.png")

    # Slice autocorrelation and save it as well
    autocorr = raw_lmc_data[:,5]
    autocorr = np.reshape(autocorr, (number_of_intervals, length_of_interval))
    autocorr = np.mean(autocorr, axis = 0)
    np.savetxt('autocorr_from_lmc.out', np.column_stack((time_for_msd, autocorr)), header="t / ps\tautocorrelation")

    # For experiment's sake write D to file
    with open("D_random", "a+") as f:
        f.write(f"{popt_lmc[0]}\n")

    fig.savefig("msd_lmc.png")



def main():
        logging.basicConfig(filename='misp.log',  level=logging.DEBUG)
        logging.debug('command line:')
        logging.debug(sys.argv)
        #logging.debug(readwrite.get_git_version())
        logging.debug(readwrite.log_git_version())
        
        readwrite.get_git_version()
        parser = argparse.ArgumentParser('This script simulates long range ion transfer. It requires a (possibly time dependet) grid and a (possibly time dependent) jump rate matrix. Standard cmd/lmc simulations are performed is parameter proton_propagation is set true. For this case pickle_jump_mat_proc.p is the jump matrix. Otherwise pickle_count_jump_mat.p is used to calculate ion transfer probabilities. ')
        parser.add_argument("path1", help="path to xyz trajec")
        parser.add_argument("pbc", help="path to pbc numpy mat")
        parser.add_argument("com", help="remove center of mass movement" , type=boolean_string)
        parser.add_argument("wrap", help="wrap trajectory, unwrap trajectory or do nothing", choices=["wrap", "unwrap", "nowrap"])
        parser.add_argument("lattice_atoms", help="lattice atoms", nargs='+')
        parser.add_argument("noji", help="number of jumping ions", type = int)
        parser.add_argument("md_timestep", help="timestep of underlying MD simulation in fs", type = float)
        parser.add_argument("equilibration_sweeps", help="number of equilibration sweeps", type = int)
        parser.add_argument("sweeps", help="number of sweeps", type = int)
        parser.add_argument("output", help="name of output file")
        parser.add_argument("reset_freq", help="set reset frequency", type = int)
        parser.add_argument("print_freq", help="set print frequency", type = int)
        parser.add_argument("proton_propagation", help="if True standard lmc potocol is applied, if False lithium dynamics on a fixed grid is simulated" , type=boolean_string)

        parser.add_argument("--verbose", help="verbosity?" , action='store_true')
        parser.add_argument("--seed", type=int, help="Predetermined seed for reproducable runs")
        parser.add_argument("--write_xyz", help="print only msd or xyz trajectory, if True xyz trajectory is written" , action="store_true")
        parser.add_argument("--custom", help = "use custom_lmc.py to alter trajectory or jump probability", action = "store_true")
        parser.add_argument("--clip", help = "clip trajectory by discarding timesteps with a higher index")

        args = parser.parse_args()
        pbc_mat = np.loadtxt(args.pbc)
        noji = args.noji
        coord, atom = readwrite.easy_read(args.path1, pbc_mat, args.com, args.wrap)
        if args.custom:
            sys.path.append(os.getcwd())
            print(f"Loading function prepare_trajectory from {os.getcwd()}custom_lmc.py")
            from custom_lmc import prepare_trajectory
            prepare_trajectory(coord, atom, pbc_mat)
#            from custom_lmc import prepare_custom_prob
#            prepare_custom_prob(helper)

        coord_o = coord[:, np.isin(atom, args.lattice_atoms), :]
        if args.proton_propagation:
            jump_mat = pickle.load(open( "pickle_jump_mat_proc.p", "rb" ))
        else:
            jump_mat = pickle.load(open( "pickle_count_jump_mat.p", "rb"))
            tmp = sparse.coo_matrix(jump_mat[0].shape)
            for i in range(len(jump_mat)):
                tmp += jump_mat[i]
            tmp = tmp/ len(jump_mat)
            jump_mat = np.array([tmp.tocoo()])

        if args.clip:
            coord = coord[:int(args.clip)]
            jump_mat = jump_mat[:int(args.clip)]

        if len(jump_mat) !=  coord.shape[0]:
            print("Warning! number of steps is not equal for jumpmatrix and trajectory")
        rng = np.random.default_rng(args.seed)   # If no seed is specified, default_rng(None) will use OS entropy
        np.random.seed(args.seed)

        nols = np.count_nonzero( np.isin(atom, args.lattice_atoms) ) #number of lattice sites

        settings = SettingsManager(noji = noji,
                                   nols = nols,
                                   sweeps = args.sweeps,
                                   equilibration_sweeps = args.equilibration_sweeps,
                                   verbose = args.verbose,
                                   xyz_output = args.write_xyz,
                                   print_freq = args.print_freq,
                                   reset_freq = args.reset_freq,
                                   seed = args.seed if args.seed is not None else 0)
        #breakpoint()
        lattice = initialize_oxygen_lattice(nols, noji, rng)
        helper= Helper(jump_mat, noji, nols)

        # open output file, thus deleting old one
        output = open( args.output, "w+" )
        if not settings.xyz_output:
            print(
                "#     Sweeps       Time                 MSD_x              MSD_y              "
                "MSD_z Autocorr      Jumps   Sweeps/Sec",
                file=output,
            )
        output.close()

        msd_lmc, autocorr_lmc, jump_mat_recalc = cmd_lmc_run(coord_o, pbc_mat, lattice, helper, settings, args.md_timestep, args.output)
        if args.write_xyz:
            coord_lmc, atom_lmc = readwrite.easy_read("lmc.out", pbc_mat, False, False)    
            atom_cut = atom[np.isin(atom, args.lattice_atoms)]
            atom_cut = list(atom_cut) + ["H"] * noji 
            print(atom_cut, len(atom_cut), coord_lmc.shape)
            readwrite.easy_write(coord_lmc, np.array(atom_cut),"lmc_final.xyz")
        else:    
            process_lmc_results(settings.sweeps, settings.reset_freq, settings.print_freq, args.md_timestep, msd_lmc, autocorr_lmc)

def print_observable_names(self):
    if self.variance_per_proton:
        print(
            "#     Sweeps       Time              MSD_x              MSD_y              MSD_z "
            "           MSD_x_var          MSD_y_var          MSD_z_var Autocorr      Jumps   "
            "Sweeps/Sec",
            file=self.output,
        )
    else:
        print(
            "#     Sweeps       Time                 MSD_x              MSD_y              "
            "MSD_z Autocorr      Jumps   Sweeps/Sec",
            file=self.output,
        )
