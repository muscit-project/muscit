import argparse
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse


from trajec_io import readwrite


@dataclass
class SettingsManager:
    noji : int
    nols : int
    md_timestep : float
    sweeps : int
    equilibration_sweeps : int
    verbose : bool
    xyz_output : bool
    print_freq : bool
    reset_freq : bool
    output_path : str
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
    def __init__(self, jump_mat, noji, nols):
        self.jump_mat = jump_mat
        self.noji = noji
        self.nols = nols
        self.jumps = 0
        self.jump_mat_recalc = np.zeros((nols, nols))


    def get_jump_probability(self, frame, lattice):
        return np.copy(self.jump_mat[frame].todense())


    #def experimental_sweep(self, frame, lattice, rng):
    def sweep(self, frame, lattice, rng):
        # Dense matrix of jump probabilities
        prob = self.get_jump_probability(frame, lattice)
        # Random array of shape noli x noli (same as prob)
        rand_mat1 = rng.uniform(0,1, self.nols **2).reshape(self.nols, self.nols)
        prob[rand_mat1 >  prob] = 0    # prevent jumps based on their probability
        prob[lattice != 0, :] = 0       # prevent jumps if destination site is occupied at the beginning of sweep
        prob[:, lattice == 0] = 0       # prevent jumps if souce site is empty at the beginning of sweep

        #L = [1,2,3,4,5,6]
        #i = random.randrange(len(L)) # get random index
        #L[i], L[-1] = L[-1], L[i]    # swap with the last element
        #x = L.pop()                  # pop last element O(1)


        destination, source = np.where(prob)
        destination = list(destination)
        source = list(source)
        banned_destination =[]
        banned_source=[]
        while source:
            i = rng.integers(len(source)) # get random index
            source[i], source[-1] = source[-1], source[i]
            destination[i], destination[-1] = destination[-1], destination[i]
            final_source = source.pop()
            final_destination = destination.pop()
            if (final_source not in banned_source) and (final_destination not in banned_destination):
                banned_destination.append(final_destination)
                banned_source.append(final_source)
                lattice[final_destination] = lattice[final_source]
                lattice[final_source] = 0
                self.jumps += 1
                self.jump_mat_recalc[final_destination , final_source ] += 1
            #else:
            #     banned_destination.append(final_destination)
            #     banned_source.append(final_source)
#

    def asweep(self, frame, lattice, rng):
    #def sweep(self, frame, lattice, rng):
        # Dense matrix of jump probabilities
        prob = self.get_jump_probability(frame, lattice)
        # Random array of shape noli x noli (same as prob)
        rand_mat1 = rng.uniform(0,1, self.nols **2).reshape(self.nols, self.nols)
        prob[rand_mat1 >  prob] = 0    # prevent jumps based on their probability
        prob[lattice != 0, :] = 0       # prevent jumps if destination site is occupied at the beginning of sweep
        prob[:, lattice == 0] = 0       # prevent jumps if souce site is empty at the beginning of sweep
        
        #breakpoint()
        
        for j in range(prob.shape[0]):
         parallel_destinations= np.nonzero(prob[j,:])[0]
         if len(parallel_destinations) > 1:
            dest_index = rng.choice(parallel_destinations)
            tmp = prob[j, dest_index] 
            prob[j,:] = 0
            prob[j, dest_index] = tmp
        
        #breakpoint()

        destination, source = np.where(prob)
        for specific_source in set(source):
            possible_destinations = destination[ source == specific_source ]
            final_destination = rng.choice( possible_destinations )
            final_source = specific_source
            lattice[final_destination] = lattice[specific_source]
            lattice[specific_source] = 0
            self.jumps += 1
            self.jump_mat_recalc[final_destination , final_source ] += 1
        #breakpoint()



    def sweep_old(self, frame, lattice, rng):
        start_lattice = np.copy(lattice)
        jump_frame = self.jump_mat[frame]
        destinations, sources, probabilities = jump_frame.row, jump_frame.col, jump_frame.data
        no_of_pairs = len(probabilities)
        for i in range(no_of_pairs):
            random_pair = rng.integers(0, no_of_pairs)
            destination, source, probability = destinations[random_pair], sources[random_pair], probabilities[random_pair]
            if rng.uniform(0,1) < probability:
                if lattice[destination] == 0 and lattice[source] != 0:
                    if start_lattice[destination] == 0 and start_lattice[source] != 0:
                        lattice[destination] = lattice[source]
                        lattice[source] = 0
                        self.jumps += 1
                        self.jump_mat_recalc[destination , source ] += 1

def cmd_lmc_run(oxygen_trajectory, oxygen_lattice, helper, observable_manager, settings, rng):
    """Main function. """
    verbose = settings.verbose

    # Equilibration
    for sweep in range(settings.equilibration_sweeps):
        #if not settings.xyz_output:
        #    if sweep % 1000 == 0:
        #        print("# Equilibration sweep {}/{}".format(sweep, settings.equilibration_sweeps), end='\r', file=settings.output)
        #breakpoint()
        helper.sweep(sweep % oxygen_trajectory.shape[0], oxygen_lattice, rng)
        #breakpoint()
    if not settings.xyz_output:
        observable_manager.print_observable_names()
    #breakpoint()
    # Run
    observable_manager.start_timer()
    for sweep in range(0, settings.sweeps):
        frame = sweep % oxygen_trajectory.shape[0]
        if sweep % settings.reset_freq == 0:
            observable_manager.reset_observables(frame)

        if sweep % settings.print_freq == 0:
            if settings.xyz_output:
                #breakpoint()
                observable_manager.print_xyz(
                    oxygen_trajectory[frame], oxygen_lattice, sweep
                )
            else:
                observable_manager.calculate_displacement(frame)
                observable_manager.calculate_msd()
                observable_manager.calculate_auto_correlation()
                observable_manager.print_observables(sweep)

        # if settings.jumpmatrix_filename is not None:
        #    helper.sweep_with_jumpmatrix(frame, oxygen_lattice)
        #else:
        #helper.sweep_old(frame, oxygen_lattice, rng)
        helper.sweep(frame, oxygen_lattice, rng)
        #print(oxygen_lattice)
        if sweep % 100 == 0:
            print(sweep, " / ", settings.sweeps, end = '\r')

    # if settings.jumpmatrix_filename is not None:
    #    np.savetxt(settings.jumpmatrix_filename, helper.jumpmatrix)
    np.savetxt("jump_mat_sampled_from_lmc_run", helper.jump_mat_recalc)
    np.savetxt("overall_number_of_jumps", np.array([helper.jump_mat_recalc.sum()]))
    print("overall_number_of_jumps: " + str(helper.jump_mat_recalc.sum()))

def diff_coef(t, D, n):
    return 6 * D * t + n


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
        parser.add_argument("--clip", type = int)
        parser.add_argument("--pickle_matrix", help = "pickled list of sparse matrices containing jump probabilities if proton_propagation is true or jump counts if proton_propagation is false")

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

        if args.pickle_matrix is None:
            pickle_path = "pickle_jump_mat_proc.p" if args.proton_propagation is True else "pickle_count_jump_mat.p"
        else:
            pickle_path = args.pickle_matrix
        if args.proton_propagation:
            jump_mat = pickle.load(open( pickle_path, "rb" ))
        else:
            jump_mat = pickle.load(open( pickle_path, "rb"))
            tmp = sparse.coo_matrix(jump_mat[0].shape)
            for i in range(len(jump_mat)):
                tmp += jump_mat[i]
            tmp = tmp/ len(jump_mat)
            jump_mat = np.array([tmp.tocoo()])

        if args.clip:
            coord = coord[:args.clip, :, :]
            jump_mat = jump_mat[:args.clip]
        #coord_o = coord[:, atom == args.lattice_atoms, :]
        coord_o = coord[:, np.isin(atom, args.lattice_atoms), :]
        if len(jump_mat) !=  coord_o.shape[0]:
            print("Warning! number of steps is not equal for jumpmatrix and trajectory")
        rng = np.random.default_rng(args.seed)   # If no seed is specified, default_rng(None) will use OS entropy


        nols = coord_o.shape[1] #number of lattice sites

        settings = SettingsManager(noji, nols, args.md_timestep, args.sweeps, args.equilibration_sweeps, args.verbose, args.write_xyz, args.print_freq, args.reset_freq, args.output, args.seed)
        #breakpoint()
        lattice = initialize_oxygen_lattice(nols, noji, rng)
        helper= Helper(jump_mat, noji, nols)
        output = open(settings.output_path, "w+")
        observable_manager =  ObservableManager(coord_o, lattice, nols, noji, settings.md_timestep, settings.sweeps, pbc_mat, helper, msd_mode=None, variance_per_proton=False, output=output)

        cmd_lmc_run(coord_o, lattice, helper, observable_manager, settings, rng)
        output.close()
        if args.write_xyz:

            coord_lmc, atom_lmc = readwrite.easy_read("lmc.out", pbc_mat, False, False)    
            atom_cut = atom[np.isin(atom, args.lattice_atoms)]
            atom_cut = list(atom_cut) + ["H"] * noji 
            print(atom_cut, len(atom_cut), coord_lmc.shape)
            readwrite.easy_write(coord_lmc, np.array(atom_cut),"lmc_final.xyz")
        else:    
            post_process_lmc(settings.sweeps, settings.reset_freq, settings.print_freq, settings.md_timestep, args.output)



class ObservableManager:
    def __init__(
        self,
        coord_o,
        lattice,
        nols,
        noji,
        md_timestep,
        sweeps,
        pbc,
        helper,
        msd_mode=None,
        variance_per_proton=False,
        output=sys.stdout,
    ):
        self.oxygen_trajectory = coord_o
        self.oxygen_lattice = lattice
        self.proton_number = noji
        self.md_timestep = md_timestep
        self.sweeps = sweeps
        self.output = output
        self.variance_per_proton = variance_per_proton
        self.displacement = np.zeros((self.proton_number, 3))
        self.proton_pos_snapshot = np.zeros((self.proton_number, 3))
        self.oxygen_lattice_snapshot = np.array(lattice)
        self.pbc = pbc
        self.helper = helper
        self.output = output

        self.format_strings = [
            "{:10d}",  # Sweeps
            "{:>10}",  # Time steps
            "{:18.8f}",  # MSD x component
            "{:18.8f}",  # MSD y component
            "{:18.8f}",  # MSD z component
            "{:}",  # MSD higher order
            "{:8d}",  # OH bond autocorrelation
            "{:10d}",  # Number of proton jumps
            "{:10.2f}",  # Simulation speed
            "{:}",
        ]  # Remaining time

        if msd_mode == "higher_msd":
            self.mean_square_displacement = np.zeros((4, 3))
            self.msd_variance = np.zeros((4, 3))
            self.calculate_msd = self.calculate_msd_higher_orders
        else:
            self.mean_square_displacement = np.zeros((1, 3))
            self.msd_variance = np.zeros(3)
            self.calculate_msd = self.calculate_msd_standard

    def calculate_displacement(self, frame):
        proton_pos_new = np.zeros(self.proton_pos_snapshot.shape)
        for oxygen_index, proton_index in enumerate(self.oxygen_lattice):
            # proton_pos_new = self.oxygen_trajectory[frame, self.oxygen_lattice =!= 0, :]
            if proton_index > 0:
                proton_pos_new[proton_index - 1] = self.oxygen_trajectory[
                    frame, oxygen_index, :
                ]
        self.displacement += (
            readwrite.pbc_dist2(self.proton_pos_snapshot, proton_pos_new, self.pbc)
            .diagonal(0, 0, 1)
            .T
        )
        self.proton_pos_snapshot[:] = proton_pos_new

    def calculate_msd_standard(self):
        self.mean_square_displacement[:] = (self.displacement ** 2).sum(
            axis=0
        ) / self.displacement.shape[0]
        return self.mean_square_displacement

    def calculate_msd_higher_orders(self):
        self.mean_square_displacement[:] = 0
        self.mean_square_displacement[0] = (self.displacement ** 2).sum(axis=0)
        self.mean_square_displacement[1] = self.mean_square_displacement[0].sum() ** 0.5
        self.mean_square_displacement[2] = self.mean_square_displacement[1].sum() ** 3
        self.mean_square_displacement[3] = self.mean_square_displacement[1].sum() ** 4
        self.mean_square_displacement /= self.displacement.shape[0]

        return self.mean_square_displacement

    def calculate_auto_correlation(self):
        self.autocorrelation = np.logical_and(
            self.oxygen_lattice == self.oxygen_lattice_snapshot,
            self.oxygen_lattice != 0,
        ).sum()

    def reset_observables(self, frame):
        for oxy_ind, prot_ind in enumerate(self.oxygen_lattice):
            if prot_ind > 0:
                self.proton_pos_snapshot[prot_ind - 1] = self.oxygen_trajectory[
                    frame, oxy_ind, :
                ]
        self.oxygen_lattice_snapshot = np.copy(self.oxygen_lattice)
        self.displacement[:] = 0
        self.helper.jumps = 0

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
            # file=settings.output)

    def print_observables(self, sweep):
        speed = float(sweep) / (time.time() - self.start_time)
        if sweep != 0:
            remaining_time_hours = int((self.sweeps - sweep) / speed / 3600)
            remaining_time_min = int((((self.sweeps - sweep) / speed) % 3600) / 60)
            remaining_time = "{:02d}:{:02d}".format(
                remaining_time_hours, remaining_time_min
            )
        else:
            remaining_time = "-01:-01"
        if self.mean_square_displacement.shape == (4, 3):
            msd2, msd3, msd4 = self.mean_square_displacement[1:]
            msd_higher = "{:18.8f} {:18.8f} {:18.8f}".format(
                msd2.sum(), msd3.sum(), msd4.sum()
            )
        else:
            msd_higher = ""

        jump_counter = self.helper.jumps

        output = (
            sweep,
            sweep * self.md_timestep,
            *self.mean_square_displacement[0],
            msd_higher,
            self.autocorrelation,
            jump_counter,
            speed,
            remaining_time,
        )
        with open("lmc.out", "a") as f:
            for i, (fmt_str, value) in enumerate(zip(self.format_strings, output)):
                print(fmt_str.format(value), end=" ", file=self.output)
            print(file=self.output)
        #with open("lmc.out", "a") as f:
        #    for i, (fmt_str, value) in enumerate(zip(self.format_strings, output)):
        #        print(fmt_str.format(value), end=" ", file=f)
        #    print(file=f)

        # for i, (fmt_str, value) in enumerate(zip(self.format_strings, output)):
        #    print(fmt_str.format(value), end=" ", file=self.output)
        #    #print(fmt_str.format(value), end=" ", file=settings.output)
        # print(file=self.output)
        # print(file=settings.output)

    def start_timer(self):
        self.start_time = time.time()

    def print_observables_var(
        self,
        sweep,
        autocorrelation,
        helper,
        timestep_fs,
        start_time,
        MSD,
        msd_var,
        msd2=None,
        msd3=None,
        msd4=None,
    ):
        speed = float(sweep) / (time.time() - start_time)
        if sweep != 0:
            remaining_time_hours = int((self.sweeps - sweep) / speed / 3600)
            remaining_time_min = int((((self.sweeps - sweep) / speed) % 3600) / 60)
            remaining_time = "{:02d}:{:02d}".format(
                remaining_time_hours, remaining_time_min
            )
        else:
            remaining_time = "-01:-01"
        if (msd2, msd3, msd4) != (None, None, None):
            msd_higher = "{:18.8f} {:18.8f} {:18.8f}".format(msd2, msd3, msd4)
        else:
            msd_higher = ""
        # print(" {:>10} {:>10}    "
        #      "{:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f}  "
        #      "{msd_higher:}  "
        #      "{:8d} {:10d} {:10.2f} {:10}".format(sweep, sweep * timestep_fs,
        #                                           MSD[0], MSD[1], MSD[2],
        #                                           msd_var[0], msd_var[1], msd_var[2],
        #                                           autocorrelation, helper.get_jumps(), speed,
        #                                           remaining_time, msd_higher=msd_higher),
        #      file=self.output)
        # with open (proxy_file, 'a') as f:
        # with open ("lmc.out", 'a') as f:
        # print(" {:>10} {:>10}    "
        #      "{:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f} {:18.8f}  "
        ##      "{msd_higher:}  "
        #      "{:8d} {:10d} {:10.2f} {:10}".format(sweep, sweep * timestep_fs,
        #                                           MSD[0], MSD[1], MSD[2],
        #                                           msd_var[0], msd_var[1], msd_var[2],
        #                                           autocorrelation, helper.get_jumps(), speed,
        #                                          remaining_time, msd_higher=msd_higher))
        #     #file=f)

        self.averaged_results[(sweep % self.reset_freq) / self.print_freq, 2:] += (
            MSD[0],
            MSD[1],
            MSD[2],
            autocorrelation,
            helper.get_jumps(),
        )

    def print_xyz(self, Os, oxygen_lattice, sweep):
        #breakpoint()
        #breakpoint()
        #print(oxygen_lattice, " in print step")
        #proton_indices = np.where(oxygen_lattice > 0)[0]
        sorter = np.argsort(oxygen_lattice)
        sort1 = np.arange(np.count_nonzero(oxygen_lattice > 0)) + 1
        proton_indices = sorter[np.searchsorted(oxygen_lattice, sort1, sorter=sorter)]
        #print(proton_indices, " proton indices")
        print(Os.shape[0] + self.proton_number, file=self.output)
        print("Time:", sweep * self.md_timestep, file=self.output)
        for i in range(Os.shape[0]):
            print(
                "O        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[i]),
                file=self.output,
            )
        for index in proton_indices:
            print(
                "H        {:20.8f}   {:20.8f}   {:20.8f}".format(*Os[index]),
                file=self.output,
            )
