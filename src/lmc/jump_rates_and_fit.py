import argparse
import logging
import os
import pickle
import sys
import time
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix

from trajec_io import readwrite
from lmc.prepare_lmc import Settings, AnalysisHelper, boolean_string

#def boolean_string(s):
#    if s not in {"False", "True"}:
#        raise ValueError("Not a valid boolean string")
#    return s == "True"

def main():
        #print("command line:")
        #print(sys.argv)
        #readwrite.get_git_version()
        readwrite.start_logging()
        logging.info(sys.argv)
        parser = argparse.ArgumentParser('This script calculates the jump rate function from the output of the  evaluate_jumps_on_grid script. If --fit is chosen, a fermi fit to the jump rate funtion is applied')
        parser.add_argument("path", help="path to trajek")
        parser.add_argument("pbc_path", help="path to pbc numpy mat")
        parser.add_argument("com", help="remove center of mass movement" , type=boolean_string, choices=[True, False])
        parser.add_argument("wrap", help="wrap trajectory, unwrap trajectory or do nothing", choices=["wrap", "unwrap", "nowrap"])
        parser.add_argument("lattice_types", help="specify atom type for grid", nargs='+')
        parser.add_argument("--jumping_ion_type", help="which atoms are transfered?", default = "H")
        parser.add_argument("--speed", help="every i-th step from trajec is used for neighbor mat", type = int, default = 1)
        parser.add_argument("--angle_atoms", help="atoms for angle definiton", nargs = '+')
        parser.add_argument("--normal_occupation", help="usual number of ions attached to each lattice point", nargs = '+')
    #    parser.add_argument("--dynamic",  action="store_true", help="update grid from trajectory in every step")
    #    parser.add_argument("--asymmetric", action="store_true", help = "calculate jump probabilities asymmetrically")
        parser.add_argument("--custom", action="store_true", help = "use custom_lmc.py for custom function prepare_trajectory")
        parser.add_argument("--jumpfile",  help="npy-file containing information on all jumps generated by create_jump_mat_li (default: jump_info.npy)", default="jump_info.npy")
        parser.add_argument("--neighfile",  help="npy-file containing information on all next-neighbors  of the jumping ion generated by create_jump_mat_li", default="jumping_ion_neighbors.npy")
        parser.add_argument("--fermi_out", help="path to store fitted fermi parameters in", default = "fermi_param")
        parser.add_argument("--distances_out", help = "path to store distances (as pickle object) in", default = "pickle_dist_mat_angle.p" )
        parser.add_argument("--fit",  action="store_true", help="fit probability curve")

        args = parser.parse_args()
        # Load pbc, trajectory and fixed lattice coordinates if supplied
        pbc_mat = np.loadtxt(args.pbc_path)
        traj = readwrite.easy_read(args.path, pbc_mat, args.com, args.wrap)
        # put passed arguments into Settings object
        fixed_lattice = None
        settings = Settings(args.jumping_ion_type, args.lattice_types, args.angle_atoms, args.normal_occupation, args.speed, fixed_lattice)

        # apply custom function to trajectory if supplied
        if args.custom:
            sys.path.append(os.getcwd())
            print(f"Loading function prepare_trajectory from {os.getcwd()}custom_lmc.py")
            from custom_lmc import prepare_trajectory
            prepare_trajectory(traj)

        # construct AnalysisHelper and calculate everything else from it
        analysishelper = AnalysisHelper(traj, settings)
        #jump_info = find_jumps( analysishelper, fixed_lattice )
        find_jumpprobs_and_fit(analysishelper, neighborfile = args.neighfile, jumpfile = args.jumpfile, fermi_file = args.fermi_out, out_distances = args.distances_out, fit = args.fit )

def fermi(x,a,b,c):
    return a/(1+ np.exp((x-b)*c))

def fit_jump_hist_to_fermi_old(distance_probability_histogram, identifier, fermi_guess = (0.04, 2.3, 30), start_fit = 25):
            #path_hist = "oo_jump_histogram"
            hist_data = distance_probability_histogram[:,0:2]
            popt, pcov = curve_fit(fermi, hist_data[start_fit:,0], hist_data[start_fit:,1], p0 = fermi_guess)
            np.savetxt("fermi_param_" + identifier, popt)
            fig, ax = plt.subplots()
            ax.plot(hist_data[start_fit:,0], hist_data[start_fit:,1], label = 'jump prob from AIMD')
            ax.plot(hist_data[start_fit:,0], fermi(hist_data[start_fit:,0], popt[0], popt[1], popt[2]), label='fermi fit') 
            ax.set(xlabel='OO distance [A]', ylabel='prob  [fs-1]',
                   title='Fermi fit of jump probabilites')
            legend = ax.legend(loc = 'best')
            ax.grid()
            fig.savefig("eval_jump_fit_" + identifier + ".png")

def fit_jump_hist_to_fermi(distance_probability_histogram, identifier, fermi_guess = (0.04, 2.3, 30), start_fit = 25):
            #hist_data = distance_probability_histogram[:,0:2]
            hist_data = distance_probability_histogram
            #popt, pcov = curve_fit(fermi, hist_data[start_fit:,0], hist_data[start_fit:,1], p0 = fermi_guess)
            #fit_fermi(average_d, jump_probs, lattice_distances_hist, fermi_guess = (0.04, 2.3, 30), start_fit = 25)            
            jump_probs = {}
            jump_probs[identifier] = hist_data[:,1]
            average_d = hist_data[:,0]
            lattice_distances_hist = {}
            lattice_distances_hist[identifier] = hist_data[:,2]
            fermi_params, fermi_params_weighted = fit_fermi(average_d, jump_probs, lattice_distances_hist, fermi_guess, start_fit)
            #fermi_params, fermi_params_weighted
            print(fermi_params)
            print(fermi_params_weighted)
            for key  in fermi_params.keys():
                #np.savetxt("fermi_param_no_weight" + identifier, fermi_params[key])
                np.savetxt("fermi_param_no_weight" + key, fermi_params[key])
            for key  in fermi_params_weighted.keys():
                #np.savetxt("fermi_param_weight" + identifier, fermi_params_weighted[key])
                np.savetxt("fermi_param_weight" + key, fermi_params_weighted[key])

def calculate_and_fit_jump_distances(coord, atom, pbc_mat, lattice_types, fit = True, jumpfile = "jump_info.npy", nonsymmetric = False, neighfile = "jumping_ion_neighbors.npy", angle_types = None, normal_occupation = None, jumping_ion_type = "H"):
    # fit and nonsymmetric a re ignored
    traj = readwrite.Trajectory(coord, atom, pbc_mat)
    settings = Settings(jumping_ion_type, lattice_types, angle_types = angle_types, normal_occupation = normal_occupation, speed = 1, fixed_lattice = None)
    helper = AnalysisHelper(traj, settings)
    return find_jumpprobs_and_fit(helper, neighfile, jumpfile, fermi_file = "fermi_param", fit = fit)

def find_jumpprobs_and_fit(analysishelper, neighborfile = "jumping_ion_neighbors.npy", jumpfile = "jump_info.npy", fermi_file = "fermi_param", out_distances = "pickle_dist_mat_angle.p", fit = True):
    # details of the distance and propability histograms
    bins = 100
    dlimits = (2, 3.0)
    d = np.histogram( [], range = dlimits, bins = bins)[1]
    average_d = (d[1:] + d[:-1])/2
    
    # get protonation states
    try:
        ion_neighbors = np.load(neighborfile)
    except (FileNotFoundError, TypeError):
        ion_neighbors = neighbors_in_traj(analysishelper)
    protonations = neighbors_to_protonation(ion_neighbors, analysishelper)
    try:
        jump_info = np.load(jumpfile)
    except (FileNotFoundError, TypeError):
        jump_info = find_jumps( analysishelper, neighbors = ion_neighbors)

    # initialize lists and dictionaries
    #   all_distances_angle: all pair-wise lattice distances between 2 and 3 A fulfilling the angle criteria
    #   all_lattice_distances_hist: for each lattice type pair -> histogram of all_distances_angle entries in each timestep
    #   all_jump_distances_hist: for each lattice type pair -> histogram of all jump distances
    all_distances_angle = []
    lattice_distances_hist = { pair: np.zeros( (bins), dtype=np.int64) for pair in analysishelper.pair_strings }

    for i in range(analysishelper.frame_no - 1):
        if i % 100 == 0:
            print( f"{i} / {analysishelper.frame_no - 1}", end = '\r')
        # Calculate jump distances, apply angle criterium and attach results to all_distances and all_distances_angle
        distances = all_distances_in_frame(analysishelper.lattice_traj, i)
        distances[ distances > dlimits[1] ] = 0
        distances[ distances < dlimits[0] ] = 0
        distances_angle = all_distances_angle_criterium_in_frame(analysishelper, i, distances)
        all_distances_angle.append( coo_matrix(distances_angle))

        # bin all lattice distances and jump distances and store results in all_lattice_distances_hist and all_jump_distances_hist
        bin_distances_by_types_and_protonations(analysishelper, distances_angle, protonations[i], dlimits, bins, out = lattice_distances_hist)

        #correct_protonated_distances = np.linalg.norm(
        #                                analysishelper.lattice_traj.pbc_dist2(
        #                                    analysishelper.lattice_traj.coords[i, np.logical_not(protonations[i]), :],
        #                                    analysishelper.lattice_traj.coords[i, protonations[i], :]
        #                                    )
        #                                    , axis = -1
        #                                )
        #pair_identifiers = analysishelper.pair_identifiers[ ~protonations[i],: ][:, protonations[i] ]
        #for pair in analysishelper.pair_strings:
        #    lattice_distances_hist[pair] += np.histogram( correct_protonated_distances[ pair_identifiers == pair ], range = dlimits, bins = bins )[0]

    jump_distances_hist = bin_jumpinfo_by_type(analysishelper, jump_info, dlimits = dlimits, bins = bins)

    # for each lattice type pair:
    #   average lattice distances and jump distances over all timesteps
    #   calculate jump probabilities by dividing jump histogram with lattice histogram
    jump_probs = {}
    for pair in analysishelper.pair_generator():
        jump_probs[pair] = np.divide(
                                jump_distances_hist[pair],
                                lattice_distances_hist[pair],
                                out = np.zeros_like(jump_distances_hist[pair], dtype = float),
                                where=lattice_distances_hist[pair]!=0
                           )
        output_histogram = np.column_stack((average_d, jump_probs[pair], lattice_distances_hist[pair], jump_distances_hist[pair]))
        np.savetxt("jumpprob_histogram_" + ''.join( pair.split('->') ), output_histogram, header="d / A, jump prob, oo-pairs, jumps", fmt = ['%8.3f', '%8f', '%8i', '%8i'])

    if out_distances is not None:
        pickle.dump(all_distances_angle,  open( out_distances, "wb" ))

    if fit:
        # fit fermi parameters (for now both without weights and weighted by number of lattice_distances
        fermi_params, fermi_params_weighted = fit_fermi(average_d, jump_probs, lattice_distances_hist)
        if fermi_file is not None:
            fermi_to_oldformat(fermi_params_weighted, fermi_file)
            fermi_to_oldformat(fermi_params, fermi_file + "_no_weight")
            fermi_to_json(fermi_params_weighted, fermi_file + ".json")
            fermi_to_json(fermi_params, fermi_file + "_no_weights.json")
        return fermi_params, fermi_params_weighted, all_distances_angle, jump_distances_hist, lattice_distances_hist, jump_probs
    return

# turn neighbor indices into protonation count for each lattice_point
def neighbors_to_protonation(neighbors, helper, fixed_lattice = None):
    # if fixed_lattice is included in helper, then this becomes unnecessary
    if fixed_lattice is None:
        lattice_no = helper.lattice_no
    else:
        lattice_no = fixed_lattice.shape[0]
    protonations = np.zeros((helper.frame_no, lattice_no))
    for i in range(helper.frame_no):
        protonations[i] = np.bincount(neighbors[i], minlength = lattice_no)
    protonations -= helper.normal_protonation
    return protonations.astype(bool)

def fermi(x,a,b,c):
    return a/(1+ np.exp((x-b)*c))

def fit_fermi(average_d, jump_probs, lattice_distances_hist, fermi_guess = (0.04, 2.3, 30), start_fit = 25):
            #path_hist = "oo_jump_histogram"
    fermi_params = {}
    fermi_params_weighted = {}
    for pair in jump_probs.keys():
#helper.pair_generator():
#>>>>>>> fermi fit can be used by old post processing scripts
        fig, ax = plt.subplots()
        ax.plot(average_d[start_fit:], jump_probs[pair][start_fit:], label = 'jump prob from AIMD')
        try:
            # fit without weights
            popt, pcov = curve_fit(fermi, average_d[start_fit:], jump_probs[pair][start_fit:], p0 = fermi_guess)
            fermi_params[pair] = popt
            # fit with weights
            sigmas = 1 / np.sqrt(lattice_distances_hist[pair])
            popt_weighted, pcov_weighted = curve_fit(fermi, average_d[start_fit:], jump_probs[pair][start_fit:], p0 = fermi_guess, sigma = sigmas[start_fit:])
            fermi_params_weighted[pair] = popt_weighted
            # plot jump probabaility and fitted fermi functions
            ax.plot(average_d[start_fit:], fermi(average_d[start_fit:], popt[0], popt[1], popt[2]), label='fermi fit')
            ax.plot(average_d[start_fit:], fermi(average_d[start_fit:], popt_weighted[0], popt_weighted[1], popt_weighted[2]), label='fermi fit weighted')
            ax.set(xlabel='OO distance [A]', ylabel='prob  [fs-1]',
                     title='Fermi fit of jump probabilites')
            legend = ax.legend(loc = 'best')
            ax.grid()
            fig.savefig("eval_jump_fit_" + pair + ".png")
        except RuntimeError:
            print(f"Could not fit fermi function to {pair} jump rates!")
    return fermi_params, fermi_params_weighted

# calculate pair-wise distances between all atoms in a trajectory timestep
def all_distances_in_frame(traj, timestep):
    return np.linalg.norm( traj.pbc_dist2(traj.coords[timestep], traj.coords[timestep]), axis = -1)

def angle1(v1, v2, acute):
    # v1 is your first vector
    # v2 is your second vector
    angle = np.arccos(
        np.clip(
        np.sum(v1 * v2, axis=1)
        / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        , 0, 1)
    )
    if acute == True:
        return angle
    else:
        return 2 * np.pi - angle


# edits distances_in_frame in place -> perhaps copy first
def all_distances_angle_criterium_in_frame(analysishelper, timestep, distances_in_frame = None):
    if distances_in_frame is None:
        distances_in_frame = all_distances_in_frame(analysishelper.lattice_traj, timestep)
    # Store coordinates of angle criterium neighbor for each lattice point
    # if no angle criterium, then the neighbors coordinates will be np.nan -> calculated angle will be nan as well
    angle_neighbors = np.full((analysishelper.lattice_no, 3), np.nan)
    for angle_type in analysishelper.angle_types:
        if angle_type == "None":
            continue
        angle_traj = analysishelper.angle_trajs[angle_type][timestep]
        angle_neighbor_indices = analysishelper.lattice_traj.next_neighbor2( analysishelper.lattice_traj.coords[timestep, analysishelper.angle_labels == angle_type], angle_traj.coords)[0]
        angle_neighbor_coords = angle_traj.coords[angle_neighbor_indices]
        angle_neighbors[ analysishelper.angle_labels == angle_type ] = angle_neighbor_coords
    # check angle criterium on source-site
    sources, destinations = np.where(distances_in_frame)
    PO_vectors = analysishelper.lattice_traj.pbc_dist(analysishelper.lattice_traj.coords[timestep, sources], angle_neighbors[sources])
    OO_vectors = analysishelper.lattice_traj.pbc_dist(analysishelper.lattice_traj.coords[timestep, sources], analysishelper.lattice_traj.coords[timestep, destinations])
    # returns false for all lattice_points without angle_criterium (because np.nan < 90 is False) -> angle_allowed is automatically true for them
    angle_forbidden = angle1(PO_vectors, OO_vectors, True) * 180 / np.pi < 90
    forbidden_sources, forbidden_destinations = sources[angle_forbidden], destinations[angle_forbidden]
    distances_in_frame[forbidden_sources, forbidden_destinations] = 0
    distances_in_frame[forbidden_destinations, forbidden_sources] = 0
    return distances_in_frame

def my_histogram( arr, bins, dlimits ):
    hist_step = ( dlimits[1] - dlimits[0] ) / bins
    return np.bincount( np.floor( (arr - dlimits[0] ) / hist_step ).astype(int) , minlength = bins )

def bin_distances_by_types_and_protonations(analysishelper, distances, protonations, dlimits = (2.0, 3.0), bins = 100, out = None):
    distances[ (distances < dlimits[0]) | (distances > dlimits[1]) ] = 0
    sources, destinations = np.where(distances)
    correct_protonations = np.logical_and( protonations[sources], np.logical_not(protonations[destinations]) )
    sources, destinations = sources[correct_protonations], destinations[correct_protonations]
    distances_values = distances[sources, destinations]
    sources_destinations_identifiers = analysishelper.pair_identifiers[sources, destinations]

    if out is None:
        out = { pair: np.zeros( (bins), dtype = int ) for pair in analysishelper.pair_strings }
    for pair in analysishelper.pair_strings:
        distances_pair = distances_values[ sources_destinations_identifiers == pair ]
        out[pair] += my_histogram(distances_pair, bins, dlimits)
    return out

def bin_jumpinfo_by_type(analysishelper, jump_info, dlimits = (2.0, 3.0), bins = 100):
    timesteps, sources, destinations = jump_info.T
    start_positions = analysishelper.lattice_traj.coords[timesteps, sources]
    end_positions = analysishelper.lattice_traj.coords[timesteps, destinations]
    distances = np.linalg.norm( analysishelper.lattice_traj.pbc_dist( start_positions, end_positions), axis = -1 )
    pair_identifiers = analysishelper.pair_identifiers[ sources, destinations ]
    hists = {}
    for pair in analysishelper.pair_strings:
        hists[pair] = np.histogram( distances[ pair_identifiers == pair], range = dlimits, bins = bins)[0]
    return hists


def fermi_to_json(fermi_params, path = "fermi.json"):
    # convert values in fermi_params from ndarray to list, so they can be handled by json
    new_dict = {k: v.tolist() for k, v in fermi_params.items()}
    with open(path, "w+") as f:
        json.dump(new_dict, f, indent=4)
    return

def json_to_fermi(path = "fermi.json"):
    with open(path, "r") as f:
        fermi_params = json.load(f)
    # convert values in fermi_params from list to ndarray
    fermi_params = { k : np.array(v) for k, v in fermi_params.items() }
    return fermi_params

def fermi_to_oldformat(fermi_params, path = "fermi_param"):
    for key, value in fermi_params.items():
        save_name = path + '_' + ''.join( key.split('->') )
        np.savetxt( save_name, value[None, :], delimiter='\t', header='omega(d) = a / (1 + exp( (d - b) * c))\na\t\tb\t\tc', fmt='%.5e')
    return

def oldformat_to_fermi(helper, path = "fermi_param"):
    fermi_params = {}
    for pair in helper.pair_generator():
        save_name = path + '_' + ''.join( pair.split('->') )
        fermi_params[pair] = np.loadtxt(save_name)

    return fermi_params
