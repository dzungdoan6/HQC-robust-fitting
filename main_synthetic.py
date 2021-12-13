import numpy as np
import random
import pickle5 as pkl
from matplotlib import pyplot as plt
from dwave.cloud import Client 
from qubovert.sim import anneal_qubo
from hyperedge_sampler import sampleHyperedge, hyperedgeExists
from qubo import create_qubo, embed_qubo, unwrap_response_QA, unwrap_response_SA
from linf_solvers import test_feasibility
from lp_solvers import lp_solver
from vis_utils import vis_error_bound, vis_line_fitting


def loadSynthetic(file_name):
    with open(file_name, 'rb') as f:
        data = pkl.load(f);

    return data['X'], data['Y'], data['total_outliers'];

if __name__ == "__main__":

    # obtain D-Wave solver
    TOKEN = ''
    if TOKEN:
        client = Client.from_config(token=TOKEN)
        solvers = client.get_solvers(num_qubits__gt=3000)
        solver = solvers[0];
  
    
    ## Load synthetic data
    file_name = 'data/X_Y_1d_n20_r0.2.pkl';
    X, Y, num_outliers = loadSynthetic(file_name);

    # parameters
    max_trials_hyperedge_sampling = 10; # maximum number of trials for generating hyperedge
    max_iter = 30; # maximum number of iterations (= total number of hyperedges)
    epsilon = 0.1; # inlier threshold
    num_anneals = 1000;
    d = 1 if X.ndim == 1 else X.shape[1];
    total_points = len(Y);

    # penalty and decay parameters
    penalty = 1;
    decay_ratio = 1;
    decay_limit = 0.01;
    decay_freq = 10;

    # all data indices
    IDS = np.arange(total_points);
    outliers = None;
    hyperedges_set = None;
    lb_fea = False;
    pass_test = False;
    z_best = np.ones(total_points);

    # Lists to store all results
    all_z_qubo = list();
    all_z_lower_bound = list();
    all_feasibility_status = list();

    # Algorithm
    for iter in range(max_iter):
        print("\n============================================================");
        print("Iter = %d" % iter);
        
        #################### SAMPLE HYPEREDGE ####################
        print("Sample hyperedge");
        trial_time = 0;

        # the while loop minimises the duplicate hyperedges
        while(1):
            trial_time += 1;
            if trial_time % max_trials_hyperedge_sampling == 0:
                print("\tTried %d times but hyperedge exists, break" % max_trials_hyperedge_sampling);
                break;
            
            hyperedge = sampleHyperedge(pass_test, IDS, outliers, X, Y, d, epsilon);

            # if the newly sampled hyperedge does not exist within the current set of hyperedges, we break the while loop
            if not hyperedgeExists(hyperedges_set, hyperedge):
                break;
        
        hyperedges_set = np.vstack((hyperedges_set, hyperedge)) if hyperedges_set is not None else np.expand_dims(hyperedge, axis=0);

        #################### DECAY PENALTY IF NEEDED ####################
        if iter > 0 and iter % decay_freq == 0:
            print("Current number of hyperedges = %d, decay penalty by %.2f" % (hyperedges_set.shape[0], decay_ratio));
            if penalty * decay_ratio < decay_limit:
                print("\tPenalty reaches limit = %f, do not decay" % decay_limit);
            else:
                penalty = penalty * decay_ratio;
            print("\tCurrent penalty = %f" % penalty);

        #################### FORMULATE VERTEX COVER PROBLEM TO QUBO ####################
        Q_dict, total_variables, Q_mat = create_qubo(hyperedges_set, total_points, penalty, display=False);
        
        #################### SOLVE QUBO ####################
        print("Solve QUBO");

        # If the program found the TOKEN from D-Wave, QUBO will be solved by quantum annealing
        # Otherwise, QUBO will be solved by simulated annealing
        if TOKEN:
            print("\tTOKEN found, using quantum annealing (number of anneals = %d)" % num_anneals);
            target_qubo, emb = embed_qubo(solver, Q_dict);
            response = solver.sample_qubo(target_qubo, num_reads=num_anneals, label='Line fitting');
            z = unwrap_response_QA(response, emb, total_variables, total_points);
        else:
            print("\tTOKEN not found, using simulated annealing (number of anneals = %d)" % num_anneals);
            response = anneal_qubo(Q_dict, num_anneals=num_anneals);
            z = unwrap_response_SA(response, total_variables, total_points);
        
        all_z_qubo.append(z);

        #################### SOLVE LOWER BOUND ####################
        print("Solve lower bound");
        z_lower_bound, _ = lp_solver(hyperedges_set, total_points);
        all_z_lower_bound.append(z_lower_bound);
        
        #################### APPLY FEASIBILITY TEST ####################
        print("Apply feasibility test");
        pass_test, outliers = test_feasibility(X, Y, d, z, epsilon, IDS);
        all_feasibility_status.append(pass_test);

        # if we pass the feasibility test, which means we have found a consensus set
        if pass_test is True:
            print("\tFound consensus set");

            # if the current consensus is the best, store it!
            if z_best.sum() > z.sum():
                print("\tFound best consensus, store it!");
                z_best = z;
    
    #################### VISUALISATION ####################
    # plot error bound (Fig 3 in the paper)
    error_bound_plot = plt.figure(1);
    vis_error_bound(all_z_qubo, all_z_lower_bound, all_feasibility_status);

    # plot fitted geometric model
    line_fit_plot = plt.figure(2);
    vis_line_fitting(X, Y, d, z_best, IDS);

    # show plots
    plt.show();