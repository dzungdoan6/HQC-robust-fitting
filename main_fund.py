import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt 
from qubovert.sim import anneal_qubo
from hyperedge_sampler import sampleHyperedge, hyperedgeExists
from qubo import create_qubo, unwrap_response_SA
from linf_solvers import test_feasibility
from lp_solvers import lp_solver
from vis_utils import vis_error_bound, vis_matches
from vision_utils import lineariseFundMatrix

def load_data(file_name):
    data = loadmat(file_name);
    x1 = data['x1_nml'];
    x2 = data['x2_nml'];
    X, Y = lineariseFundMatrix(x1, x2);
    return X, Y, data;
if __name__ == "__main__":

    file_name = 'data/castleA_castleB_Lowe2.0.mat'
    X, Y, raw_data = load_data(file_name);
    total_points, d = X.shape;
    IDS = np.arange(total_points);

    # parameters
    max_trials_hyperedge_sampling = 10; # maximum number of trials for generating hyperedge
    max_iter = 300; # maximum number of iterations (= total number of hyperedges)
    epsilon = 0.03; # inlier threshold
    num_anneals = 100;
    total_points, d = X.shape;
    IDS = np.arange(total_points);  # all data indices
    
    # penalty and decay parameters
    penalty = 1;
    decay_ratio = 0.5;
    decay_limit = 0.01;
    decay_freq = 50;
    
    # Some initial variables
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
        print("Solve QUBO using simulated annealing (number of anneals = %d)" % num_anneals);
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
    # plot error bound (Fig 5 in the paper)
    error_bound_plot = plt.figure(1);
    vis_error_bound(all_z_qubo, all_z_lower_bound, all_feasibility_status);

    # visualise image with inliers and outliers
    inliers_outliers_plot = plt.figure(2);
    outliers = np.argwhere(z_best >= 0.9999);
    outliers = np.squeeze(outliers);
    inliers = np.setdiff1d(IDS, outliers); 
    vis_img = vis_matches(raw_data, inliers);
    plt.imshow(vis_img)
    
    # show plots
    plt.show();