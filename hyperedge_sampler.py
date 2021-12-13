import numpy as np
import random
from robust_fitting_solvers import fitLinearRANSAC
from linf_solvers import myFitTchebycheff


def genHyperedge(X, Y, epsilon):
    Y = np.expand_dims(Y, axis=1) if Y.ndim == 1 else Y;
    total_points, d = X.shape;
    all_ids = np.arange(total_points);
    [F, inliers_ids] = fitLinearRANSAC(X, Y, epsilon, max_iterations=100);
    outliers_ids = np.setdiff1d(all_ids, inliers_ids);
    inliers_ids = list(inliers_ids);
    outliers_ids = list(outliers_ids)

    hyperedge = genHyperedgeLinfFromSuboptimal(X, Y, d, inliers_ids, outliers_ids, init_model=F);
    return hyperedge;


def genHyperedgeLinfFromSuboptimal(X, Y, d, inliers_ids, outliers_ids, init_model = None):
    Y = np.expand_dims(Y, axis=1) if Y.ndim == 1 else Y;
    
    # initially set number of added inliers = total outliers
    num_added = len(outliers_ids);

    # if number of added inliers + total outliers < size of hyperedge
    if len(outliers_ids) + num_added < d + 1:
        num_added = d+1 - len(outliers_ids);

    # subsets = outliers_ids + random.sample(list(inliers_ids), k=num_added)
    if len(inliers_ids) > num_added:
        subsets = outliers_ids + random.sample(list(inliers_ids), k=num_added);
    else:
        subsets = outliers_ids + inliers_ids;

    subsets = np.array(subsets);
    x = X[subsets,:];
    y = Y[subsets,:];
    hyperedge, xn, max_res = myFitTchebycheff(x, y, xn = init_model);
    
    hyperedge = subsets[hyperedge];
    return hyperedge;

def hyperedgeExists(hyperedge_set, hyperedge):
    if hyperedge_set is None:
        return False;
        
    n = hyperedge_set.shape[0];
    for i in range(n):
        arr = hyperedge_set[i,:] - hyperedge;
        if np.all((arr == 0)) == True:
            return True;
    return False;

def sampleHyperedge(found_consensus, IDS, outliers, X, Y, d, epsilon):
    inliers = np.setdiff1d(IDS, outliers); 
            
    # maintain an feasible subsets for sampling hyperedges
    if found_consensus is False:
        subsets = inliers;
    else:
        subsets = list(outliers) + random.sample(list(inliers), k=2*(d+1));
        subsets = np.array(subsets);
    
    x = X[subsets, :] if d > 1 else X[subsets];
    y = Y[subsets];
    x = np.expand_dims(x, axis=1) if d == 1 else x; 
    hyperedge = genHyperedge(x, y, epsilon);
    hyperedge = subsets[hyperedge];
    
    hyperedge = np.sort(hyperedge);

    return hyperedge;