import numpy as np
import random
import warnings
from linf_solvers import myFitTchebycheff
warnings.filterwarnings( 'ignore' )

def lineptdist(xn, X, Y, eps):
    res = np.abs(np.matmul(X,xn) - Y);
    res = np.squeeze(res) if res.ndim > 1 else res;
    inliers = np.argwhere(res < eps);
    inliers = np.squeeze(inliers) if inliers.ndim > 1 else inliers;
    return inliers;

def fitLinearRANSAC(X, Y, epsilon, max_iterations=1000):
    
    assert(X.shape[0] == Y.shape[0])
    Y = np.expand_dims(Y, axis=1) if Y.ndim == 1 else Y;
    d = X.shape[1];
    max_inliers = 0;
    best_model = None;
    total_points = X.shape[0]
    iter = 0;
    while (1):
        ids = random.sample(range(total_points), k=d+1);
        _, xn, _ = myFitTchebycheff(X[ids,:], Y[ids,:])

        num_inliers = 0;
        inls = list();
        for i in range(total_points):
            is_inliers = lineptdist(xn, X[i,:], Y[i], eps=epsilon);
            if is_inliers.size > 0:
                num_inliers += 1;
                

        if num_inliers > max_inliers:
            max_inliers = num_inliers;
            best_model = xn;
        
        iter += 1;
        if iter >= max_iterations and best_model is not None:
            break;
           
    # perform final fitting
    inliers = lineptdist(best_model, X, Y, eps=epsilon);
    return best_model, inliers