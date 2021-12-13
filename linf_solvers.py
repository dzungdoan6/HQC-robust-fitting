import numpy as np
from numpy import random as rnd
import gurobipy as gp
from gurobipy import GRB

def myFitTchebycheff(X, Y, xn = None):
    [num_points, dimen] = X.shape;
    if xn is None:
        xn = rnd.rand(dimen, 1);
    X = np.vstack((X, -X));
    Y = np.vstack((Y, -Y));

    res = np.matmul(X, xn) - Y;
    res = np.squeeze(res);
    
    max_res = np.amax(res);
    M = np.argwhere(np.abs(res - max_res) < 1e-9);
    M = np.squeeze(M, axis=1);
    res[M] = max_res;

    while (M.size < dimen + 1):
        A = X[M, :];
        if A.ndim == 1:
            A = np.expand_dims(A, axis=0);

        tmp1 = -np.matmul(A, A.transpose());
        c = np.linalg.lstsq(tmp1, np.ones((M.size, 1)));
        c = c[0];
        y = np.matmul(A.transpose(), c);

        lambda1 = np.matmul(X, y) + 1;
        lambda1 = np.squeeze(lambda1);
        
        lambda1 = np.divide(max_res - res, lambda1);
        lambda1[np.argwhere(np.isnan(lambda1))] = 0;
        lambda1[np.argwhere(lambda1 <= 0)] = np.Inf;
        j = np.argmin(lambda1);
        lambda1_j = lambda1[j];
        xn += lambda1_j * y;

        M = np.concatenate((M, [j]));

        res = np.matmul(X, xn) - Y;
        res = np.squeeze(res);
        max_res = np.amax(res);
        res[M] = max_res;

    C = np.ones((dimen+1,1))
    C = np.hstack((C, X[M,:]));
    try:
        C = np.linalg.inv(C);
    except:
        C = np.linalg.pinv(C);
    C1 = C[0,:]
    Ch = np.divide(C1, np.abs(C1));
    itrn = 1e6;

    while np.sum(Ch) < dimen and itrn:
        p1 = np.argmin(C1);
        y = C[1:,p1] / C[0,p1];
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1);
        
        
        t = np.matmul(X, y) + 1;
        t = np.squeeze(t);
        t = np.divide(max_res - res, t);
        t[np.argwhere(np.isnan(t))] = 0;
        t[np.argwhere(t <= 0)] = np.Inf;
        j = np.argmin(t);
        t_j = t[j];
        xn += t_j * y;

        tmp = np.concatenate(([1], X[j,:]));
        tmp = np.expand_dims(tmp, axis=0);
        lambda1 = np.matmul(tmp, C);

        beta = p1;
        Cb = C[:, beta] / lambda1[0, beta];
        Cb = np.expand_dims(Cb, axis=1);

        C -= np.multiply(np.matmul(np.ones((dimen+1, 1)), lambda1), np.matmul(Cb, np.ones((1, dimen+1))));
        C[:, beta] = Cb.squeeze();
        M[beta] = j;

        res = np.matmul(X, xn) - Y;
        res = np.squeeze(res);
        max_res = np.amax(res);
        res[M] = max_res;

        C1 = C[0,:];
        Ch = np.divide(C1, np.abs(C1));
        itrn -= 1;

    l = Y.size/2;
    id = np.argwhere(M >= l);
    id = np.squeeze(id, axis=1);
    for i in id:
        M[i] -= l;
    return M, xn, max_res;


def is_feasible(x, y, eps):
    assert(x.shape[0] == y.size)
    
    num_points = x.shape[0];
    d = 1 if x.ndim == 1 else x.shape[1];

    model = gp.Model();
    
    theta = model.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta');
    
    for i in range(num_points):
        v = 0;
        if d > 1:
            for j in range(d):
                v += x[i, j] * theta[j];
        else:
            v += x[i] * theta[0];
        v = v <= (eps + y[i]);
        model.addConstr(v);

    for i in range(num_points):
        v = 0;
        if d > 1:
            for j in range(d):
                v += -x[i, j] * theta[j];
        else:
            v += -x[i] * theta[0];
        v = v <= (eps - y[i]);
        model.addConstr(v);
    
    model.setObjective(0, GRB.MINIMIZE);
    model.Params.OutputFlag = 0;
    model.optimize();
    if model.status == GRB.OPTIMAL:
        return True;
    else:
        return False;

def test_feasibility(X, Y, d, z, epsilon, IDS):
    outliers = np.argwhere(z >= 0.9999);
    outliers = np.squeeze(outliers);
        
    inliers = np.setdiff1d(IDS, outliers); 
    x = X[inliers, :] if d > 1 else X[inliers];
    y = Y[inliers];
    pass_test = is_feasible(x, y, epsilon);
    return pass_test, outliers;
