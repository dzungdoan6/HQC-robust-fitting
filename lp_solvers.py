import numpy as np
import gurobipy as gp
from gurobipy import GRB

def lp_solver(B, total_points):
    total_hyperedges, hyperedge_size = B.shape;
    model = gp.Model();
    z = model.addVars(total_points, lb = 0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name='z');
    model.setObjective(z.sum(), GRB.MINIMIZE);
    for i in range(total_hyperedges):
        v = 0;
        for j in range(hyperedge_size):
            v += z[B[i, j]];
        v = v >= 1;
        model.addConstr(v);
    model.Params.OutputFlag = 0;
    # model = set_default_model_params(model);
    model.optimize();
    z_opt = np.array(model.getAttr('x', z.values()));
    status = "optimal" if model.status == GRB.OPTIMAL else "suboptimal";
    return z_opt, status;
