import numpy as np;
import numpy.matlib as matlib;
from collections import defaultdict
import networkx as nx
from minorminer import find_embedding
import dwave.embedding

def hyperedges_to_matrix(hyperedges_set, total_points):
    [total_hyperedges, hyperedge_size] = hyperedges_set.shape;
    rows = np.arange(total_hyperedges);
    rows = np.expand_dims(rows, axis=1);
    R = matlib.repmat(rows, 1, hyperedge_size);
    R = R.flatten();
    hyperedges_set = hyperedges_set.flatten();
    mat = np.zeros((total_hyperedges, total_points));
    mat[R, hyperedges_set] = 1;
    return mat;

def matrix_to_upper_triangular(Q):
    assert(Q.shape[0] == Q.shape[1]);
    Q_lo = np.tril(Q, -1);
    Q = np.triu(Q);
    return Q + Q_lo.transpose();

def create_Q(B, total_points, penalty):
    total_hyperedges, hyperedge_size = B.shape;
    total_variables = total_points + (hyperedge_size-1)* total_hyperedges;
    
    Q1 = np.zeros((total_variables + 1, total_variables + 1));
    Q1[0:total_points, 0:total_points] = np.diag(np.ones(total_points));
    
    A = hyperedges_to_matrix(B, total_points);
    S = np.kron(np.eye(total_hyperedges), np.ones((1, (hyperedge_size-1))));

    tmp = np.hstack((A, -S, -np.ones((total_hyperedges, 1))));
    Q2 = penalty * np.matmul(tmp.transpose(), tmp);
    Q = Q1 + Q2;
    Q = matrix_to_upper_triangular(Q);

    return Q;


def restructure_Q(Q):
    assert(Q.shape[0] == Q.shape[1]);

    # take last column but ignore the last element Q[-1,-1]
    last_col = Q[0:-1, -1];

    # delete last column
    Q = np.delete(Q, -1, axis=1);

    # delete last row
    Q = np.delete(Q, -1, axis=0);

    assert(Q.shape[0] == Q.shape[1]);
    n = Q.shape[0];
    ids = np.diag_indices(n);
    Q[ids] += last_col;
    return Q;

def create_qubo(hyperedges_set, total_points, penalty, display = True):
    if display:
        print("Create QUBO, penalty = %.2f" % penalty)
    Q_mat = create_Q(hyperedges_set, total_points, penalty);
    Q_mat = restructure_Q(Q_mat);
    assert(Q_mat.shape[0] == Q_mat.shape[1]);
    total_variables = Q_mat.shape[0];

    # Initialize our Q matrix
    Q_dict = defaultdict(int)
    
    for i in range(total_variables):
        for j in range(total_variables):
            if Q_mat[i,j] != 0:
                Q_dict[(i,j)] = Q_mat[i,j];

    return Q_dict, total_variables, Q_mat;

def unwrap_response_SA(response, total_variables, total_points):
    response_result = response.best.state;
            
    zt = -np.ones(total_variables)
    for i in range(total_variables):
        zt[i] = response_result[i];

    z = zt[0:total_points];
    z = np.where(z>0.5, 1, 0);
    return z;

def unwrap_response_QA(response, emb, total_variables, total_points):
    response_result = response.result()
            
    zt = -np.ones(total_variables)
    for i in range(total_variables):
        _results_ = []
        for j in emb[i]:
            _results_.append(response_result['solutions'][0][j])
        zt[i] = np.mean(_results_)

    z = zt[0:total_points];
    return z;

def embed_qubo(solver, Q_dict, chain_strength=None):
    G = nx.Graph()
    G.add_edges_from( solver.edges );
    emb = find_embedding(Q_dict, G.edges, random_seed=10) 
    
    target_qubo = dwave.embedding.embed_qubo(Q_dict, emb, G.adj, chain_strength=chain_strength)
    return target_qubo, emb;