import numpy as np

def lineariseFundMatrix(x1, x2):
    A = list();
    b = list();
    assert(x1.shape[1] == x2.shape[1]);
    total_points = x1.shape[1];
    for i in range(total_points):
        x = x1[0,i];
        y = x1[1,i];

        xp = x2[0,i];
        yp = x2[1,i];
        a = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, 1];
        A.append(a);
        b.append(-y);
    return np.array(A), np.array(b);