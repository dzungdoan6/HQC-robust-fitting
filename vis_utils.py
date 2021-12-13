from matplotlib import pyplot as plt
import numpy as np
import cv2
from linf_solvers import myFitTchebycheff

def vis_error_bound(all_z, all_lower_bound, all_fea_stt):
    
    x_axis = np.arange(len(all_z));
    num_outliers = [z.sum() for z in all_z];
    lower_bound = [lb.sum() for lb in all_lower_bound];
    feasible_ids = [i for i, x in enumerate(all_fea_stt) if x == True]

    num_outliers = np.array(num_outliers);
    lower_bound = np.array(lower_bound);
    feasible_ids = np.array(feasible_ids);
    
    ax = plt.gca();  # or any other way to get an axis object
    plt.plot(x_axis, num_outliers, color="#f1c40f", label="$|| \mathbf{z}||_1$");
    plt.plot(x_axis, lower_bound, color="#e74c3c", label="$LP(A)$ (lower bound)");
    plt.scatter(x_axis[feasible_ids], num_outliers[feasible_ids], \
        s=40, c="#2ecc71", linewidths=1.5, edgecolors='#34495e', label="Feasible $\mathbf{z}$");
    plt.xlabel("Number of hyperedges", fontsize=18);
    plt.ylabel("Number of outliers", fontsize=18);
    ax.legend(prop={'size': 14});

def vis_line_fitting(X, Y, d, z, IDS):
    outliers = np.argwhere(z >= 0.9999);
    outliers = np.squeeze(outliers);
    inliers = np.setdiff1d(IDS, outliers); 

    ax = plt.gca();  # or any other way to get an axis object
    x = X[inliers, :] if d > 1 else X[inliers];
    y = Y[inliers];
    x = np.expand_dims(x, axis=1) if d == 1 else x;
    y = np.expand_dims(y, axis=1);
    _, m, _ = myFitTchebycheff(x, y);

    plt.plot(np.array([[0], [1.01]]), m*np.array([[0], [1.01]]), color='#34495e', label="1D line");
    plt.plot(X[:], Y, 'x', markersize=6, color="#2ecc71", label="Data points");
    
    plt.scatter(X[outliers], Y[outliers], s=150, facecolors='none', edgecolors='r', label="Outliers");
    plt.xlabel("X", fontsize=18);
    plt.ylabel("Y", fontsize=18);
    ax.legend(prop={'size': 12});

def vis_matches(data, inliers=None, line_width=2):
    imA = data['im1'];
    imB = data['im2'];
    X1 = data['X1'];
    X2 = data['X2'];
    assert(X1.shape[1] == X2.shape[1]);
    total_points = X1.shape[1];
    # initialize the output visualization image
    (hA, wA) = imA.shape[:2]
    (hB, wB) = imB.shape[:2]
    ncA = 3 if imA.ndim>2 else 1;
    ncB = 3 if imB.ndim>2 else 1;
    assert(ncA == ncB);
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8") if ncA == 3 else np.zeros((max(hA, hB), wA + wB), dtype="uint8")
    vis[0:hA, 0:wA] = imA
    vis[0:hB, wA:] = imB

    if ncA == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    all_ids = np.arange(total_points);
    if inliers is not None:
        outliers = np.setdiff1d(all_ids, inliers); 
    else:
        inliers = all_ids;
        outliers = [];

    for i in inliers:
        # draw the match
        ptA = (int(X1[0,i]), int(X1[1,i]))
        ptB = (int(X2[0,i]) + wA, int(X2[1,i]))
        cv2.line(vis, ptA, ptB, (0, 255, 0), line_width)
       
        cv2.drawMarker(vis, ptA, color=(0, 0, 255), markerSize=5,thickness=2);
        cv2.drawMarker(vis, ptB, color=(0, 0, 255), markerSize=5,thickness=2);
        

    for i in outliers:
        # draw the match
        ptA = (int(X1[0,i]), int(X1[1,i]))
        ptB = (int(X2[0,i]) + wA, int(X2[1,i]))
        cv2.line(vis, ptA, ptB, (255, 0, 0), line_width)
        cv2.circle(vis, ptA, color=(255, 0, 0), radius=3,thickness=2);
        cv2.circle(vis, ptB, color=(255, 0, 0), radius=3,thickness=2);
        
    # plt.imshow(vis),plt.show()
    return vis