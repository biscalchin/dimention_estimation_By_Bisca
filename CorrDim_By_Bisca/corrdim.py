import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def corr_dim(X, k=10):
    """
    Estimates the Correlation Dimension of a dataset X using the CorrDim algorithm.
    
    Parameters:
        X (ndarray): A 2D array where each row represents a data point and each column a dimension.
        k (int): The number of points to consider for the local estimate.
    
    Returns:
        float: The estimated correlation dimension.
    """
    # Step 1: Compute pairwise distances
    distances = squareform(pdist(X, 'euclidean'))

    # Step 2: Prepare the range of radius values r
    r_vals = np.logspace(-2, 1, 50)
    c_r = []

    # Step 3: Compute the correlation sum for each radius value
    for r in r_vals:
        c_r.append(np.mean(np.heaviside(r - distances, 0)))

    # Step 4: Estimate the slope (correlation dimension) in the linear region of the log-log plot
    c_r = np.array(c_r)
    log_r_vals = np.log(r_vals)
    log_c_r = np.log(c_r)

    # Perform linear regression in the log-log space
    idx = (log_c_r > -np.inf) & (log_c_r < np.inf)  # Filter out -inf and inf values
    slope = np.polyfit(log_r_vals[idx], log_c_r[idx], 1)[0]

    return slope
