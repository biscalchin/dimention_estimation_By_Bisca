import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def packing_numbers(X, r):
    """
    Estimate the intrinsic dimensionality of a dataset X using Packing Numbers.
    
    Parameters:
        X (ndarray): A 2D array where each row represents a data point and each column a dimension.
        r (float): Radius of the hyper-spheres for packing.
    
    Returns:
        int: The estimated number of hyper-spheres that can be packed.
    """
    # Create a nearest neighbors model and fit it to the data
    nbrs = NearestNeighbors(radius=r).fit(X)
    
    # Find the points that are isolated within the radius r
    # i.e., points that do not have any other points within a distance r
    radii = nbrs.radius_neighbors(X, return_distance=True)
    
    # Count the number of points that do not have any neighbors within radius r
    is_isolated = np.array([len(indices) == 0 for distances, indices in zip(*radii)])
    
    # The sum of isolated points will give us the packing number estimate
    packing_number = np.sum(is_isolated)
    
    return packing_number


def local_dim(X, n_neighbors=10, threshold=0.1):
    """
    Estimates the local dimensionality of a dataset X at each point using local PCA.

    Parameters:
        X (ndarray): A 2D array where each row represents a data point and each column a dimension.
        n_neighbors (int): The number of nearest neighbors to consider for the local PCA.
        threshold (float): The threshold for determining the significance of eigenvalues.

    Returns:
        ndarray: An array of estimated local dimensions for each point in X.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    local_dimensions = []

    for i, point in enumerate(X):
        # Find the k-nearest neighbors including the point itself
        distances, indices = nbrs.kneighbors([point])
        
        # Extract the neighbors
        neighbors = X[indices[0]]
        
        # Center the neighbors
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        
        # Compute the local covariance matrix
        cov_matrix = np.cov(neighbors_centered, rowvar=False)
        
        # Compute the eigenvalues (the variances of the data along the principal components)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Sort the eigenvalues in descending order
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        
        # The local dimension can be estimated as the number of significant eigenvalues
        significant_eigenvalues = eigenvalues_sorted[eigenvalues_sorted > threshold * eigenvalues_sorted[0]]
        local_dimension = len(significant_eigenvalues)
        
        local_dimensions.append(local_dimension)
        
    return np.array(local_dimensions)
