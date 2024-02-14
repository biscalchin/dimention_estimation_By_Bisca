import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.linalg import lstsq


def intrinsic_dim(X, method='MLE'):
    """
    Estimate the intrinsic dimensionality of dataset X.
    """
    # Ensure X is a numpy array with unique rows, zero mean, and unit variance
    X = np.unique(X, axis=0).astype(np.float64)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0) + 1e-7

    if method == 'CorrDim':
        return corr_dim(X)
    elif method == 'NearNbDim':
        return near_nb_dim(X)
    elif method == 'PackingNumbers':
        return packing_numbers(X)
    elif method == 'GMST':
        return gmst(X)
    elif method == 'EigValue':
        return eig_value(X)
    elif method == 'MLE':
        return mle(X)
    else:
        raise ValueError(f"Unknown method: {method}")


def corr_dim(X):
    """
    Compute the correlation dimension estimation of dataset X.

    Parameters:
    - X: NumPy array of shape (n_samples, n_features)

    Returns:
    - no_dims: Estimated intrinsic dimensionality based on correlation dimension.
    """
    n = X.shape[0]
    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=6).fit(X)  # Using 6 because it includes the point itself
    distances, _ = nn.kneighbors(X)
    val = distances[:, 1:].flatten()  # Exclude the first column which is distance to itself

    # Determine r1 and r2
    r1 = np.median(val)
    r2 = np.max(val)

    # Initialize s1 and s2
    s1, s2 = 0, 0
    XX = np.sum(X ** 2, axis=1)
    for i in range(n):
        dist = np.sqrt(XX + XX[i] - 2 * np.dot(X, X[i, :]))
        s1 += np.sum(dist < r1)
        s2 += np.sum(dist < r2)

    # Adjust counts for self-matching
    s1 -= n
    s2 -= n

    # Calculate correlation sums
    Cr1 = 2.0 * s1 / (n * (n - 1))
    Cr2 = 2.0 * s2 / (n * (n - 1))

    # Estimate intrinsic dimensionality
    no_dims = (np.log(Cr2) - np.log(Cr1)) / (np.log(r2) - np.log(r1))

    return no_dims


def near_nb_dim(X):
    """
    Compute the nearest neighbor dimension estimation of dataset X.

    Parameters:
    - X: NumPy array of shape (n_samples, n_features)

    Returns:
    - no_dims: Estimated intrinsic dimensionality based on nearest neighbor dimension.
    """
    k1 = 6
    k2 = 12

    # Compute nearest neighbors
    nn = NearestNeighbors(n_neighbors=k2).fit(X)
    distances, indices = nn.kneighbors(X)

    # Calculate Tk for k in the range [k1, k2)
    Tk = np.zeros(k2 - k1)
    for k in range(k1, k2):
        Tk[k - k1] = np.sum(distances[:, k])

    # Normalize Tk by the number of samples
    Tk = Tk / X.shape[0]

    # Estimate intrinsic dimensionality
    no_dims = (np.log(Tk[-1]) - np.log(Tk[0])) / (np.log(k2 - 1) - np.log(k1 - 1))

    return no_dims


def packing_numbers(X):
    """
    Estimate the intrinsic dimensionality of dataset X using the Packing Numbers method.

    Parameters:
    - X: NumPy array of shape (n_samples, n_features)

    Returns:
    - no_dims: Estimated intrinsic dimensionality.
    """
    r = np.array([0.1, 0.5])  # Radii for estimation
    epsilon = 0.01
    max_iter = 20
    done = False
    l = 0
    L = np.zeros((2, max_iter))  # Store log of packing numbers for each radius

    while not done and l < max_iter:
        l += 1
        perm = np.random.permutation(X.shape[0])
        X_perm = X[perm, :]

        for k in range(2):  # For each radius
            C = []  # Indices of centers of non-overlapping spheres
            for i in range(X.shape[0]):
                is_separated = True
                for j in C:
                    if np.sqrt(np.sum((X_perm[i, :] - X_perm[j, :]) ** 2)) < r[k]:
                        is_separated = False
                        break
                if is_separated:
                    C.append(i)
            L[k, l - 1] = np.log(len(C))  # Log of packing number

        # Estimate intrinsic dimension
        no_dims = -((np.mean(L[1, :l]) - np.mean(L[0, :l])) / (np.log(r[1]) - np.log(r[0])))

        # Check for convergence
        if l > 10:
            if 1.65 * (np.sqrt(np.var(L[0, :l]) + np.var(L[1, :l])) / np.sqrt(l) / (
                    np.log(r[1]) - np.log(r[0]))) < epsilon:
                done = True

    return no_dims


def gmst(X):
    """
    Estimate the intrinsic dimensionality of dataset X using the Geodesic Minimum Spanning Tree method.

    Parameters:
    - X: NumPy array of shape (n_samples, n_features)

    Returns:
    - no_dims: Estimated intrinsic dimensionality.
    """
    gamma = 1
    M = 1  # Number of estimates
    N = 10  # Number of random permutations
    samp_points = np.arange(X.shape[0] - 10, X.shape[0])  # Sample points
    k = 6  # Number of nearest neighbors
    Q = len(samp_points)
    knnlenavg_vec = np.zeros((M, Q))
    knnlenstd_vec = np.zeros((M, Q))
    dvec = np.zeros(M)

    # Compute Euclidean distance matrix using NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k * 10)
    nn.fit(X)
    D, _ = nn.kneighbors(X)

    for i in range(M):
        j = 0
        for n in samp_points:
            knnlen1 = 0
            knnlen2 = 0
            for trial in range(N):
                # Random permutation of data
                indices = np.random.permutation(X.shape[0])[:n]
                Dr = D[indices][:, indices]

                # Sum of distances to k nearest neighbors
                L = np.sum(Dr[:, 1:k + 1], axis=1)  # Exclude the distance to itself

                knnlen1 += np.sum(L)
                knnlen2 += np.sum(L ** 2)

            # Compute average and standard deviation over N trials
            knnlenavg_vec[i, j] = knnlen1 / (N * n)
            knnlenstd_vec[i, j] = np.sqrt((knnlen2 - (knnlen1 / N) ** 2) / (N - 1))
            j += 1

        # Least squares estimate of intrinsic dimensionality
        A = np.vstack([np.log(samp_points), np.ones(Q)]).T
        sol1, _, _, _ = lstsq(A, np.log(knnlenavg_vec[i, :]))
        dvec[i] = gamma / (1 - sol1[0])

    no_dims = np.mean(np.abs(dvec))
    return no_dims


def eig_value(X):
    # PCA Eigenvalue Analysis
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)
    lambda_ = pca.explained_variance_ratio_
    no_dims = np.sum(lambda_ > 0.025)
    return no_dims


def mle(X):
    # Maximum Likelihood Estimation
    k1, k2 = 6, 12
    n = X.shape[0]
    d = X.shape[1]
    X2 = np.sum(X**2, axis=1)
    knnmatrix = np.zeros((k2, n))
    for i in range(n):
        distance = np.sort(X2 + X2[i] - 2 * np.dot(X, X[i,:]))
        knnmatrix[:,i] = 0.5 * np.log(distance[1:k2+1])
    S = np.cumsum(knnmatrix, axis=0)
    indexk = np.tile(np.arange(k1, k2+1).reshape(k2-k1+1, 1), (1, n))
    dhat = -(indexk - 2) / (S[k1-1:k2,:] - knnmatrix[k1-1:k2,:] * indexk)
    return np.mean(dhat)
