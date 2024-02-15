import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.linalg import lstsq
import sys
from scipy.special import gamma
from scipy.stats import linregress


def loading_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar.

    Parameters:
        iteration (int): Current iteration.
        total (int): Total iterations.
        prefix (str): Prefix string.
        suffix (str): Suffix string.
        decimals (int): Positive number of decimals in percent complete.
        length (int): Character length of bar.
        fill (str): Bar fill character.
        print_end (str): End character (e.g. "\\r", "\\r\\n").
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    sys.stdout.flush()  # Flush the output buffer, forcing an update to the display
    # Print New Line on Complete
    if iteration == total:
        print()


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
    nn = NearestNeighbors(n_neighbors=6).fit(X)  # Using 6 because it includes the point itself
    distances, _ = nn.kneighbors(X)
    val = distances[:, 1:].flatten()  # Exclude the first column which is distance to itself

    r1 = np.median(val)
    r2 = np.max(val)

    s1, s2 = 0, 0
    XX = np.sum(X ** 2, axis=1)
    print("Calculating correlation dimension...")
    for i in range(n):
        dist = np.sqrt(np.maximum(XX + XX[i] - 2 * np.dot(X, X[i, :]), 0))
        s1 += np.sum(dist < r1)
        s2 += np.sum(dist < r2)

        # Update the loading bar for each iteration
        loading_bar(i + 1, n, prefix='Progress:', suffix='Complete', length=50)

    s1 -= n
    s2 -= n

    Cr1 = 2.0 * s1 / (n * (n - 1))
    Cr2 = 2.0 * s2 / (n * (n - 1))

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
    total_iterations = max_iter * 2  # Total iterations for progress bar (assuming max_iter * 2 for simplicity)

    current_iteration = 0  # Initialize current iteration for progress bar

    print("Starting Packing Numbers calculation...")
    while not done and l < max_iter:
        l += 1
        perm = np.random.permutation(X.shape[0])
        X_perm = X[perm, :]

        for k in range(2):  # For each radius
            C = []  # Indices of centers of non-overlapping spheres
            for i in range(X.shape[0]):
                is_separated = True
                for j in C:
                    if np.sqrt(np.maximum(np.sum((X_perm[i, :] - X_perm[j, :]) ** 2), 0)) < r[k]:
                        is_separated = False
                        break
                if is_separated:
                    C.append(i)
            L[k, l - 1] = np.log(len(C))  # Log of packing number

            # Update current iteration and display the progress bar
            current_iteration += 1
            loading_bar(current_iteration, total_iterations, prefix='Progress:', suffix='Complete', length=50)

        # Estimate intrinsic dimension
        no_dims = -((np.mean(L[1, :l]) - np.mean(L[0, :l])) / (np.log(r[1]) - np.log(r[0])))

        # Check for convergence
        if l > 10:
            if 1.65 * (np.sqrt(np.maximum(np.var(L[0, :l]) + np.var(L[1, :l]), 0)) / np.sqrt(l) / (
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
    print("Preparing data for GMST calculation...")

    gamma = 1
    M = 1  # Assuming M is always 1 for simplification
    N = 10  # Number of random permutations
    samp_points = np.arange(X.shape[0] - 10, X.shape[0])  # Sample points
    k = 6  # Nearest neighbors to consider
    Q = len(samp_points)
    total_iterations = Q * N  # Total iterations for progress bar updates

    knnlenavg_vec = np.zeros(Q)
    knnlenstd_vec = np.zeros(Q)
    dvec = np.zeros(M)

    print("Beginning GMST calculation...")

    current_iteration = 0  # Initialize current iteration for progress bar

    for j, n in enumerate(samp_points):
        knnlen1 = 0
        knnlen2 = 0
        for trial in range(N):
            # Update current iteration and display the progress bar at the beginning of each trial
            current_iteration += 1
            loading_bar(current_iteration, total_iterations, prefix='Progress:', suffix='Complete', length=50)

            # Select a random subset of points
            indices = np.random.permutation(X.shape[0])[:n]
            subset_X = X[indices]

            # Recalculate distances for the current subset
            nn_subset = NearestNeighbors(n_neighbors=k + 1)
            nn_subset.fit(subset_X)
            distances_subset, _ = nn_subset.kneighbors(subset_X)

            # Use distances to the k nearest neighbors (excluding the point itself)
            L = np.sum(distances_subset[:, 1:], axis=1)

            knnlen1 += np.sum(L)
            knnlen2 += np.sum(L ** 2)

        # Compute average and standard deviation over N trials for each sample point size
        knnlenavg_vec[j] = knnlen1 / (N * n)
        variance = (knnlen2 - (knnlen1 / N) ** 2) / (N - 1)
        knnlenstd_vec[j] = np.sqrt(np.maximum(variance, 0))  # Ensures non-negative input to sqrt

    # Least squares estimate of intrinsic dimensionality after iterating through all sample point sizes
    A = np.vstack([np.log(samp_points), np.ones(Q)]).T
    sol, _, _, _ = lstsq(A, np.log(knnlenavg_vec))
    dvec[0] = gamma / (1 - sol[0])

    no_dims = np.mean(np.abs(dvec))
    return no_dims


def eig_value(X):
    """
    Estimate the intrinsic dimensionality of dataset X using Eigenvalue Analysis via PCA.

    Parameters:
    - X : np.ndarray
        A NumPy array of shape (n_samples, n_features) representing the dataset.

    Returns:
    - int
        The estimated intrinsic dimensionality of the dataset, determined as the number
        of principal components (eigenvalues) that account for a significant portion of
        the variance (more than 2.5%).

    This method performs PCA on the dataset to identify the eigenvalues, then counts
    how many of these eigenvalues are significant, i.e., each representing more than
    2.5% of the total variance. The count of these significant eigenvalues is used
    as an estimate of the dataset's intrinsic dimensionality.
    """
    # Perform PCA on the dataset to find its eigenvalues
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)

    # Calculate the ratio of variance explained by each component
    lambda_ = pca.explained_variance_ratio_

    # Count the number of components that explain more than 2.5% of the variance
    no_dims = np.sum(lambda_ > 0.025)

    # Return the count as the estimated dimensionality
    return no_dims


def mle(X):
    """
    Estimate the intrinsic dimensionality of dataset X using Maximum Likelihood Estimation (MLE).

    Parameters:
    - X : np.ndarray
        A NumPy array of shape (n_samples, n_features) representing the dataset.

    Returns:
    - float
        The estimated intrinsic dimensionality of the dataset.

    The MLE method computes the log distances to the k-nearest neighbors for each point in the dataset,
    then uses these distances to estimate the intrinsic dimensionality. The estimation is performed over
    a range of k values, from k1 to k2, and the final dimensionality estimate is the average over all points
    and all values of k in the specified range.
    """
    k1, k2 = 6, 12  # Range of neighborhood sizes to consider
    n = X.shape[0]  # Number of samples
    X2 = np.sum(X**2, axis=1)  # Precompute the squared norms of data points

    # Initialize the matrix to store log distances to k-nearest neighbors
    knnmatrix = np.zeros((k2, n))

    # Compute the matrix of log distances
    for i in range(n):
        # Calculate squared Euclidean distance from point i to all points
        distance = np.sort(X2 + X2[i] - 2 * np.dot(X, X[i,:]))

        # Store the log of distances, excluding the distance to the point itself (hence starting from index 1)
        knnmatrix[:, i] = 0.5 * np.log(distance[1:k2+1])

    # Calculate the cumulative sum of log distances, which is used in the MLE formula
    S = np.cumsum(knnmatrix, axis=0)

    # Prepare an array of the k values considered, for use in the dimensionality estimation formula
    indexk = np.tile(np.arange(k1, k2+1).reshape(k2-k1+1, 1), (1, n))

    # Apply the MLE formula to estimate the dimensionality
    dhat = -(indexk - 2) / (S[k1-1:k2,:] - knnmatrix[k1-1:k2,:] * indexk)

    # Return the average estimated dimensionality across all points and k values
    return np.mean(dhat)


def idpettis(ydist, n, K=10):
    """
    Estimates the intrinsic dimensionality of the data using the IDPettis algorithm.

    Parameters
    ----------
    ydist : ndarray
        A 2D array containing the nearest neighbor distances for each point, sorted in ascending order.
    n : int
        The sample size.
    K : int, optional
        The maximum number of nearest neighbors to consider.

    Returns
    -------
    idhat : float
        The estimate of the intrinsic dimensionality of the data.
    """
    # Step 2: Determine all the distances r_{k, x_i}
    # Assuming ydist is already sorted and contains distances for up to K nearest neighbors

    # Step 3: Remove outliers
    m_max = np.max(ydist[:, K - 1])
    s_max = np.std(ydist[:, K - 1])
    valid_indices = np.where(ydist[:, K - 1] <= m_max + s_max)[0]
    ydist_filtered = ydist[valid_indices, :]

    # Add a small constant to avoid log(0)
    epsilon = 1e-10
    ydist_filtered = np.maximum(ydist_filtered, epsilon)  # Ensures all values are above 0

    # Step 4: Calculate log(mean(r_k))
    log_rk_means = np.log(np.mean(ydist_filtered[:, :K], axis=0))

    # Step 5: Initial estimate d0
    k_values = np.arange(1, K + 1)
    slope, _, _, _, _ = linregress(np.log(k_values), log_rk_means)
    d_hat = 1 / slope

    # Initialize variables for iteration
    d_prev = d_hat
    convergence_threshold = 1e-3
    max_iterations = 100
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Step 6: Calculate log(G_{k, d_hat})
        G_k_d = (k_values ** (1 / d_prev)) * gamma(k_values) / gamma(k_values + (1 / d_prev))
        log_G_k_d = np.log(G_k_d)

        # Step 7: Update estimate of intrinsic dimensionality
        combined_log = log_G_k_d + log_rk_means
        slope, _, _, _, _ = linregress(np.log(k_values), combined_log)
        d_hat = 1 / slope

        # Check for convergence
        if np.abs(d_hat - d_prev) < convergence_threshold:
            break

        d_prev = d_hat

    return d_hat


def generate_helix_data(num_points=1000):
    """
    Generates data points along a helix.

    Parameters
    ----------
    num_points : int, optional
        Number of data points to generate.

    Returns
    -------
    data : ndarray
        Data points along the helix.
    """
    theta = np.random.uniform(low=0, high=4 * np.pi, size=num_points)
    x = np.cos(theta)
    y = np.sin(theta)
    z = 0.1 * theta

    return np.vstack((x, y, z)).T
