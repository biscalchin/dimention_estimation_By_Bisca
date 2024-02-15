import numpy as np
from IntrinsicDimEstimator import intrinsic_dim, idpettis, generate_helix_data
from scipy.spatial.distance import pdist, squareform


def genLDdata():
    # Generate data from different geometries
    X1, X2, X3 = np.random.randn(3, 1000)  # Sphere surface samples
    lambda_ = np.sqrt(X1 ** 2 + X2 ** 2 + X3 ** 2)
    X = np.vstack((X1 / lambda_, X2 / lambda_, X3 / lambda_)).T  # Sphere
    XX = np.random.rand(1000, 3) + 2  # Cube
    L1 = np.vstack((np.zeros(1000), np.zeros(1000), 2 * np.random.rand(1000) + 1)).T
    L2 = np.vstack((np.zeros(1000), np.zeros(1000), -2 * np.random.rand(1000) - 1)).T
    L3 = np.vstack((np.zeros(1000), 2 * np.random.rand(1000) + 1, np.zeros(1000))).T
    L4 = np.vstack((np.zeros(1000), -2 * np.random.rand(1000) - 1, np.zeros(1000))).T
    A = np.vstack((X, XX, L1, L2, L3, L4))  # Combined dataset
    return A


# Generate the synthetic dataset
A = genLDdata()

# Methods to test
methods = ['CorrDim', 'NearNbDim', 'PackingNumbers', 'GMST', 'EigValue', 'MLE']

# Noise level to test
noise_level = 0.05  # Example noise level

# Add noise to the dataset
noisy_A = A + noise_level * np.random.randn(*A.shape)

# Estimate intrinsic dimensionality using each method
for method in methods:
    estimated_dim = intrinsic_dim(noisy_A, method=method)
    print(f'Estimated dimension using {method}: {estimated_dim}')


print("Changing dataset...")
# Generate a helix dataset
data = generate_helix_data(num_points=1000)
print("An Helix shaped dataset has been correctly generated.")

# Compute the pairwise distances
ydist = squareform(pdist(data))

# Sort distances for each point to get nearest neighbors, excluding the distance to itself
ydist_sorted = np.sort(ydist, axis=1)[:, 1:]

# Estimate the intrinsic dimensionality
idhat = idpettis(ydist_sorted, n=data.shape[0], K=10)

print(f'Estimated intrinsic dimensionality: {idhat}')
