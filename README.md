# IntrinsicDimEstimator by Eng. Alberto Biscalchin
The IntrinsicDimEstimator module provides tools for estimating the intrinsic dimensionality of datasets using various methods. It supports correlation dimension, nearest neighbor dimension, packing numbers, geodesic minimum spanning tree, eigenvalue analysis, and maximum likelihood estimation methods.

## Installation
Install IntrinsicDimEstimator using pip:
```Bash
pip install IntrinsicDimEstimator
```

## Usage
Below is an example demonstrating how to generate a synthetic dataset and use the IntrinsicDimEstimator to estimate its intrinsic dimensionality with varying levels of noise added to the dataset.

```Python
import numpy as np
from IntrinsicDimEstimator import intrinsic_dim

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

# Noise levels to test
noise_levels = [0, 0.01, 0.05, 0.1, 0.5]

for noise_level in noise_levels:
    noisy_A = A + noise_level * np.random.randn(*A.shape)  # Add noise
    estimated_dim = intrinsic_dim(noisy_A, method='MLE')  # Estimate intrinsic dimension
    print(f'Estimated dimension with noise level {noise_level}: {estimated_dim}')

```

This example generates a dataset composed of points from the surface of a sphere, a cube, and lines attached to a sphere. It then adds varying levels of Gaussian noise to this dataset and estimates the intrinsic dimensionality using the Maximum Likelihood Estimation (MLE) method provided by the IntrinsicDimEstimator.

## Contributing
Contributions to the IntrinsicDimEstimator are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.
