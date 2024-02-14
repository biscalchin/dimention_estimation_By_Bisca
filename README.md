# IntrinsicDimEstimator by Eng. Alberto Biscalchin
The IntrinsicDimEstimator module provides tools for estimating the intrinsic dimensionality of datasets using various methods. It supports correlation dimension, nearest neighbor dimension, packing numbers, geodesic minimum spanning tree, eigenvalue analysis, and maximum likelihood estimation methods.

## Installation
Install IntrinsicDimEstimator using pip:
```Bash
pip install IntrinsicDimEstimator
```

## Available Methods
The IntrinsicDimEstimator module offers a variety of methods to estimate the intrinsic dimensionality of a dataset. Each method has its own approach and is suitable for different types of data. Below is a brief overview of each method:

## CorrDim (Correlation Dimension)
- Description: Estimates the intrinsic dimensionality based on the correlation dimension, which measures how the number of close pairs of points (within a certain distance) scales with that distance.
- Best for: Datasets where you suspect a fractal-like structure or want to understand scaling properties.
## NearNbDim (Nearest Neighbor Dimension)
- Description: Utilizes the distances to the k-th nearest neighbors of each point to estimate the dimension. It assumes that the volume of the space occupied by the data grows at a rate indicative of the intrinsic dimensionality.
- Best for: General datasets, especially when the focus is on local structures and scaling behaviors.
## PackingNumbers
- Description: Estimates dimensionality by calculating the maximum number of non-overlapping spheres of a certain radius that can fit within the dataset. It then examines how this number changes as the radius changes.
- Best for: Analyzing the dataset's global geometric structure and its compactness or sparsity.
## GMST (Geodesic Minimum Spanning Tree)
- Description: Based on analyzing the geodesic minimum spanning tree of the dataset. It focuses on the scaling properties of the tree's length as a function of the number of points.
- Best for: Datasets where the geodesic (rather than Euclidean) distances between points are important, such as manifold learning.
## EigValue (Eigenvalue Analysis)
- Description: Performs Principal Component Analysis (PCA) and estimates the dimensionality by counting the number of significant eigenvalues. It's a direct measure of how many principal components are needed to represent the data's variance.
- Best for: Linearly structured data or when interested in capturing the variance with minimal dimensions.
## MLE (Maximum Likelihood Estimation)
- Description: Uses a statistical approach to estimate the intrinsic dimension by maximizing the likelihood of the distances between nearest neighbors under a model for the data distribution.
- Best for: A broad range of datasets, offering a robust and theoretically grounded estimation.

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

```

This example generates a dataset composed of points from the surface of a sphere, a cube, and lines attached to a sphere. It then adds varying levels of Gaussian noise to this dataset and estimates the intrinsic dimensionality using the Maximum Likelihood Estimation (MLE) method provided by the IntrinsicDimEstimator.

## Contributing
Contributions to the IntrinsicDimEstimator are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.
