# CorrDim Correlation Dimension Estimation By Bisca

This repository hosts a Python implementation of the CorrDim algorithm, aimed at estimating the Correlation Dimension of datasets. The Correlation Dimension is a measure used to characterize the complexity of fractal objects or structures in data, indicating the minimum number of variables needed to describe the statistical properties of the dataset. This implementation is inspired by methods commonly employed in the field of chaos theory and fractal analysis.

## Features

- Efficient computation of the Correlation Dimension using the CorrDim algorithm.
- Utilization of NumPy and SciPy libraries for high-performance numerical operations and pairwise distance computations.
- Example scripts demonstrating the application of the CorrDim algorithm on synthetic and real-world datasets.
- Support for customizing the range of radius values and the number of points considered for local estimates.

## Requirements

- Python 3.6 or newer
- NumPy
- SciPy

## Installation

```Batch
pip install CorrDim-By-Bisca
```
## Usage

To estimate the Correlation Dimension of a dataset using the CorrDim algorithm, follow these steps:

1. Format your dataset as a 2D NumPy array where each row corresponds to a data point and each column to a dimension.
2. Import the `corr_dim` function from the module.
3. Compute the Correlation Dimension of your dataset.

Example:

```python
import numpy as np
from CorrDim-By-Bisca import corr_dim
from scipy.spatial.distance import pdist, squareform

# Example dataset: 1000 points in a 3D space
data = np.random.rand(1000, 3)

# Estimate the Correlation Dimension
dimension_estimate = corr_dim(data, k=10)

print(f'Estimated Correlation Dimension: {dimension_estimate}')
```

## Contributing
We welcome contributions to enhance the functionality of this implementation or to extend its applicability. If you have suggestions or improvements, please feel free to submit a pull request or open an issue to discuss your ideas.
