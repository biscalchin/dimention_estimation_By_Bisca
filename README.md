# CorrDim Correlation Dimension Estimation By Bisca

This repository hosts a Python implementation of the CorrDim algorithm, aimed at estimating the Correlation Dimension of datasets. Additionally, it includes functions for estimating the local intrinsic dimensionality and packing numbers of a dataset, which are useful in various data analysis scenarios. The Correlation Dimension is a measure used to characterize the complexity of fractal objects or structures in data, indicating the minimum number of variables needed to describe the statistical properties of the dataset. This implementation is inspired by methods commonly employed in the field of chaos theory and fractal analysis.

## Features

- Efficient computation of the Correlation Dimension using the CorrDim algorithm.
- Calculation of local intrinsic dimensionality to understand the complexity at different points in the dataset.
- Estimation of packing numbers to assess the spread and density of the dataset.
- Utilization of NumPy and SciPy libraries for high-performance numerical operations and pairwise distance computations.
- Example scripts demonstrating the application of the CorrDim algorithm and additional functions on synthetic and real-world datasets.
- Support for customizing the range of radius values and the number of points considered for local estimates.

## Requirements

- Python 3.6 or newer
- NumPy
- SciPy
- scikit-learn

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
from CorrDim_By_Bisca import corr_dim
from scipy.spatial.distance import pdist, squareform

# Example dataset: 1000 points in a 3D space
data = np.random.rand(1000, 3)

# Estimate the Correlation Dimension
dimension_estimate = corr_dim(data, k=10)

print(f'Estimated Correlation Dimension: {dimension_estimate}')
```

## Estimating Local Intrinsic Dimensionality and Packing Numbers
To analyze the complexity and density of specific regions within your dataset, you can use the local_dim and packing_numbers functions. These functions provide additional insight into the structure of your data.

Example:
```python
from CorrDim_By_Bisca import local_dim, packing_numbers

# Estimate the local intrinsic dimensionality
local_dimensions = local_dim(data, k=10)

# Estimate the packing numbers
packing_nums = packing_numbers(data, radius=1.0)
```

## Contributing
We welcome contributions to enhance the functionality of this implementation or to extend its applicability. If you have suggestions or improvements, please feel free to submit a pull request or open an issue to discuss your ideas.
