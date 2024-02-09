from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'A Python module for estimating the Correlation Dimension of datasets using the CorrDim algorithm'
LONG_DESCRIPTION = 'This Python module introduces a robust method for estimating the Correlation Dimension of any given dataset, leveraging the CorrDim algorithm. It computes pairwise distances between points in the dataset, evaluates the correlation sum for a range of radius values, and estimates the slope in a log-log plot to determine the correlation dimension. This package is designed for researchers, data scientists, and anyone interested in fractal dimensions and chaos theory. It utilizes NumPy for efficient numerical computations and SciPy for calculating pairwise distances, ensuring high performance and accuracy.'

# Setting up
setup(
    name="CorrDim_By_Bisca",
    version=VERSION,
    author="Eng. Alberto Biscalchin",
    author_email="biscalchin.mau.se@gmail.com",  
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION, 
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],  
    keywords = ["correlation dimension", "CorrDim algorithm", "fractal analysis", "chaos theory", "data analysis", "NumPy", "SciPy", "multidimensional scaling", "time-series analysis"],
    classifiers=[
        "Development Status :: 4 - Beta",  
        "Intended Audience :: Science/Research",  
        "Programming Language :: Python :: 3",  
        "Operating System :: OS Independent",
    ]
)
