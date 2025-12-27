# fdars-py

[![Python CI](https://github.com/sipemu/fdars/actions/workflows/python-ci.yml/badge.svg)](https://github.com/sipemu/fdars/actions/workflows/python-ci.yml)
[![codecov](https://codecov.io/gh/sipemu/fdars/graph/badge.svg)](https://codecov.io/gh/sipemu/fdars)
[![PyPI version](https://badge.fury.io/py/fdars-py.svg)](https://badge.fury.io/py/fdars-py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Functional Data Analysis in Python**

A Python package for functional data analysis, powered by a high-performance Rust backend (`fdars-core`). This package provides tools for analyzing curves, time series, and surfaces as functional data.

## Features

- **Functional Data Representation**: `FData` class for managing functional data with evaluation points, metadata, and convenient operations
- **Depth Measures**: Fraiman-Muniz, Modal, Band, Modified Band, Random Projection, Tukey, and Spatial depths
- **Distance Metrics**: Lp, Hausdorff, DTW, and semimetrics (Fourier, horizontal shift)
- **Basis Representations**: B-splines, Fourier, P-spline smoothing with automatic selection
- **Clustering**: K-means and Fuzzy C-means for functional data
- **Regression**: Functional PCA, Functional PLS, Ridge regression
- **Smoothing**: Nadaraya-Watson, Local Linear, Local Polynomial regression
- **Seasonal Analysis**: Period estimation, peak detection, seasonal strength, change detection
- **Detrending**: Linear, polynomial, LOESS, spline, and automatic detrending
- **Outlier Detection**: Depth-based and likelihood ratio test methods

## Installation

### From PyPI

```bash
pip install fdars-py
```

### From source

Requires Rust toolchain and maturin:

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install from source
git clone https://github.com/sipemu/fdars.git
cd fdars/fdars-py
maturin develop --release
```

## Quick Start

```python
import numpy as np
from fdapy import FData, depth, metric_lp, kmeans, fpca

# Create functional data
t = np.linspace(0, 1, 100)
X = np.array([np.sin(2 * np.pi * t + phase) for phase in np.random.randn(20)])
fd = FData(X, argvals=t)

# Compute mean and center
mean_func = fd.mean()
fd_centered = fd.center()

# Compute depth
depths = depth(fd, method="FM")

# Compute distance matrix
D = metric_lp(fd, p=2)

# Clustering
result = kmeans(fd, n_clusters=3)
labels = result["labels"]
centers = result["centers"]

# Functional PCA
pca_result = fpca(fd, n_components=3)
scores = pca_result["scores"]
components = pca_result["components"]
```

## API Overview

### Core Class

```python
from fdapy import FData

# Create from array
fd = FData(data, argvals=t, rangeval=(0, 1))

# Operations
fd.mean()           # Mean function
fd.center()         # Centered data
fd.deriv(nderiv=1)  # Derivative
fd.norm(p=2)        # Lp norms
fd.plot()           # Visualization
```

### Depth Measures

```python
from fdapy import depth, depth_fm, depth_mbd

# Unified interface
depths = depth(fd, method="FM")  # or "mode", "RP", "RT", "BD", "MBD", "MEI"

# Direct functions
depths = depth_fm(fd)
depths = depth_mbd(fd)
```

### Metrics

```python
from fdapy import metric_lp, metric_dtw, metric_hausdorff

D = metric_lp(fd, p=2)           # L2 distances
D = metric_dtw(fd)               # Dynamic Time Warping
D = metric_hausdorff(fd)         # Hausdorff distances
```

### Basis Representations

```python
from fdapy import fdata_to_basis, basis_to_fdata, pspline, select_basis_auto

# Project to basis
coefs = fdata_to_basis(fd, nbasis=15, basis_type="bspline")

# Reconstruct
fd_reconstructed = basis_to_fdata(coefs, t, nbasis=15)

# P-spline smoothing
result = pspline(fd, nbasis=20, lambda_=1.0)

# Automatic selection
result = select_basis_auto(fd, nbasis_range=(5, 30), criterion="GCV")
```

### Clustering

```python
from fdapy import kmeans, fcm

# K-means
result = kmeans(fd, n_clusters=3)
labels = result["labels"]
centers = result["centers"]  # FData object

# Fuzzy C-means
result = fcm(fd, n_clusters=3, m=2.0)
membership = result["membership"]
```

### Regression

```python
from fdapy import fpca, fpls, ridge

# Functional PCA
result = fpca(fd, n_components=5)
scores = result["scores"]
components = result["components"]
explained_var = result["explained_variance_ratio"]

# Functional PLS
result = fpls(X=fd, y=response, n_components=3)
```

### Seasonal Analysis

```python
from fdapy import estimate_period, detect_peaks, seasonal_strength, detrend, decompose

# Period estimation
result = estimate_period(fd, method="fft")
periods = result["periods"]

# Peak detection
result = detect_peaks(fd, min_prominence=0.1)

# Seasonal decomposition
result = decompose(fd, period=12, method="additive")
trend = result["trend"]
seasonal = result["seasonal"]
remainder = result["remainder"]
```

### Outlier Detection

```python
from fdapy import outliers_depth, outliers_lrt

# Depth-based
result = outliers_depth(fd, quantile=0.01)
outlier_mask = result["outliers"]

# Likelihood ratio test
result = outliers_lrt(fd, alpha=0.05)
```

## Comparison with R Package (fdars)

This package provides a Python API that mirrors the R package `fdars`:

| R | Python |
|---|--------|
| `fdata(X, argvals)` | `FData(X, argvals)` |
| `mean.fdata(fd)` | `fd.mean()` |
| `depth.FM(fd)` | `depth_fm(fd)` or `depth(fd, method="FM")` |
| `metric.lp(fd1, fd2)` | `metric_lp(fd1, fd2)` |
| `fdata2basis(fd, nbasis)` | `fdata_to_basis(fd, nbasis)` |
| `cluster.kmeans(fd, ncl)` | `kmeans(fd, n_clusters)` |
| `fregre.pc(fd, y)` | `fpca(fd)` |

## Requirements

- Python >= 3.9
- NumPy >= 1.20

Optional dependencies:
- matplotlib (for plotting)
- pandas (for DataFrame conversion)

## License

MIT License
