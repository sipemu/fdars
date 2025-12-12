# fdars

**Functional Data Analysis in Rust** - A high-performance R package for functional data analysis with a Rust backend.

## Overview

fdars provides a comprehensive suite of tools for functional data analysis, leveraging Rust for computationally intensive operations. The package offers significant performance improvements over pure R implementations while maintaining full compatibility with the R ecosystem.

## Features

### Core Functionality (97 exported functions)

| Category | Functions | Description |
|----------|-----------|-------------|
| **Data Structures** | `fdata`, `fdata.cen`, `fdata.deriv`, `func.mean`, `norm.fdata` | Create and manipulate functional data objects |
| **Depth Functions** | `depth.FM`, `depth.mode`, `depth.RP`, `depth.RT`, `depth.FSD`, `depth.KFSD`, `depth.RPD` | Compute functional depth measures |
| **Statistics** | `func.med.*`, `func.trim.*`, `func.var`, `func.trimvar.*` | Functional medians, trimmed means, and variances |
| **Metrics** | `metric.lp`, `metric.hausdorff`, `metric.DTW`, `metric.kl` | Distance metrics for functional data |
| **Semimetrics** | `semimetric.pca`, `semimetric.deriv`, `semimetric.basis`, `semimetric.fourier`, `semimetric.hshift` | Semimetric measures |
| **Regression** | `fregre.pc`, `fregre.basis`, `fregre.np` (+ CV variants) | Functional regression models |
| **Tests** | `flm.test`, `fmean.test.fdata` | Statistical hypothesis tests |
| **Outliers** | `outliers.depth.pond`, `outliers.depth.trim`, `outliers.lrt`, `outliers.thres.lrt` | Outlier detection methods |
| **Kernels** | `Ker.*`, `AKer.*`, `IKer.*`, `Kernel`, `Kernel.asymmetric`, `Kernel.integrate` | Symmetric, asymmetric, and integrated kernel functions |
| **Smoothing** | `S.NW`, `S.LLR`, `S.LPR`, `S.LCR`, `S.KNN`, `h.default`, `CV.S`, `GCV.S`, `optim.np` | Kernel smoothing and bandwidth selection |
| **Clustering** | `kmeans.fd`, `kmeans.center.ini` | Functional k-means clustering |
| **Utilities** | `int.simpson`, `inprod.fdata`, `r.ou`, `r.brownian`, `r.bridge`, `pred.*` | Integration, random processes, prediction metrics |

### Performance

The Rust backend provides significant speedups for computationally intensive operations:

| Operation | Expected Speedup |
|-----------|-----------------|
| Depth computation | 10-50x |
| Distance matrices | 20-100x |
| Smoothing matrices | 10-50x |
| K-means clustering | 50-200x |

## Installation

### Prerequisites

- R (>= 4.0)
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- A C compiler (gcc, clang)

### From Source

```r
# Install devtools if needed
install.packages("devtools")

# Install fdars from source
devtools::install_local("path/to/fdars")
```

### Building

```bash
cd fdars
R CMD build .
R CMD INSTALL fdars_0.1.0.tar.gz
```

## Quick Start

```r
library(fdars)

# Create functional data
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 20, 100)
for (i in 1:20) {
  X[i, ] <- sin(2 * pi * t) + rnorm(100, sd = 0.1)
}
fd <- fdata(X, argvals = t)

# Compute depth
depths <- depth.FM(fd)

# Find median
median_curve <- func.med.FM(fd)

# Functional PCA
pc <- fdata2pc(fd, ncomp = 3)

# Regression
y <- rowMeans(X) + rnorm(20, sd = 0.1)
model <- fregre.pc(fd, y, ncomp = 3)

# Clustering
clusters <- kmeans.fd(fd, ncl = 2)

# Smoothing
S <- S.NW(t, h = 0.1)
smoothed <- S %*% X[1, ]
```

## Comparison with fda.usc

fdars is designed to be largely compatible with [fda.usc](https://cran.r-project.org/package=fda.usc), offering:

- **Same API**: Most functions have the same names and arguments
- **Same output format**: Results are structured similarly for easy migration
- **Better performance**: Rust backend for compute-intensive operations
- **Modern codebase**: Clean implementation with comprehensive tests

### Coverage

fdars implements 100% of core FDA functionality from fda.usc:
- All depth functions
- All functional statistics
- All distance metrics
- Functional regression
- Hypothesis tests
- Outlier detection
- Kernel smoothing
- Clustering

## Documentation

Each function includes comprehensive documentation accessible via `?function_name` in R.

## Testing

The package includes 338 tests covering all functionality:

```r
# Run tests
testthat::test_local()
```

## Architecture

```
fdars/
├── R/                    # R wrapper functions
│   ├── fdata.R          # Core data structures
│   ├── depth.R          # Depth functions
│   ├── metric.R         # Distance metrics
│   ├── fregre.R         # Regression
│   ├── tests.R          # Statistical tests
│   ├── outliers.R       # Outlier detection
│   ├── kernels.R        # Kernel functions
│   ├── smoothing.R      # Smoothing matrices
│   ├── clustering.R     # K-means clustering
│   └── utility.R        # Utilities
├── src/
│   └── rust/
│       └── src/
│           └── lib.rs   # Rust implementation (~3900 lines)
└── tests/
    └── testthat/        # Test suite
```

## License

MIT

## Author

Simon Müller

## Acknowledgments

- Inspired by and compatible with [fda.usc](https://cran.r-project.org/package=fda.usc)
- Built with [extendr](https://extendr.github.io/) for R-Rust integration
- Uses [rayon](https://github.com/rayon-rs/rayon) for parallelization
