# fdars

**Functional Data Analysis in Rust** - A high-performance R package for functional data analysis with a Rust backend.

## What is Functional Data Analysis?

Functional Data Analysis (FDA) is a branch of statistics that deals with data where each observation is a function, curve, or surface rather than a single number or vector. Examples include:

- **Temperature curves**: Daily temperature recordings over a year for multiple weather stations
- **Growth curves**: Height measurements of children tracked over time
- **Spectroscopy data**: Absorbance spectra measured across wavelengths
- **Financial trajectories**: Stock price movements over trading days
- **Medical signals**: ECG, EEG, or fMRI time series

Traditional statistical methods treat each time point as a separate variable, losing the inherent smoothness and continuity of the data. FDA treats the entire curve as a single observation, enabling more powerful and interpretable analyses.

## Why fdars?

fdars provides tools to:

1. **Represent functional data** - Store and manipulate collections of curves with the `fdata` class
2. **Measure centrality** - Find representative curves using depth functions (Fraiman-Muniz, modal, random projection)
3. **Detect outliers** - Identify unusual curves that deviate from the population
4. **Compute distances** - Measure similarity between curves using various metrics (Lp, Hausdorff, DTW)
5. **Perform regression** - Predict scalar responses from functional predictors
6. **Cluster curves** - Group similar functional observations together
7. **Smooth noisy data** - Apply kernel smoothing to reduce noise while preserving signal

The Rust backend provides 10-200x speedups over pure R implementations for computationally intensive operations like depth computation, distance matrices, and clustering.

## Installation

### Prerequisites

- R (>= 4.0)
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- A C compiler (gcc, clang)

### From GitHub

```r
# Install remotes if needed
install.packages("remotes")

# Install fdars
remotes::install_github("sipemu/fdars")
```

### From Source

```bash
cd fdars
R CMD build .
R CMD INSTALL fdars_0.1.0.tar.gz
```

## Quick Start

```r
library(fdars)

# Create functional data from a matrix (rows = observations, cols = time points)
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 20, 100)
for (i in 1:20) {
  X[i, ] <- sin(2 * pi * t) + rnorm(100, sd = 0.1)
}
fd <- fdata(X, argvals = t)

# Compute depth - measures how "central" each curve is
depths <- depth.FM(fd)

# Find the functional median (most central curve)
median_curve <- median.FM(fd)

# Detect outliers
outliers <- outliers.depth.trim(fd, trim = 0.1)

# Functional regression: predict scalar y from functional X
y <- rowMeans(X) + rnorm(20, sd = 0.1)
model <- fregre.pc(fd, y, ncomp = 3)
predictions <- predict(model, fd)

# Cluster curves into groups
clusters <- kmeans.fd(fd, ncl = 2)

# Smooth noisy curves
S <- S.NW(t, h = 0.1)  # Nadaraya-Watson smoother
smoothed <- S %*% X[1, ]
```

## Key Concepts

### Functional Data Objects (`fdata`)

The `fdata` class stores functional data as a matrix where rows are observations and columns are evaluation points:

```r
fd <- fdata(data_matrix, argvals = time_points, rangeval = c(0, 1))
```

### Depth Functions

Depth measures how "central" or "typical" a curve is relative to a sample. Higher depth = more central:

- `depth.FM` - Fraiman-Muniz depth (integrates univariate depths)
- `depth.BD` - Band depth (proportion of pairs where curve is enveloped)
- `depth.MBD` - Modified band depth (more robust, allows partial envelopment)
- `depth.mode` - Modal depth (based on kernel density estimation)
- `depth.RP` - Random projection depth
- `depth.RT` - Random Tukey depth

### Functional Regression

Predict a scalar response from functional predictors:

- `fregre.pc` - Principal component regression
- `fregre.basis` - Basis expansion regression
- `fregre.np` - Nonparametric kernel regression

All models support `predict()` for new data.

### Distance Metrics

Measure similarity between curves:

- `metric.lp` - Lp distance (L2 = Euclidean)
- `metric.hausdorff` - Hausdorff distance
- `metric.DTW` - Dynamic time warping

### Outlier Detection

Identify unusual curves:

- `outliers.depth.trim` - Trimmed depth-based detection
- `outliers.depth.pond` - Weighted depth-based detection
- `outliers.lrt` - Likelihood ratio test
- `outliers.boxplot` - Functional boxplot-based detection
- `MS.plot` - Magnitude-Shape plot for visualizing outliers

### Functional Statistics

- `mean(fd)` - Functional mean (S3 method)
- `var.fdata(fd)` - Functional variance
- `sd.fdata(fd)` - Functional standard deviation
- `cov.fdata(fd)` - Functional covariance
- `gmed(fd)` - Geometric median (L1 median via Weiszfeld algorithm)

### Depth-Based Medians and Trimmed Means

- `median.FM`, `median.MBD`, `median.BD`, `median.mode`, `median.RP`, `median.RPD`, `median.RT` - Depth-based medians
- `trimmed.FM`, `trimmed.MBD`, `trimmed.BD`, `trimmed.mode`, `trimmed.RP`, `trimmed.RPD`, `trimmed.RT` - Trimmed means
- `trimvar.FM`, `trimvar.RP`, `trimvar.RPD`, `trimvar.RT`, `trimvar.mode` - Trimmed variances

### Visualization

- `boxplot.fdata` - Functional boxplot with depth-based envelopes
- `MS.plot` - Magnitude-Shape plot for outlier visualization

### Clustering

- `kmeans.fd` - K-means clustering for functional data
- `optim.kmeans.fd` - Optimal k selection
- `fuzzycmeans.fd` - Fuzzy C-means clustering with soft membership

### Curve Registration

- `register.fd` - Shift registration using cross-correlation

### Feature Extraction

- `localavg.fdata` - Extract local average features from curves

### 2D Functional Data (Surfaces)

fdars supports 2D functional data (surfaces/images) with most statistical functions:

```r
# Create 2D functional data (e.g., 10 surfaces on a 20x30 grid)
n <- 10
m1 <- 20
m2 <- 30
s <- seq(0, 1, length.out = m1)
t <- seq(0, 1, length.out = m2)

# Generate surfaces: f(s,t) = sin(2*pi*s) * cos(2*pi*t) + noise
X <- array(0, dim = c(n, m1, m2))
for (i in 1:n) {
  for (si in 1:m1) {
    for (ti in 1:m2) {
      X[i, si, ti] <- sin(2*pi*s[si]) * cos(2*pi*t[ti]) + rnorm(1, sd = 0.1)
    }
  }
}

fd2d <- fdata(X, argvals = list(s, t), fdata2d = TRUE)

# All these work with 2D data:
mean_surface <- mean(fd2d)           # Mean surface
var_surface <- var.fdata(fd2d)       # Pointwise variance
depths <- depth.FM(fd2d)             # Depth values
median_surface <- median.FM(fd2d)    # Depth-based median
gmed_surface <- gmed(fd2d)           # Geometric median

# Plot 2D data (heatmap + contours)
plot(fd2d)
```

## Documentation

All functions are documented. Access help with `?function_name`:

```r
?fdata
?depth.FM
?fregre.pc
?kmeans.fd
```

## Performance

The Rust backend provides significant speedups:

| Operation | Speedup vs Pure R |
|-----------|------------------|
| Depth computation | 10-50x |
| Distance matrices | 20-100x |
| Smoothing matrices | 10-50x |
| K-means clustering | 50-200x |

## Compatibility

fdars is designed to be compatible with [fda.usc](https://cran.r-project.org/package=fda.usc). Most functions have the same names, arguments, and output structures, making migration straightforward.

## Comparison with fda.usc

### Extended Functionality

fdars provides several features not available in fda.usc:

| Feature | Description |
|---------|-------------|
| **10-200x Performance** | Rust backend with parallel processing for computationally intensive operations |
| **Band Depth Functions** | `depth.BD()` and `depth.MBD()` with Rust backend for fast computation |
| **Functional Boxplot** | `boxplot.fdata()` for depth-based functional boxplots |
| **MS Plot** | `MS.plot()` for magnitude-shape outlier visualization |
| **Fuzzy C-Means** | `fuzzycmeans.fd()` for soft clustering with membership degrees |
| **Geometric Median** | `gmed()` L1 median via Weiszfeld algorithm |
| **Curve Registration** | `register.fd()` shift registration using cross-correlation |
| **Local Averages** | `localavg.fdata()` for feature extraction |
| **Optimal k Selection** | `optim.kmeans.fd()` automatically finds optimal clusters using silhouette, Calinski-Harabasz, or elbow methods |
| **Local k-NN Bandwidth** | `fregre.np()` supports local cross-validation (`kNN.lCV`) for adaptive bandwidth per observation |
| **2D Functional Data** | Native support for surfaces/images as functional data with 2D plotting (heatmap + contours) |
| **Modern Visualizations** | All plots use ggplot2 instead of base R graphics |
| **Dynamic Time Warping** | Built-in `metric.DTW()` for time series alignment |

### Not Yet Implemented

The following fda.usc features are not yet available in fdars:

| Category | Functions |
|----------|-----------|
| **Classification** | `classif.depth`, `classif.DD`, `classif.np`, `classif.glm`, `classif.gkam`, `classif.gsam` |
| **PLS Regression** | `fregre.pls`, `fregre.pls.cv` |
| **GLM/GAM Models** | `fregre.glm`, `fregre.gkam`, `fregre.gsam`, `fregre.lm`, `fregre.plm` |
| **Functional ANOVA** | `fanova.onefactor`, `fanova.hetero`, `fanova.RPm` |
| **Functional Response** | `fregre.basis.fr` (functional-on-functional regression) |
| **Multivariate FDA** | `depth.mdata`, `depth.mfdata` |
| **Additional Tests** | `dfv.test`, `fEqDistrib.test`, `fEqMoments.test` |

Contributions are welcome! See the [GitHub repository](https://github.com/sipemu/fdars) to get involved.

## License

MIT

## Author

Simon Mueller

## Acknowledgments

- API compatible with [fda.usc](https://cran.r-project.org/package=fda.usc) by Manuel Febrero-Bande and Manuel Oviedo de la Fuente
- Built with [extendr](https://extendr.github.io/) for R-Rust integration
- Uses [rayon](https://github.com/rayon-rs/rayon) for parallelization
