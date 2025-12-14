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

# Install fdars (with documentation)
remotes::install_github("sipemu/fdars", build_vignettes = TRUE)
```

**Note:** On Windows, you may need [Rtools](https://cran.r-project.org/bin/windows/Rtools/) installed.

### From Source

```bash
cd fdars
R CMD build .
R CMD INSTALL fdars_0.4.0.tar.gz
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
depths <- depth(fd)  # default: FM method
depths <- depth(fd, method = "mode")  # or specify method

# Find the functional median (most central curve)
median_curve <- median(fd)  # default: FM method

# Detect outliers
outliers <- outliers.depth.trim(fd, trim = 0.1)

# Functional regression: predict scalar y from functional X
y <- rowMeans(X) + rnorm(20, sd = 0.1)
model <- fregre.pc(fd, y, ncomp = 3)
predictions <- predict(model, fd)

# Cluster curves into groups
clusters <- cluster.kmeans(fd, ncl = 2)

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

#### Identifiers and Metadata

You can attach identifiers and metadata (covariates) to functional data objects:

```r
# Create fdata with IDs and metadata
meta <- data.frame(
  group = factor(c("control", "treatment", ...)),
  age = c(25, 32, ...),
  response = c(0.5, 0.8, ...)
)
fd <- fdata(X, id = paste0("patient_", 1:n), metadata = meta)

# Access fields
fd$id              # Character vector of identifiers
fd$metadata$group  # Access metadata columns

# Subsetting preserves metadata
fd_sub <- fd[1:10, ]  # id and metadata are also subsetted

# View metadata info
print(fd)    # Shows metadata columns
summary(fd)  # Shows metadata types and ranges
```

**Note:** If metadata contains an `id` column or has non-default row names, they must match the fdata identifiers. An error is thrown on mismatch.

### Depth Functions

Depth measures how "central" or "typical" a curve is relative to a sample. Higher depth = more central.

Use the unified `depth()` function with a `method` parameter:

```r
depth(fd, method = "FM")     # Fraiman-Muniz depth (default)
depth(fd, method = "BD")     # Band depth
depth(fd, method = "MBD")    # Modified band depth
depth(fd, method = "mode")   # Modal depth (kernel density)
depth(fd, method = "RP")     # Random projection depth
depth(fd, method = "RT")     # Random Tukey depth
depth(fd, method = "FSD")    # Functional spatial depth
depth(fd, method = "KFSD")   # Kernel functional spatial depth
depth(fd, method = "RPD")    # Random projection with derivatives
```

### Functional Regression

Predict a scalar response from functional predictors:

- `fregre.pc` - Principal component regression
- `fregre.basis` - Basis expansion regression
- `fregre.np` - Nonparametric kernel regression

All models support `predict()` for new data.

### Distance Metrics

Measure similarity between curves using `metric()` with a method parameter:

```r
metric(fd, method = "lp")        # Lp distance (default, L2 = Euclidean)
metric(fd, method = "hausdorff") # Hausdorff distance
metric(fd, method = "dtw")       # Dynamic time warping
metric(fd, method = "pca")       # PCA-based semimetric
metric(fd, method = "deriv")     # Derivative-based semimetric
```

Individual functions are also available: `metric.lp`, `metric.hausdorff`, `metric.DTW`, `semimetric.pca`, `semimetric.deriv`.

### Outlier Detection

Identify unusual curves:

- `outliers.depth.trim` - Trimmed depth-based detection
- `outliers.depth.pond` - Weighted depth-based detection
- `outliers.lrt` - Likelihood ratio test
- `outliers.boxplot` - Functional boxplot-based detection
- `MS.plot` - Magnitude-Shape plot for visualizing outliers
- `outliergram` - Outliergram (MEI vs MBD plot)

#### Labeling Outliers by ID or Metadata

Both `MS.plot` and `plot.outliergram` support labeling points by ID or metadata columns:

```r
# Create fdata with IDs and metadata
fd <- fdata(X, id = paste0("patient_", 1:n),
            metadata = data.frame(subject_id = paste0("S", 1:n)))

# Outliergram with custom labels
og <- outliergram(fd)
plot(og, label = "id")           # Label outliers with patient IDs
plot(og, label = "subject_id")   # Label with metadata column
plot(og, label_all = TRUE)       # Label ALL points, not just outliers

# MS.plot with custom labels
MS.plot(fd, label = "id")        # Label outliers with patient IDs
MS.plot(fd, label = NULL)        # No labels
```

### Functional Statistics

- `mean(fd)` - Functional mean
- `var(fd)` - Functional variance
- `sd(fd)` - Functional standard deviation
- `cov(fd)` - Functional covariance
- `gmed(fd)` - Geometric median (L1 median via Weiszfeld algorithm)

### Covariance Functions and Gaussian Process Generation

Generate synthetic functional data from Gaussian processes with various covariance kernels:

```r
# Smooth samples with Gaussian (squared exponential) kernel
fd_smooth <- make_gaussian_process(n = 20, t = seq(0, 1, 100),
                                   cov = cov.Gaussian(length_scale = 0.2))

# Rough samples with Matern kernel
fd_rough <- make_gaussian_process(n = 20, t = seq(0, 1, 100),
                                  cov = cov.Matern(nu = 1.5))

# Periodic samples
fd_periodic <- make_gaussian_process(n = 10, t = seq(0, 2, 200),
                                     cov = cov.Periodic(period = 0.5))

# Combine kernels: signal + noise
cov_total <- cov.add(cov.Gaussian(variance = 1), cov.WhiteNoise(variance = 0.1))
fd_noisy <- make_gaussian_process(n = 10, t = seq(0, 1, 100), cov = cov_total)
```

Available covariance functions:
- `cov.Gaussian` - Squared exponential (RBF) kernel, infinitely smooth
- `cov.Exponential` - Exponential kernel (Matern ν=0.5), rough
- `cov.Matern` - Matern family with smoothness parameter ν
- `cov.Brownian` - Brownian motion covariance (1D only)
- `cov.Linear` - Linear kernel
- `cov.Polynomial` - Polynomial kernel
- `cov.WhiteNoise` - Independent noise at each point
- `cov.Periodic` - Periodic kernel (1D only)
- `cov.add` - Combine kernels by addition
- `cov.mult` - Combine kernels by multiplication

### Depth-Based Medians and Trimmed Means

Use the unified functions with a `method` parameter:

```r
# Median (curve with maximum depth)
median(fd)                          # default: FM method
median(fd, method = "mode")         # modal depth-based median

# Trimmed mean (mean of deepest curves)
trimmed(fd, trim = 0.1)             # default: FM method
trimmed(fd, trim = 0.1, method = "RP")  # RP depth-based trimmed mean

# Trimmed variance
trimvar(fd, trim = 0.1)             # default: FM method
trimvar(fd, trim = 0.1, method = "mode")
```

### Visualization

- `plot(fd, color = ...)` - Plot curves with coloring by numeric or categorical variables
  - `show.mean = TRUE` - Overlay group mean curves
  - `show.ci = TRUE` - Show confidence interval ribbons per group
- `boxplot.fdata` - Functional boxplot with depth-based envelopes
- `MS.plot` - Magnitude-Shape plot for outlier visualization
- `outliergram` - Outliergram for shape outlier detection (MEI vs MBD plot)
- `plot.fdata2pc` - FPCA visualization (components, variance, scores)

### Group Comparison

- `group.distance` - Compute distances between groups (centroid, Hausdorff, depth-based)
- `group.test` - Permutation test for significant group differences
- `plot.group.distance` - Visualize group distances (heatmap, dendrogram)

### Clustering

- `cluster.kmeans` - K-means clustering for functional data
- `cluster.optim` - Optimal k selection using silhouette, CH, or elbow
- `cluster.fcm` - Fuzzy C-means clustering with soft membership
- `cluster.init` - K-means++ center initialization

### Curve Registration

- `register.fd` - Shift registration using cross-correlation

### Feature Extraction

- `localavg.fdata` - Extract local average features from curves

### 2D Functional Data (Surfaces)

fdars supports 2D functional data (surfaces/images). The following functions have full 2D support:

| Category | Functions |
|----------|-----------|
| **Depth** | `depth` (methods: FM, mode, RP, RT, FSD, KFSD) |
| **Distance** | `metric.lp`, `metric.hausdorff`, `semimetric.pca`, `semimetric.deriv` |
| **Statistics** | `mean`, `var`, `sd`, `cov`, `gmed`, `deriv` |
| **Centrality** | `median`, `trimmed`, `trimvar` (all methods except BD, MBD, RPD) |
| **Regression** | `fregre.np` (nonparametric) |
| **Visualization** | `plot` (heatmap + contours) |

**Note:** Band depths (BD, MBD), RPD, and DTW do not support 2D data.

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
var_surface <- var(fd2d)             # Pointwise variance
depths <- depth(fd2d)                # Depth values
median_surface <- median(fd2d)       # Depth-based median
gmed_surface <- gmed(fd2d)           # Geometric median

# Plot 2D data (heatmap + contours)
plot(fd2d)
```

#### Converting DataFrames to 2D fdata

Use `df_to_fdata2d()` to convert long-format DataFrames to 2D functional data:

```r
# DataFrame structure: id column, s-index column, t-value columns
df <- data.frame(
  id = rep(c("surf1", "surf2"), each = 5),
  s = rep(1:5, 2),
  t1 = rnorm(10), t2 = rnorm(10), t3 = rnorm(10)
)

# Convert to 2D fdata
fd2d <- df_to_fdata2d(df, id_col = 1, s_col = 2)
```

## Documentation

All functions are documented. Access help with `?function_name`:

```r
?fdata
?depth
?fregre.pc
?cluster.kmeans
```

## Performance

The Rust backend provides significant speedups:

| Operation | Speedup vs Pure R |
|-----------|------------------|
| Depth computation | 10-50x |
| Distance matrices | 20-100x |
| Smoothing matrices | 10-50x |
| K-means clustering | 50-200x |

## API Reference

### Unified Functions

fdars uses a clean, unified API with method parameters:

| Function | Description |
|----------|-------------|
| `fdata(X, id, metadata)` | Create fdata with optional IDs and metadata |
| `df_to_fdata2d(df)` | Convert DataFrame to 2D fdata |
| `depth(fd, method = "FM")` | Compute depth (FM, BD, MBD, MEI, mode, RP, RT, FSD, KFSD, RPD) |
| `metric(fd, method = "lp")` | Compute distance matrix (lp, hausdorff, dtw, pca, deriv) |
| `median(fd, method = "FM")` | Depth-based median |
| `trimmed(fd, trim, method = "FM")` | Trimmed mean |
| `trimvar(fd, trim, method = "FM")` | Trimmed variance |
| `cluster.kmeans(fd, ncl)` | K-means clustering |
| `cluster.fcm(fd, ncl)` | Fuzzy C-means clustering |
| `cluster.optim(fd, ncl.range)` | Optimal k selection |
| `make_gaussian_process(n, t, cov)` | Generate GP samples |
| `cov.Gaussian()`, `cov.Matern()`, etc. | Covariance kernel functions |
| `outliergram(fd)` | Outliergram for shape outlier detection |
| `plot(outliergram, label)` | Plot outliergram with ID/metadata labels |
| `MS.plot(fd, label)` | MS plot with ID/metadata labels |
| `plot(fdata2pc_obj)` | FPCA visualization |
| `plot(fd, color = groups)` | Plot with coloring by groups/values |
| `group.distance(fd, groups)` | Distance between groups (centroid, hausdorff, depth) |
| `group.test(fd, groups)` | Permutation test for group differences |

## License

MIT

## Author

Simon Mueller

## Acknowledgments

- Built with [extendr](https://extendr.github.io/) for R-Rust integration
- Uses [rayon](https://github.com/rayon-rs/rayon) for parallelization
