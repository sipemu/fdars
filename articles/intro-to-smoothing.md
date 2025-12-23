# Introduction to Smoothing Functional Data

## Why Smooth Functional Data?

Real-world functional data almost always contains measurement noise. Raw
observations are discrete samples of an underlying smooth process,
contaminated by:

- **Measurement error**: Instrument precision limits
- **Sampling noise**: Random variation at each time point
- **Digitization artifacts**: Finite resolution of recording devices

Smoothing transforms noisy discrete observations into smooth functional
objects, revealing the true underlying signal. This is essential for:

- Computing derivatives (noise amplifies dramatically)
- Meaningful curve comparisons
- Interpretable visualizations
- Reliable statistical inference

``` r
library(fdars)
#> 
#> Attaching package: 'fdars'
#> The following objects are masked from 'package:stats':
#> 
#>     cov, deriv, median, sd, var
#> The following object is masked from 'package:base':
#> 
#>     norm
set.seed(42)
```

## Loading Real Data: The Berkeley Growth Study

The Berkeley Growth Study tracked the heights of 54 girls and 39 boys
from ages 1 to 18. This classic dataset is available in the `fda`
package.

``` r
# Check if fda package is available
if (!requireNamespace("fda", quietly = TRUE)) {
  message("Install 'fda' package for real data examples: install.packages('fda')")
  # Create synthetic growth-like data as fallback
  age <- seq(1, 18, length.out = 31)
  n_girls <- 20
  heights <- matrix(0, n_girls, length(age))
  for (i in 1:n_girls) {
    # Gompertz growth curve with individual variation
    A <- rnorm(1, 170, 5)  # asymptotic height
    b <- rnorm(1, 2.5, 0.2)
    c <- rnorm(1, 0.15, 0.02)
    heights[i, ] <- A * exp(-b * exp(-c * age)) + rnorm(length(age), sd = 1.5)
  }
} else {
  data(growth, package = "fda")
  age <- growth$age
  heights <- t(growth$hgtf)  # Girls' heights (rows = curves, cols = time)
  n_girls <- nrow(heights)
}

# Create fdata object
fd_raw <- fdata(heights, argvals = age)
cat("Dataset:", n_girls, "growth curves measured at", length(age), "ages\n")
#> Dataset: 54 growth curves measured at 31 ages
```

### Visualizing the Raw Data

``` r
plot(fd_raw, main = "Berkeley Growth Study: Girls' Heights",
     xlab = "Age (years)", ylab = "Height (cm)")
```

![](intro-to-smoothing_files/figure-html/plot-raw-1.png)

The data appears relatively smooth because height was carefully
measured. Let’s add some realistic measurement noise to demonstrate
smoothing techniques:

``` r
# Add measurement noise to simulate less precise instruments
noise_sd <- 2.0  # cm of measurement error
heights_noisy <- heights + matrix(rnorm(length(heights), sd = noise_sd),
                                   nrow = nrow(heights))
fd_noisy <- fdata(heights_noisy, argvals = age)

# Compare original and noisy
par(mfrow = c(1, 2))
plot(fd_raw[1:5, ], main = "Original (5 curves)", ylim = range(heights_noisy[1:5, ]))
```

![](intro-to-smoothing_files/figure-html/add-noise-1.png)

``` r
plot(fd_noisy[1:5, ], main = "With Noise (5 curves)", ylim = range(heights_noisy[1:5, ]))
```

![](intro-to-smoothing_files/figure-html/add-noise-2.png)

``` r
par(mfrow = c(1, 1))
```

## Method 1: P-spline Smoothing

Penalized splines (P-splines) are a powerful and flexible smoothing
method. They balance two objectives:

1.  **Fit the data closely** (minimize residuals)
2.  **Keep the curve smooth** (penalize roughness)

The smoothing parameter $\lambda$ controls this trade-off: - Small
$\lambda$: Follows data closely, may be wiggly - Large $\lambda$: Very
smooth, may miss real features

### Automatic Smoothing with P-splines

``` r
# pspline() automatically selects optimal lambda via GCV
fd_pspline <- pspline(fd_noisy)
print(fd_pspline)
#> P-spline Smoothing Results
#> ==========================
#> Number of curves: 54 
#> Number of basis functions: 20 
#> Penalty order: 2 
#> Lambda: 1e+00 
#> Effective df: 8.54 
#> GCV: 1.466e+02
```

### Visualizing the Result

``` r
# Compare noisy vs smoothed for first curve
idx <- 1
plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = "P-spline Smoothing", xlab = "Age (years)", ylab = "Height (cm)")
lines(age, fd_pspline$fdata$data[idx, ], col = "blue", lwd = 2)
lines(age, fd_raw$data[idx, ], col = "red", lwd = 1, lty = 2)
legend("bottomright", c("Noisy data", "P-spline smooth", "Original"),
       col = c("gray60", "blue", "red"), pch = c(16, NA, NA),
       lty = c(NA, 1, 2), lwd = c(NA, 2, 1))
```

![](intro-to-smoothing_files/figure-html/plot-pspline-1.png)

### Effect of Smoothing Parameter

``` r
# Different smoothing levels
fd_less <- pspline(fd_noisy, lambda = 0.001)   # Less smoothing
fd_more <- pspline(fd_noisy, lambda = 100)     # More smoothing

par(mfrow = c(1, 3))
idx <- 1
ylim <- range(fd_noisy$data[idx, ])

plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = expression(lambda == 0.001), ylim = ylim)
lines(age, fd_less$fdata$data[idx, ], col = "blue", lwd = 2)

plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = "Optimal (GCV)", ylim = ylim)
lines(age, fd_pspline$fdata$data[idx, ], col = "blue", lwd = 2)

plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = expression(lambda == 100), ylim = ylim)
lines(age, fd_more$fdata$data[idx, ], col = "blue", lwd = 2)
```

![](intro-to-smoothing_files/figure-html/lambda-comparison-1.png)

``` r

par(mfrow = c(1, 1))
```

## Method 2: Kernel Smoothers

Kernel smoothers estimate the value at each point using a weighted
average of nearby observations. The bandwidth $h$ controls the
neighborhood size.

### Nadaraya-Watson Smoother

``` r
# Compute smoother matrix with Nadaraya-Watson
# S.NW takes evaluation points (tt) as first argument
h <- h.default(fd_noisy)  # Default bandwidth based on data

S_nw <- S.NW(tt = age, h = h)

# Apply smoother to each curve
fd_kernel <- fd_noisy
for (i in 1:nrow(fd_kernel$data)) {
  fd_kernel$data[i, ] <- as.vector(S_nw %*% fd_noisy$data[i, ])
}
```

### Local Linear Regression

Local linear regression often performs better at boundaries:

``` r
S_llr <- S.LLR(tt = age, h = h)
fd_llr <- fd_noisy
for (i in 1:nrow(fd_llr$data)) {
  fd_llr$data[i, ] <- as.vector(S_llr %*% fd_noisy$data[i, ])
}
```

### Comparing Kernel Methods

``` r
idx <- 1
plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = "Kernel Smoother Comparison", xlab = "Age", ylab = "Height (cm)")
lines(age, fd_kernel$data[idx, ], col = "blue", lwd = 2)
lines(age, fd_llr$data[idx, ], col = "darkgreen", lwd = 2)
lines(age, fd_raw$data[idx, ], col = "red", lty = 2)
legend("bottomright", c("Data", "Nadaraya-Watson", "Local Linear", "Original"),
       col = c("gray60", "blue", "darkgreen", "red"),
       pch = c(16, NA, NA, NA), lty = c(NA, 1, 1, 2), lwd = c(NA, 2, 2, 1))
```

![](intro-to-smoothing_files/figure-html/compare-kernels-1.png)

### Bandwidth Selection

The default bandwidth from
[`h.default()`](https://sipemu.github.io/fdars/reference/h.default.md)
is based on the data:

``` r
cat("Default bandwidth:", h, "\n")
#> Default bandwidth: 1.5
cat("This is computed as the 15th percentile of pairwise distances.\n")
#> This is computed as the 15th percentile of pairwise distances.
```

## Method 3: Basis Expansion

Represent curves as linear combinations of basis functions. This
provides dimensionality reduction along with smoothing.

### B-spline Basis

``` r
# Project to B-spline basis
coefs <- fdata2basis(fd_noisy, nbasis = 12, type = "bspline")
fd_basis <- basis2fdata(coefs, argvals = age, type = "bspline")

# Compare
idx <- 1
plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60",
     main = "B-spline Basis (K=12)", xlab = "Age", ylab = "Height (cm)")
lines(age, fd_basis$data[idx, ], col = "purple", lwd = 2)
lines(age, fd_raw$data[idx, ], col = "red", lty = 2)
legend("bottomright", c("Data", "B-spline", "Original"),
       col = c("gray60", "purple", "red"),
       pch = c(16, NA, NA), lty = c(NA, 1, 2), lwd = c(NA, 2, 1))
```

![](intro-to-smoothing_files/figure-html/basis-bspline-1.png)

### Automatic Basis Selection

How many basis functions? Cross-validation finds the optimal number:

``` r
# Cross-validation for optimal number of basis functions
cv_basis <- fdata2basis.cv(fd_noisy, type = "bspline",
                           nbasis.range = 5:20)
print(cv_basis)
#> Basis Cross-Validation Results
#> ==============================
#> Criterion: GCV 
#> Optimal nbasis: 5 
#> Score at optimal: Inf 
#> Range tested: 5 - 20
plot(cv_basis)
```

![](intro-to-smoothing_files/figure-html/basis-cv-1.png)

### Information Criteria

The `fdata2basis.cv` function supports different criteria: - **GCV**
(default): Generalized Cross-Validation - **CV**: Leave-one-out
Cross-Validation - **AIC**: Akaike Information Criterion - **BIC**:
Bayesian Information Criterion

``` r
# Get optimal basis from our CV result
optimal_nbasis <- cv_basis$optimal.nbasis
cat("Optimal number of basis functions (GCV):", optimal_nbasis, "\n")
#> Optimal number of basis functions (GCV): 5
```

## Comparing All Methods

``` r
idx <- 3  # Pick a different curve

# Get smoothed versions
smooth_pspline <- fd_pspline$fdata$data[idx, ]
smooth_kernel <- fd_llr$data[idx, ]

# Use optimal basis from CV
coefs_opt <- fdata2basis(fd_noisy, nbasis = optimal_nbasis, type = "bspline")
smooth_basis <- basis2fdata(coefs_opt, argvals = age, type = "bspline")$data[idx, ]

# Plot comparison
plot(age, fd_noisy$data[idx, ], pch = 16, col = "gray60", cex = 0.8,
     main = "Smoothing Method Comparison",
     xlab = "Age (years)", ylab = "Height (cm)")
lines(age, smooth_pspline, col = "blue", lwd = 2)
lines(age, smooth_kernel, col = "darkgreen", lwd = 2)
lines(age, smooth_basis, col = "purple", lwd = 2)
lines(age, fd_raw$data[idx, ], col = "red", lty = 2, lwd = 1.5)

legend("bottomright",
       c("Noisy data", "P-spline", "Local Linear", "B-spline basis", "Original"),
       col = c("gray60", "blue", "darkgreen", "purple", "red"),
       pch = c(16, NA, NA, NA, NA), lty = c(NA, 1, 1, 1, 2),
       lwd = c(NA, 2, 2, 2, 1.5), cex = 0.9)
```

![](intro-to-smoothing_files/figure-html/compare-all-1.png)

## Mean and Variability

After smoothing, compute summary statistics:

``` r
# Use P-spline smoothed data
fd_smooth <- fd_pspline$fdata

# Mean curve
mean_curve <- mean(fd_smooth)

# Standard deviation at each point
sd_curve <- sd(fd_smooth)

# Plot mean with confidence band
plot(age, mean_curve$data[1, ], type = "l", lwd = 2, col = "blue",
     main = "Mean Growth Curve with Variability Band",
     xlab = "Age (years)", ylab = "Height (cm)",
     ylim = c(min(mean_curve$data - 2*sd_curve$data),
              max(mean_curve$data + 2*sd_curve$data)))

polygon(c(age, rev(age)),
        c(mean_curve$data[1, ] - 2*sd_curve$data[1, ],
          rev(mean_curve$data[1, ] + 2*sd_curve$data[1, ])),
        col = rgb(0, 0, 1, 0.2), border = NA)

lines(age, mean_curve$data[1, ], lwd = 2, col = "blue")
legend("bottomright", c("Mean", "Mean +/- 2 SD"),
       col = c("blue", rgb(0, 0, 1, 0.2)), lwd = c(2, 10))
```

![](intro-to-smoothing_files/figure-html/mean-variability-1.png)

## Summary: When to Use Each Method

| Method               | Best For                              | Key Parameter         |
|----------------------|---------------------------------------|-----------------------|
| **P-splines**        | General purpose, automatic tuning     | $\lambda$ (smoothing) |
| **Kernel smoothers** | Non-parametric, local adaptation      | $h$ (bandwidth)       |
| **Basis expansion**  | Dimensionality reduction, derivatives | $K$ (number of basis) |

**Recommendations:**

1.  **Start with P-splines** - automatic $\lambda$ selection works well
    in most cases
2.  **Use basis expansion** when you need derivatives (smoother
    derivatives)
3.  **Use kernel smoothers** for highly irregular sampling or adaptive
    smoothing

## References

- Ramsay, J.O. and Silverman, B.W. (2005). *Functional Data Analysis*.
  Springer.
- Eilers, P.H.C. and Marx, B.D. (1996). Flexible Smoothing with
  B-splines and Penalties. *Statistical Science*, 11(2), 89-121.
- Fan, J. and Gijbels, I. (1996). *Local Polynomial Modelling and Its
  Applications*. Chapman and Hall.
