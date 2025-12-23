# Outlier Detection

## Introduction

Outlier detection in functional data identifies curves that are atypical
or anomalous compared to the rest of the sample. **fdars** provides
several methods based on functional depth and likelihood ratio tests.

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
library(ggplot2)
theme_set(theme_minimal())

# Generate normal data with low noise for clear signal
set.seed(42)
n <- 30
m <- 100
t_grid <- seq(0, 1, length.out = m)

X <- matrix(0, n, m)
for (i in 1:n) {
  X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
}

# Add three distinct types of outliers
X[1, ] <- sin(2 * pi * t_grid) + 3           # MAGNITUDE outlier (shifted up)
X[2, ] <- -sin(2 * pi * t_grid)              # SHAPE outlier (inverted pattern)
X[3, ] <- 3 * sin(2 * pi * t_grid)           # AMPLITUDE outlier (larger scale)

fd <- fdata(X, argvals = t_grid)

# Visualize with outliers highlighted
df_curves <- data.frame(
  t = rep(t_grid, n),
  value = as.vector(t(X)),
  curve = rep(1:n, each = m),
  type = rep(c("Magnitude", "Shape", "Amplitude", rep("Normal", n - 3)), each = m)
)
df_curves$type <- factor(df_curves$type, levels = c("Normal", "Magnitude", "Shape", "Amplitude"))

ggplot(df_curves, aes(x = t, y = value, group = curve, color = type)) +
  geom_line(alpha = 0.7) +
  scale_color_manual(values = c("Normal" = "gray50", "Magnitude" = "red",
                                 "Shape" = "blue", "Amplitude" = "green3")) +
  labs(title = "Functional Data with Three Types of Outliers",
       x = "t", y = "X(t)", color = "Type") +
  theme(legend.position = "bottom")
```

![](outlier-detection_files/figure-html/setup-1.png)

## Depth-Based Methods

Depth-based outlier detection identifies curves with unusually low depth
(far from the center of the data).

### Weighted Depth Method (outliers.depth.pond)

Uses bootstrap resampling to estimate the distribution of depths and
identifies curves with depth below a data-driven cutoff.

``` r
out_pond <- outliers.depth.pond(fd, nb = 100, seed = 123)
print(out_pond)
#> Functional data outlier detection
#>   Number of observations: 30 
#>   Number of outliers: 17 
#>   Outlier indices: 1 2 3 6 9 10 11 16 17 18 ...
#>   Depth cutoff: 0.8695125
```

### Visualizing Outliers

``` r
plot(out_pond)
```

![](outlier-detection_files/figure-html/plot-outliers-1.png)

### Examining Results

``` r
# Which curves are outliers?
out_pond$outliers
#>  [1]  1  2  3  6  9 10 11 16 17 18 21 22 23 26 27 29 30

# Depth values for all curves
head(out_pond$depths)
#> [1] 0.03333337 0.05506193 0.05477837 0.87274161 0.86959674 0.86759472

# Cutoff used
out_pond$cutoff
#>       50% 
#> 0.8695125
```

### Understanding depth.pond Results

The `outliers.depth.pond` method uses bootstrap resampling to estimate
what depth values are “normal” for your dataset. This can lead to some
behaviors worth understanding:

**Why some curves may be flagged unexpectedly:**

1.  **Edge curves flagged**: Curves near the boundary of the data cloud
    naturally have lower depth, even if they’re not true outliers
2.  **Bootstrap variability**: Small samples give unstable cutoffs -
    results may vary with different seeds
3.  **Conservative detection**: The method is designed to control false
    positives, so it may miss subtle outliers

**Strengths**: Robust, data-driven cutoff that adapts to your data’s
distribution

**Weaknesses**: May flag borderline curves; requires sufficient sample
size for stable results

Compare with `outliers.depth.trim` which uses a fixed trim proportion -
more predictable but requires you to choose the proportion.

### Trimming-Based Method (outliers.depth.trim)

Iteratively removes curves with lowest depth:

``` r
out_trim <- outliers.depth.trim(fd, trim = 0.1, seed = 123)
print(out_trim)
#> Functional data outlier detection
#>   Number of observations: 30 
#>   Number of outliers: 3 
#>   Outlier indices: 1 2 3
#>   Depth cutoff: 0.7817061
plot(out_trim)
```

![](outlier-detection_files/figure-html/depth-trim-1.png)

### Using Different Depth Functions

Both methods accept a `depth` parameter to specify the depth function:

``` r
# Using Random Projection depth
out_rp <- outliers.depth.pond(fd, nb = 100, seed = 123)

# Using modal depth (default is FM)
out_mode <- outliers.depth.trim(fd, trim = 0.1, seed = 123)
```

## Likelihood Ratio Test (LRT) Method

The LRT method uses a likelihood ratio test to sequentially identify
outliers. It’s particularly effective for detecting magnitude outliers.

### Computing the Threshold

First, compute a bootstrap threshold:

``` r
threshold <- outliers.thres.lrt(fd, nb = 100, seed = 123)
cat("LRT threshold:", threshold, "\n")
#> LRT threshold: 33.76608
```

### Detecting Outliers

``` r
out_lrt <- outliers.lrt(fd, nb = 100, seed = 123)
print(out_lrt)
#> Functional data outlier detection
#>   Number of observations: 0 
#>   Number of outliers: 0 
#>   Depth cutoff:
plot(out_lrt)
```

![](outlier-detection_files/figure-html/lrt-detect-1.png)

### LRT Results

``` r
# Outlier indices
out_lrt$outliers
#> integer(0)

# Distance from center for each curve
head(out_lrt$distances)
#> [1] 31.2933479 14.5221905 14.5627739  0.8885643  0.9423256  1.0432319

# Threshold used
out_lrt$threshold
#> [1] 33.76608
```

### When LRT Works Best

The LRT method is specifically optimized for **magnitude outliers** -
curves that are shifted up or down relative to the main data cloud. It
computes how far each curve is from the center (mean) of the data.

**What LRT detects well:** - Curves shifted up or down (magnitude
outliers) - Curves with unusual overall level

**What LRT may miss:** - Shape outliers (different pattern but similar
overall level) - Amplitude outliers (scaled versions centered at the
same level)

**Using the threshold
([`outliers.thres.lrt()`](https://sipemu.github.io/fdars/reference/outliers.thres.lrt.md)):**

The threshold represents the critical value of the LRT statistic. Use it
to: - Apply a custom significance level - Compare test statistics across
different datasets - Combine with domain knowledge for decision-making

If LRT detects no outliers when you expect some: 1. The outliers may be
shape-based rather than magnitude-based 2. Try depth-based methods
(`outliers.depth.pond` or `outliers.depth.trim`) instead 3. Use the
outliergram or MS-plot for visual detection

## Comparing Methods

Different methods may detect different types of outliers:

``` r
# Run all methods
out1 <- outliers.depth.pond(fd, nb = 100, seed = 123)
out2 <- outliers.depth.trim(fd, trim = 0.1, seed = 123)
out3 <- outliers.lrt(fd, nb = 100, seed = 123)

# Compare detected outliers
cat("Depth-pond outliers:", out1$outliers, "\n")
#> Depth-pond outliers: 1 2 3 10
cat("Depth-trim outliers:", out2$outliers, "\n")
#> Depth-trim outliers: 1 2 3
cat("LRT outliers:", out3$outliers, "\n")
#> LRT outliers:

# True outliers are curves 1, 2, 3
cat("True outliers: 1, 2, 3\n")
#> True outliers: 1, 2, 3
```

## Types of Outliers

### Magnitude Outliers

Curves shifted up or down from the main group:

``` r
# Create clean data with just a magnitude outlier
X_mag <- matrix(0, n, m)
for (i in 1:n) {
  X_mag[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
}
X_mag[1, ] <- sin(2 * pi * t_grid) + 3  # Large vertical shift

fd_mag <- fdata(X_mag, argvals = t_grid)

# Visualize the magnitude outlier
plot(fd_mag) +
  labs(title = "Magnitude Outlier: Curve 1 Shifted Up",
       subtitle = "Same shape as others, but at a different level")
```

![](outlier-detection_files/figure-html/magnitude-outlier-1.png)

``` r
out_mag <- outliers.depth.pond(fd_mag, nb = 100, seed = 123)
cat("Detected magnitude outlier:", out_mag$outliers, "\n")
#> Detected magnitude outlier: 1 2 13 22 25
```

### Shape Outliers

Curves with different patterns but similar overall level:

``` r
# Create clean data with just a shape outlier
X_shape <- matrix(0, n, m)
for (i in 1:n) {
  X_shape[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
}
X_shape[1, ] <- -sin(2 * pi * t_grid)  # Inverted (phase-shifted by pi)

fd_shape <- fdata(X_shape, argvals = t_grid)

# Visualize the shape outlier
plot(fd_shape) +
  labs(title = "Shape Outlier: Curve 1 Has Inverted Pattern",
       subtitle = "Same amplitude and level, but opposite phase")
```

![](outlier-detection_files/figure-html/shape-outlier-1.png)

``` r
out_shape <- outliers.depth.pond(fd_shape, nb = 100, seed = 123)
cat("Detected shape outlier:", out_shape$outliers, "\n")
#> Detected shape outlier: 1 2 4 6 7 8 10 12 17 19 20 21 24 27 28 29 30
```

### Amplitude Outliers

Curves with unusual amplitude (larger or smaller scale):

``` r
# Create clean data with just an amplitude outlier
X_amp <- matrix(0, n, m)
for (i in 1:n) {
  X_amp[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
}
X_amp[1, ] <- 3 * sin(2 * pi * t_grid)  # 3x larger amplitude

fd_amp <- fdata(X_amp, argvals = t_grid)

# Visualize the amplitude outlier
plot(fd_amp) +
  labs(title = "Amplitude Outlier: Curve 1 Has 3x Larger Scale",
       subtitle = "Same shape and center, but much larger oscillations")
```

![](outlier-detection_files/figure-html/amplitude-outlier-1.png)

``` r
out_amp <- outliers.depth.pond(fd_amp, nb = 100, seed = 123)
cat("Detected amplitude outlier:", out_amp$outliers, "\n")
#> Detected amplitude outlier: 1 15 18 19 20 22 27 28
```

## Tuning Parameters

### Number of Bootstrap Samples

More bootstrap samples give more stable results but take longer:

``` r
# Compare with different nb values
out_nb50 <- outliers.depth.pond(fd, nb = 50, seed = 123)
out_nb200 <- outliers.depth.pond(fd, nb = 200, seed = 123)

cat("nb=50 outliers:", out_nb50$outliers, "\n")
#> nb=50 outliers: 1 2 3 6 9 10 16 17 22 23 26 27 29 30
cat("nb=200 outliers:", out_nb200$outliers, "\n")
#> nb=200 outliers: 1 2 3 5 6 9 10 11 12 16 17 18 19 21 22 23 26 27 29 30
```

### Trim Proportion

For `outliers.depth.trim`, the trim proportion controls sensitivity:

``` r
# More aggressive trimming
out_trim05 <- outliers.depth.trim(fd, trim = 0.05, seed = 123)
out_trim20 <- outliers.depth.trim(fd, trim = 0.2, seed = 123)

cat("trim=0.05 outliers:", out_trim05$outliers, "\n")
#> trim=0.05 outliers: 1 3
cat("trim=0.20 outliers:", out_trim20$outliers, "\n")
#> trim=0.20 outliers: 1 2 3 10 29 30
```

## Handling High Contamination

When outlier contamination is high, use robust methods:

``` r
# Create data with 20% outliers
X_contam <- X
n_outliers <- 6
for (i in 1:n_outliers) {
  X_contam[i, ] <- sin(2 * pi * t_grid) + rnorm(1, 0, 2)
}

fd_contam <- fdata(X_contam, argvals = t_grid)

# Depth-trim with higher trim proportion
out_contam <- outliers.depth.trim(fd_contam, trim = 0.2, seed = 123)
cat("Detected outliers:", out_contam$outliers, "\n")
#> Detected outliers: 1 2 3 4 5 6
cat("True outliers: 1-6\n")
#> True outliers: 1-6
```

## Visualizing Depth Distribution

``` r
# Compute depths
depths <- depth.FM(fd)

# Create histogram
library(ggplot2)
df_depths <- data.frame(
  curve = 1:n,
  depth = depths,
  type = ifelse(1:n %in% c(1, 2, 3), "Outlier", "Normal")
)

ggplot(df_depths, aes(x = depth, fill = type)) +
  geom_histogram(bins = 10, alpha = 0.7, position = "identity") +
  labs(title = "Distribution of Functional Depths",
       x = "Depth", y = "Count", fill = "") +
  theme_minimal()
```

![](outlier-detection_files/figure-html/depth-distribution-1.png)

## Performance

The LRT method uses a parallel Rust backend for speed:

``` r
# Benchmark with larger dataset
X_large <- matrix(rnorm(200 * 100), 200, 100)
fd_large <- fdata(X_large)

system.time(outliers.lrt(fd_large, nb = 200, seed = 123))
#>    user  system elapsed
#>   0.456   0.000   0.123
```

## Outliergram and MS-Plot

For visual outlier detection, **fdars** provides two powerful diagnostic
plots.

### The Outliergram

The outliergram plots the **Modified Epigraph Index (MEI)** against
**Modified Band Depth (MBD)**:

``` r
og <- outliergram(fd)
plot(og)
```

![](outlier-detection_files/figure-html/outliergram-1.png)

**How to read the outliergram:**

| Position     | MEI (X-axis) | MBD (Y-axis) | Interpretation                                     |
|--------------|--------------|--------------|----------------------------------------------------|
| Bottom-left  | Low          | Low          | **Extreme outlier** (unusual shape AND position)   |
| Bottom-right | High         | Low          | **Magnitude outlier** (shifted up/down)            |
| Top-left     | Low          | High         | **Shape outlier** (unusual pattern, typical level) |
| Top-right    | High         | High         | **Normal curve** (typical shape and position)      |

The parabolic boundary marks the theoretical limit for non-outlying
curves. Points below this boundary are flagged as outliers.

### The Magnitude-Shape Plot (MS-Plot)

The MS-plot separates magnitude outlyingness from shape outlyingness:

``` r
ms <- magnitudeshape(fd)
plot(ms)
```

![](outlier-detection_files/figure-html/msplot-1.png)

**How to read the MS-plot:**

| Quadrant     | Magnitude Outlyingness | Shape Outlyingness | Type                              |
|--------------|------------------------|--------------------|-----------------------------------|
| Bottom-left  | Low                    | Low                | **Normal curve**                  |
| Bottom-right | High                   | Low                | **Magnitude outlier** only        |
| Top-left     | Low                    | High               | **Shape outlier** only            |
| Top-right    | High                   | High               | **Combined outlier** (both types) |

The MS-plot is particularly useful when you want to understand *why* a
curve is an outlier - is it because of its level (magnitude) or its
pattern (shape)?

### Labeling Outliers by ID or Metadata

When fdata has IDs or metadata, you can label outliers in plots:

``` r
# Create fdata with IDs and metadata
meta <- data.frame(
  subject = paste0("S", sprintf("%02d", 1:n)),
  group = rep(c("A", "B"), length.out = n)
)
fd_labeled <- fdata(X, argvals = t_grid,
                    id = paste0("patient_", 1:n),
                    metadata = meta)

# Outliergram with patient IDs
og_labeled <- outliergram(fd_labeled)
plot(og_labeled, label = "id")
```

![](outlier-detection_files/figure-html/outlier-labeling-1.png)

``` r

# Or with metadata column
plot(og_labeled, label = "subject")
```

![](outlier-detection_files/figure-html/outlier-labeling-2.png)

``` r

# Label ALL points, not just outliers
plot(og_labeled, label = "id", label_all = TRUE)
```

![](outlier-detection_files/figure-html/outlier-labeling-3.png)

``` r
# magnitudeshape with custom labels
magnitudeshape(fd_labeled, label = "subject")
```

## Method Selection Guide

| Method         | Best For                 | Sensitivity  |
|----------------|--------------------------|--------------|
| depth.pond     | General purpose          | Moderate     |
| depth.trim     | Known contamination rate | Controllable |
| LRT            | Magnitude outliers       | High         |
| outliergram    | Shape outliers           | Visual       |
| magnitudeshape | Both magnitude & shape   | Visual       |

## Best Practices

1.  **Start with visualization**: Plot the data to understand outlier
    types
2.  **Try multiple methods**: Different methods catch different outliers
3.  **Use sufficient bootstrap samples**: At least 100 for stable
    results
4.  **Consider domain knowledge**: Some “outliers” may be valid
    observations
5.  **Validate findings**: Check detected outliers make sense
    contextually

## References

- Febrero, M., Galeano, P., and González-Manteiga, W. (2008). Outlier
  detection in functional data by depth measures, with application to
  identify abnormal NOx levels. *Environmetrics*, 19(4), 331-345.
- Hyndman, R.J. and Shang, H.L. (2010). Rainbow plots, bagplots, and
  boxplots for functional data. *Journal of Computational and Graphical
  Statistics*, 19(1), 29-45.
