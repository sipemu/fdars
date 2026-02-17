#!/usr/bin/env Rscript
# Generate shared test datasets for fdars-core validation.
# Writes JSON files to validation/data/
#
# Usage: Rscript validation/R/generate_test_data.R

library(jsonlite)

# Resolve script directory robustly
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  return(getwd())
}

out_dir <- file.path(dirname(get_script_dir()), "data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write_json_file <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  write_json(obj, path, digits = 17, auto_unbox = TRUE)
  message(sprintf("  wrote %s", path))
}

set.seed(42)

# ---------------------------------------------------------------------------
# 1. Standard functional dataset: 50 curves on 101 grid points
# ---------------------------------------------------------------------------
message("Generating standard functional dataset (50x101)...")
n <- 50L
m <- 101L
argvals <- seq(0, 1, length.out = m)

# Generate curves: mean + random Fourier components + noise
mean_fn <- sin(2 * pi * argvals)
curves <- matrix(0, nrow = n, ncol = m)
for (i in 1:n) {
  a1 <- rnorm(1, 0, 0.3)
  a2 <- rnorm(1, 0, 0.2)
  a3 <- rnorm(1, 0, 0.1)
  curves[i, ] <- mean_fn + a1 * sin(2 * pi * argvals) +
    a2 * cos(4 * pi * argvals) +
    a3 * sin(6 * pi * argvals) +
    rnorm(m, 0, 0.05)
}

# Convert to column-major flat vector (matching Rust layout: data[i + j*n])
data_colmajor <- as.vector(curves)  # R stores matrices column-major

write_json_file(list(
  n = n, m = m,
  argvals = argvals,
  data = data_colmajor
), "standard_50x101")

# ---------------------------------------------------------------------------
# 2. Three-cluster dataset: 60 curves (20 per cluster) on 51 grid points
# ---------------------------------------------------------------------------
message("Generating 3-cluster dataset (60x51)...")
n_clust <- 60L
m_clust <- 51L
k <- 3L
argvals_clust <- seq(0, 1, length.out = m_clust)

cluster_means <- list(
  sin(2 * pi * argvals_clust),
  cos(2 * pi * argvals_clust),
  0.5 * sin(4 * pi * argvals_clust)
)

curves_clust <- matrix(0, nrow = n_clust, ncol = m_clust)
true_labels <- integer(n_clust)
for (c_idx in 1:k) {
  for (j in 1:20) {
    i <- (c_idx - 1) * 20 + j
    true_labels[i] <- c_idx
    curves_clust[i, ] <- cluster_means[[c_idx]] + rnorm(m_clust, 0, 0.15)
  }
}

write_json_file(list(
  n = n_clust, m = m_clust, k = k,
  argvals = argvals_clust,
  data = as.vector(curves_clust),
  true_labels = true_labels
), "clusters_60x51")

# ---------------------------------------------------------------------------
# 3. Noisy sine for smoothing validation: 201 points
# ---------------------------------------------------------------------------
message("Generating noisy sine (201 points)...")
m_smooth <- 201L
x_smooth <- seq(0, 1, length.out = m_smooth)
y_true <- sin(2 * pi * x_smooth)
y_noisy <- y_true + rnorm(m_smooth, 0, 0.2)

write_json_file(list(
  x = x_smooth,
  y_true = y_true,
  y_noisy = y_noisy,
  m = m_smooth
), "noisy_sine_201")

# ---------------------------------------------------------------------------
# 4. Time series with known period for seasonal validation
# ---------------------------------------------------------------------------
message("Generating seasonal time series...")
n_ts <- 200L
t_ts <- seq(0, 10, length.out = n_ts)
period <- 2.0
ts_pure <- sin(2 * pi * t_ts / period)
ts_noisy <- ts_pure + rnorm(n_ts, 0, 0.1)
ts_trend <- ts_pure + 0.3 * t_ts + rnorm(n_ts, 0, 0.1)
ts_multi <- sin(2 * pi * t_ts / 2.0) + 0.5 * sin(2 * pi * t_ts / 0.7) + rnorm(n_ts, 0, 0.05)

write_json_file(list(
  t = t_ts,
  pure_sine = ts_pure,
  noisy_sine = ts_noisy,
  with_trend = ts_trend,
  multi_period = ts_multi,
  n = n_ts,
  period = period
), "seasonal_200")

# ---------------------------------------------------------------------------
# 5. Regression dataset: 30 curves + scalar response
# ---------------------------------------------------------------------------
message("Generating regression dataset (30x51)...")
n_reg <- 30L
m_reg <- 51L
argvals_reg <- seq(0, 1, length.out = m_reg)

curves_reg <- matrix(0, nrow = n_reg, ncol = m_reg)
y_reg <- numeric(n_reg)
for (i in 1:n_reg) {
  a <- rnorm(1, 0, 1)
  b <- rnorm(1, 0, 0.5)
  curves_reg[i, ] <- a * sin(2 * pi * argvals_reg) + b * cos(2 * pi * argvals_reg) + rnorm(m_reg, 0, 0.05)
  y_reg[i] <- 2 * a + 0.5 * b + rnorm(1, 0, 0.1)
}

write_json_file(list(
  n = n_reg, m = m_reg,
  argvals = argvals_reg,
  data = as.vector(curves_reg),
  y = y_reg
), "regression_30x51")

# ---------------------------------------------------------------------------
# 6. Outlier dataset: 50 curves with 3 outliers
# ---------------------------------------------------------------------------
message("Generating outlier dataset (50x101)...")
n_out <- 50L
m_out <- 101L
argvals_out <- seq(0, 1, length.out = m_out)

curves_out <- matrix(0, nrow = n_out, ncol = m_out)
outlier_indices <- c(5L, 23L, 41L)
for (i in 1:n_out) {
  if (i %in% outlier_indices) {
    curves_out[i, ] <- sin(2 * pi * argvals_out) + 3.0 + rnorm(m_out, 0, 0.05)
  } else {
    curves_out[i, ] <- sin(2 * pi * argvals_out) + rnorm(1, 0, 0.2) + rnorm(m_out, 0, 0.05)
  }
}

write_json_file(list(
  n = n_out, m = m_out,
  argvals = argvals_out,
  data = as.vector(curves_out),
  outlier_indices = outlier_indices
), "outliers_50x101")

# ---------------------------------------------------------------------------
# 7. Decomposition dataset: trend + seasonal + noise
# ---------------------------------------------------------------------------
message("Generating decomposition dataset...")
n_dec <- 10L
m_dec <- 120L
argvals_dec <- seq(0, 10, length.out = m_dec)
period_dec <- 2.5

curves_dec <- matrix(0, nrow = n_dec, ncol = m_dec)
for (i in 1:n_dec) {
  trend_i <- 0.5 * argvals_dec + rnorm(1, 0, 0.2)
  seasonal_i <- sin(2 * pi * argvals_dec / period_dec) * (1 + rnorm(1, 0, 0.1))
  curves_dec[i, ] <- trend_i + seasonal_i + rnorm(m_dec, 0, 0.1)
}

write_json_file(list(
  n = n_dec, m = m_dec,
  argvals = argvals_dec,
  data = as.vector(curves_dec),
  period = period_dec
), "decomposition_10x120")

message("\n=== All test data generated ===")
