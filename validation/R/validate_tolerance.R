#!/usr/bin/env Rscript
# Validate tolerance band methods against R reference implementations.
# Requires: fda.usc, KernSmooth, jsonlite
#
# Usage: Rscript validation/R/validate_tolerance.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)
library(KernSmooth)

message("=== Tolerance band validation ===")

d <- load_data("standard_50x101")
mat <- to_matrix(d$data, d$n, d$m)
argvals <- d$argvals

result <- list()

# ---------------------------------------------------------------------------
# 1. FPCA tolerance band: center = pointwise mean (deterministic)
# ---------------------------------------------------------------------------
message("  Computing FPCA band center...")
result$fpca_center <- as.vector(colMeans(mat))

# Also compute top 3 FPCA eigenvalues for reference
message("  Computing FPCA eigenvalues...")
fd <- fdata(mat, argvals = argvals)
pc <- fdata2pc(fd, ncomp = 3)
result$fpca_eigenvalues <- as.vector(pc$d[1:3]^2 / (d$n - 1))

# ---------------------------------------------------------------------------
# 2. Conformal prediction band: training-set mean and conformal quantile
# ---------------------------------------------------------------------------
message("  Computing conformal prediction band...")
set.seed(42)
n <- d$n
cal_frac <- 0.2
n_cal <- max(1, min(n - 2, round(n * cal_frac)))
n_train <- n - n_cal

# Random permutation (same logic as Rust: StdRng::seed_from_u64(42))
perm <- sample.int(n)
train_idx <- perm[1:n_train]
cal_idx <- perm[(n_train + 1):n]

train_mean <- colMeans(mat[train_idx, , drop = FALSE])
result$conformal_center <- as.vector(train_mean)

# Sup-norm non-conformity scores on calibration set
cal_scores <- apply(mat[cal_idx, , drop = FALSE], 1, function(row) max(abs(row - train_mean)))
# Conformal quantile: ceil((n_cal + 1) * coverage) / n_cal
coverage <- 0.95
level <- min(1, ceiling((n_cal + 1) * coverage) / n_cal)
result$conformal_quantile <- as.numeric(quantile(cal_scores, probs = level, type = 1))

# ---------------------------------------------------------------------------
# 3. Degras SCB: smoothed mean via locpoly + Gaussian multiplier bootstrap
# ---------------------------------------------------------------------------
message("  Computing Degras SCB...")
raw_mean <- colMeans(mat)

# Local polynomial smoothing (degree=1, bw=0.15, Epanechnikov kernel)
lp_fit <- locpoly(argvals, raw_mean, bandwidth = 0.15, degree = 1,
                  gridsize = length(argvals), range.x = range(argvals))
# locpoly returns on its own grid; interpolate back to argvals
smoothed_mean <- approx(lp_fit$x, lp_fit$y, xout = argvals, rule = 2)$y
result$degras_center <- as.vector(smoothed_mean)

# Residual sigma
sigma_hat <- sqrt(colMeans((mat - matrix(smoothed_mean, nrow = n, ncol = d$m, byrow = TRUE))^2))
sigma_hat <- pmax(sigma_hat, 1e-15)

# Gaussian multiplier bootstrap for critical value
nb <- 500
sqrt_n <- sqrt(n)
set.seed(42)
sup_stats <- numeric(nb)
for (b in 1:nb) {
  weights <- rnorm(n)
  z <- numeric(d$m)
  for (j in 1:d$m) {
    z[j] <- abs(sum(weights * (mat[, j] - smoothed_mean[j])) / (sqrt_n * sigma_hat[j]))
  }
  sup_stats[b] <- max(z)
}
confidence <- 0.95
result$degras_critical_value <- as.numeric(quantile(sup_stats, probs = confidence, type = 1))

save_expected(result, "tolerance_expected")
message("=== Tolerance band validation complete ===")
