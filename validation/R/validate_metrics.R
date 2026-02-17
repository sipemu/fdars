#!/usr/bin/env Rscript
# Validate metric/distance functions against R reference implementations.
# Computes Lp distances, DTW distances, Fourier semimetric, and hshift
# semimetric, then saves expected outputs for comparison with Rust
# implementations.
#
# Reference packages: fda.usc, dtw
# Usage: Rscript validation/R/validate_metrics.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)
library(dtw)

message("=== Validating metric/distance functions ===\n")

# Load standard test data (50 curves, 101 grid points)
dat <- load_data("standard_50x101")
n <- dat$n
m <- dat$m
argvals <- dat$argvals

# Build fda.usc::fdata object (full dataset)
fdataobj <- to_fdata(dat$data, n, m, argvals)

# Build raw matrix for subsetting
mat <- to_matrix(dat$data, n, m)

results <- list()

# ---- (a) Lp self-distances (L2) on first 10 curves --------------------------
message("  Computing Lp (L2) distance matrix on first 10 curves...")
results$lp_l2 <- tryCatch({
  fdata_sub10 <- fda.usc::fdata(mat[1:10, ], argvals = argvals)
  dist_mat <- fda.usc::metric.lp(fdata_sub10, lp = 2)
  dist_mat <- as.matrix(dist_mat)
  list(
    n = nrow(dist_mat),
    data = as.numeric(dist_mat)
  )
}, error = function(e) {
  warning(sprintf("Lp L2 distance failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) DTW distances for curves 1 and 2 -----------------------------------
message("  Computing DTW distance (symmetric2) for curves 1 and 2...")
results$dtw_symmetric2 <- tryCatch({
  x <- mat[1, ]
  y <- mat[2, ]
  alignment <- dtw::dtw(x, y, step.pattern = symmetric2)
  alignment$distance
}, error = function(e) {
  warning(sprintf("DTW symmetric2 failed: %s", conditionMessage(e)))
  NULL
})

message("  Computing DTW distance (Sakoe-Chiba window=5) for curves 1 and 2...")
results$dtw_sakoechiba <- tryCatch({
  x <- mat[1, ]
  y <- mat[2, ]
  alignment <- dtw::dtw(x, y, window.type = "sakoechiba", window.size = 5)
  alignment$distance
}, error = function(e) {
  warning(sprintf("DTW Sakoe-Chiba failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Fourier semimetric on first 10 curves ------------------------------
message("  Computing Fourier semimetric (nderiv=0, nbasis=5) on first 10 curves...")
results$semimetric_fourier <- tryCatch({
  fdata_sub10 <- fda.usc::fdata(mat[1:10, ], argvals = argvals)
  dist_mat <- fda.usc::semimetric.fourier(fdata_sub10, fdata_sub10, nderiv = 0, nbasis = 5)
  dist_mat <- as.matrix(dist_mat)
  list(
    n = nrow(dist_mat),
    data = as.numeric(dist_mat)
  )
}, error = function(e) {
  warning(sprintf("Fourier semimetric failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) Hshift semimetric on first 5 curves (slow) -------------------------
message("  Computing hshift semimetric on first 5 curves (this may be slow)...")
results$semimetric_hshift <- tryCatch({
  fdata_sub5 <- fda.usc::fdata(mat[1:5, ], argvals = argvals)
  dist_mat <- fda.usc::semimetric.hshift(fdata_sub5, fdata_sub5)
  dist_mat <- as.matrix(dist_mat)
  list(
    n = nrow(dist_mat),
    data = as.numeric(dist_mat)
  )
}, error = function(e) {
  warning(sprintf("Hshift semimetric failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ------------------------------------------------------------
message("\n  Saving expected metric values...")
save_expected(results, "metrics_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Metrics validation complete: %d/%d computations succeeded ===", computed, total))
