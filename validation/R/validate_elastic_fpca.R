#!/usr/bin/env Rscript
# Validate elastic FPCA (vert/horiz/joint) against R's fdasrvf.
#
# Reference package: fdasrvf
# Usage: Rscript validation/R/validate_elastic_fpca.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fdasrvf)

message("=== Validating elastic FPCA ===\n")

results <- list()

# Generate test data: shifted sine curves
set.seed(42)
n <- 15
m <- 51
argvals <- seq(0, 1, length.out = m)
data_mat <- matrix(0, nrow = m, ncol = n)  # fdasrvf uses m x n format
for (i in 1:n) {
  shift <- 0.1 * (i - n / 2)
  scale <- 1.0 + 0.2 * (i / n)
  data_mat[, i] <- scale * sin(2 * pi * (argvals + shift))
}

# Store input data
results$data <- as.numeric(data_mat)
results$n <- n
results$m <- m
results$argvals <- argvals

# First run time_warping to get aligned data
message("  Running time_warping alignment...")
warp_result <- tryCatch({
  time_warping(data_mat, argvals, max_iter = 10)
}, error = function(e) {
  warning(sprintf("time_warping failed: %s", conditionMessage(e)))
  NULL
})

if (!is.null(warp_result)) {
  # Store alignment results for Rust comparison
  results$aligned_data <- as.numeric(warp_result$fn)
  # Warping functions: m x n matrix (rows=grid points, cols=curves)
  gam_mat <- warp_result$warping_functions
  results$gammas <- as.numeric(gam_mat)  # column-major: [gam1_pts, gam2_pts, ...]
  results$mean <- as.numeric(warp_result$fmean)
  results$mean_srsf <- as.numeric(warp_result$mqn)
  # Also store aligned SRSFs
  results$aligned_srsfs <- as.numeric(warp_result$qn)

  # ---- (a) Vertical FPCA ---
  message("  Computing vertical FPCA...")
  results$vert_fpca <- tryCatch({
    vfpca <- vertFPCA(warp_result, no = 3, showplot = FALSE)
    list(
      ncomp = 3,
      scores = as.numeric(vfpca$coef),
      scores_nrow = nrow(vfpca$coef),
      scores_ncol = ncol(vfpca$coef),
      eigenvalues = vfpca$latent,
      cumulative_variance = cumsum(vfpca$latent) / sum(vfpca$latent)
    )
  }, error = function(e) {
    warning(sprintf("Vertical FPCA failed: %s", conditionMessage(e)))
    NULL
  })

  # ---- (b) Horizontal FPCA ---
  message("  Computing horizontal FPCA...")
  results$horiz_fpca <- tryCatch({
    hfpca <- horizFPCA(warp_result, no = 3, showplot = FALSE)
    list(
      scores = as.numeric(hfpca$coef),
      scores_nrow = nrow(hfpca$coef),
      scores_ncol = ncol(hfpca$coef),
      eigenvalues = hfpca$latent,
      cumulative_variance = cumsum(hfpca$latent) / sum(hfpca$latent)
    )
  }, error = function(e) {
    warning(sprintf("Horizontal FPCA failed: %s", conditionMessage(e)))
    NULL
  })

  # ---- (c) Joint FPCA ---
  message("  Computing joint FPCA...")
  results$joint_fpca <- tryCatch({
    jfpca <- jointFPCA(warp_result, no = 3, showplot = FALSE)
    list(
      scores = as.numeric(jfpca$coef),
      scores_nrow = nrow(jfpca$coef),
      scores_ncol = ncol(jfpca$coef),
      eigenvalues = jfpca$latent,
      cumulative_variance = cumsum(jfpca$latent) / sum(jfpca$latent),
      balance_c = jfpca$C
    )
  }, error = function(e) {
    warning(sprintf("Joint FPCA failed: %s", conditionMessage(e)))
    NULL
  })
} else {
  message("  Skipping FPCA (alignment failed)")
}

save_expected(results, "elastic_fpca")
message("Done.\n")
