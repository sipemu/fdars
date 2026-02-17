#!/usr/bin/env Rscript
# Validate regression-related computations against R reference implementations.
# Computes FPCA via SVD, ridge regression via glmnet, and PLS regression via pls
# on the regression_30x51 dataset, then saves expected outputs for comparison
# with Rust implementations.
#
# Reference packages: glmnet, pls
# Usage: Rscript validation/R/validate_regression.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(glmnet)
library(pls)

message("=== Validating regression computations ===\n")

# Load regression test data (30 curves, 51 grid points, with response y)
dat <- load_data("regression_30x51")
n <- dat$n
m <- dat$m
argvals <- dat$argvals
y <- dat$y

# Build raw matrix (n x m)
mat <- to_matrix(dat$data, n, m)

results <- list()

# ---- (a) FPCA via SVD on centered matrix -------------------------------------
message("  Computing FPCA via SVD on centered matrix...")
results$fpca_svd <- tryCatch({
  # Center the matrix (subtract column means)
  col_means <- colMeans(mat)
  centered <- sweep(mat, 2, col_means)

  # SVD
  sv <- svd(centered)

  # Extract first 3 components
  ncomp <- 3
  singular_values <- sv$d[1:ncomp]
  scores <- sv$u[, 1:ncomp, drop = FALSE]  # n x ncomp
  loadings <- sv$v[, 1:ncomp, drop = FALSE]  # m x ncomp

  # Proportion of variance explained
  total_var <- sum(sv$d^2)
  prop_var <- (sv$d[1:ncomp]^2) / total_var

  message(sprintf("    Singular values (first 3): %.6f, %.6f, %.6f",
                  singular_values[1], singular_values[2], singular_values[3]))
  message(sprintf("    Proportion variance (first 3): %.6f, %.6f, %.6f",
                  prop_var[1], prop_var[2], prop_var[3]))

  list(
    singular_values = singular_values,
    scores = as.numeric(scores),       # n x ncomp, column-major flat
    loadings = as.numeric(loadings),   # m x ncomp, column-major flat
    col_means = col_means,
    proportion_variance = prop_var
  )
}, error = function(e) {
  warning(sprintf("FPCA SVD failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Ridge regression (alpha=0, lambda=1.0) -----------------------------
message("  Computing ridge regression (alpha=0, lambda=1.0)...")
results$ridge <- tryCatch({
  fit <- glmnet::glmnet(x = mat, y = y, alpha = 0, lambda = 1.0,
                        intercept = TRUE, standardize = FALSE)
  intercept <- as.numeric(fit$a0)
  coefficients <- as.numeric(fit$beta)

  message(sprintf("    Intercept: %.6f", intercept))
  message(sprintf("    Coefficients: %d values (first 3: %.6f, %.6f, %.6f)",
                  length(coefficients),
                  coefficients[1], coefficients[2], coefficients[3]))

  list(
    intercept = intercept,
    coefficients = coefficients
  )
}, error = function(e) {
  warning(sprintf("Ridge regression failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) PLS regression (2 components) ---------------------------------------
message("  Computing PLS regression (2 components, oscorespls)...")
results$pls <- tryCatch({
  ncomp <- 2
  X <- mat
  fit <- pls::plsr(y ~ X, ncomp = ncomp, method = "oscorespls", validation = "none")

  # Extract scores (n x ncomp), loadings (m x ncomp), weights (m x ncomp)
  scores_mat <- fit$scores                     # n x ncomp
  loadings_mat <- fit$loadings                 # m x ncomp
  weights_mat <- fit$loading.weights           # m x ncomp (projection weights)

  message(sprintf("    Scores: %d x %d", nrow(scores_mat), ncol(scores_mat)))
  message(sprintf("    Loadings: %d x %d", nrow(loadings_mat), ncol(loadings_mat)))
  message(sprintf("    Weights: %d x %d", nrow(weights_mat), ncol(weights_mat)))

  list(
    scores = as.numeric(scores_mat),       # n x ncomp, column-major flat
    loadings = as.numeric(loadings_mat),   # m x ncomp, column-major flat
    weights = as.numeric(weights_mat)      # m x ncomp, column-major flat
  )
}, error = function(e) {
  warning(sprintf("PLS regression failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ------------------------------------------------------------
message("\n  Saving expected regression values...")
save_expected(results, "regression_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Regression validation complete: %d/%d computations succeeded ===", computed, total))
