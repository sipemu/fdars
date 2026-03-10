#!/usr/bin/env Rscript
# Validate basis-penalized smoothing against R's fda::smooth.basis.
#
# Reference package: fda
# Usage: Rscript validation/R/validate_smooth_basis.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda)

message("=== Validating smooth.basis ===\n")

results <- list()

# Common grid
argvals <- seq(0, 1, length.out = 101)
n <- 5
m <- length(argvals)

# Generate test data: noisy sine curves
set.seed(42)
data_mat <- matrix(0, nrow = n, ncol = m)
for (i in 1:n) {
  data_mat[i, ] <- sin(2 * pi * argvals) + rnorm(m, sd = 0.1)
}

# ---- (a) B-spline smooth.basis with lambda=1e-4 ---
message("  Computing smooth.basis with B-splines...")
results$bspline_smooth <- tryCatch({
  bspl_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 15, norder = 4)
  fdpar_obj <- fdPar(bspl_basis, Lfdobj = 2, lambda = 1e-4)
  sm <- smooth.basis(argvals, t(data_mat), fdpar_obj)

  # Extract results
  coefs <- t(sm$fd$coefs)  # n x nbasis
  fitted <- t(eval.fd(argvals, sm$fd))  # n x m

  # Penalty matrix
  pen_mat <- eval.penalty(bspl_basis, 2)

  list(
    data = as.numeric(t(data_mat)),
    n = n,
    m = m,
    nbasis = 15,
    lambda = 1e-4,
    coefficients = as.numeric(t(coefs)),
    fitted = as.numeric(t(fitted)),
    edf = sm$df,
    gcv = sm$gcv,
    penalty_matrix = as.numeric(pen_mat),
    penalty_dim = nrow(pen_mat)
  )
}, error = function(e) {
  warning(sprintf("B-spline smooth.basis failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Fourier smooth.basis ---
message("  Computing smooth.basis with Fourier basis...")
results$fourier_smooth <- tryCatch({
  four_basis <- create.fourier.basis(rangeval = c(0, 1), nbasis = 7, period = 1.0)
  fdpar_obj <- fdPar(four_basis, Lfdobj = 2, lambda = 1e-6)
  sm <- smooth.basis(argvals, t(data_mat), fdpar_obj)

  coefs <- t(sm$fd$coefs)
  fitted <- t(eval.fd(argvals, sm$fd))
  pen_mat <- eval.penalty(four_basis, 2)

  list(
    data = as.numeric(t(data_mat)),
    nbasis = 7,
    period = 1.0,
    lambda = 1e-6,
    coefficients = as.numeric(t(coefs)),
    fitted = as.numeric(t(fitted)),
    edf = sm$df,
    gcv = sm$gcv,
    penalty_matrix = as.numeric(pen_mat),
    penalty_dim = nrow(pen_mat)
  )
}, error = function(e) {
  warning(sprintf("Fourier smooth.basis failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) GCV-optimal lambda ---
message("  Computing GCV-optimal smooth.basis...")
results$gcv_optimal <- tryCatch({
  bspl_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 15, norder = 4)

  log_lambdas <- seq(-8, 4, length.out = 25)
  gcv_values <- numeric(length(log_lambdas))

  for (k in seq_along(log_lambdas)) {
    lam <- 10^log_lambdas[k]
    fdpar_obj <- fdPar(bspl_basis, Lfdobj = 2, lambda = lam)
    sm <- smooth.basis(argvals, t(data_mat), fdpar_obj)
    gcv_values[k] <- mean(sm$gcv)
  }

  best_idx <- which.min(gcv_values)

  list(
    log_lambdas = log_lambdas,
    gcv_values = gcv_values,
    best_log_lambda = log_lambdas[best_idx],
    best_gcv = gcv_values[best_idx]
  )
}, error = function(e) {
  warning(sprintf("GCV optimization failed: %s", conditionMessage(e)))
  NULL
})

save_expected(results, "smooth_basis")
message("Done.\n")
