#!/usr/bin/env Rscript
# Validate basis function computations against R reference implementations.
# Computes B-spline and Fourier basis matrices, difference matrices,
# P-spline smoothing, and Fourier fitting, then saves expected outputs
# for comparison with Rust implementations.
#
# Reference package: fda
# Usage: Rscript validation/R/validate_basis.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda)

message("=== Validating basis functions ===\n")

results <- list()

# Common evaluation grid
argvals <- seq(0, 1, length.out = 101)

# ---- (a) B-spline basis matrix (101 x 14) -----------------------------------
message("  Computing B-spline basis matrix (nknots=10, order=4, nbasis=14)...")
results$bspline_matrix <- tryCatch({
  bspl_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 14, norder = 4)
  bspl_mat <- eval.basis(argvals, bspl_basis)
  # Flatten column-major (R default) to match Rust expectations
  list(
    nrow = nrow(bspl_mat),
    ncol = ncol(bspl_mat),
    data = as.numeric(bspl_mat)
  )
}, error = function(e) {
  warning(sprintf("B-spline basis matrix failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Fourier basis matrix (101 x 7) -------------------------------------
message("  Computing Fourier basis matrix (nbasis=7, period=1.0)...")
results$fourier_matrix <- tryCatch({
  four_basis <- create.fourier.basis(rangeval = c(0, 1), nbasis = 7, period = 1.0)
  four_mat <- eval.basis(argvals, four_basis)
  list(
    nrow = nrow(four_mat),
    ncol = ncol(four_mat),
    data = as.numeric(four_mat)
  )
}, error = function(e) {
  warning(sprintf("Fourier basis matrix failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Difference matrices -------------------------------------------------
message("  Computing difference matrices (n=10, order=1 and order=2)...")
results$diff_matrix_order1 <- tryCatch({
  d1 <- diff(diag(10), differences = 1)
  list(
    nrow = nrow(d1),
    ncol = ncol(d1),
    data = as.numeric(d1)
  )
}, error = function(e) {
  warning(sprintf("Difference matrix order 1 failed: %s", conditionMessage(e)))
  NULL
})

results$diff_matrix_order2 <- tryCatch({
  d2 <- diff(diag(10), differences = 2)
  list(
    nrow = nrow(d2),
    ncol = ncol(d2),
    data = as.numeric(d2)
  )
}, error = function(e) {
  warning(sprintf("Difference matrix order 2 failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) P-spline fit on noisy sine data ------------------------------------
message("  Computing P-spline fit (nbasis=15, lambda=0.01, Lfdobj=2)...")
results$pspline_fit <- tryCatch({
  sine_data <- load_data("noisy_sine_201")
  x <- sine_data$x
  y <- sine_data$y_noisy

  pspline_basis <- create.bspline.basis(rangeval = range(x), nbasis = 15, norder = 4)
  pspline_fdPar <- fdPar(pspline_basis, Lfdobj = 2, lambda = 0.01)
  pspline_smooth <- smooth.basis(argvals = x, y = y, fdParobj = pspline_fdPar)

  coefs <- as.numeric(pspline_smooth$fd$coefs)
  fitted <- as.numeric(eval.fd(x, pspline_smooth$fd))
  gcv <- as.numeric(pspline_smooth$gcv)

  list(
    coefficients = coefs,
    fitted_values = fitted,
    gcv = gcv
  )
}, error = function(e) {
  warning(sprintf("P-spline fit failed: %s", conditionMessage(e)))
  NULL
})

# ---- (e) Fourier fit on first curve of standard data -------------------------
message("  Computing Fourier fit (nbasis=7, first curve of standard_50x101)...")
results$fourier_fit <- tryCatch({
  std_data <- load_data("standard_50x101")
  n <- std_data$n
  m <- std_data$m
  grid <- std_data$argvals

  # Extract first curve from column-major data
  mat <- to_matrix(std_data$data, n, m)
  y1 <- mat[1, ]

  four_fit_basis <- create.fourier.basis(rangeval = range(grid), nbasis = 7, period = 1.0)
  four_smooth <- smooth.basis(argvals = grid, y = y1, fdParobj = four_fit_basis)

  coefs <- as.numeric(four_smooth$fd$coefs)
  fitted <- as.numeric(eval.fd(grid, four_smooth$fd))

  list(
    coefficients = coefs,
    fitted_values = fitted
  )
}, error = function(e) {
  warning(sprintf("Fourier fit failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ------------------------------------------------------------
message("\n  Saving expected basis values...")
save_expected(results, "basis_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Basis validation complete: %d/%d computations succeeded ===", computed, total))
