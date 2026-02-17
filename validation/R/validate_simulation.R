#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Validating simulation eigenfunctions and eigenvalues ===\n")

results <- list()

# Common grid
t_grid <- seq(0, 1, length.out = 101)
n_grid <- length(t_grid)

# ---- (a) Fourier eigenfunctions (101 x 5) --------------------------------------
message("  Computing Fourier eigenfunctions (nbasis=5)...")
results$fourier_eigenfunctions <- tryCatch({
  nbasis <- 5L

  # phi_1(t) = 1 (constant)
  # phi_2(t) = sin(2*pi*t)
  # phi_3(t) = cos(2*pi*t)
  # phi_4(t) = sin(4*pi*t)
  # phi_5(t) = cos(4*pi*t)
  phi_mat <- matrix(0, nrow = n_grid, ncol = nbasis)
  phi_mat[, 1] <- 1
  phi_mat[, 2] <- sin(2 * pi * t_grid)
  phi_mat[, 3] <- cos(2 * pi * t_grid)
  phi_mat[, 4] <- sin(4 * pi * t_grid)
  phi_mat[, 5] <- cos(4 * pi * t_grid)

  message(sprintf("    Matrix: %d x %d", nrow(phi_mat), ncol(phi_mat)))

  list(
    nrow = n_grid,
    ncol = nbasis,
    data = as.numeric(phi_mat)  # column-major flat
  )
}, error = function(e) {
  warning(sprintf("Fourier eigenfunctions failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Wiener eigenfunctions (101 x 5) ---------------------------------------
message("  Computing Wiener eigenfunctions (nbasis=5)...")
results$wiener_eigenfunctions <- tryCatch({
  nbasis <- 5L

  # phi_k(t) = sqrt(2) * sin((k - 0.5) * pi * t) for k = 1..5
  phi_mat <- matrix(0, nrow = n_grid, ncol = nbasis)
  for (k in 1:nbasis) {
    phi_mat[, k] <- sqrt(2) * sin((k - 0.5) * pi * t_grid)
  }

  message(sprintf("    Matrix: %d x %d", nrow(phi_mat), ncol(phi_mat)))

  list(
    nrow = n_grid,
    ncol = nbasis,
    data = as.numeric(phi_mat)  # column-major flat
  )
}, error = function(e) {
  warning(sprintf("Wiener eigenfunctions failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Eigenvalues -----------------------------------------------------------
message("  Computing eigenvalue sequences (n=10)...")
results$eigenvalues <- tryCatch({
  n_eig <- 10L
  k_seq <- 1:n_eig

  # Linear: 1/k for k=1..10
  eig_linear <- 1.0 / k_seq

  # Exponential: exp(-(k-1)) for k=1..10
  eig_exponential <- exp(-(k_seq - 1))

  # Wiener: 1/((k-0.5)*pi)^2 for k=1..10
  eig_wiener <- 1.0 / ((k_seq - 0.5) * pi)^2

  message(sprintf("    Linear (first 3): %.6f, %.6f, %.6f",
                  eig_linear[1], eig_linear[2], eig_linear[3]))
  message(sprintf("    Exponential (first 3): %.6f, %.6f, %.6f",
                  eig_exponential[1], eig_exponential[2], eig_exponential[3]))
  message(sprintf("    Wiener (first 3): %.6f, %.6f, %.6f",
                  eig_wiener[1], eig_wiener[2], eig_wiener[3]))

  list(
    linear = as.numeric(eig_linear),
    exponential = as.numeric(eig_exponential),
    wiener = as.numeric(eig_wiener)
  )
}, error = function(e) {
  warning(sprintf("Eigenvalue computation failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results --------------------------------------------------------------
message("\n  Saving expected simulation values...")
save_expected(results, "simulation_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Simulation validation complete: %d/%d computations succeeded ===", computed, total))
