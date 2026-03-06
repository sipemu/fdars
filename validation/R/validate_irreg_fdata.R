#!/usr/bin/env Rscript
# Validate irregular functional data methods.
# Generates test data on irregular grids and computes expected values.
#
# Usage: Rscript validation/R/validate_irreg_fdata.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Validating irregular functional data methods ===\n")

set.seed(42)

results <- list()

# Generate 5 curves on irregular grids (random subsets of [0,1])
n_curves <- 5
n_points_per_curve <- c(30, 25, 35, 28, 32)
argvals_list <- list()
values_list <- list()

for (i in 1:n_curves) {
  t_i <- sort(runif(n_points_per_curve[i], min = 0, max = 1))
  # Ensure 0 and 1 are included
  t_i[1] <- 0.0
  t_i[length(t_i)] <- 1.0
  y_i <- sin(2 * pi * t_i) + (i - 1) * 0.3 + rnorm(length(t_i), sd = 0.05)
  argvals_list[[i]] <- t_i
  values_list[[i]] <- y_i
}

# Save the generated data
results$n_curves <- n_curves
results$n_points <- n_points_per_curve
results$argvals <- argvals_list
results$values <- values_list

# ---- (a) Trapezoidal integration on irregular grid ---------------------------
message("  Computing trapezoidal integrals...")
results$integrate <- tryCatch({
  integrals <- numeric(n_curves)
  for (i in 1:n_curves) {
    t_i <- argvals_list[[i]]
    y_i <- values_list[[i]]
    n_i <- length(t_i)
    # Trapezoidal rule
    integral <- 0
    for (j in 2:n_i) {
      integral <- integral + (t_i[j] - t_i[j-1]) * (y_i[j] + y_i[j-1]) / 2
    }
    integrals[i] <- integral
  }

  message(sprintf("    Integrals: %s", paste(sprintf("%.6f", integrals), collapse = ", ")))

  list(integrals = integrals)
}, error = function(e) {
  warning(sprintf("Integration failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) L2 norm on irregular grid ------------------------------------------
message("  Computing L2 norms...")
results$norm_l2 <- tryCatch({
  norms <- numeric(n_curves)
  for (i in 1:n_curves) {
    t_i <- argvals_list[[i]]
    y_i <- values_list[[i]]
    n_i <- length(t_i)
    integral_sq <- 0
    for (j in 2:n_i) {
      integral_sq <- integral_sq + (t_i[j] - t_i[j-1]) * (y_i[j]^2 + y_i[j-1]^2) / 2
    }
    norms[i] <- sqrt(integral_sq)
  }

  message(sprintf("    L2 norms: %s", paste(sprintf("%.6f", norms), collapse = ", ")))

  list(norms = norms)
}, error = function(e) {
  warning(sprintf("L2 norm failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Mean after interpolation to common grid -----------------------------
message("  Computing mean on common grid...")
results$mean_curve <- tryCatch({
  target_grid <- seq(0, 1, length.out = 51)
  interp_mat <- matrix(0, nrow = n_curves, ncol = length(target_grid))

  for (i in 1:n_curves) {
    interp_mat[i, ] <- approx(argvals_list[[i]], values_list[[i]],
                               xout = target_grid, rule = 2)$y
  }

  mean_values <- colMeans(interp_mat)

  message(sprintf("    Mean curve range: [%.4f, %.4f]", min(mean_values), max(mean_values)))

  list(
    target_grid = target_grid,
    mean_values = as.numeric(mean_values)
  )
}, error = function(e) {
  warning(sprintf("Mean curve failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) Interpolation to regular grid (linear) -----------------------------
message("  Computing interpolation to regular grid...")
results$to_regular <- tryCatch({
  target_grid <- seq(0, 1, length.out = 51)
  interp_mat <- matrix(0, nrow = n_curves, ncol = length(target_grid))

  for (i in 1:n_curves) {
    interp_mat[i, ] <- approx(argvals_list[[i]], values_list[[i]],
                               xout = target_grid, rule = 2)$y
  }

  list(
    target_grid = target_grid,
    data = as.numeric(interp_mat),  # column-major
    n = n_curves,
    m = length(target_grid)
  )
}, error = function(e) {
  warning(sprintf("Interpolation to regular grid failed: %s", conditionMessage(e)))
  NULL
})

# ---- (e) Pairwise L2 distances on irregular grids ----------------------------
message("  Computing pairwise L2 distances...")
results$metric_lp <- tryCatch({
  # To compute L2 distance between curves on different irregular grids,
  # interpolate both to a fine common grid, then integrate
  fine_grid <- seq(0, 1, length.out = 201)
  interp_mat <- matrix(0, nrow = n_curves, ncol = length(fine_grid))
  for (i in 1:n_curves) {
    interp_mat[i, ] <- approx(argvals_list[[i]], values_list[[i]],
                               xout = fine_grid, rule = 2)$y
  }

  # Pairwise L2 distance
  h <- fine_grid[2] - fine_grid[1]
  dist_mat <- matrix(0, nrow = n_curves, ncol = n_curves)
  for (i in 1:n_curves) {
    for (j in 1:n_curves) {
      if (i < j) {
        diff_sq <- (interp_mat[i, ] - interp_mat[j, ])^2
        d_ij <- sqrt(sum(diff_sq) * h)
        dist_mat[i, j] <- d_ij
        dist_mat[j, i] <- d_ij
      }
    }
  }

  message(sprintf("    Distance matrix shape: %dx%d", nrow(dist_mat), ncol(dist_mat)))

  list(
    n = n_curves,
    data = as.numeric(dist_mat)
  )
}, error = function(e) {
  warning(sprintf("Pairwise L2 distances failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ----------------------------------------------------------
message("\n  Saving expected irreg_fdata values...")
save_expected(results, "irreg_fdata_expected")

computed <- sum(!vapply(results[c("integrate", "norm_l2", "mean_curve", "to_regular", "metric_lp")], is.null, logical(1)))
total <- 5
message(sprintf("\n=== Irreg fdata validation complete: %d/%d computations succeeded ===", computed, total))
