# Validation script for B-spline and P-spline functions
# Compares fdars (Rust) vs mgcv (R) implementations

library(mgcv)
library(fdars)

# Set seed for reproducibility
set.seed(42)

cat("========================================\n")
cat("B-spline and P-spline Validation\n")
cat("Comparing fdars (Rust) vs mgcv (R)\n")
cat("========================================\n\n")

# -----------------------------------------------------------------------------
# Test 1: B-spline basis comparison at boundaries
# -----------------------------------------------------------------------------
cat("----------------------------------------\n")
cat("Test 1: B-spline Basis at Boundaries\n")
cat("----------------------------------------\n\n")

# Create evaluation points including exact boundaries
t_grid <- seq(0, 1, length.out = 101)
n_knots <- 10

# mgcv B-spline basis (cubic, k = n_knots + 3 for cubic splines)
# mgcv uses different parameterization, so we compare behavior not exact values
mgcv_basis <- smoothCon(s(x, bs = "ps", k = n_knots + 3, m = c(2, 2)),
                        data = data.frame(x = t_grid),
                        absorb.cons = FALSE)[[1]]$X

# fdars B-spline basis (via pspline with lambda=0 for pure B-spline fit)
# We'll test by fitting a known function and checking boundary behavior
true_func <- sin(2 * pi * t_grid)
fd_test <- fdata(matrix(true_func, nrow = 1), argvals = t_grid)

# Check that basis sums to approximately 1 (partition of unity)
# For properly constructed B-splines, sum of basis functions at any point = 1
mgcv_row_sums <- rowSums(mgcv_basis)
cat("mgcv basis row sums (should be ~1):\n")
cat("  Min:", min(mgcv_row_sums), "\n")
cat("  Max:", max(mgcv_row_sums), "\n")
cat("  At t=0:", mgcv_row_sums[1], "\n")
cat("  At t=1:", mgcv_row_sums[length(mgcv_row_sums)], "\n\n")

# -----------------------------------------------------------------------------
# Test 2: P-spline smoothing comparison
# -----------------------------------------------------------------------------
cat("----------------------------------------\n")
cat("Test 2: P-spline Smoothing Comparison\n")
cat("----------------------------------------\n\n")

# Generate noisy data
n_obs <- 100
t_obs <- seq(0, 1, length.out = n_obs)
true_signal <- sin(2 * pi * t_obs)
noise <- rnorm(n_obs, sd = 0.3)
y_noisy <- true_signal + noise

# Fit with mgcv
mgcv_fit <- gam(y ~ s(t, bs = "ps", k = 20, m = c(2, 2)),
                data = data.frame(y = y_noisy, t = t_obs),
                method = "GCV.Cp")
mgcv_fitted <- predict(mgcv_fit)
mgcv_edf <- sum(mgcv_fit$edf)

# Fit with fdars
fd_noisy <- fdata(matrix(y_noisy, nrow = 1), argvals = t_obs)
fdars_result <- pspline(fd_noisy, nbasis = 20, lambda.select = TRUE)
fdars_fitted <- as.vector(fdars_result$fdata$data)
fdars_edf <- fdars_result$edf

cat("Effective degrees of freedom:\n")
cat("  mgcv:", round(mgcv_edf, 2), "\n")
cat("  fdars:", round(fdars_edf, 2), "\n\n")

# Compare fitted values
correlation <- cor(mgcv_fitted, fdars_fitted)
max_diff <- max(abs(mgcv_fitted - fdars_fitted))
rmse <- sqrt(mean((mgcv_fitted - fdars_fitted)^2))

cat("Fitted values comparison:\n")
cat("  Correlation:", round(correlation, 4), "\n")
cat("  Max absolute difference:", round(max_diff, 4), "\n")
cat("  RMSE:", round(rmse, 4), "\n\n")

# Check boundary values specifically
cat("Boundary fitted values:\n")
cat("  mgcv at t=0:", round(mgcv_fitted[1], 4), "\n")
cat("  fdars at t=0:", round(fdars_fitted[1], 4), "\n")
cat("  True at t=0:", round(true_signal[1], 4), "\n")
cat("  mgcv at t=1:", round(mgcv_fitted[n_obs], 4), "\n")
cat("  fdars at t=1:", round(fdars_fitted[n_obs], 4), "\n")
cat("  True at t=1:", round(true_signal[n_obs], 4), "\n\n")

# Calculate MSE to true signal at boundaries vs interior
boundary_idx <- c(1:5, (n_obs-4):n_obs)
interior_idx <- 20:80

mse_mgcv_boundary <- mean((mgcv_fitted[boundary_idx] - true_signal[boundary_idx])^2)
mse_mgcv_interior <- mean((mgcv_fitted[interior_idx] - true_signal[interior_idx])^2)
mse_fdars_boundary <- mean((fdars_fitted[boundary_idx] - true_signal[boundary_idx])^2)
mse_fdars_interior <- mean((fdars_fitted[interior_idx] - true_signal[interior_idx])^2)

cat("MSE to true signal:\n")
cat("  mgcv boundary (first/last 5 points):", round(mse_mgcv_boundary, 4), "\n")
cat("  mgcv interior:", round(mse_mgcv_interior, 4), "\n")
cat("  fdars boundary:", round(mse_fdars_boundary, 4), "\n")
cat("  fdars interior:", round(mse_fdars_interior, 4), "\n\n")

# -----------------------------------------------------------------------------
# Test 3: Specific boundary behavior test
# -----------------------------------------------------------------------------
cat("----------------------------------------\n")
cat("Test 3: Boundary Extrapolation Test\n")
cat("----------------------------------------\n\n")

# Use a function that is non-zero at boundaries
test_func <- function(t) t * (1 - t) * sin(3 * pi * t)
y_test <- test_func(t_obs)

# mgcv fit
mgcv_fit2 <- gam(y ~ s(t, bs = "ps", k = 25, m = c(2, 2)),
                 data = data.frame(y = y_test, t = t_obs),
                 sp = 0.01)  # Fixed smoothing parameter
mgcv_fitted2 <- predict(mgcv_fit2)

# fdars fit with same lambda
fd_test2 <- fdata(matrix(y_test, nrow = 1), argvals = t_obs)
fdars_result2 <- pspline(fd_test2, nbasis = 25, lambda = 0.01)
fdars_fitted2 <- as.vector(fdars_result2$fdata$data)

cat("Exact boundary fit (should match true = 0):\n")
cat("  True at t=0:", round(test_func(0), 6), "\n")
cat("  True at t=1:", round(test_func(1), 6), "\n")
cat("  mgcv at t=0:", round(mgcv_fitted2[1], 6), "\n")
cat("  mgcv at t=1:", round(mgcv_fitted2[n_obs], 6), "\n")
cat("  fdars at t=0:", round(fdars_fitted2[1], 6), "\n")
cat("  fdars at t=1:", round(fdars_fitted2[n_obs], 6), "\n\n")

# -----------------------------------------------------------------------------
# Test 4: Multiple curves
# -----------------------------------------------------------------------------
cat("----------------------------------------\n")
cat("Test 4: Multiple Curves P-spline\n")
cat("----------------------------------------\n\n")

n_curves <- 10
X_multi <- matrix(0, n_curves, n_obs)
for (i in 1:n_curves) {
  phase <- runif(1, 0, pi)
  X_multi[i, ] <- sin(2 * pi * t_obs + phase) + rnorm(n_obs, sd = 0.2)
}

fd_multi <- fdata(X_multi, argvals = t_obs)
fdars_multi <- pspline(fd_multi, nbasis = 15, lambda = 1)

cat("Multi-curve P-spline results:\n")
cat("  Number of curves processed:", nrow(fdars_multi$fdata$data), "\n")
cat("  EDF:", round(fdars_multi$edf, 2), "\n")
cat("  GCV:", round(fdars_multi$gcv, 6), "\n\n")

# Check each curve at boundaries
cat("Boundary values for each curve:\n")
for (i in 1:min(5, n_curves)) {
  cat(sprintf("  Curve %d: t=0: %.4f, t=1: %.4f\n",
              i, fdars_multi$fdata$data[i, 1], fdars_multi$fdata$data[i, n_obs]))
}
cat("\n")

# -----------------------------------------------------------------------------
# Test 5: Derivative estimation at boundaries
# -----------------------------------------------------------------------------
cat("----------------------------------------\n")
cat("Test 5: GCV/Lambda Selection Comparison\n")
cat("----------------------------------------\n\n")

# Generate data with known optimal smoothness
set.seed(123)
y_smooth <- sin(4 * pi * t_obs) + rnorm(n_obs, sd = 0.15)

# mgcv with GCV
mgcv_gcv <- gam(y ~ s(t, bs = "ps", k = 30),
                data = data.frame(y = y_smooth, t = t_obs),
                method = "GCV.Cp")

# fdars with GCV selection
fd_smooth <- fdata(matrix(y_smooth, nrow = 1), argvals = t_obs)
fdars_gcv <- pspline(fd_smooth, nbasis = 30, lambda.select = TRUE, criterion = "GCV")

cat("GCV-selected smoothing:\n")
cat("  mgcv selected lambda:", round(mgcv_gcv$sp, 4), "\n")
cat("  fdars selected lambda:", round(fdars_gcv$lambda, 4), "\n")
cat("  mgcv EDF:", round(sum(mgcv_gcv$edf), 2), "\n")
cat("  fdars EDF:", round(fdars_gcv$edf, 2), "\n")

mgcv_gcv_fitted <- predict(mgcv_gcv)
fdars_gcv_fitted <- as.vector(fdars_gcv$fdata$data)

cat("  Correlation of fits:", round(cor(mgcv_gcv_fitted, fdars_gcv_fitted), 4), "\n\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("========================================\n")
cat("Summary\n")
cat("========================================\n")

# Overall assessment
pass_correlation <- correlation > 0.95
pass_boundary_left <- abs(fdars_fitted2[1]) < 0.1
pass_boundary_right <- abs(fdars_fitted2[n_obs]) < 0.1

cat("P-spline fit correlation (vs mgcv):", round(correlation, 4),
    ifelse(pass_correlation, " [PASS]", " [CHECK]"), "\n")
cat("Left boundary accuracy:", round(abs(fdars_fitted2[1]), 6),
    ifelse(pass_boundary_left, " [PASS]", " [CHECK]"), "\n")
cat("Right boundary accuracy:", round(abs(fdars_fitted2[n_obs]), 6),
    ifelse(pass_boundary_right, " [PASS]", " [CHECK]"), "\n")

if (pass_correlation && pass_boundary_left && pass_boundary_right) {
  cat("\nAll tests PASSED!\n")
} else {
  cat("\nSome tests need attention - see details above.\n")
}
