#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(FNN)

message("=== Validating smoothing methods ===\n")

# Load noisy sine data (201 points)
dat <- load_data("noisy_sine_201")
x <- dat$x
y <- dat$y_noisy
m <- dat$m

results <- list()

# ---- (a) Nadaraya-Watson (local constant) with Gaussian kernel ----------------
# Use EXACT Nadaraya-Watson (not locpoly binning) to match Rust's implementation
message("  Computing exact Nadaraya-Watson smoother (bandwidth=0.05, degree=0)...")
results$nadaraya_watson <- tryCatch({
  bw <- 0.05
  # Compute exact NW using kernel weights
  nw_weights <- dnorm(outer(x, x, "-") / bw)
  nw_weights <- nw_weights / rowSums(nw_weights)
  as.numeric(nw_weights %*% y)
}, error = function(e) {
  warning(sprintf("Nadaraya-Watson failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Local linear with Gaussian kernel ------------------------------------
# Use exact weighted least squares instead of locpoly binning
message("  Computing exact local linear smoother (bandwidth=0.05, degree=1)...")
results$local_linear <- tryCatch({
  bw <- 0.05
  n <- length(x)
  fitted <- numeric(n)
  for (i in 1:n) {
    u <- (x - x[i]) / bw
    w <- dnorm(u)
    d <- x - x[i]
    s0 <- sum(w)
    s1 <- sum(w * d)
    s2 <- sum(w * d^2)
    t0 <- sum(w * y)
    t1 <- sum(w * y * d)
    det <- s0 * s2 - s1^2
    if (abs(det) > 1e-10) {
      fitted[i] <- (s2 * t0 - s1 * t1) / det
    } else if (s0 > 1e-10) {
      fitted[i] <- t0 / s0
    } else {
      fitted[i] <- 0
    }
  }
  as.numeric(fitted)
}, error = function(e) {
  warning(sprintf("Local linear failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) KNN smoother with k=5 -----------------------------------------------
message("  Computing KNN smoother (k=5)...")
results$knn_k5 <- tryCatch({
  pred <- FNN::knn.reg(train = matrix(x, ncol = 1),
                       test = matrix(x, ncol = 1),
                       y = y, k = 5)$pred
  as.numeric(pred)
}, error = function(e) {
  warning(sprintf("KNN smoother failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) Local polynomial (degree=2) with Gaussian kernel --------------------
# Use exact weighted least squares instead of locpoly binning
message("  Computing exact local polynomial smoother (bandwidth=0.05, degree=2)...")
results$local_polynomial <- tryCatch({
  bw <- 0.05
  n <- length(x)
  p <- 3  # degree 2 + 1
  fitted <- numeric(n)
  for (i in 1:n) {
    u <- (x - x[i]) / bw
    w <- dnorm(u)
    d <- x - x[i]
    # Build weighted normal equations
    xtx <- matrix(0, p, p)
    xty <- numeric(p)
    for (j in 1:p) {
      w_dj <- w * d^(j-1)
      for (k in 1:p) {
        xtx[j, k] <- sum(w_dj * d^(k-1))
      }
      xty[j] <- sum(w_dj * y)
    }
    coefs <- tryCatch(solve(xtx, xty), error = function(e) c(sum(w * y) / sum(w), rep(0, p-1)))
    fitted[i] <- coefs[1]
  }
  as.numeric(fitted)
}, error = function(e) {
  warning(sprintf("Local polynomial failed: %s", conditionMessage(e)))
  NULL
})

# ---- (e) Smoothing matrix NW: row sums and one row --------------------------
message("  Computing NW smoothing matrix for evaluation point x[101]...")
results$smoothing_matrix_nw <- tryCatch({
  # Compute explicit NW weights for a single evaluation point x[101] (midpoint)
  eval_point <- x[101]  # should be ~0.5
  bw <- 0.05
  # Gaussian kernel weights: K((x_i - eval_point) / bw) / sum(K(...))
  raw_weights <- dnorm((x - eval_point) / bw)
  weights <- raw_weights / sum(raw_weights)

  message(sprintf("    Eval point: %.4f, Row sum: %.12f", eval_point, sum(weights)))

  list(
    eval_point = eval_point,
    bandwidth = bw,
    weights = as.numeric(weights),
    row_sum = sum(weights)
  )
}, error = function(e) {
  warning(sprintf("Smoothing matrix NW failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results -------------------------------------------------------------
save_expected(results, "smoothing_expected")
message("\nDone.\n")
