#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(KernSmooth)
library(FNN)

message("=== Validating smoothing methods ===\n")

# Load noisy sine data (201 points)
dat <- load_data("noisy_sine_201")
x <- dat$x
y <- dat$y_noisy
m <- dat$m

results <- list()

# ---- (a) Nadaraya-Watson (local constant) with Gaussian kernel ----------------
message("  Computing Nadaraya-Watson smoother (bandwidth=0.05, degree=0)...")
results$nadaraya_watson <- tryCatch({
  fit <- KernSmooth::locpoly(x, y, bandwidth = 0.05, degree = 0,
                             gridsize = 201, range.x = c(0, 1))
  as.numeric(fit$y)
}, error = function(e) {
  warning(sprintf("Nadaraya-Watson failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Local linear with Gaussian kernel ------------------------------------
message("  Computing local linear smoother (bandwidth=0.05, degree=1)...")
results$local_linear <- tryCatch({
  fit <- KernSmooth::locpoly(x, y, bandwidth = 0.05, degree = 1,
                             gridsize = 201, range.x = c(0, 1))
  as.numeric(fit$y)
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
message("  Computing local polynomial smoother (bandwidth=0.05, degree=2)...")
results$local_polynomial <- tryCatch({
  fit <- KernSmooth::locpoly(x, y, bandwidth = 0.05, degree = 2,
                             gridsize = 201, range.x = c(0, 1))
  as.numeric(fit$y)
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
