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

# ---- Save results -------------------------------------------------------------
save_expected(results, "smoothing_expected")
message("\nDone.\n")
