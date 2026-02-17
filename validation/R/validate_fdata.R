#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)

message("=== Validating fdata operations ===\n")

# Load standard test data (50 curves, 101 grid points)
dat <- load_data("standard_50x101")
n <- dat$n
m <- dat$m
argvals <- dat$argvals

# Build fda.usc::fdata object
fdataobj <- to_fdata(dat$data, n, m, argvals)

# Build raw matrix for manual operations
mat <- to_matrix(dat$data, n, m)

results <- list()

# ---- (a) Mean function -------------------------------------------------------
message("  Computing functional mean (func.mean)...")
results$mean <- tryCatch({
  mean_fdata <- fda.usc::func.mean(fdataobj)
  # Extract the $data row as a numeric vector of length m
  as.numeric(mean_fdata$data[1, ])
}, error = function(e) {
  warning(sprintf("func.mean failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Centered data -------------------------------------------------------
message("  Computing centered curves (each curve minus the mean)...")
results$centered <- tryCatch({
  mean_vec <- as.numeric(fda.usc::func.mean(fdataobj)$data[1, ])
  # Subtract the mean from each curve (row)
  centered_mat <- sweep(mat, 2, mean_vec, "-")
  # Flatten column-major to match Rust expectations
  as.numeric(centered_mat)
}, error = function(e) {
  warning(sprintf("Centering failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Lp norm (L2) --------------------------------------------------------
message("  Computing L2 norms (norm.fdata, lp=2)...")
results$norm_l2 <- tryCatch({
  norms <- fda.usc::norm.fdata(fdataobj, lp = 2)
  as.numeric(norms)
}, error = function(e) {
  warning(sprintf("norm.fdata L2 failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results -------------------------------------------------------------
save_expected(results, "fdata_expected")
message("\nDone.\n")
