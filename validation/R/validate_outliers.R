#!/usr/bin/env Rscript
# Validate outlier detection against R reference implementations.
# Computes Fraiman-Muniz depth values and identifies lowest-depth curves
# as outliers on the outliers_50x101 dataset, then saves expected outputs
# for comparison with Rust implementations.
#
# Reference packages: fda.usc
# Usage: Rscript validation/R/validate_outliers.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)

message("=== Validating outlier detection ===\n")

# Load outlier test data (50 curves, 101 grid points, 3 known outliers)
dat <- load_data("outliers_50x101")
n <- dat$n
m <- dat$m
argvals <- dat$argvals
known_outlier_indices <- dat$outlier_indices

message(sprintf("  Known outlier indices (1-based): %s",
                paste(known_outlier_indices, collapse = ", ")))

# Build fda.usc::fdata object
fdataobj <- to_fdata(dat$data, n, m, argvals)

results <- list()

# ---- (a) Depth-based outlier detection (Fraiman-Muniz) -----------------------
message("  Computing Fraiman-Muniz depths...")
results$depth_fm <- tryCatch({
  res <- fda.usc::depth.FM(fdataobj)
  depth_values <- as.numeric(res$dep)

  # Find the 3 curves with lowest depth (most likely outliers)
  n_outliers <- 3
  sorted_indices <- order(depth_values)
  lowest_depth_indices <- sorted_indices[1:n_outliers]

  message(sprintf("    Depth range: [%.6f, %.6f]", min(depth_values), max(depth_values)))
  message(sprintf("    3 lowest-depth indices (1-based): %s",
                  paste(lowest_depth_indices, collapse = ", ")))
  message(sprintf("    Known outlier indices (1-based):  %s",
                  paste(sort(known_outlier_indices), collapse = ", ")))

  list(
    depth_values = depth_values,
    lowest_depth_indices = sort(lowest_depth_indices)
  )
}, error = function(e) {
  warning(sprintf("Fraiman-Muniz depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Trimmed depth outlier detection (if available) ----------------------
message("  Attempting outliers.depth.trim...")
results$outliers_depth_trim <- tryCatch({
  out_result <- fda.usc::outliers.depth.trim(fdataobj)

  # Extract detected outlier indices
  # outliers.depth.trim returns a list with 'outliers' containing row names
  detected <- out_result$outliers
  if (is.null(detected) || length(detected) == 0) {
    message("    No outliers detected by outliers.depth.trim")
    list(
      detected_indices = integer(0),
      n_detected = 0L
    )
  } else {
    # Convert row names to integer indices
    detected_indices <- as.integer(detected)
    message(sprintf("    Detected outlier indices (1-based): %s",
                    paste(sort(detected_indices), collapse = ", ")))
    list(
      detected_indices = sort(detected_indices),
      n_detected = length(detected_indices)
    )
  }
}, error = function(e) {
  warning(sprintf("outliers.depth.trim failed: %s", conditionMessage(e)))
  message("    Skipping outliers.depth.trim (not available or failed)")
  NULL
})

# ---- Save results ------------------------------------------------------------
message("\n  Saving expected outlier detection values...")
save_expected(results, "outliers_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Outlier detection validation complete: %d/%d computations succeeded ===", computed, total))
