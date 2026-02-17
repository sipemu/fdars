#!/usr/bin/env Rscript
# Validate depth measures against R reference implementations.
# Computes all supported depth functions on the standard_50x101 dataset
# and saves expected outputs for comparison with Rust implementations.
#
# Reference packages: fda.usc, roahd
# Usage: Rscript validation/R/validate_depth.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)
library(roahd)

message("=== Validating depth measures ===\n")

# Load standard test data (50 curves, 101 grid points)
dat <- load_data("standard_50x101")
n <- dat$n
m <- dat$m
argvals <- dat$argvals

# Build fda.usc::fdata object
fdataobj <- to_fdata(dat$data, n, m, argvals)

# Build raw matrix for roahd (n x m)
mat <- to_matrix(dat$data, n, m)

results <- list()

# ---- Fraiman-Muniz depth (fda.usc) ----------------------------------------
message("  Computing Fraiman-Muniz depth...")
results$fraiman_muniz <- tryCatch({
  res <- fda.usc::depth.FM(fdataobj)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Fraiman-Muniz depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Band depth (roahd) ---------------------------------------------------
message("  Computing band depth...")
results$band <- tryCatch({
  as.numeric(roahd::BD(mat))
}, error = function(e) {
  warning(sprintf("Band depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Modified band depth (roahd) ------------------------------------------
message("  Computing modified band depth...")
results$modified_band <- tryCatch({
  as.numeric(roahd::MBD(mat))
}, error = function(e) {
  warning(sprintf("Modified band depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Modified epigraph index (roahd) --------------------------------------
message("  Computing modified epigraph index...")
results$modified_epigraph <- tryCatch({
  as.numeric(roahd::MEI(mat))
}, error = function(e) {
  warning(sprintf("Modified epigraph index failed: %s", conditionMessage(e)))
  NULL
})

# ---- Modal depth (fda.usc) ------------------------------------------------
message("  Computing modal depth...")
results$modal <- tryCatch({
  res <- fda.usc::depth.mode(fdataobj)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Modal depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Random projection depth (fda.usc) ------------------------------------
message("  Computing random projection depth (nproj=50, seed=123)...")
results$random_projection <- tryCatch({
  set.seed(123)
  res <- fda.usc::depth.RP(fdataobj, nproj = 50)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Random projection depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Random Tukey depth (fda.usc) -----------------------------------------
message("  Computing random Tukey depth (nproj=50, seed=123)...")
results$random_tukey <- tryCatch({
  set.seed(123)
  res <- fda.usc::depth.RT(fdataobj, nproj = 50)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Random Tukey depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Functional spatial depth (fda.usc) -----------------------------------
message("  Computing functional spatial depth...")
results$functional_spatial <- tryCatch({
  res <- fda.usc::depth.FSD(fdataobj)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Functional spatial depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Kernel functional spatial depth (fda.usc) ----------------------------
message("  Computing kernel functional spatial depth...")
results$kernel_functional_spatial <- tryCatch({
  res <- fda.usc::depth.KFSD(fdataobj)
  as.numeric(res$dep)
}, error = function(e) {
  warning(sprintf("Kernel functional spatial depth failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ----------------------------------------------------------
message("\n  Saving expected depth values...")
save_expected(results, "depth_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Depth validation complete: %d/%d measures computed ===", computed, total))
