#!/usr/bin/env Rscript
# Install all R packages needed for fdars-core validation
# Usage: Rscript validation/R/install_deps.R

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(sprintf("  %s already installed", pkg))
  }
}

message("=== Installing fdars-core validation dependencies ===\n")

# Core FDA
message("-- Core FDA packages --")
for (pkg in c("fda", "fda.usc", "roahd", "ddalpha")) install_if_missing(pkg)

# Statistics / Regression
message("\n-- Regression packages --")
for (pkg in c("pls", "glmnet", "MASS")) install_if_missing(pkg)

# Smoothing
message("\n-- Smoothing packages --")
for (pkg in c("KernSmooth", "locpol", "FNN")) install_if_missing(pkg)

# Clustering
message("\n-- Clustering packages --")
for (pkg in c("e1071", "cluster", "fpc")) install_if_missing(pkg)

# Time Series / Seasonal
message("\n-- Time series packages --")
for (pkg in c("lomb", "Rssa", "tsmp")) install_if_missing(pkg)

# Utilities
message("\n-- Utility packages --")
for (pkg in c("pracma", "jsonlite")) install_if_missing(pkg)

message("\n=== All dependencies installed ===")
