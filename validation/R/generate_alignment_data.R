#!/usr/bin/env Rscript
# Generate test datasets for elastic alignment and equivalence testing.
# Writes JSON files to validation/data/
#
# Usage: Rscript validation/R/generate_alignment_data.R

library(jsonlite)

# Resolve script directory robustly
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  return(getwd())
}

out_dir <- file.path(dirname(get_script_dir()), "data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write_json_file <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  write_json(obj, path, digits = 17, auto_unbox = TRUE)
  message(sprintf("  wrote %s", path))
}

# ---------------------------------------------------------------------------
# 1. Alignment dataset: 30 curves on 51 grid points with phase variability
# ---------------------------------------------------------------------------
message("Generating alignment dataset (30x51)...")
set.seed(42)

n <- 30L
m <- 51L
argvals <- seq(0, 1, length.out = m)

# Base shape: sin(2*pi*t) + 0.5*cos(4*pi*t)
base_fn <- sin(2 * pi * argvals) + 0.5 * cos(4 * pi * argvals)

curves <- matrix(0, nrow = n, ncol = m)
for (i in 1:n) {
  # Phase variability: gamma(t) = t + a*sin(pi*t), a ~ U(-0.15, 0.15)
  a <- runif(1, -0.15, 0.15)
  gamma <- argvals + a * sin(pi * argvals)
  # Clamp to [0, 1]
  gamma <- pmin(pmax(gamma, 0), 1)

  # Amplitude noise: scale + shift
  amp <- 1 + rnorm(1, 0, 0.1)
  shift <- rnorm(1, 0, 0.05)

  # Interpolate base function at warped time
  warped_base <- approx(argvals, base_fn, xout = gamma, rule = 2)$y

  # Add pointwise noise
  curves[i, ] <- amp * warped_base + shift + rnorm(m, 0, 0.02)
}

# Store as column-major flat vector (matching Rust layout)
write_json_file(list(
  n = n, m = m,
  argvals = argvals,
  data = as.vector(curves)
), "alignment_30x51")

# ---------------------------------------------------------------------------
# 2. Equivalence test dataset: two groups of 30 curves, 51 grid points
# ---------------------------------------------------------------------------
message("Generating equivalence groups dataset...")
set.seed(123)

n_eq <- 30L
m_eq <- 51L
argvals_eq <- seq(0, 1, length.out = m_eq)

# Group 1: centered at sin(2*pi*t)
mu1 <- sin(2 * pi * argvals_eq)
curves1 <- matrix(0, nrow = n_eq, ncol = m_eq)
for (i in 1:n_eq) {
  curves1[i, ] <- mu1 + rnorm(1, 0, 0.2) * sin(2 * pi * argvals_eq) +
    rnorm(1, 0, 0.1) * cos(4 * pi * argvals_eq) +
    rnorm(m_eq, 0, 0.03)
}

# Group 2: centered at sin(2*pi*t) + 0.15*cos(2*pi*t)
mu2 <- mu1 + 0.15 * cos(2 * pi * argvals_eq)
curves2 <- matrix(0, nrow = n_eq, ncol = m_eq)
for (i in 1:n_eq) {
  curves2[i, ] <- mu2 + rnorm(1, 0, 0.2) * sin(2 * pi * argvals_eq) +
    rnorm(1, 0, 0.1) * cos(4 * pi * argvals_eq) +
    rnorm(m_eq, 0, 0.03)
}

write_json_file(list(
  n = n_eq, m = m_eq,
  argvals = argvals_eq,
  data1 = as.vector(curves1),
  data2 = as.vector(curves2)
), "equivalence_groups")

message("\n=== Alignment test data generated ===")
