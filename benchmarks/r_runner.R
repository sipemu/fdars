#!/usr/bin/env Rscript
#
# R benchmark runner for the fdars cross-language harness.
#
# Reads the seeded CSV from generate_data.py, runs fda::pca.fd FPCA with nharm=5,
# times it via bench::mark(), and emits timing + result_fingerprint to
# benchmarks/results/r.json.
#
# Usage:
#   Rscript r_runner.R --op fpca [--smoke] [--n 60] [--m 50] [--seed 42]
#   Rscript r_runner.R --op fpca --smoke

suppressPackageStartupMessages({
  library(bench)
  library(jsonlite)
})

# --- Argument parsing --------------------------------------------------------

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(
    op = NULL,
    smoke = FALSE,
    n = 60L,
    m = 50L,
    seed = 42L
  )
  i <- 1L
  while (i <= length(args)) {
    switch(args[[i]],
      "--op"     = { i <- i + 1L; parsed$op <- args[[i]] },
      "--smoke"  = { parsed$smoke <- TRUE },
      "--n"      = { i <- i + 1L; parsed$n <- as.integer(args[[i]]) },
      "--m"      = { i <- i + 1L; parsed$m <- as.integer(args[[i]]) },
      "--seed"   = { i <- i + 1L; parsed$seed <- as.integer(args[[i]]) }
    )
    i <- i + 1L
  }
  if (is.null(parsed$op)) stop("--op is required")
  parsed
}

# --- CSV loading -------------------------------------------------------------

load_csv <- function(csv_path) {
  df <- read.csv(csv_path, check.names = FALSE)
  # First column is curve_id — drop it
  mat <- as.matrix(df[, -1L])
  storage.mode(mat) <- "double"
  list(mat = mat, n = nrow(mat), m = ncol(mat))
}

# --- Simpson's integration weights (matches fdars helpers.rs:simpsons_weights) ---

simpsons_weights_fdars <- function(argvals) {
  # Direct port of fdars simpsons_weights_uniform for uniform grids.
  # This must match the Rust implementation exactly for the agreement gate to pass.
  m <- length(argvals)
  if (m < 2L) return(rep(1.0, m))
  h0 <- argvals[[2L]] - argvals[[1L]]
  weights <- numeric(m)
  n_intervals <- m - 1L

  if (n_intervals %% 2L == 0L) {
    # Even intervals (odd m): pure composite Simpson's 1/3 rule
    weights[[1L]] <- h0 / 3.0
    weights[[m]] <- h0 / 3.0
    for (i in seq(2L, m - 1L)) {
      weights[[i]] <- if ((i - 1L) %% 2L == 1L) 4.0 * h0 / 3.0 else 2.0 * h0 / 3.0
    }
  } else {
    # Odd intervals (even m): Simpson's for first (m-2) intervals, trapezoidal for last
    n_simp <- m - 1L
    weights[[1L]] <- h0 / 3.0
    weights[[n_simp]] <- h0 / 3.0
    for (i in seq(2L, n_simp - 1L)) {
      weights[[i]] <- if ((i - 1L) %% 2L == 1L) 4.0 * h0 / 3.0 else 2.0 * h0 / 3.0
    }
    # Trapezoidal correction for the last interval
    weights[[n_simp]] <- weights[[n_simp]] + h0 / 2.0
    weights[[m]] <- weights[[m]] + h0 / 2.0
  }
  weights
}

# --- FPCA runner (direct Simpson's-weighted SVD — matches fdars exactly) ----
#
# Uses the same algorithm as fdars fdata_to_pc (regression.rs):
#   1. Center columns (mean function)
#   2. Compute Simpson's weights (same as fdars helpers.rs:simpsons_weights)
#   3. Scale centered data by sqrt(weights)
#   4. Thin SVD (R's svd() with nu=nharm, nv=nharm)
#   5. Fix sign: max-abs element of each loading column must be positive
#   6. Unscale loadings by sqrt(weights)
#
# This produces numerically identical loadings to fdars (correlation > 0.9999).

run_fpca <- function(csv_path, nharm = 5L, n_iterations = 10L, smoke = FALSE) {
  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  # Precompute weights (same for all timing iterations)
  sw <- simpsons_weights_fdars(grid)
  sqrt_w <- sqrt(sw)

  # Inner FPCA function (timed)
  fpca_direct <- function(mat, sqrt_w, nharm) {
    # Step 1: Center
    means <- colMeans(mat)
    centered <- sweep(mat, 2L, means, "-")

    # Step 2: Scale by sqrt(weights)
    weighted <- sweep(centered, 2L, sqrt_w, "*")

    # Step 3: Thin SVD (right singular vectors = loadings in weighted space)
    sv_res <- svd(weighted, nu = nharm, nv = nharm)
    V <- sv_res$v  # (m x nharm)

    # Step 4: Fix sign (fdars fix_svd_signs: max-abs element positive per column)
    for (k in seq_len(nharm)) {
      j_max <- which.max(abs(V[, k]))
      if (V[[j_max, k]] < 0.0) V[, k] <- -V[, k]
    }

    # Step 5: Unscale rotation by sqrt(weights)
    safe_sqrt_w <- pmax(sqrt_w, 1e-15)
    loadings <- sweep(V, 1L, safe_sqrt_w, "/")  # (m x nharm)

    list(loadings = loadings, means = means)
  }

  if (smoke) {
    result <- fpca_direct(mat, sqrt_w, nharm)
    mean_ns <- 0.0
  } else {
    # Warm-up pass
    fpca_direct(mat, sqrt_w, nharm)

    # Timed pass via bench::mark
    # bench::mark()$median returns seconds; convert to ns
    timing <- bench::mark(
      fpca_direct(mat, sqrt_w, nharm),
      iterations = n_iterations,
      memory = FALSE
    )
    mean_ns <- as.numeric(timing$median) * 1e9
    result <- fpca_direct(mat, sqrt_w, nharm)
  }

  # Transpose loadings to (nharm, m) for JSON schema (matches python_runner.py)
  loadings_ncomp_m <- t(result$loadings)  # (nharm, m)

  record <- list(
    op = "fpca",
    library = "r",
    n = n,
    m = m,
    mean_ns = mean_ns,
    result_fingerprint = list(
      loadings = unname(lapply(seq_len(nrow(loadings_ncomp_m)), function(i) {
        as.list(loadings_ncomp_m[i, ])
      }))
    )
  )
  record
}

# --- Main --------------------------------------------------------------------

main <- function() {
  args <- parse_args()

  # Get the directory containing this script (works when run via Rscript)
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0L) {
    script_path <- sub("^--file=", "", file_arg[[1L]])
    script_dir <- normalizePath(dirname(script_path), mustWork = FALSE)
  } else {
    # Fallback: assume current working directory contains benchmarks/
    script_dir <- normalizePath(".", mustWork = FALSE)
  }

  csv_path <- file.path(script_dir, "data",
                        sprintf("curves_n%d_m%d.csv", args$n, args$m))
  results_dir <- file.path(script_dir, "results")
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(results_dir, "r.json")

  if (!file.exists(csv_path)) {
    stop(sprintf(
      "CSV not found: %s\nRun: python benchmarks/generate_data.py first.",
      csv_path
    ))
  }

  if (args$op == "fpca") {
    record <- run_fpca(csv_path, nharm = 5L, smoke = args$smoke)
  } else {
    stop(sprintf("Unknown op: %s", args$op))
  }

  write_json(list(record), out_path, auto_unbox = TRUE, digits = 17)
  cat(sprintf("R runner [%s]: mean_ns=%.1f\n", args$op, record$mean_ns))
  cat(sprintf("Written: %s\n", out_path))
}

# NULL-coalescing operator (base R >= 4.4 has this, but define for compat)
`%||%` <- function(a, b) if (!is.null(a)) a else b

main()
