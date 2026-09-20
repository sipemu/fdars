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
  library(fda)
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

# --- FPCA runner (fda::pca.fd) -----------------------------------------------

run_fpca <- function(csv_path, nharm = 5L, n_iterations = 10L, smoke = FALSE) {
  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  # Smooth raw data onto a B-spline basis (same approach as the R template in RESEARCH.md)
  nbasis <- min(20L, m)
  basis <- create.bspline.basis(c(0, 1), nbasis = nbasis)
  fd_smooth <- smooth.basis(grid, t(mat), basis)$fd

  if (smoke) {
    # Smoke run: single pass to verify packages load and computation succeeds
    result_obj <- pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE)
    mean_ns <- 0.0
  } else {
    # Warm-up pass (not timed)
    pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE)

    # Timed pass via bench::mark
    # bench::mark returns a tibble; $median is a bench_time object (ns internally)
    timing <- bench::mark(
      pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE),
      iterations = n_iterations,
      memory = FALSE
    )
    # bench::mark()$median is a bench_time object; as.numeric() returns SECONDS.
    # Convert to nanoseconds to match the shared comparison schema.
    mean_ns <- as.numeric(timing$median) * 1e9
    result_obj <- pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE)
  }

  # Extract loadings: pca.fd returns $harmonics which is an fd object.
  # Evaluate on the original grid to get (m x nharm) matrix.
  harm_fd <- result_obj$harmonics
  loadings_m_ncomp <- eval.fd(grid, harm_fd)  # shape: (m, nharm)

  # Transpose to (nharm, m) for consistent schema with python_runner.py
  loadings_ncomp_m <- t(loadings_m_ncomp)  # shape: (nharm, m)

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
