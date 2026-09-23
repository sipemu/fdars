#!/usr/bin/env Rscript
#
# R benchmark runner for the fdars cross-language harness.
#
# Reads the seeded CSV from generate_data.py, runs R FDA operations, times
# them via bench::mark(), and emits timing + result_fingerprint to
# benchmarks/results/r.json.
#
# Supported ops (--op):
#   fpca             - FPCA via hand-rolled Simpson's-weighted SVD (matches fdars)
#   basis_smoothing  - B-spline basis smoothing via create.bspline.basis + smooth.basis, lambda=0
#   depth            - Functional depth via roahd::MBD (deterministic)
#   functional_boxplot - roahd::fbplot(Depths="MBD", Fvalue=1.5, display=FALSE)
#   fpcr             - FPCR regression via fda.usc::fregre.pc
#   elastic          - fdasrvf::pair_align_functions (lambda=0) + time_warping (Karcher mean)
#
# Usage:
#   Rscript r_runner.R --op fpca [--smoke] [--n 60] [--m 50] [--seed 42]
#   Rscript r_runner.R --smoke   # runs all ops once each

suppressPackageStartupMessages({
  library(bench)
  library(jsonlite)
})

ALL_OPS <- c("fpca", "basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic")

# --- Argument parsing --------------------------------------------------------

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(
    op = NULL,
    smoke = FALSE,
    n = 60L,
    m = 50L,
    seed = 42L,
    pairwise_only = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    switch(args[[i]],
      "--op"             = { i <- i + 1L; parsed$op <- args[[i]] },
      "--smoke"          = { parsed$smoke <- TRUE },
      "--n"              = { i <- i + 1L; parsed$n <- as.integer(args[[i]]) },
      "--m"              = { i <- i + 1L; parsed$m <- as.integer(args[[i]]) },
      "--seed"           = { i <- i + 1L; parsed$seed <- as.integer(args[[i]]) },
      "--pairwise-only"  = { parsed$pairwise_only <- TRUE }
    )
    i <- i + 1L
  }
  # When --smoke without --op: run all ops
  if (is.null(parsed$op) && !parsed$smoke) {
    stop("--op is required unless --smoke is used (which runs all ops)")
  }
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

load_response <- function(y_path) {
  df <- read.csv(y_path, check.names = FALSE)
  as.numeric(df[, "y"])
}

# --- Timing helper -----------------------------------------------------------
#
# bench::mark()$median returns a bench_time object; as.numeric() gives seconds.
# Multiply by 1e9 to convert to nanoseconds for consistency with Python/Rust.

time_fn <- function(fn, n_iterations = 10L, smoke = FALSE) {
  result <- fn()  # warm-up / single run for smoke
  if (smoke) {
    return(list(mean_ns = 0.0, result = result))
  }
  timing <- bench::mark(fn(), iterations = n_iterations, memory = FALSE)
  mean_ns <- as.numeric(timing$median) * 1e9  # convert seconds -> nanoseconds
  result <- fn()   # re-run for a fresh result object
  list(mean_ns = mean_ns, result = result)
}

# --- Simpson's integration weights (matches fdars helpers.rs:simpsons_weights) ---

simpsons_weights_fdars <- function(argvals) {
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

  sw <- simpsons_weights_fdars(grid)
  sqrt_w <- sqrt(sw)

  fpca_direct <- function(mat, sqrt_w, nharm) {
    means <- colMeans(mat)
    centered <- sweep(mat, 2L, means, "-")
    weighted <- sweep(centered, 2L, sqrt_w, "*")
    sv_res <- svd(weighted, nu = nharm, nv = nharm)
    V <- sv_res$v  # (m x nharm)
    for (k in seq_len(nharm)) {
      j_max <- which.max(abs(V[, k]))
      if (V[[j_max, k]] < 0.0) V[, k] <- -V[, k]
    }
    safe_sqrt_w <- pmax(sqrt_w, 1e-15)
    loadings <- sweep(V, 1L, safe_sqrt_w, "/")  # (m x nharm)
    list(loadings = loadings, means = means)
  }

  res <- time_fn(function() fpca_direct(mat, sqrt_w, nharm),
                 n_iterations = n_iterations, smoke = smoke)
  result <- res$result

  loadings_ncomp_m <- t(result$loadings)  # (nharm, m)

  record <- list(
    op = "fpca",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    result_fingerprint = list(
      loadings = unname(lapply(seq_len(nrow(loadings_ncomp_m)), function(i) {
        as.list(loadings_ncomp_m[i, ])
      }))
    )
  )
  record
}

# --- Native R FPCA via fda::pca.fd (ecosystem-realistic timing, NOT gated) ----
#
# fda::pca.fd operates in B-spline coefficient space (not the raw grid space used
# by fdars). This produces a DIFFERENT numerical result from the Simpson's-SVD
# branch above; it is included only as a "what an R user actually runs" timing.
# Op label: "fpca_native_r" — not gated for numeric agreement.
#
# IMPORTANT: smooth.basis() (preprocessing) is EXCLUDED from the timed call.
# Only pca.fd() is timed (Pitfall 5 from RESEARCH.md).
#
# NOT added to ALL_OPS — invoked explicitly via --op fpca_native_r only.
# This avoids breaking the existing --smoke all-ops contract.

run_fpca_native_r <- function(csv_path, nharm = 5L, n_basis = 20L,
                               n_iterations = 10L, smoke = FALSE) {
  suppressPackageStartupMessages(library(fda))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  # B-spline smoothing is PREPROCESSING — computed OUTSIDE the timed lambda
  basis_obj <- create.bspline.basis(c(0, 1), nbasis = n_basis, norder = 4L)
  fd_par <- fdPar(basis_obj, Lfdobj = int2Lfd(2L), lambda = 0)
  fd_smooth <- smooth.basis(grid, t(mat), fd_par)$fd  # preprocessing — NOT timed

  # Time ONLY the pca.fd() call
  res <- time_fn(function() pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE),
                 n_iterations = n_iterations, smoke = smoke)

  # No result_fingerprint — not gated (different numerical result from SVD branch)
  list(
    op = "fpca_native_r",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns
  )
}

# --- Basis smoothing (fda::create.bspline.basis + smooth.basis, lambda=0) ----
#
# Matches scikit-fda BSplineBasis + BasisSmoother(smoothing_parameter=0).
# lambda=0 → no penalty → interpolation / pure B-spline fit.
# Fingerprint: smoothed curve values evaluated back on the original grid.

run_basis_smoothing <- function(csv_path, n_basis = 10L, n_iterations = 10L,
                                smoke = FALSE) {
  suppressPackageStartupMessages(library(fda))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  basis_obj <- create.bspline.basis(c(0, 1), nbasis = n_basis, norder = 4L)
  # lambda=0 → no roughness penalty
  fd_par <- fdPar(basis_obj, Lfdobj = int2Lfd(2L), lambda = 0)

  smooth_fn <- function() {
    # smooth.basis: argvals, y (m x n matrix, columns = curves), fdParobj
    # Note: smooth.basis expects y as (m x n) — columns = observations
    fd_smooth <- smooth.basis(grid, t(mat), fd_par)
    # Evaluate the smoothed fd object on the original grid
    eval.fd(grid, fd_smooth$fd)  # (m x n) matrix
  }

  res <- time_fn(smooth_fn, n_iterations = n_iterations, smoke = smoke)
  smoothed_vals <- res$result  # (m x n)

  record <- list(
    op = "basis_smoothing",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    result_fingerprint = list(
      # smoothed_values: n rows each of length m (transpose from m x n)
      smoothed_values = unname(lapply(seq_len(n), function(i) {
        as.list(smoothed_vals[, i])
      }))
    )
  )
  record
}

# --- Functional depth (roahd::MBD — deterministic MBD) ----------------------
#
# Uses roahd::MBD (Modified Band Depth) which is deterministic.
# Matches scikit-fda ModifiedBandDepth on the Python side.
# Avoids random projection (depth.RP) in the gated path — per RESEARCH
# Pitfall 5: random projection depth with different projection distributions
# (Gaussian vs vexponential) cannot be directly compared across libraries.

run_depth <- function(csv_path, n_iterations = 10L, smoke = FALSE) {
  suppressPackageStartupMessages(library(roahd))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  fdata_obj <- fData(grid, mat)

  depth_fn <- function() {
    as.numeric(MBD(fdata_obj))
  }

  res <- time_fn(depth_fn, n_iterations = n_iterations, smoke = smoke)
  depths <- res$result

  record <- list(
    op = "depth",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    result_fingerprint = list(
      depths = as.list(depths)
    )
  )
  record
}

# --- Functional boxplot (roahd::fbplot, MBD, Fvalue=1.5) --------------------
#
# fbplot() with display=FALSE returns list(Depth, Fvalue, ID_outliers).
# Fingerprint: the Depth vector (MBD values) — same quantity as
# scikit-fda Boxplot + ModifiedBandDepth on the Python side.
#
# Note: roahd 1.4.3 fbplot() signature (verified via live R session):
#   fbplot(Data, Depths="MBD", Fvalue=1.5, display=FALSE)
# where Data is an fData object.

run_functional_boxplot <- function(csv_path, n_iterations = 10L, smoke = FALSE) {
  suppressPackageStartupMessages(library(roahd))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  fdata_obj <- fData(grid, mat)

  boxplot_fn <- function() {
    fbplot(fdata_obj, Depths = "MBD", Fvalue = 1.5, display = FALSE)
  }

  res <- time_fn(boxplot_fn, n_iterations = n_iterations, smoke = smoke)
  bp_result <- res$result

  # bp_result$Depth is the MBD depth vector (length n)
  depths <- as.numeric(bp_result$Depth)

  record <- list(
    op = "functional_boxplot",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    result_fingerprint = list(
      depths = as.list(depths)
    )
  )
  record
}

# --- FPCR regression (fda.usc::fregre.pc) ------------------------------------
#
# fda.usc::fregre.pc fits a functional PC regression:
#   fregre.pc(fdataobj, y, l=1:ncomp)
# where fdataobj is an fdata object and y is the scalar response.
# Fingerprint: fitted.values from the model.
#
# Response loaded from y_n{n}.csv (seeded with seed+100 by generate_data.py).

run_fpcr <- function(csv_path, y_path, ncomp = 5L, n_iterations = 10L,
                     smoke = FALSE) {
  suppressPackageStartupMessages(library(fda.usc))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  y <- load_response(y_path)

  # fda.usc fdata object
  fdataobj <- fdata(mat, argvals = grid)

  fpcr_fn <- function() {
    model <- fregre.pc(fdataobj, y, l = seq_len(ncomp))
    as.numeric(model$fitted.values)
  }

  res <- time_fn(fpcr_fn, n_iterations = n_iterations, smoke = smoke)
  fitted_vals <- res$result

  record <- list(
    op = "fpcr",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    result_fingerprint = list(
      fitted_values = as.list(fitted_vals)
    )
  )
  record
}

# --- Elastic alignment / Karcher mean (fdasrvf) ------------------------------
#
# Two sub-operations:
# 1. Pairwise alignment via pair_align_functions(f1, f2, time, lambda=0,
#    pen="roughness", method="DP") — Fisher-Rao metric, no warp penalty.
#    Fingerprint: warping function values (gam) on the time grid.
#
# 2. Karcher mean via time_warping(f, time, lambda=0, max_iter=20) —
#    Fréchet mean in the SRSF / Fisher-Rao sense.
#    Fingerprint: Karcher mean curve values (fn) on the time grid.
#
# Both use lambda=0 to match fdars elastic_align_pair(lambda=0.0) and
# scikit-fda FisherRaoElasticRegistration(penalty=0.0).
#
# fdasrvf 2.4.4 verified function signatures (from live R session):
#   pair_align_functions(f1, f2, time, lambda=0, pen="roughness", method="DP")
#   time_warping(f, time, lambda=0, max_iter=20, ...)

run_elastic <- function(csv_path, elastic_csv_path, n_iterations = 5L,
                        smoke = FALSE, pairwise_only = FALSE) {
  suppressPackageStartupMessages(library(fdasrvf))

  # Full curve set for Karcher mean
  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  # 2-curve pair for pairwise alignment
  dat_pair <- load_csv(elastic_csv_path)
  mat_pair <- dat_pair$mat
  stopifnot(nrow(mat_pair) == 2L && ncol(mat_pair) == m)

  f1 <- as.numeric(mat_pair[1L, ])
  f2 <- as.numeric(mat_pair[2L, ])
  # fdasrvf expects f as (m x n) matrix for time_warping (columns = curves)
  f_mat <- t(mat)  # (m x n)

  # Phase 113 Plan 02 (user-approved 2026-09-23): in grid baseline mode
  # (pairwise_only = TRUE), TIME only the fast, N-independent pairwise alignment;
  # EXCLUDE the slow fdasrvf time_warping (Karcher mean) from timing — it is
  # noise-gated out of the ranking (RSD 29-46%) and takes ~10 min/cell at m=200.
  # The Karcher fingerprint is still computed ONCE (untimed) so the gate's
  # secondary karcher check has data; only its TIMING is dropped.
  if (isTRUE(pairwise_only)) {
    pairwise_fn <- function() {
      pair_res <- pair_align_functions(f1, f2, grid,
                                       lambda = 0, pen = "roughness", method = "DP")
      as.numeric(pair_res$gam)  # length m
    }
    res <- time_fn(pairwise_fn, n_iterations = n_iterations, smoke = smoke)
    warping_vals <- res$result

    # Karcher fingerprint computed ONCE, untimed (excluded from grid timing).
    km_res <- time_warping(f_mat, grid, lambda = 0, max_iter = 20L)
    karcher_vals <- as.numeric(km_res$fn[, 1L])

    record <- list(
      op = "elastic",
      library = "r",
      n = n,
      m = m,
      mean_ns = res$mean_ns,
      elastic_timing_scope = "pairwise_only",
      result_fingerprint = list(
        warping_values = as.list(warping_vals),
        karcher_mean   = as.list(karcher_vals)
      )
    )
    return(record)
  }

  # Legacy single-cell path: time pairwise + Karcher together.
  elastic_fn <- function() {
    # Pairwise alignment
    pair_res <- pair_align_functions(f1, f2, grid,
                                     lambda = 0, pen = "roughness", method = "DP")
    warping_vals <- as.numeric(pair_res$gam)  # length m

    # Karcher mean
    km_res <- time_warping(f_mat, grid, lambda = 0, max_iter = 20L)
    karcher_vals <- as.numeric(km_res$fn[, 1L])  # first column = Karcher mean curve

    list(warping_vals = warping_vals, karcher_vals = karcher_vals)
  }

  res <- time_fn(elastic_fn, n_iterations = n_iterations, smoke = smoke)
  warping_vals <- res$result$warping_vals
  karcher_vals <- res$result$karcher_vals

  record <- list(
    op = "elastic",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns,
    elastic_timing_scope = "pairwise_plus_karcher",
    result_fingerprint = list(
      warping_values = as.list(warping_vals),
      karcher_mean   = as.list(karcher_vals)
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
    script_dir <- normalizePath(".", mustWork = FALSE)
  }

  data_dir      <- file.path(script_dir, "data")
  csv_path      <- file.path(data_dir, sprintf("curves_n%d_m%d.csv", args$n, args$m))
  y_path        <- file.path(data_dir, sprintf("y_n%d.csv", args$n))
  elastic_path  <- file.path(data_dir, sprintf("elastic_n2_m%d.csv", args$m))
  results_dir   <- file.path(script_dir, "results")
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
  out_path      <- file.path(results_dir, "r.json")

  if (!file.exists(csv_path)) {
    stop(sprintf("CSV not found: %s\nRun: python benchmarks/generate_data.py first.", csv_path))
  }

  ops_to_run <- if (is.null(args$op)) ALL_OPS else args$op
  records <- list()

  for (op in ops_to_run) {
    record <- switch(op,
      fpca = run_fpca(csv_path, nharm = 5L, smoke = args$smoke),
      fpca_native_r = run_fpca_native_r(csv_path, nharm = 5L, n_basis = 20L,
                                         smoke = args$smoke),
      basis_smoothing = run_basis_smoothing(csv_path, n_basis = 10L, smoke = args$smoke),
      depth = run_depth(csv_path, smoke = args$smoke),
      functional_boxplot = run_functional_boxplot(csv_path, smoke = args$smoke),
      fpcr = {
        if (!file.exists(y_path)) {
          stop(sprintf("Response CSV not found: %s\nRun: python benchmarks/generate_data.py first.", y_path))
        }
        run_fpcr(csv_path, y_path, ncomp = 5L, smoke = args$smoke)
      },
      elastic = {
        if (!file.exists(elastic_path)) {
          stop(sprintf("Elastic pair CSV not found: %s\nRun: python benchmarks/generate_data.py first.", elastic_path))
        }
        run_elastic(csv_path, elastic_path, smoke = args$smoke,
                    pairwise_only = args$pairwise_only)
      },
      stop(sprintf("Unknown op: %s", op))
    )

    records[[length(records) + 1L]] <- record
    cat(sprintf("R runner [%s]: mean_ns=%.1f\n", op, record$mean_ns))
  }

  write_json(records, out_path, auto_unbox = TRUE, digits = 17)
  cat(sprintf("Written: %s\n", out_path))
}

# NULL-coalescing operator (base R >= 4.4 has this, but define for compat)
`%||%` <- function(a, b) if (!is.null(a)) a else b

main()
