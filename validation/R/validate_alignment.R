#!/usr/bin/env Rscript
# Validate elastic alignment against fdasrvf reference implementation.
# Requires: fdasrvf, jsonlite
#
# Usage: Rscript validation/R/validate_alignment.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fdasrvf)

message("=== Alignment validation ===")

d <- load_data("alignment_30x51")
mat <- to_matrix(d$data, d$n, d$m)
argvals <- d$argvals

result <- list()

# Helper: trapezoidal weights (matches Rust's simpsons_weights)
trap_weights <- function(t) {
  n <- length(t)
  h <- diff(t)
  w <- numeric(n)
  w[1] <- h[1] / 2
  w[n] <- h[n - 1] / 2
  for (i in 2:(n - 1)) w[i] <- (h[i - 1] + h[i]) / 2
  w
}

# Helper: L2 SRSF distance (matches Rust's elastic_distance)
l2_srsf_distance <- function(f1, f2, t) {
  q1 <- f_to_srvf(f1, t)
  q2 <- f_to_srvf(f2, t)
  # Align q2 to q1 via pair_align_functions, get aligned curve
  pa <- pair_align_functions(f1, f2, t)
  q2_aligned <- f_to_srvf(pa$f2tilde, t)
  w <- trap_weights(t)
  sqrt(sum(w * (q1 - q2_aligned)^2))
}

# ---------------------------------------------------------------------------
# 1. SRSF transform for rows 0 and 1 (R uses 1-based indexing)
# ---------------------------------------------------------------------------
message("  Computing SRSF transforms...")
q0 <- f_to_srvf(mat[1, ], argvals)
q1 <- f_to_srvf(mat[2, ], argvals)
result$srsf_row0 <- as.vector(q0)
result$srsf_row1 <- as.vector(q1)

# ---------------------------------------------------------------------------
# 2. SRSF round-trip (SRSF -> inverse -> back)
# ---------------------------------------------------------------------------
message("  Computing SRSF round-trip...")
f0_reconstructed <- srvf_to_f(q0, argvals, mat[1, 1])
result$srsf_roundtrip_row0 <- as.vector(f0_reconstructed)

# ---------------------------------------------------------------------------
# 3. Elastic distance between curves 0 and 1
#    Rust uses L2 distance in SRSF space after alignment, not arccos.
# ---------------------------------------------------------------------------
message("  Computing elastic distance (L2 SRSF)...")
pa_01 <- pair_align_functions(mat[1, ], mat[2, ], argvals)
q0_srsf <- f_to_srvf(mat[1, ], argvals)
q1_aligned_srsf <- f_to_srvf(pa_01$f2tilde, argvals)
w <- trap_weights(argvals)
result$elastic_distance_01 <- sqrt(sum(w * (q0_srsf - q1_aligned_srsf)^2))

# ---------------------------------------------------------------------------
# 4. Pairwise alignment: align curve 2 to curve 1
# ---------------------------------------------------------------------------
message("  Computing pairwise alignment...")
result$pair_align_gamma <- as.vector(pa_01$gam)
result$pair_align_f_aligned <- as.vector(pa_01$f2tilde)

# ---------------------------------------------------------------------------
# 5. Karcher mean (max_iter=20)
# ---------------------------------------------------------------------------
message("  Computing Karcher mean (this may take a moment)...")
# time_warping expects: m x n matrix (columns are curves), time vector
mat_t <- t(mat)  # m x n
tw <- time_warping(mat_t, argvals, max_iter = 20L)
result$karcher_mean <- as.vector(tw$fmean)
result$karcher_mean_srsf <- as.vector(tw$mqn)

# ---------------------------------------------------------------------------
# 6. Distance matrix for first 5 curves (L2 SRSF distance)
# ---------------------------------------------------------------------------
message("  Computing 5x5 distance matrix...")
k <- 5
dist_mat <- matrix(0, nrow = k, ncol = k)
for (i in 1:k) {
  for (j in 1:k) {
    if (i < j) {
      d_ij <- l2_srsf_distance(mat[i, ], mat[j, ], argvals)
      dist_mat[i, j] <- d_ij
      dist_mat[j, i] <- d_ij
    }
  }
}
result$distance_matrix_5x5 <- list(
  n = k,
  data = as.vector(dist_mat)  # column-major
)

save_expected(result, "alignment_expected")
message("=== Alignment validation complete ===")
