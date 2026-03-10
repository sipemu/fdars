#!/usr/bin/env Rscript
# Validate elastic changepoint detection against R's fdasrvf.
#
# Reference package: fdasrvf
# Usage: Rscript validation/R/validate_elastic_changepoint.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fdasrvf)

message("=== Validating elastic changepoint detection ===\n")

results <- list()

# Generate data with amplitude changepoint at n/2
set.seed(42)
n <- 30
m <- 51
argvals <- seq(0, 1, length.out = m)
cp <- 15  # true changepoint

data_mat <- matrix(0, nrow = m, ncol = n)  # fdasrvf: m x n
for (i in 1:n) {
  amp <- if (i <= cp) 1.0 else 2.0
  data_mat[, i] <- amp * sin(2 * pi * argvals)
}

results$data_amp <- as.numeric(data_mat)
results$n <- n
results$m <- m
results$argvals <- argvals
results$true_changepoint <- cp

# Check if elastic.changepoint exists
has_changepoint <- exists("elastic.changepoint", where = asNamespace("fdasrvf"))

if (has_changepoint) {
  # ---- (a) Amplitude changepoint ---
  message("  Computing amplitude changepoint...")
  results$amp_changepoint <- tryCatch({
    acp <- elastic.changepoint(data_mat, time = argvals, type = "amplitude",
                                MCMCiter = 200)
    list(
      detected_changepoint = acp$change,
      test_statistic = acp$Tn,
      p_value = acp$pvalue
    )
  }, error = function(e) {
    warning(sprintf("Amplitude changepoint failed: %s", conditionMessage(e)))
    NULL
  })

  # ---- (b) Phase changepoint ---
  message("  Computing phase changepoint...")
  data_phase <- matrix(0, nrow = m, ncol = n)
  for (i in 1:n) {
    shift <- if (i <= cp) 0.0 else 0.15
    data_phase[, i] <- sin(2 * pi * (argvals + shift))
  }
  results$data_phase <- as.numeric(data_phase)

  results$ph_changepoint <- tryCatch({
    pcp <- elastic.changepoint(data_phase, time = argvals, type = "phase",
                                MCMCiter = 200)
    list(
      detected_changepoint = pcp$change,
      test_statistic = pcp$Tn,
      p_value = pcp$pvalue
    )
  }, error = function(e) {
    warning(sprintf("Phase changepoint failed: %s", conditionMessage(e)))
    NULL
  })
} else {
  message("  elastic.changepoint not available in this fdasrvf version")
  message("  Saving basic reference values only")

  # Provide basic CUSUM reference values computed manually
  # Align using time_warping first
  warp <- tryCatch(time_warping(data_mat, argvals, max_iter = 5), error = function(e) NULL)
  if (!is.null(warp)) {
    aligned <- warp$fn  # m x n

    # Manual CUSUM computation for amplitude
    total_sum <- rowSums(aligned)
    cusum_vals <- numeric(n - 1)
    running_sum <- rep(0, m)
    for (k in 1:(n-1)) {
      running_sum <- running_sum + aligned[, k]
      diff_vec <- running_sum - (k / n) * total_sum
      cusum_vals[k] <- sum(diff_vec^2) / (m * n^2)
    }
    results$amp_changepoint <- list(
      detected_changepoint = which.max(cusum_vals),
      cusum_values = cusum_vals,
      test_statistic = max(cusum_vals)
    )
  }
}

save_expected(results, "elastic_changepoint")
message("Done.\n")
