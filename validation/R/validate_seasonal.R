#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Validating seasonal / spectral methods ===\n")

results <- list()

# Load seasonal dataset
seasonal_data <- load_data("seasonal_200")
t_vals <- seasonal_data$t
pure_sine <- seasonal_data$pure_sine
noisy_sine <- seasonal_data$noisy_sine
n_ts <- seasonal_data$n
true_period <- seasonal_data$period

# ---- (a) Periodogram / FFT ----------------------------------------------------
message("  Computing periodogram (FFT) on pure_sine...")
results$periodogram <- tryCatch({
  spec_result <- stats::spectrum(pure_sine, method = "pgram", plot = FALSE)

  freq <- as.numeric(spec_result$freq)
  spec <- as.numeric(spec_result$spec)

  # Detect peak frequency
  peak_idx <- which.max(spec)
  peak_freq <- freq[peak_idx]

  message(sprintf("    Frequencies: %d values, range [%.4f, %.4f]",
                  length(freq), min(freq), max(freq)))
  message(sprintf("    Peak frequency: %.6f (index %d)", peak_freq, peak_idx))

  list(
    freq = freq,
    spec = spec,
    peak_freq = peak_freq,
    peak_index = peak_idx
  )
}, error = function(e) {
  warning(sprintf("Periodogram failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) ACF ------------------------------------------------------------------
message("  Computing ACF on noisy_sine (lag.max=100)...")
results$acf <- tryCatch({
  acf_result <- stats::acf(noisy_sine, lag.max = 100, plot = FALSE)
  acf_values <- as.numeric(acf_result$acf)

  message(sprintf("    ACF values: %d (including lag 0)", length(acf_values)))

  list(
    acf = acf_values
  )
}, error = function(e) {
  warning(sprintf("ACF failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Lomb-Scargle ----------------------------------------------------------
message("  Computing Lomb-Scargle periodogram on noisy_sine...")
results$lomb_scargle <- tryCatch({
  if (!requireNamespace("lomb", quietly = TRUE)) {
    message("    lomb package not available, skipping")
    NULL
  } else {
    lsp_result <- lomb::lsp(noisy_sine, times = t_vals, type = "period",
                            plot = FALSE, ofac = 4)

    scanned <- as.numeric(lsp_result$scanned)
    power <- as.numeric(lsp_result$power)
    peak_period <- as.numeric(lsp_result$peak.at[1])

    message(sprintf("    Scanned periods: %d values", length(scanned)))
    message(sprintf("    Peak period: %.6f (true: %.1f)", peak_period, true_period))

    list(
      scanned_periods = scanned,
      power = power,
      peak_period = peak_period
    )
  }
}, error = function(e) {
  warning(sprintf("Lomb-Scargle failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) Peak detection -------------------------------------------------------
message("  Computing peak detection with pracma::findpeaks...")
results$peak_detection <- tryCatch({
  if (!requireNamespace("pracma", quietly = TRUE)) {
    message("    pracma package not available, skipping")
    NULL
  } else {
    # Create test signal: sin(2*pi*x/0.2) on x = seq(0, 2, length=200)
    x_peaks <- seq(0, 2, length.out = 200)
    signal <- sin(2 * pi * x_peaks / 0.2)

    peaks <- pracma::findpeaks(signal, minpeakheight = 0.5, minpeakdistance = 5)
    # findpeaks returns matrix: columns are [height, index, start, end]

    peak_heights <- as.numeric(peaks[, 1])
    peak_indices <- as.integer(peaks[, 2])

    message(sprintf("    Found %d peaks", length(peak_indices)))
    message(sprintf("    First 3 indices: %s", paste(head(peak_indices, 3), collapse = ", ")))

    list(
      signal = as.numeric(signal),
      x = x_peaks,
      peak_indices = peak_indices,
      peak_heights = peak_heights
    )
  }
}, error = function(e) {
  warning(sprintf("Peak detection failed: %s", conditionMessage(e)))
  NULL
})

# ---- (e) Period estimation ground truth ----------------------------------------
message("  Validating period estimation from FFT on pure_sine...")
results$period_estimation <- tryCatch({
  spec_result <- stats::spectrum(pure_sine, method = "pgram", plot = FALSE)

  freq <- as.numeric(spec_result$freq)
  spec <- as.numeric(spec_result$spec)

  peak_idx <- which.max(spec)
  peak_freq <- freq[peak_idx]

  # The spectrum() frequency is in cycles per observation.
  # To convert to actual period: period = 1 / (peak_freq * sampling_rate_in_time_units)
  # Sampling rate: n_ts observations over (max(t) - min(t)) time units
  dt <- (max(t_vals) - min(t_vals)) / (n_ts - 1)
  detected_period_fft <- 1.0 / (peak_freq / dt)

  # Alternative: directly from frequency in cycles per sample
  # detected_period_fft = 1/peak_freq (in sample units) * dt (to time units)
  detected_period_samples <- 1.0 / peak_freq
  detected_period_time <- detected_period_samples * dt

  message(sprintf("    Detected period (FFT): %.6f", detected_period_time))
  message(sprintf("    True period: %.1f", true_period))
  message(sprintf("    Absolute error: %.6f", abs(detected_period_time - true_period)))

  list(
    detected_period_fft = detected_period_time,
    true_period = true_period,
    peak_freq_cycles_per_sample = peak_freq,
    dt = dt
  )
}, error = function(e) {
  warning(sprintf("Period estimation failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results --------------------------------------------------------------
message("\n  Saving expected seasonal values...")
save_expected(results, "seasonal_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Seasonal validation complete: %d/%d computations succeeded ===", computed, total))
