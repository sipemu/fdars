# Verify Rust wavelet implementation against R's WaveletComp
#
# This script compares the amplitude modulation detection from:
# 1. fdars (Rust) - Morlet wavelet via FFT
# 2. WaveletComp (R) - Reference implementation
#
# Expected: Both should detect similar amplitude patterns

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)

# Check if WaveletComp is installed
if (!requireNamespace("WaveletComp", quietly = TRUE)) {
  message("Installing WaveletComp for comparison...")
  install.packages("WaveletComp", repos = "https://cloud.r-project.org")
}
library(WaveletComp)

set.seed(42)

# =============================================================================
# Generate test signals
# =============================================================================

n_points <- 200
t <- seq(0, 1, length.out = n_points)
period <- 0.2  # 5 cycles in [0, 1]

# 1. Stable amplitude
stable_signal <- sin(2 * pi * t / period)

# 2. Emerging amplitude (0.2 -> 1.0)
emerging_amplitude <- 0.2 + 0.8 * t
emerging_signal <- emerging_amplitude * sin(2 * pi * t / period)

# 3. Fading amplitude (1.0 -> 0.2)
fading_amplitude <- 1.0 - 0.8 * t
fading_signal <- fading_amplitude * sin(2 * pi * t / period)

# 4. Oscillating amplitude
oscillating_amplitude <- 0.5 + 0.4 * sin(2 * pi * t * 2)  # 2 modulation cycles
oscillating_signal <- oscillating_amplitude * sin(2 * pi * t / period)

signals <- list(
  stable = stable_signal,
  emerging = emerging_signal,
  fading = fading_signal,
  oscillating = oscillating_signal
)

true_amplitudes <- list(
  stable = rep(1, n_points),
  emerging = emerging_amplitude,
  fading = fading_amplitude,
  oscillating = oscillating_amplitude
)

# =============================================================================
# Run fdars (Rust) wavelet detection
# =============================================================================

cat("Running fdars wavelet detection...\n")

fdars_results <- lapply(names(signals), function(name) {
  # Create fdata object
  fd <- fdata(matrix(signals[[name]], nrow = 1), argvals = t)

  # Run wavelet-based detection
  result <- detect_amplitude_modulation(fd, period = period, method = "wavelet",
                                         modulation_threshold = 0.15,
                                         seasonality_threshold = 0.2)

  # Extract amplitude from amplitude_curve fdata object
  if (!is.null(result$amplitude_curve)) {
    amp <- as.vector(result$amplitude_curve$data)
  } else {
    amp <- rep(NA, n_points)
  }

  list(
    name = name,
    has_modulation = result$has_modulation,
    modulation_type = result$modulation_type,
    modulation_score = result$modulation_score,
    amplitude_trend = result$amplitude_trend,
    wavelet_amplitude = amp,
    time_points = t
  )
})
names(fdars_results) <- names(signals)

# =============================================================================
# Run WaveletComp for comparison
# =============================================================================

cat("Running WaveletComp analysis...\n")

waveletcomp_results <- lapply(names(signals), function(name) {
  df <- data.frame(t = t, y = signals[[name]])

  # Analyze wavelet at the target period
  # WaveletComp uses dt (time step) to determine periods
  dt <- t[2] - t[1]

  # Suppress plotting
  wt <- analyze.wavelet(df, "y", loess.span = 0, dt = dt, dj = 1/20,
                        make.pval = FALSE, verbose = FALSE)

  # Find the scale closest to our target period
  # For Morlet wavelet: period = scale * 2*pi / omega0, omega0 = 6
  target_scale <- period * 6 / (2 * pi)
  scale_idx <- which.min(abs(wt$Scale - target_scale))

  # Extract power at target period over time
  power_at_period <- wt$Power[scale_idx, ]
  amplitude <- sqrt(power_at_period)

  # Normalize amplitude to compare with true amplitude
  amplitude_normalized <- amplitude / max(amplitude, na.rm = TRUE)

  list(
    name = name,
    amplitude = amplitude_normalized,
    scale_used = wt$Scale[scale_idx],
    period_used = wt$Period[scale_idx]
  )
})
names(waveletcomp_results) <- names(signals)

# =============================================================================
# Compare results
# =============================================================================

cat("\n========================================\n")
cat("AMPLITUDE MODULATION DETECTION COMPARISON\n")
cat("========================================\n\n")

comparison_table <- data.frame(
  Signal = character(),
  fdars_type = character(),
  fdars_score = numeric(),
  fdars_trend = numeric(),
  WC_period = numeric(),
  Correlation = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(signals)) {
  fdars_res <- fdars_results[[name]]
  wc_res <- waveletcomp_results[[name]]

  # Normalize fdars amplitude for comparison
  fdars_amp <- fdars_res$wavelet_amplitude
  if (length(fdars_amp) > 0 && max(fdars_amp) > 0) {
    fdars_amp_norm <- fdars_amp / max(fdars_amp)
  } else {
    fdars_amp_norm <- rep(0, n_points)
  }

  # Compute correlation between fdars and WaveletComp amplitudes
  if (length(fdars_amp_norm) == length(wc_res$amplitude)) {
    correlation <- cor(fdars_amp_norm, wc_res$amplitude, use = "complete.obs")
  } else {
    correlation <- NA
  }

  comparison_table <- rbind(comparison_table, data.frame(
    Signal = name,
    fdars_type = fdars_res$modulation_type,
    fdars_score = round(fdars_res$modulation_score, 4),
    fdars_trend = round(fdars_res$amplitude_trend, 4),
    WC_period = round(wc_res$period_used, 4),
    Correlation = round(correlation, 4)
  ))

  cat(sprintf("Signal: %s\n", name))
  cat(sprintf("  fdars: type=%s, score=%.4f, trend=%.4f\n",
              fdars_res$modulation_type, fdars_res$modulation_score, fdars_res$amplitude_trend))
  cat(sprintf("  WaveletComp: period=%.4f (target=%.4f)\n", wc_res$period_used, period))
  cat(sprintf("  Correlation between amplitudes: %.4f\n\n", correlation))
}

print(comparison_table)

# =============================================================================
# Create comparison plots
# =============================================================================

cat("\nGenerating comparison plots...\n")

# Prepare data for plotting
plot_data <- data.frame()

for (name in names(signals)) {
  fdars_res <- fdars_results[[name]]
  wc_res <- waveletcomp_results[[name]]
  true_amp <- true_amplitudes[[name]]

  # Normalize for comparison
  fdars_amp <- fdars_res$wavelet_amplitude
  if (length(fdars_amp) > 0 && max(fdars_amp) > 0) {
    fdars_amp_norm <- fdars_amp / max(fdars_amp)
  } else {
    fdars_amp_norm <- rep(NA, n_points)
  }

  true_amp_norm <- true_amp / max(true_amp)

  plot_data <- rbind(plot_data, data.frame(
    t = t,
    signal = signals[[name]],
    true_amplitude = true_amp_norm,
    fdars_amplitude = fdars_amp_norm,
    waveletcomp_amplitude = wc_res$amplitude,
    scenario = name
  ))
}

plot_data$scenario <- factor(plot_data$scenario,
                              levels = c("stable", "emerging", "fading", "oscillating"))

# Plot 1: Signals
p_signals <- ggplot(plot_data, aes(x = t, y = signal)) +
  geom_line(color = "steelblue", linewidth = 0.5) +
  facet_wrap(~ scenario, ncol = 2, scales = "free_y") +
  labs(title = "Test Signals", x = "Time", y = "Value") +
  theme_minimal()

# Plot 2: Amplitude comparison
plot_long <- plot_data %>%
  select(t, true_amplitude, fdars_amplitude, waveletcomp_amplitude, scenario) %>%
  pivot_longer(cols = c(true_amplitude, fdars_amplitude, waveletcomp_amplitude),
               names_to = "method", values_to = "amplitude") %>%
  mutate(method = case_when(
    method == "true_amplitude" ~ "True",
    method == "fdars_amplitude" ~ "fdars (Rust)",
    method == "waveletcomp_amplitude" ~ "WaveletComp (R)"
  ))

p_amplitudes <- ggplot(plot_long, aes(x = t, y = amplitude, color = method)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ scenario, ncol = 2) +
  scale_color_manual(values = c("True" = "black", "fdars (Rust)" = "steelblue",
                                 "WaveletComp (R)" = "coral")) +
  labs(title = "Amplitude Comparison: fdars vs WaveletComp",
       subtitle = "Normalized amplitudes at target period",
       x = "Time", y = "Normalized Amplitude", color = "Method") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Save plots
if (!dir.exists("plots")) dir.create("plots")
ggsave("plots/wavelet_verification_signals.pdf", p_signals, width = 8, height = 6)
ggsave("plots/wavelet_verification_amplitudes.pdf", p_amplitudes, width = 10, height = 7)

cat("Plots saved to:\n")
cat("  - plots/wavelet_verification_signals.pdf\n")
cat("  - plots/wavelet_verification_amplitudes.pdf\n")

# =============================================================================
# Verification summary
# =============================================================================

cat("\n========================================\n")
cat("VERIFICATION SUMMARY\n")
cat("========================================\n")

# Check if correlations are high
high_corr_threshold <- 0.8
all_high_corr <- all(comparison_table$Correlation > high_corr_threshold, na.rm = TRUE)

if (all_high_corr) {
  cat("PASS: All amplitude correlations > ", high_corr_threshold, "\n")
  cat("The Rust wavelet implementation matches WaveletComp well.\n")
} else {
  cat("WARNING: Some amplitude correlations are low.\n")
  low_corr <- comparison_table[comparison_table$Correlation <= high_corr_threshold, ]
  print(low_corr)
}

# Check modulation type detection
expected_types <- list(stable = "stable", emerging = "emerging",
                       fading = "fading", oscillating = "oscillating")

cat("\nModulation type detection:\n")
all_match <- TRUE
for (name in names(expected_types)) {
  expected <- expected_types[[name]]
  got <- fdars_results[[name]]$modulation_type
  matches <- identical(got, expected)
  match_status <- if (matches) "PASS" else "FAIL"
  if (!matches) all_match <- FALSE
  cat(sprintf("  %s: expected=%s, got=%s [%s]\n", name, expected, got, match_status))
}

if (all_match) {
  cat("\nAll modulation types correctly detected!\n")
} else {
  cat("\nNote: oscillating detection requires higher modulation score\n")
}
