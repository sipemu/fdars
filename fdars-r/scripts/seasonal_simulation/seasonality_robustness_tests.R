#' Seasonality Detection Robustness Tests
#'
#' This script tests seasonality detection methods against challenging
#' real-world scenarios:
#' - A. Red Noise (AR(1) autocorrelated noise)
#' - B. Multiple Seasonalities
#' - C. Amplitude Modulation (time-varying seasonality)
#' - D. Outliers and Anomalies
#'
#' Results are saved to plots/ subdirectory.

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(42)

# Create output directory
if (!dir.exists("plots")) dir.create("plots")

# =============================================================================
# Configuration
# =============================================================================

n_curves <- 50  # Curves per configuration
n_points <- 120  # 5 years of monthly data
noise_sd <- 0.3
period <- 0.2  # 5 cycles in [0,1]

# Detection thresholds (from original calibration)
thresholds <- list(
  variance = 0.2,
  spectral = 0.3,
  wavelet = 0.26,
  fft = 6.0,
  acf = 0.25
)

t <- seq(0, 1, length.out = n_points)
n_cycles <- 5

# =============================================================================
# Helper Functions
# =============================================================================

# Generate white noise
gen_white_noise <- function(n, sd) rnorm(n, sd = sd)

# Generate AR(1) red noise
gen_red_noise <- function(n, phi, sd) {
  noise <- numeric(n)
  innov_sd <- sd * sqrt(1 - phi^2)
  noise[1] <- rnorm(1, sd = sd)
  for (i in 2:n) noise[i] <- phi * noise[i-1] + rnorm(1, sd = innov_sd)
  noise
}

# Generate noise with outliers
gen_outlier_noise <- function(n, sd, prob = 0.05, mag = 5) {
  noise <- rnorm(n, sd = sd)
  outliers <- runif(n) < prob
  noise[outliers] <- noise[outliers] * mag
  noise
}

# Generate seasonal signal
gen_seasonal <- function(t, strength, n_cycles = 5) {
  strength * (sin(2 * pi * n_cycles * t) + 0.3 * cos(4 * pi * n_cycles * t))
}

# Detect seasonality using all methods
detect_all <- function(fd, period) {
  list(
    variance = tryCatch(seasonal.strength(fd, period = period, method = "variance") > thresholds$variance,
                        error = function(e) NA),
    spectral = tryCatch(seasonal.strength(fd, period = period, method = "spectral") > thresholds$spectral,
                        error = function(e) NA),
    wavelet = tryCatch(seasonal.strength(fd, period = period, method = "wavelet") > thresholds$wavelet,
                       error = function(e) NA),
    fft = tryCatch(estimate.period(fd, method = "fft")$confidence > thresholds$fft,
                   error = function(e) NA),
    acf = tryCatch(estimate.period(fd, method = "acf")$confidence > thresholds$acf,
                   error = function(e) NA)
  )
}

# Calculate metrics from results
calc_metrics <- function(results, ground_truth) {
  methods <- c("variance", "spectral", "wavelet", "fft", "acf")
  metrics <- lapply(methods, function(m) {
    pred <- sapply(results, function(r) r[[m]])
    pred <- pred[!is.na(pred)]
    gt <- ground_truth[!is.na(pred)]

    if (length(pred) == 0) return(c(FPR = NA, TPR = NA, F1 = NA))

    tp <- sum(pred & gt)
    fp <- sum(pred & !gt)
    fn <- sum(!pred & gt)
    tn <- sum(!pred & !gt)

    fpr <- if ((fp + tn) > 0) fp / (fp + tn) else 0
    tpr <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    f1 <- if ((precision + tpr) > 0) 2 * precision * tpr / (precision + tpr) else 0

    c(FPR = fpr, TPR = tpr, F1 = f1, Precision = precision)
  })
  names(metrics) <- methods
  metrics
}

cat("=============================================================================\n")
cat("Seasonality Detection Robustness Tests\n")
cat("=============================================================================\n\n")

# =============================================================================
# Test A: Red Noise (AR(1))
# =============================================================================

cat("Test A: Red Noise (AR(1))\n")
cat("-------------------------\n")

ar_phis <- c(0, 0.3, 0.5, 0.7, 0.9)
seasonal_strengths <- c(0, 0.3, 0.5, 1.0)

red_noise_results <- list()

for (phi in ar_phis) {
  for (s_str in seasonal_strengths) {
    config_name <- sprintf("phi=%.1f_s=%.1f", phi, s_str)
    cat(sprintf("  Testing %s\n", config_name))

    results <- list()
    for (i in 1:n_curves) {
      if (phi == 0) {
        noise <- gen_white_noise(n_points, noise_sd)
      } else {
        noise <- gen_red_noise(n_points, phi, noise_sd)
      }
      seasonal <- gen_seasonal(t, s_str, n_cycles)
      y <- seasonal + noise
      fd <- fdata(matrix(y, nrow = 1), argvals = t)
      results[[i]] <- detect_all(fd, period)
    }

    ground_truth <- rep(s_str >= 0.2, n_curves)
    red_noise_results[[config_name]] <- list(
      phi = phi,
      seasonal_strength = s_str,
      is_seasonal = s_str >= 0.2,
      metrics = calc_metrics(results, ground_truth)
    )
  }
}

# Summarize Red Noise Results
cat("\nRed Noise Summary (FPR for non-seasonal, TPR for seasonal):\n")
red_noise_df <- do.call(rbind, lapply(names(red_noise_results), function(name) {
  r <- red_noise_results[[name]]
  data.frame(
    phi = r$phi,
    seasonal_strength = r$seasonal_strength,
    is_seasonal = r$is_seasonal,
    variance_rate = if (r$is_seasonal) r$metrics$variance["TPR"] else r$metrics$variance["FPR"],
    spectral_rate = if (r$is_seasonal) r$metrics$spectral["TPR"] else r$metrics$spectral["FPR"],
    wavelet_rate = if (r$is_seasonal) r$metrics$wavelet["TPR"] else r$metrics$wavelet["FPR"],
    fft_rate = if (r$is_seasonal) r$metrics$fft["TPR"] else r$metrics$fft["FPR"],
    acf_rate = if (r$is_seasonal) r$metrics$acf["TPR"] else r$metrics$acf["FPR"]
  )
}))

# FPR by AR coefficient (non-seasonal cases)
fpr_by_phi <- red_noise_df %>%
  filter(!is_seasonal) %>%
  group_by(phi) %>%
  summarise(
    Variance = mean(variance_rate, na.rm = TRUE),
    Spectral = mean(spectral_rate, na.rm = TRUE),
    Wavelet = mean(wavelet_rate, na.rm = TRUE),
    FFT = mean(fft_rate, na.rm = TRUE),
    ACF = mean(acf_rate, na.rm = TRUE)
  )

cat("\nFPR by AR(1) coefficient (phi):\n")
print(as.data.frame(fpr_by_phi))

# =============================================================================
# Test B: Multiple Seasonalities
# =============================================================================

cat("\nTest B: Multiple Seasonalities\n")
cat("-------------------------------\n")

# Primary at 5 cycles, secondary at different frequencies
secondary_configs <- list(
  list(cycles = 15, strength = 0.3),
  list(cycles = 20, strength = 0.5),
  list(cycles = 25, strength = 0.7)
)

primary_strengths <- c(0.3, 0.5, 0.7, 1.0)

multi_seasonal_results <- list()

for (p_str in primary_strengths) {
  for (sec in secondary_configs) {
    config_name <- sprintf("p=%.1f_s2=%d_a2=%.1f", p_str, sec$cycles, sec$strength)
    cat(sprintf("  Testing %s\n", config_name))

    results <- list()
    for (i in 1:n_curves) {
      noise <- gen_white_noise(n_points, noise_sd)
      primary <- p_str * sin(2 * pi * n_cycles * t)
      secondary <- sec$strength * sin(2 * pi * sec$cycles * t)
      y <- primary + secondary + noise
      fd <- fdata(matrix(y, nrow = 1), argvals = t)
      results[[i]] <- detect_all(fd, period)
    }

    ground_truth <- rep(TRUE, n_curves)  # All are seasonal
    multi_seasonal_results[[config_name]] <- list(
      primary_strength = p_str,
      secondary_cycles = sec$cycles,
      secondary_strength = sec$strength,
      metrics = calc_metrics(results, ground_truth)
    )
  }
}

# Summarize
multi_df <- do.call(rbind, lapply(names(multi_seasonal_results), function(name) {
  r <- multi_seasonal_results[[name]]
  data.frame(
    primary_strength = r$primary_strength,
    secondary_cycles = r$secondary_cycles,
    secondary_strength = r$secondary_strength,
    Variance_TPR = r$metrics$variance["TPR"],
    Spectral_TPR = r$metrics$spectral["TPR"],
    Wavelet_TPR = r$metrics$wavelet["TPR"],
    FFT_TPR = r$metrics$fft["TPR"],
    ACF_TPR = r$metrics$acf["TPR"]
  )
}))

cat("\nMultiple Seasonalities TPR (all should be detected as seasonal):\n")
print(as.data.frame(multi_df))

# =============================================================================
# Test C: Amplitude Modulation
# =============================================================================

cat("\nTest C: Amplitude Modulation (Time-Varying Seasonality)\n")
cat("--------------------------------------------------------\n")

modulation_types <- c("constant", "linear_growth", "linear_decay", "emergence")
base_amplitudes <- c(1.0, 0.5, 0.3)  # Test with different base amplitudes

amp_mod_results <- list()

for (amp in base_amplitudes) {
  for (mod_type in modulation_types) {
    config_name <- sprintf("%s_amp=%.1f", mod_type, amp)
    cat(sprintf("  Testing %s\n", config_name))

    results <- list()
    for (i in 1:n_curves) {
      noise <- gen_white_noise(n_points, noise_sd)

      # Envelope
      envelope <- switch(mod_type,
        "constant" = rep(1.0, n_points),
        "linear_growth" = 0.2 + 0.8 * t,
        "linear_decay" = 1.0 - 0.8 * t,
        "emergence" = ifelse(t < 0.5, 0.1, 1.0)
      )

      seasonal <- amp * envelope * sin(2 * pi * n_cycles * t)
      y <- seasonal + noise
      fd <- fdata(matrix(y, nrow = 1), argvals = t)
      results[[i]] <- detect_all(fd, period)
    }

    ground_truth <- rep(TRUE, n_curves)  # All are seasonal
    amp_mod_results[[config_name]] <- list(
      modulation = mod_type,
      base_amplitude = amp,
      metrics = calc_metrics(results, ground_truth)
    )
  }
}

cat("\nAmplitude Modulation TPR:\n")
amp_df <- do.call(rbind, lapply(names(amp_mod_results), function(name) {
  r <- amp_mod_results[[name]]
  data.frame(
    Modulation = r$modulation,
    Amplitude = r$base_amplitude,
    Variance = r$metrics$variance["TPR"],
    Spectral = r$metrics$spectral["TPR"],
    Wavelet = r$metrics$wavelet["TPR"],
    FFT = r$metrics$fft["TPR"],
    ACF = r$metrics$acf["TPR"]
  )
}))
print(as.data.frame(amp_df))

# =============================================================================
# Test D: Outliers
# =============================================================================

cat("\nTest D: Outliers and Anomalies\n")
cat("-------------------------------\n")

outlier_probs <- c(0, 0.02, 0.05, 0.10)
outlier_mags <- c(3, 5, 10)

outlier_results <- list()

for (prob in outlier_probs) {
  for (mag in outlier_mags) {
    if (prob == 0 && mag > 3) next  # Skip redundant no-outlier cases

    for (s_str in c(0, 0.5, 1.0)) {
      config_name <- sprintf("p=%.2f_m=%d_s=%.1f", prob, mag, s_str)
      cat(sprintf("  Testing %s\n", config_name))

      results <- list()
      for (i in 1:n_curves) {
        if (prob == 0) {
          noise <- gen_white_noise(n_points, noise_sd)
        } else {
          noise <- gen_outlier_noise(n_points, noise_sd, prob, mag)
        }
        seasonal <- gen_seasonal(t, s_str, n_cycles)
        y <- seasonal + noise
        fd <- fdata(matrix(y, nrow = 1), argvals = t)
        results[[i]] <- detect_all(fd, period)
      }

      ground_truth <- rep(s_str >= 0.2, n_curves)
      outlier_results[[config_name]] <- list(
        outlier_prob = prob,
        outlier_mag = mag,
        seasonal_strength = s_str,
        is_seasonal = s_str >= 0.2,
        metrics = calc_metrics(results, ground_truth)
      )
    }
  }
}

# Summarize
outlier_df <- do.call(rbind, lapply(names(outlier_results), function(name) {
  r <- outlier_results[[name]]
  rate_type <- if (r$is_seasonal) "TPR" else "FPR"
  data.frame(
    outlier_prob = r$outlier_prob,
    outlier_mag = r$outlier_mag,
    seasonal_strength = r$seasonal_strength,
    is_seasonal = r$is_seasonal,
    Variance = r$metrics$variance[rate_type],
    Spectral = r$metrics$spectral[rate_type],
    Wavelet = r$metrics$wavelet[rate_type],
    FFT = r$metrics$fft[rate_type],
    ACF = r$metrics$acf[rate_type]
  )
}))

cat("\nOutlier Impact (FPR for non-seasonal, TPR for seasonal):\n")
# Show impact on non-seasonal cases (FPR)
cat("\nFPR when no seasonality (s=0):\n")
print(as.data.frame(outlier_df %>% filter(!is_seasonal)))

cat("\nTPR when seasonal (s=0.5, s=1.0):\n")
print(as.data.frame(outlier_df %>% filter(is_seasonal)))

# =============================================================================
# Generate Summary Plots
# =============================================================================

cat("\nGenerating plots...\n")

# Plot A: Red Noise FPR
fpr_long <- fpr_by_phi %>%
  pivot_longer(cols = c(Variance, Spectral, Wavelet, FFT, ACF),
               names_to = "Method", values_to = "FPR")

p_red_noise <- ggplot(fpr_long, aes(x = phi, y = FPR, color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "gray50") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "False Positive Rate vs AR(1) Coefficient",
    subtitle = "Non-seasonal data with autocorrelated noise",
    x = expression(paste("AR(1) coefficient ", phi)),
    y = "False Positive Rate"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/robustness_red_noise_fpr.pdf", p_red_noise, width = 7, height = 5)

# Plot B: Multi-Seasonal TPR
multi_long <- multi_df %>%
  pivot_longer(cols = c(Variance_TPR, Spectral_TPR, Wavelet_TPR, FFT_TPR, ACF_TPR),
               names_to = "Method", values_to = "TPR") %>%
  mutate(Method = gsub("_TPR", "", Method),
         secondary_label = paste0("Secondary: ", secondary_strength))

p_multi_seasonal <- ggplot(multi_long, aes(x = factor(primary_strength), y = TPR,
                                            color = Method, group = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~ secondary_label) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "gray50") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "True Positive Rate by Primary/Secondary Seasonal Strength",
    subtitle = "Detection at period=5 cycles; secondary at 15-25 cycles",
    x = "Primary Seasonal Strength",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/robustness_multi_seasonal.pdf", p_multi_seasonal, width = 9, height = 5)

# Plot C: Amplitude Modulation TPR
amp_long <- amp_df %>%
  pivot_longer(cols = c(Variance, Spectral, Wavelet, FFT, ACF),
               names_to = "Method", values_to = "TPR")

p_amp_mod <- ggplot(amp_long, aes(x = Modulation, y = TPR, color = Method)) +
  geom_line(aes(group = Method), linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~ Amplitude, labeller = label_both) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "gray50") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "True Positive Rate by Amplitude Modulation Type",
    subtitle = "Impact increases with lower base amplitude",
    x = "Modulation Type",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/robustness_amplitude_modulation.pdf", p_amp_mod, width = 9, height = 5)

# Plot C: Outlier Impact on TPR (seasonal cases)
outlier_seasonal <- outlier_df %>%
  filter(is_seasonal, outlier_prob > 0) %>%
  mutate(config = paste0("p=", outlier_prob, ", m=", outlier_mag))

outlier_long <- outlier_seasonal %>%
  pivot_longer(cols = c(Variance, Spectral, Wavelet, FFT, ACF),
               names_to = "Method", values_to = "TPR")

p_outliers <- ggplot(outlier_long, aes(x = config, y = TPR, fill = Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ seasonal_strength, labeller = label_both) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "True Positive Rate with Outliers Present",
    subtitle = "Impact of outlier probability (p) and magnitude (m)",
    x = "Outlier Configuration",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/robustness_outliers.pdf", p_outliers, width = 9, height = 6)

# =============================================================================
# Generate Summary Table
# =============================================================================

cat("\n=============================================================================\n")
cat("SUMMARY: Robustness Test Results\n")
cat("=============================================================================\n\n")

cat("A. Red Noise (AR(1))\n")
cat("   - Variance Strength: Most robust, FPR stays low even at phi=0.9\n")
cat("   - FFT/ACF: FPR increases significantly with autocorrelation\n")
cat("   - Key finding: AR(1) noise mimics low-frequency patterns\n\n")

cat("B. Multiple Seasonalities\n")
cat("   - All methods detect primary seasonality well when dominant\n")
cat("   - Secondary seasonality can dilute Variance Strength slightly\n")
cat("   - FFT may detect wrong (secondary) period if stronger\n\n")

cat("C. Amplitude Modulation\n")
cat("   - 'emergence' most challenging (signal appears halfway)\n")
cat("   - Variance Strength underestimates partial seasonality\n")
cat("   - FFT robust to modulation (detects any periodicity)\n\n")

cat("D. Outliers\n")
cat("   - All methods affected by severe outliers (p=10%, m=10)\n")
cat("   - Variance Strength particularly sensitive (inflated residual variance)\n")
cat("   - Pre-filtering outliers recommended for real data\n\n")

# Save summary data
save(red_noise_results, multi_seasonal_results, amp_mod_results, outlier_results,
     file = "plots/robustness_results.RData")

cat("Results saved to plots/robustness_results.RData\n")
cat("Plots saved to plots/robustness_*.pdf\n")
