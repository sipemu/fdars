#!/usr/bin/env Rscript
# Seasonality Detection with Different Trend Types
#
# Tests robustness of detection methods across various trend shapes:
# - None (flat)
# - Linear
# - Quadratic
# - Cubic
# - Exponential
# - Logarithmic
# - Sigmoid (S-curve)
# - Slow sine (non-seasonal oscillation)

library(fdars)
library(ggplot2)
library(tidyr)
library(dplyr)

set.seed(42)

# --- Configuration ---
n_seasonal_strengths <- 5    # Number of seasonal strength levels
n_trend_strengths <- 4       # Number of trend strength levels per type
n_curves_per_combo <- 20     # Number of curves per combination
n_years <- 5                 # Number of years of monthly data
n_months <- n_years * 12     # Total number of monthly observations
noise_sd <- 0.3              # Standard deviation of noise

# Detection thresholds (calibrated)
detection_thresholds <- list(
  aic_comparison = 0,
  fft_confidence = 6.0,
  acf_confidence = 0.25,
  strength_spectral = 0.3
)

# Parameter ranges
seasonal_strengths <- seq(0, 1, length.out = n_seasonal_strengths)
trend_strengths <- seq(0, 1, length.out = n_trend_strengths)

# Time grid
t <- seq(0, 1, length.out = n_months)

# --- Trend functions ---
trend_functions <- list(
  none = function(t, strength) rep(0, length(t)),

  linear = function(t, strength) strength * (t - 0.5),

  quadratic = function(t, strength) strength * ((t - 0.5)^2 - 0.25),

  cubic = function(t, strength) strength * 2 * (t - 0.5)^3,

  exponential = function(t, strength) strength * (exp(2 * t) / exp(2) - 0.5),

  logarithmic = function(t, strength) strength * (log(t + 0.1) - log(0.1)) / (log(1.1) - log(0.1)) - 0.5 * strength,

  sigmoid = function(t, strength) strength * (1 / (1 + exp(-10 * (t - 0.5))) - 0.5),

  slow_sine = function(t, strength) strength * sin(2 * pi * t)  # One full cycle, NOT seasonal
)

trend_names <- names(trend_functions)

# --- Generate seasonal component ---
generate_seasonal <- function(t, strength) {
  n_cycles <- length(t) / 12  # 5 annual cycles
  seasonal <- strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + strength * 0.3 * cos(4 * pi * n_cycles * t)
  return(seasonal)
}

# --- Detection functions ---
detect_aic_comparison <- function(fd_single, fourier_nbasis_range = seq(5, 21, by = 2),
                                   pspline_nbasis = 20) {
  fourier_aics <- sapply(fourier_nbasis_range, function(k) {
    tryCatch(basis.aic(fd_single, nbasis = k, type = "fourier"), error = function(e) NA)
  })
  best_fourier_aic <- min(fourier_aics, na.rm = TRUE)

  pspline_result <- tryCatch({
    pspline(fd_single, nbasis = pspline_nbasis, lambda.select = TRUE, criterion = "AIC")
  }, error = function(e) NULL)

  if (is.null(pspline_result) || is.na(best_fourier_aic)) {
    return(list(score = NA, detected = NA))
  }

  score <- pspline_result$aic - best_fourier_aic
  detected <- score > 0
  return(list(score = score, detected = detected))
}

detect_fft <- function(fd_single) {
  result <- tryCatch({
    estimate_period(fd_single, method = "fft", detrend = "linear")
  }, error = function(e) NULL)

  if (is.null(result)) return(list(score = NA, detected = NA))

  score <- result$confidence
  detected <- score > detection_thresholds$fft_confidence
  return(list(score = score, detected = detected))
}

detect_acf <- function(fd_single) {
  result <- tryCatch({
    estimate_period(fd_single, method = "acf", detrend = "linear")
  }, error = function(e) NULL)

  if (is.null(result)) return(list(score = NA, detected = NA))

  score <- result$confidence
  detected <- score > detection_thresholds$acf_confidence
  return(list(score = score, detected = detected))
}

detect_strength_spectral <- function(fd_single, period = 0.2) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "spectral", detrend = "linear")
  }, error = function(e) NA)

  if (is.na(score)) return(list(score = NA, detected = NA))

  detected <- score > detection_thresholds$strength_spectral
  return(list(score = score, detected = detected))
}

# --- Run detection across all combinations ---
cat("=== Seasonality Detection: Multiple Trend Types ===\n\n")
cat(sprintf("Trend types: %s\n", paste(trend_names, collapse = ", ")))
cat(sprintf("Seasonal strengths: %s\n", paste(seasonal_strengths, collapse = ", ")))
cat(sprintf("Trend strengths: %s\n", paste(trend_strengths, collapse = ", ")))
cat(sprintf("Curves per combination: %d\n", n_curves_per_combo))
cat(sprintf("Total combinations: %d\n",
            length(trend_names) * length(seasonal_strengths) * length(trend_strengths)))

results <- data.frame()

for (trend_name in trend_names) {
  trend_fn <- trend_functions[[trend_name]]

  for (ss in seasonal_strengths) {
    for (ts in trend_strengths) {
      cat(sprintf("\rProcessing: %s, seasonal=%.1f, trend=%.1f   ",
                  trend_name, ss, ts))

      for (j in 1:n_curves_per_combo) {
        # Generate curve
        trend <- trend_fn(t, ts)
        seasonal <- generate_seasonal(t, ss)
        noise <- rnorm(n_months, sd = noise_sd)
        x <- trend + seasonal + noise

        fd <- fdata(matrix(x, nrow = 1), argvals = t, rangeval = c(0, 1))

        # Apply detection methods
        aic_result <- detect_aic_comparison(fd)
        fft_result <- detect_fft(fd)
        acf_result <- detect_acf(fd)
        spec_result <- detect_strength_spectral(fd)

        results <- rbind(results, data.frame(
          trend_type = trend_name,
          seasonal_strength = ss,
          trend_strength = ts,
          curve_id = j,
          aic_score = aic_result$score,
          aic_detected = aic_result$detected,
          fft_score = fft_result$score,
          fft_detected = fft_result$detected,
          acf_score = acf_result$score,
          acf_detected = acf_result$detected,
          spec_score = spec_result$score,
          spec_detected = spec_result$detected
        ))
      }
    }
  }
}
cat("\n\n")

# --- Define ground truth ---
truth_threshold <- 0.2
results$ground_truth <- results$seasonal_strength >= truth_threshold

# --- Calculate metrics ---
calculate_metrics <- function(detected, ground_truth) {
  valid <- !is.na(detected) & !is.na(ground_truth)
  detected <- detected[valid]
  ground_truth <- ground_truth[valid]

  tp <- sum(detected & ground_truth)
  tn <- sum(!detected & !ground_truth)
  fp <- sum(detected & !ground_truth)
  fn <- sum(!detected & ground_truth)

  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
  f1 <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
               2 * precision * recall / (precision + recall), NA)
  fpr <- ifelse(tn + fp > 0, fp / (tn + fp), NA)

  return(data.frame(Accuracy = accuracy, Precision = precision,
                    Recall = recall, FPR = fpr, F1 = f1))
}

# Calculate metrics by trend type
metrics_by_trend_type <- lapply(trend_names, function(tt) {
  subset_data <- results %>% filter(trend_type == tt)

  metrics <- rbind(
    cbind(Method = "AIC", TrendType = tt,
          calculate_metrics(subset_data$aic_detected, subset_data$ground_truth)),
    cbind(Method = "FFT", TrendType = tt,
          calculate_metrics(subset_data$fft_detected, subset_data$ground_truth)),
    cbind(Method = "ACF", TrendType = tt,
          calculate_metrics(subset_data$acf_detected, subset_data$ground_truth)),
    cbind(Method = "Spectral", TrendType = tt,
          calculate_metrics(subset_data$spec_detected, subset_data$ground_truth))
  )
  return(metrics)
})

metrics_all <- do.call(rbind, metrics_by_trend_type)

# --- Print results ---
cat("=== F1 Scores by Method and Trend Type ===\n\n")

f1_wide <- metrics_all %>%
  select(Method, TrendType, F1) %>%
  pivot_wider(names_from = TrendType, values_from = F1)

print(f1_wide, digits = 3)

cat("\n=== False Positive Rates by Method and Trend Type ===\n\n")

fpr_wide <- metrics_all %>%
  select(Method, TrendType, FPR) %>%
  pivot_wider(names_from = TrendType, values_from = FPR)

print(fpr_wide, digits = 3)

# --- FPR at zero seasonality by trend type and strength ---
cat("\n=== False Positive Rates at Zero Seasonality ===\n")

fpr_zero_seasonal <- results %>%
  filter(seasonal_strength == 0) %>%
  group_by(trend_type, trend_strength) %>%
  summarise(
    AIC = mean(aic_detected, na.rm = TRUE),
    FFT = mean(fft_detected, na.rm = TRUE),
    ACF = mean(acf_detected, na.rm = TRUE),
    Spectral = mean(spec_detected, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nFPR by trend type and trend strength (seasonal=0):\n")
print(fpr_zero_seasonal, n = 40)

# --- Visualization ---
cat("\n=== Generating Plots ===\n")

method_colors <- c(
  "AIC" = "#2166AC",
  "FFT" = "#B2182B",
  "ACF" = "#1B7837",
  "Spectral" = "#762A83"
)

# Plot 1: F1 scores by trend type
f1_plot_data <- metrics_all %>%
  mutate(TrendType = factor(TrendType, levels = trend_names))

p1 <- ggplot(f1_plot_data, aes(x = TrendType, y = F1 * 100, fill = Method)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "F1 Score by Trend Type",
       x = "Trend Type",
       y = "F1 Score (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom")

# Plot 2: FPR at zero seasonality across trend types
fpr_plot_data <- fpr_zero_seasonal %>%
  pivot_longer(cols = c(AIC, FFT, ACF, Spectral),
               names_to = "Method", values_to = "FPR") %>%
  mutate(trend_type = factor(trend_type, levels = trend_names))

p2 <- ggplot(fpr_plot_data, aes(x = trend_strength, y = FPR * 100,
                                 color = Method, linetype = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~ trend_type, ncol = 4) +
  scale_color_manual(values = method_colors) +
  labs(title = "False Positive Rate vs Trend Strength (No Seasonality)",
       subtitle = "How much does each trend type confuse the detectors?",
       x = "Trend Strength",
       y = "False Positive Rate (%)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

# Plot 3: Example of each trend type
example_data <- data.frame(t = rep(t, length(trend_names)))
example_data$trend_type <- rep(trend_names, each = length(t))
example_data$value <- unlist(lapply(trend_names, function(tt) {
  trend_functions[[tt]](t, 1.0)
}))
example_data$trend_type <- factor(example_data$trend_type, levels = trend_names)

p3 <- ggplot(example_data, aes(x = t, y = value)) +
  geom_line(color = "steelblue", linewidth = 1) +
  facet_wrap(~ trend_type, ncol = 4, scales = "free_y") +
  labs(title = "Trend Functions (strength = 1.0)",
       x = "Time",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 4: Detection rate heatmap for each method
detection_by_combo <- results %>%
  group_by(trend_type, seasonal_strength, trend_strength) %>%
  summarise(
    AIC = mean(aic_detected, na.rm = TRUE),
    FFT = mean(fft_detected, na.rm = TRUE),
    ACF = mean(acf_detected, na.rm = TRUE),
    Spectral = mean(spec_detected, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(AIC, FFT, ACF, Spectral),
               names_to = "Method", values_to = "Detection_Rate") %>%
  mutate(trend_type = factor(trend_type, levels = trend_names))

p4 <- ggplot(detection_by_combo %>% filter(trend_strength == max(trend_strength)),
             aes(x = factor(seasonal_strength), y = trend_type,
                 fill = Detection_Rate)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.0f%%", Detection_Rate * 100)), size = 2.5) +
  facet_wrap(~ Method, ncol = 2) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0.5, limits = c(0, 1)) +
  geom_vline(xintercept = 1.5, linetype = "dashed") +
  labs(title = "Detection Rates at Maximum Trend Strength",
       subtitle = "By trend type and seasonal strength",
       x = "Seasonal Strength",
       y = "Trend Type",
       fill = "Detection\nRate") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# Save plots
ggsave("plots/seasonality_detection_trend_types_f1.pdf", p1, width = 10, height = 6)
ggsave("plots/seasonality_detection_trend_types_fpr.pdf", p2, width = 14, height = 8)
ggsave("plots/seasonality_detection_trend_types_examples.pdf", p3, width = 12, height = 6)
ggsave("plots/seasonality_detection_trend_types_heatmap.pdf", p4, width = 10, height = 8)

cat("Plots saved to:\n")
cat("  - plots/seasonality_detection_trend_types_f1.pdf\n")
cat("  - plots/seasonality_detection_trend_types_fpr.pdf\n")
cat("  - plots/seasonality_detection_trend_types_examples.pdf\n")
cat("  - plots/seasonality_detection_trend_types_heatmap.pdf\n")

# Save results
saveRDS(results, "seasonality_detection_trend_types_results.rds")
saveRDS(metrics_all, "seasonality_detection_trend_types_metrics.rds")

# --- Summary ---
cat("\n=== Summary ===\n\n")

# Find which trend types cause most problems (highest FPR at max trend strength)
problem_trends <- fpr_zero_seasonal %>%
  filter(trend_strength == max(trend_strength)) %>%
  pivot_longer(cols = c(AIC, FFT, ACF, Spectral),
               names_to = "Method", values_to = "FPR") %>%
  group_by(Method) %>%
  slice_max(FPR, n = 2) %>%
  arrange(Method, desc(FPR))

cat("Most problematic trend types (highest FPR at max strength):\n")
print(problem_trends, n = 20)

# Overall best method
best_methods <- metrics_all %>%
  group_by(Method) %>%
  summarise(
    Mean_F1 = mean(F1, na.rm = TRUE),
    Mean_FPR = mean(FPR, na.rm = TRUE),
    Min_F1 = min(F1, na.rm = TRUE),
    Max_FPR = max(FPR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_F1))

cat("\nOverall method performance (across all trend types):\n")
print(best_methods)

cat("\nAnalysis complete!\n")
