#!/usr/bin/env Rscript
# Seasonality Detection with Non-linear Trend
#
# Extension of the seasonality detection comparison that adds non-linear
# trend components to test robustness of detection methods.
#
# Compares methods across:
# - Different seasonal strengths (0 to 1)
# - Different non-linear trend strengths (0 to 1)
#
# Methods tested:
# 1. AIC comparison (Fourier vs P-spline)
# 2. FFT-based period estimation confidence
# 3. ACF-based period estimation confidence
# 4. Seasonal strength (variance method)
# 5. Seasonal strength (spectral method)
# 6. Automatic basis selection (select.basis.auto)

library(fdars)
library(ggplot2)
library(tidyr)
library(dplyr)

set.seed(42)

# --- Configuration ---
n_seasonal_strengths <- 6    # Number of seasonal strength levels
n_trend_strengths <- 6       # Number of trend strength levels
n_curves_per_combo <- 30     # Number of curves per combination
n_years <- 5                 # Number of years of monthly data
n_months <- n_years * 12     # Total number of monthly observations
noise_sd <- 0.3              # Standard deviation of noise

# Detection thresholds
detection_thresholds <- list(
  aic_comparison = 0,
  fft_confidence = 2.0,
  acf_confidence = 0.3,
  strength_variance = 0.3,
  strength_spectral = 0.3,
  basis_auto = 0.5
)

# Parameter ranges
seasonal_strengths <- seq(0, 1, length.out = n_seasonal_strengths)
trend_strengths <- seq(0, 1, length.out = n_trend_strengths)

# Time grid
t <- seq(0, 1, length.out = n_months)

# --- Non-linear trend functions ---
generate_nonlinear_trend <- function(t, trend_strength) {
  # Combination of polynomial and exponential components
  # Creates a complex non-linear trend that varies over time

  # Quadratic component
  quadratic <- 2 * (t - 0.5)^2

  # Cubic component for asymmetry
  cubic <- 0.5 * (t - 0.3)^3

  # Sigmoid-like component
  sigmoid <- 1 / (1 + exp(-10 * (t - 0.6)))

  # Combine components
  trend <- trend_strength * (quadratic + cubic + 0.3 * sigmoid - 0.5)

  return(trend)
}

# --- Function to generate seasonal time series with trend ---
generate_curve <- function(t, seasonal_strength, trend_strength, noise_sd = 0.3) {
  # Non-linear trend
  trend <- generate_nonlinear_trend(t, trend_strength)

  # Seasonal component (annual cycle = 5 complete cycles)
  n_cycles <- length(t) / 12
  seasonal <- seasonal_strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + seasonal_strength * 0.3 * cos(4 * pi * n_cycles * t)

  # Noise
  noise <- rnorm(length(t), sd = noise_sd)

  return(trend + seasonal + noise)
}

# --- Detection method functions ---

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

detect_strength_variance <- function(fd_single, period = 12) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "variance", detrend = "linear")
  }, error = function(e) NA)

  if (is.na(score)) return(list(score = NA, detected = NA))

  detected <- score > detection_thresholds$strength_variance
  return(list(score = score, detected = detected))
}

detect_strength_spectral <- function(fd_single, period = 12) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "spectral", detrend = "linear")
  }, error = function(e) NA)

  if (is.na(score)) return(list(score = NA, detected = NA))

  detected <- score > detection_thresholds$strength_spectral
  return(list(score = score, detected = detected))
}

detect_basis_auto <- function(fd_single) {
  result <- tryCatch({
    select.basis.auto(fd_single, criterion = "AIC", use.seasonal.hint = TRUE)
  }, error = function(e) NULL)

  if (is.null(result)) return(list(score = NA, detected = NA))

  detected <- result$seasonal.detected[1]
  score <- as.numeric(detected)

  return(list(score = score, detected = detected))
}

# --- Generate data and run detection ---
cat("=== Seasonality Detection with Non-linear Trend ===\n\n")
cat("Generating data...\n")
cat(sprintf("  - %d seasonal strength levels\n", n_seasonal_strengths))
cat(sprintf("  - %d trend strength levels\n", n_trend_strengths))
cat(sprintf("  - %d curves per combination\n", n_curves_per_combo))
cat(sprintf("  - Total: %d curves\n", n_seasonal_strengths * n_trend_strengths * n_curves_per_combo))

results <- data.frame(
  seasonal_strength = numeric(0),
  trend_strength = numeric(0),
  curve_id = integer(0),
  aic_score = numeric(0),
  aic_detected = logical(0),
  fft_score = numeric(0),
  fft_detected = logical(0),
  acf_score = numeric(0),
  acf_detected = logical(0),
  var_score = numeric(0),
  var_detected = logical(0),
  spec_score = numeric(0),
  spec_detected = logical(0),
  auto_score = numeric(0),
  auto_detected = logical(0)
)

total_combos <- length(seasonal_strengths) * length(trend_strengths)
combo_count <- 0

for (ss in seasonal_strengths) {
  for (ts in trend_strengths) {
    combo_count <- combo_count + 1
    cat(sprintf("\nProcessing [%d/%d] seasonal=%.1f, trend=%.1f...\n",
                combo_count, total_combos, ss, ts))

    for (j in 1:n_curves_per_combo) {
      if (j %% 10 == 0) cat(sprintf("  Curve %d/%d\n", j, n_curves_per_combo))

      # Generate curve
      x <- generate_curve(t, ss, ts, noise_sd)
      fd <- fdata(matrix(x, nrow = 1), argvals = t, rangeval = c(0, 1))

      # Apply all detection methods
      aic_result <- detect_aic_comparison(fd)
      fft_result <- detect_fft(fd)
      acf_result <- detect_acf(fd)
      var_result <- detect_strength_variance(fd)
      spec_result <- detect_strength_spectral(fd)
      auto_result <- detect_basis_auto(fd)

      results <- rbind(results, data.frame(
        seasonal_strength = ss,
        trend_strength = ts,
        curve_id = j,
        aic_score = aic_result$score,
        aic_detected = aic_result$detected,
        fft_score = fft_result$score,
        fft_detected = fft_result$detected,
        acf_score = acf_result$score,
        acf_detected = acf_result$detected,
        var_score = var_result$score,
        var_detected = var_result$detected,
        spec_score = spec_result$score,
        spec_detected = spec_result$detected,
        auto_score = auto_result$score,
        auto_detected = auto_result$detected
      ))
    }
  }
}

# --- Define ground truth ---
truth_threshold <- 0.2
results$ground_truth <- results$seasonal_strength >= truth_threshold

# --- Calculate metrics ---
cat("\n=== Detection Rate Analysis ===\n")

# Detection rates by seasonal and trend strength
detection_summary <- results %>%
  group_by(seasonal_strength, trend_strength) %>%
  summarise(
    AIC = mean(aic_detected, na.rm = TRUE),
    FFT = mean(fft_detected, na.rm = TRUE),
    ACF = mean(acf_detected, na.rm = TRUE),
    Var_Strength = mean(var_detected, na.rm = TRUE),
    Spec_Strength = mean(spec_detected, na.rm = TRUE),
    Basis_Auto = mean(auto_detected, na.rm = TRUE),
    .groups = "drop"
  )

# Classification metrics by trend strength
cat("\n=== Classification Performance by Trend Strength ===\n")

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
  specificity <- ifelse(tn + fp > 0, tn / (tn + fp), NA)

  return(data.frame(Accuracy = accuracy, Precision = precision,
                    Recall = recall, Specificity = specificity, F1 = f1))
}

metrics_by_trend <- lapply(trend_strengths, function(ts) {
  subset_data <- results %>% filter(trend_strength == ts)

  metrics <- rbind(
    cbind(Method = "AIC", Trend = ts, calculate_metrics(subset_data$aic_detected, subset_data$ground_truth)),
    cbind(Method = "FFT", Trend = ts, calculate_metrics(subset_data$fft_detected, subset_data$ground_truth)),
    cbind(Method = "ACF", Trend = ts, calculate_metrics(subset_data$acf_detected, subset_data$ground_truth)),
    cbind(Method = "Var Strength", Trend = ts, calculate_metrics(subset_data$var_detected, subset_data$ground_truth)),
    cbind(Method = "Spec Strength", Trend = ts, calculate_metrics(subset_data$spec_detected, subset_data$ground_truth)),
    cbind(Method = "Basis Auto", Trend = ts, calculate_metrics(subset_data$auto_detected, subset_data$ground_truth))
  )
  return(metrics)
})

metrics_all <- do.call(rbind, metrics_by_trend)

# Print summary
cat("\nF1 Scores by Method and Trend Strength:\n")
cat("=" , rep("=", 80), "\n", sep = "")

f1_wide <- metrics_all %>%
  select(Method, Trend, F1) %>%
  pivot_wider(names_from = Trend, values_from = F1)

print(f1_wide, digits = 3)

# --- Visualization with ggplot2 ---
cat("\n=== Generating Plots ===\n")

method_colors <- c(
  "AIC" = "#2166AC",
  "FFT" = "#B2182B",
  "ACF" = "#1B7837",
  "Var Strength" = "#E66101",
  "Spec Strength" = "#762A83",
  "Basis Auto" = "#8C510A"
)

# Plot 1: Heatmaps of detection rates for each method
detection_long <- detection_summary %>%
  pivot_longer(cols = c(AIC, FFT, ACF, Var_Strength, Spec_Strength, Basis_Auto),
               names_to = "Method", values_to = "Detection_Rate") %>%
  mutate(Method = recode(Method,
                         "Var_Strength" = "Var Strength",
                         "Spec_Strength" = "Spec Strength",
                         "Basis_Auto" = "Basis Auto"))

p_heatmaps <- ggplot(detection_long,
                     aes(x = factor(seasonal_strength),
                         y = factor(trend_strength),
                         fill = Detection_Rate)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.0f%%", Detection_Rate * 100)), size = 2.5) +
  facet_wrap(~ Method, ncol = 3) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0.5, limits = c(0, 1),
                       labels = scales::percent) +
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "black", alpha = 0.5) +
  labs(title = "Detection Rates by Seasonal and Trend Strength",
       subtitle = "Dashed line indicates ground truth threshold (seasonal >= 0.2)",
       x = "Seasonal Strength",
       y = "Trend Strength",
       fill = "Detection\nRate") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        strip.text = element_text(face = "bold"))

# Plot 2: F1 score vs trend strength for each method
f1_plot_data <- metrics_all %>%
  filter(!is.na(F1)) %>%
  mutate(Method = factor(Method, levels = c("AIC", "FFT", "ACF",
                                             "Var Strength", "Spec Strength", "Basis Auto")))

p_f1_trend <- ggplot(f1_plot_data, aes(x = Trend, y = F1 * 100, color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "F1 Score vs Trend Strength",
       subtitle = "How robust are methods to non-linear trends?",
       x = "Trend Strength",
       y = "F1 Score (%)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# Plot 3: Detection rate vs seasonal strength, faceted by trend strength
detection_by_seasonal <- results %>%
  group_by(seasonal_strength, trend_strength) %>%
  summarise(
    AIC = mean(aic_detected, na.rm = TRUE),
    FFT = mean(fft_detected, na.rm = TRUE),
    ACF = mean(acf_detected, na.rm = TRUE),
    Spec_Strength = mean(spec_detected, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(AIC, FFT, ACF, Spec_Strength),
               names_to = "Method", values_to = "Detection_Rate") %>%
  mutate(Method = recode(Method, "Spec_Strength" = "Spec Strength"))

p_detection_facet <- ggplot(detection_by_seasonal,
                            aes(x = seasonal_strength, y = Detection_Rate * 100,
                                color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_vline(xintercept = truth_threshold, linetype = "dashed", color = "gray50") +
  facet_wrap(~ paste("Trend =", trend_strength), ncol = 3) +
  scale_color_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Detection Rates by Seasonal Strength (Faceted by Trend)",
       x = "Seasonal Strength",
       y = "Detection Rate (%)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 4: Example curves showing trend + seasonal combinations
example_curves <- data.frame(t = rep(t, 9))
example_curves$curve_id <- rep(1:9, each = length(t))

params <- expand.grid(seasonal = c(0, 0.5, 1), trend = c(0, 0.5, 1))
example_curves$value <- unlist(lapply(1:9, function(i) {
  generate_curve(t, params$seasonal[i], params$trend[i], noise_sd = 0)
}))
example_curves$label <- rep(sprintf("S=%.1f, T=%.1f", params$seasonal, params$trend),
                            each = length(t))

p_examples <- ggplot(example_curves, aes(x = t, y = value)) +
  geom_line(color = "steelblue") +
  facet_wrap(~ label, ncol = 3, scales = "free_y") +
  labs(title = "Example Curves: Seasonal (S) + Trend (T) Combinations",
       subtitle = "Without noise",
       x = "Time",
       y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# Plot 5: Precision-Recall curves by trend strength (for best methods)
compute_pr_curve <- function(scores, ground_truth, method_name, n_points = 100) {
  valid <- !is.na(scores) & !is.na(ground_truth)
  scores <- scores[valid]
  ground_truth <- ground_truth[valid]

  if (length(unique(scores)) < 2) return(NULL)

  thresholds <- seq(min(scores), max(scores), length.out = n_points)

  pr_data <- lapply(thresholds, function(thresh) {
    detected <- scores > thresh
    tp <- sum(detected & ground_truth)
    fp <- sum(detected & !ground_truth)
    fn <- sum(!detected & ground_truth)

    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 1)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)

    data.frame(threshold = thresh, precision = precision, recall = recall)
  })

  pr_df <- do.call(rbind, pr_data)
  pr_df$Method <- method_name
  return(pr_df)
}

# PR curves for different trend strengths (AIC and Spectral only for clarity)
pr_by_trend <- lapply(c(0, 0.5, 1), function(ts) {
  subset_data <- results %>% filter(trend_strength == ts)

  pr_aic <- compute_pr_curve(subset_data$aic_score, subset_data$ground_truth, "AIC")
  pr_spec <- compute_pr_curve(subset_data$spec_score, subset_data$ground_truth, "Spec Strength")

  pr_combined <- rbind(pr_aic, pr_spec)
  if (!is.null(pr_combined)) {
    pr_combined$Trend <- ts
  }
  return(pr_combined)
})

pr_trend_data <- do.call(rbind, pr_by_trend)

p_pr_trend <- ggplot(pr_trend_data, aes(x = recall, y = precision, color = Method)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ paste("Trend =", Trend), ncol = 3) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Precision-Recall Curves by Trend Strength",
       subtitle = "Comparing AIC and Spectral Strength methods",
       x = "Recall",
       y = "Precision") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# Plot 6: Robustness summary - F1 drop from no trend to max trend
f1_robustness <- metrics_all %>%
  filter(!is.na(F1)) %>%
  group_by(Method) %>%
  summarise(
    F1_no_trend = F1[Trend == min(Trend)],
    F1_max_trend = F1[Trend == max(Trend)],
    F1_drop = F1_no_trend - F1_max_trend,
    F1_drop_pct = (F1_no_trend - F1_max_trend) / F1_no_trend * 100,
    .groups = "drop"
  ) %>%
  mutate(Method = factor(Method, levels = c("AIC", "FFT", "ACF",
                                             "Var Strength", "Spec Strength", "Basis Auto")))

p_robustness <- ggplot(f1_robustness, aes(x = reorder(Method, -F1_drop_pct),
                                           y = F1_drop_pct, fill = Method)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", F1_drop_pct)), vjust = -0.5, size = 3) +
  scale_fill_manual(values = method_colors) +
  labs(title = "Robustness to Non-linear Trend",
       subtitle = "F1 score drop (%) from no trend to maximum trend",
       x = "Method",
       y = "F1 Drop (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save plots
ggsave("seasonality_detection_trend_heatmaps.pdf",
       p_heatmaps, width = 12, height = 8)

ggsave("seasonality_detection_trend_comparison.pdf",
       gridExtra::grid.arrange(p_f1_trend, p_robustness,
                               p_detection_facet, p_examples,
                               ncol = 2, nrow = 2),
       width = 14, height = 12)

ggsave("seasonality_detection_trend_pr.pdf",
       gridExtra::grid.arrange(p_pr_trend, ncol = 1),
       width = 12, height = 5)

cat("Plots saved to:\n")
cat("  - seasonality_detection_trend_heatmaps.pdf\n")
cat("  - seasonality_detection_trend_comparison.pdf\n")
cat("  - seasonality_detection_trend_pr.pdf\n")

# Save results
saveRDS(results, "seasonality_detection_trend_results.rds")
saveRDS(metrics_all, "seasonality_detection_trend_metrics.rds")
cat("\nResults saved to:\n")
cat("  - seasonality_detection_trend_results.rds\n")
cat("  - seasonality_detection_trend_metrics.rds\n")

# --- Summary ---
cat("\n=== Summary ===\n\n")

cat("F1 Score Summary (No Trend vs Max Trend):\n")
cat("-" , rep("-", 60), "\n", sep = "")
cat(sprintf("%-15s %15s %15s %15s\n", "Method", "No Trend", "Max Trend", "Drop"))
cat("-" , rep("-", 60), "\n", sep = "")
for (i in 1:nrow(f1_robustness)) {
  cat(sprintf("%-15s %14.1f%% %14.1f%% %14.1f%%\n",
              as.character(f1_robustness$Method[i]),
              f1_robustness$F1_no_trend[i] * 100,
              f1_robustness$F1_max_trend[i] * 100,
              f1_robustness$F1_drop_pct[i]))
}
cat("-" , rep("-", 60), "\n", sep = "")

# Find most robust method
most_robust <- f1_robustness %>%
  filter(!is.na(F1_drop_pct)) %>%
  arrange(F1_drop_pct) %>%
  slice(1)

best_overall <- f1_robustness %>%
  filter(!is.na(F1_max_trend)) %>%
  arrange(desc(F1_max_trend)) %>%
  slice(1)

cat(sprintf("\nMost robust to trend: %s (%.1f%% F1 drop)\n",
            most_robust$Method, most_robust$F1_drop_pct))
cat(sprintf("Best F1 at max trend: %s (%.1f%% F1)\n",
            best_overall$Method, best_overall$F1_max_trend * 100))

cat("\nAnalysis complete!\n")
