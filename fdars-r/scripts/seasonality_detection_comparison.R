#!/usr/bin/env Rscript
# Seasonality Detection Method Comparison
#
# Compares different methods for detecting seasonality in functional data:
# 1. AIC comparison (Fourier vs P-spline)
# 2. FFT-based period estimation confidence
# 3. ACF-based period estimation confidence
# 4. Seasonal strength (variance method)
# 5. Seasonal strength (spectral method)
# 6. Automatic basis selection (select.basis.auto)
#
# Ground truth: simulated data with known seasonal strengths (0 to 1)

library(fdars)
library(ggplot2)
library(tidyr)
library(dplyr)

set.seed(42)

# --- Configuration ---
n_strengths <- 11           # Number of seasonal strength levels (0 to 1)
n_curves_per_strength <- 50 # Number of curves per strength level
n_years <- 5                # Number of years of monthly data
n_months <- n_years * 12    # Total number of monthly observations
noise_sd <- 0.3             # Standard deviation of noise

# Detection threshold for binary classification
# Methods return continuous scores; we threshold for detection rate
# Thresholds calibrated to ~5% false positive rate on pure noise
detection_thresholds <- list(
  aic_comparison = 0,           # Fourier AIC < P-spline AIC
  fft_confidence = 6.0,         # Power ratio threshold (95th pct of noise ~5.7)
  acf_confidence = 0.25,        # ACF peak threshold (95th pct of noise ~0.22)
  strength_variance = 0.1,      # Variance ratio threshold (use without detrend)
  strength_spectral = 0.3,      # Spectral power threshold (95th pct of noise ~0.29)
  basis_auto = 0.5              # Already binary (uses internal threshold)
)

# Seasonal strengths to test (0 = no season, 1 = full season)
seasonal_strengths <- seq(0, 1, length.out = n_strengths)

# Time grid (monthly data normalized to [0, 1])
t <- seq(0, 1, length.out = n_months)

# --- Function to generate seasonal time series ---
generate_seasonal_curve <- function(t, strength, noise_sd = 0.3) {
  # Base trend (slow variation)
  trend <- 0.5 * sin(2 * pi * t) * 0.3

  # Seasonal component (annual cycle = 5 complete cycles over our time span)
  n_cycles <- length(t) / 12
  seasonal <- strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + strength * 0.3 * cos(4 * pi * n_cycles * t)

  # Noise
  noise <- rnorm(length(t), sd = noise_sd)

  return(trend + seasonal + noise)
}

# --- Detection method functions ---

# Method 1: AIC comparison (Fourier vs P-spline)
detect_aic_comparison <- function(fd_single, fourier_nbasis_range = seq(5, 21, by = 2),
                                   pspline_nbasis = 20) {
  # Find optimal Fourier AIC
  fourier_aics <- sapply(fourier_nbasis_range, function(k) {
    tryCatch(basis.aic(fd_single, nbasis = k, type = "fourier"), error = function(e) NA)
  })
  best_fourier_aic <- min(fourier_aics, na.rm = TRUE)

  # P-spline AIC with automatic lambda

  pspline_result <- tryCatch({
    pspline(fd_single, nbasis = pspline_nbasis, lambda.select = TRUE, criterion = "AIC")
  }, error = function(e) NULL)

  if (is.null(pspline_result) || is.na(best_fourier_aic)) {
    return(list(score = NA, detected = NA))
  }

  # Score: negative difference means Fourier is better (seasonal)
  score <- pspline_result$aic - best_fourier_aic
  detected <- score > 0  # Fourier wins = seasonal detected

  return(list(score = score, detected = detected))
}

# Method 2: FFT-based period estimation
detect_fft <- function(fd_single, expected_period = 12) {
  result <- tryCatch({
    estimate_period(fd_single, method = "fft", detrend = "linear")
  }, error = function(e) NULL)

  if (is.null(result)) {
    return(list(score = NA, detected = NA))
  }

  # Use confidence as score
  score <- result$confidence
  detected <- score > detection_thresholds$fft_confidence

  return(list(score = score, detected = detected))
}

# Method 3: ACF-based period estimation
detect_acf <- function(fd_single, expected_period = 12) {
  result <- tryCatch({
    estimate_period(fd_single, method = "acf", detrend = "linear")
  }, error = function(e) NULL)

  if (is.null(result)) {
    return(list(score = NA, detected = NA))
  }

  score <- result$confidence
  detected <- score > detection_thresholds$acf_confidence

  return(list(score = score, detected = detected))
}

# Method 4: Seasonal strength (variance method)
# Note: variance method works poorly with linear detrending, use "none"
detect_strength_variance <- function(fd_single, period = 12) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "variance", detrend = "none")
  }, error = function(e) NA)

  if (is.na(score)) {
    return(list(score = NA, detected = NA))
  }

  detected <- score > detection_thresholds$strength_variance

  return(list(score = score, detected = detected))
}

# Method 5: Seasonal strength (spectral method)
detect_strength_spectral <- function(fd_single, period = 12) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "spectral", detrend = "linear")
  }, error = function(e) NA)

  if (is.na(score)) {
    return(list(score = NA, detected = NA))
  }

  detected <- score > detection_thresholds$strength_spectral

  return(list(score = score, detected = detected))
}

# Method 6: Automatic basis selection
detect_basis_auto <- function(fd_single) {
  result <- tryCatch({
    select.basis.auto(fd_single, criterion = "AIC", use.seasonal.hint = TRUE)
  }, error = function(e) NULL)

  if (is.null(result)) {
    return(list(score = NA, detected = NA))
  }

  # seasonal.detected is already boolean
  detected <- result$seasonal.detected[1]
  # Score: 1 if detected, 0 if not
  score <- as.numeric(detected)

  return(list(score = score, detected = detected))
}

# --- Generate data and run detection ---
cat("=== Seasonality Detection Method Comparison ===\n\n")
cat("Generating seasonal time series data...\n")
cat(sprintf("  - %d seasonal strength levels (0 to 1)\n", n_strengths))
cat(sprintf("  - %d curves per strength level\n", n_curves_per_strength))
cat(sprintf("  - %d monthly observations per curve (%d years)\n", n_months, n_years))

# Store all results
results <- data.frame(
  strength = numeric(0),
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

# Process each strength level
for (i in seq_along(seasonal_strengths)) {
  strength <- seasonal_strengths[i]
  cat(sprintf("\nProcessing strength = %.1f...\n", strength))

  for (j in 1:n_curves_per_strength) {
    if (j %% 10 == 0) cat(sprintf("  Curve %d/%d\n", j, n_curves_per_strength))

    # Generate curve
    x <- generate_seasonal_curve(t, strength, noise_sd)
    fd <- fdata(matrix(x, nrow = 1), argvals = t, rangeval = c(0, 1))

    # Apply all detection methods
    aic_result <- detect_aic_comparison(fd)
    fft_result <- detect_fft(fd)
    acf_result <- detect_acf(fd)
    var_result <- detect_strength_variance(fd)
    spec_result <- detect_strength_spectral(fd)
    auto_result <- detect_basis_auto(fd)

    # Store results
    results <- rbind(results, data.frame(
      strength = strength,
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

# --- Calculate detection rates ---
cat("\n=== Detection Rate Analysis ===\n")

# Define ground truth: strength >= 0.2 is "truly seasonal"
# (This is a design choice - adjust as needed)
truth_threshold <- 0.2
results$ground_truth <- results$strength >= truth_threshold

# Calculate detection rates by strength level
detection_rates <- aggregate(
  cbind(aic_detected, fft_detected, acf_detected,
        var_detected, spec_detected, auto_detected) ~ strength,
  data = results,
  FUN = function(x) mean(x, na.rm = TRUE)
)

names(detection_rates) <- c("strength", "AIC", "FFT", "ACF",
                            "Var_Strength", "Spec_Strength", "Basis_Auto")

cat("\nDetection Rates by Seasonal Strength:\n")
cat("=" , rep("=", 85), "\n", sep = "")
cat(sprintf("%-10s %12s %12s %12s %12s %12s %12s\n",
            "Strength", "AIC", "FFT", "ACF", "Var_Str", "Spec_Str", "Auto"))
cat("=" , rep("=", 85), "\n", sep = "")

for (i in 1:nrow(detection_rates)) {
  cat(sprintf("%-10.1f %11.0f%% %11.0f%% %11.0f%% %11.0f%% %11.0f%% %11.0f%%\n",
              detection_rates$strength[i],
              detection_rates$AIC[i] * 100,
              detection_rates$FFT[i] * 100,
              detection_rates$ACF[i] * 100,
              detection_rates$Var_Strength[i] * 100,
              detection_rates$Spec_Strength[i] * 100,
              detection_rates$Basis_Auto[i] * 100))
}
cat("=" , rep("=", 85), "\n", sep = "")

# --- Calculate classification metrics ---
cat("\n=== Classification Performance (Ground Truth: strength >= 0.2) ===\n\n")

calculate_metrics <- function(detected, ground_truth) {
  # Remove NAs
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

  fpr <- ifelse(tn + fp > 0, fp / (tn + fp), NA)  # False Positive Rate

  return(data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    FPR = fpr,
    F1 = f1
  ))
}

metrics <- rbind(
  cbind(Method = "AIC Comparison", calculate_metrics(results$aic_detected, results$ground_truth)),
  cbind(Method = "FFT Confidence", calculate_metrics(results$fft_detected, results$ground_truth)),
  cbind(Method = "ACF Confidence", calculate_metrics(results$acf_detected, results$ground_truth)),
  cbind(Method = "Variance Strength", calculate_metrics(results$var_detected, results$ground_truth)),
  cbind(Method = "Spectral Strength", calculate_metrics(results$spec_detected, results$ground_truth)),
  cbind(Method = "Basis Auto", calculate_metrics(results$auto_detected, results$ground_truth))
)

cat(sprintf("%-20s %10s %10s %10s %10s %10s %10s\n",
            "Method", "Accuracy", "Precision", "Recall", "FPR", "Specif.", "F1"))
cat("-" , rep("-", 80), "\n", sep = "")
for (i in 1:nrow(metrics)) {
  cat(sprintf("%-20s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
              metrics$Method[i],
              metrics$Accuracy[i] * 100,
              metrics$Precision[i] * 100,
              metrics$Recall[i] * 100,
              metrics$FPR[i] * 100,
              metrics$Specificity[i] * 100,
              metrics$F1[i] * 100))
}
cat("-" , rep("-", 80), "\n", sep = "")

# --- False Positive Rate at Zero Seasonal Strength ---
cat("\n=== False Positive Rates (at seasonal strength = 0) ===\n")
zero_strength <- results %>% filter(strength == 0)
fpr_at_zero <- data.frame(
  Method = c("AIC", "FFT", "ACF", "Var Strength", "Spec Strength", "Basis Auto"),
  FPR = c(
    mean(zero_strength$aic_detected, na.rm = TRUE),
    mean(zero_strength$fft_detected, na.rm = TRUE),
    mean(zero_strength$acf_detected, na.rm = TRUE),
    mean(zero_strength$var_detected, na.rm = TRUE),
    mean(zero_strength$spec_detected, na.rm = TRUE),
    mean(zero_strength$auto_detected, na.rm = TRUE)
  )
)
cat("\nFalse positive rate when there is NO seasonality:\n")
for (i in 1:nrow(fpr_at_zero)) {
  cat(sprintf("  %-15s: %5.1f%%\n", fpr_at_zero$Method[i], fpr_at_zero$FPR[i] * 100))
}

# --- Find optimal thresholds using ROC-like analysis ---
cat("\n=== Optimal Threshold Analysis ===\n")

find_optimal_threshold <- function(scores, ground_truth, n_thresholds = 100) {
  valid <- !is.na(scores) & !is.na(ground_truth)
  scores <- scores[valid]
  ground_truth <- ground_truth[valid]

  if (length(unique(scores)) < 2) return(list(threshold = NA, f1 = NA))

  thresholds <- seq(min(scores), max(scores), length.out = n_thresholds)
  best_f1 <- 0
  best_threshold <- NA

  for (thresh in thresholds) {
    detected <- scores > thresh
    tp <- sum(detected & ground_truth)
    fp <- sum(detected & !ground_truth)
    fn <- sum(!detected & ground_truth)

    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

    if (f1 > best_f1) {
      best_f1 <- f1
      best_threshold <- thresh
    }
  }

  return(list(threshold = best_threshold, f1 = best_f1))
}

cat("\nOptimal thresholds (maximizing F1 score):\n")
opt_aic <- find_optimal_threshold(results$aic_score, results$ground_truth)
opt_fft <- find_optimal_threshold(results$fft_score, results$ground_truth)
opt_acf <- find_optimal_threshold(results$acf_score, results$ground_truth)
opt_var <- find_optimal_threshold(results$var_score, results$ground_truth)
opt_spec <- find_optimal_threshold(results$spec_score, results$ground_truth)

cat(sprintf("  AIC Comparison:    threshold = %7.2f, F1 = %.1f%%\n", opt_aic$threshold, opt_aic$f1 * 100))
cat(sprintf("  FFT Confidence:    threshold = %7.2f, F1 = %.1f%%\n", opt_fft$threshold, opt_fft$f1 * 100))
cat(sprintf("  ACF Confidence:    threshold = %7.2f, F1 = %.1f%%\n", opt_acf$threshold, opt_acf$f1 * 100))
cat(sprintf("  Variance Strength: threshold = %7.2f, F1 = %.1f%%\n", opt_var$threshold, opt_var$f1 * 100))
cat(sprintf("  Spectral Strength: threshold = %7.2f, F1 = %.1f%%\n", opt_spec$threshold, opt_spec$f1 * 100))

# --- Visualization with ggplot2 ---
cat("\n=== Generating Plots ===\n")

# Define color palette for methods
method_colors <- c(
  "AIC" = "#2166AC",
  "FFT" = "#B2182B",
  "ACF" = "#1B7837",
  "Var Strength" = "#E66101",
  "Spec Strength" = "#762A83",
  "Basis Auto" = "#8C510A"
)

# Reshape detection rates for ggplot
detection_rates_long <- detection_rates %>%
  pivot_longer(cols = -strength, names_to = "Method", values_to = "Rate") %>%
  mutate(Method = factor(Method, levels = c("AIC", "FFT", "ACF", "Var_Strength",
                                             "Spec_Strength", "Basis_Auto"))) %>%
  mutate(Method = recode(Method,
                         "Var_Strength" = "Var Strength",
                         "Spec_Strength" = "Spec Strength",
                         "Basis_Auto" = "Basis Auto"))

# Plot 1: Detection rates by seasonal strength
p1 <- ggplot(detection_rates_long, aes(x = strength, y = Rate * 100,
                                        color = Method, shape = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_vline(xintercept = truth_threshold, linetype = "dashed", color = "gray50") +
  annotate("text", x = truth_threshold + 0.05, y = 50,
           label = "Ground truth\nthreshold", hjust = 0, size = 3) +
  scale_color_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Detection Rates by Method",
       x = "True Seasonal Strength",
       y = "Detection Rate (%)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 2: F1 score comparison
metrics_plot <- metrics %>%
  mutate(Method = recode(Method,
                         "AIC Comparison" = "AIC",
                         "FFT Confidence" = "FFT",
                         "ACF Confidence" = "ACF",
                         "Variance Strength" = "Var Strength",
                         "Spectral Strength" = "Spec Strength",
                         "Basis Auto" = "Basis Auto")) %>%
  mutate(Method = factor(Method, levels = c("AIC", "FFT", "ACF",
                                             "Var Strength", "Spec Strength", "Basis Auto")))

p2 <- ggplot(metrics_plot, aes(x = Method, y = F1 * 100, fill = Method)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", F1 * 100)), vjust = -0.5, size = 3) +
  scale_fill_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 105)) +
  labs(title = "F1 Score by Method",
       x = "Method",
       y = "F1 Score (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 3: AIC score distribution
results$ground_truth_label <- ifelse(results$ground_truth, "Seasonal", "Non-seasonal")

p3 <- ggplot(results, aes(x = ground_truth_label, y = aic_score, fill = ground_truth_label)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_fill_manual(values = c("Non-seasonal" = "#F4A582", "Seasonal" = "#92C5DE")) +
  labs(title = "AIC Score Distribution",
       x = "",
       y = "AIC Difference (P-spline - Fourier)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 4: Spectral strength distribution
p4 <- ggplot(results, aes(x = ground_truth_label, y = spec_score, fill = ground_truth_label)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = detection_thresholds$strength_spectral,
             linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("Non-seasonal" = "#F4A582", "Seasonal" = "#92C5DE")) +
  labs(title = "Spectral Strength Distribution",
       x = "",
       y = "Spectral Strength") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# --- Precision-Recall Curve ---
# Function to compute precision-recall curve for a method
compute_pr_curve <- function(scores, ground_truth, method_name, n_points = 200) {
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

# Compute PR curves for all methods
pr_aic <- compute_pr_curve(results$aic_score, results$ground_truth, "AIC")
pr_fft <- compute_pr_curve(results$fft_score, results$ground_truth, "FFT")
pr_acf <- compute_pr_curve(results$acf_score, results$ground_truth, "ACF")
pr_var <- compute_pr_curve(results$var_score, results$ground_truth, "Var Strength")
pr_spec <- compute_pr_curve(results$spec_score, results$ground_truth, "Spec Strength")

pr_all <- rbind(pr_aic, pr_fft, pr_acf, pr_var, pr_spec)
pr_all$Method <- factor(pr_all$Method,
                        levels = c("AIC", "FFT", "ACF", "Var Strength", "Spec Strength"))

# Calculate AUC-PR for each method
compute_auc_pr <- function(pr_df) {
  pr_df <- pr_df[order(pr_df$recall), ]
  pr_df <- pr_df[!duplicated(pr_df$recall), ]

  if (nrow(pr_df) < 2) return(NA)

  # Trapezoidal integration
  auc <- sum(diff(pr_df$recall) * (head(pr_df$precision, -1) + tail(pr_df$precision, -1)) / 2)
  return(auc)
}

auc_values <- pr_all %>%
  group_by(Method) %>%
  summarise(AUC = compute_auc_pr(data.frame(recall = recall, precision = precision)),
            .groups = "drop")

# Add operating points (default thresholds)
operating_points <- metrics_plot %>%
  filter(Method %in% c("AIC", "FFT", "ACF", "Var Strength", "Spec Strength")) %>%
  select(Method, Precision, Recall) %>%
  filter(!is.na(Precision) & !is.na(Recall))

# Plot 5: Precision-Recall curves
p5 <- ggplot(pr_all, aes(x = recall, y = precision, color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(data = operating_points,
             aes(x = Recall, y = Precision, color = Method),
             size = 4, shape = 18) +
  geom_text(data = auc_values,
            aes(x = 0.2, y = seq(0.4, 0.1, length.out = nrow(auc_values)),
                label = sprintf("%s: %.3f", Method, AUC),
                color = Method),
            hjust = 0, size = 3, show.legend = FALSE) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Precision-Recall Curves",
       subtitle = "Diamonds indicate default threshold operating points",
       x = "Recall (Sensitivity)",
       y = "Precision") +
  annotate("text", x = 0.2, y = 0.5, label = "AUC-PR:", fontface = "bold", hjust = 0, size = 3) +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 9))

# Plot 6: Precision vs Recall scatter with method labels
p6 <- ggplot(metrics_plot %>% filter(!is.na(Precision) & !is.na(Recall)),
             aes(x = Recall, y = Precision, color = Method)) +
  geom_point(size = 5) +
  geom_text(aes(label = Method), vjust = -1, size = 3) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(0, 1.05), ylim = c(0.7, 1.05)) +
  labs(title = "Precision vs Recall (Default Thresholds)",
       x = "Recall",
       y = "Precision") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Save main comparison plot
ggsave("seasonality_detection_comparison.pdf",
       gridExtra::grid.arrange(p1, p2, p5, p6, ncol = 2),
       width = 14, height = 10)

# --- Detailed plots ---

# Score distributions by strength for each method
results_long_scores <- results %>%
  select(strength, aic_score, fft_score, acf_score, var_score, spec_score) %>%
  pivot_longer(cols = -strength, names_to = "Method", values_to = "Score") %>%
  mutate(Method = recode(Method,
                         "aic_score" = "AIC",
                         "fft_score" = "FFT",
                         "acf_score" = "ACF",
                         "var_score" = "Var Strength",
                         "spec_score" = "Spec Strength"))

# Threshold lines for each method
threshold_lines <- data.frame(
  Method = c("AIC", "FFT", "ACF", "Var Strength", "Spec Strength"),
  threshold = c(0, detection_thresholds$fft_confidence,
                detection_thresholds$acf_confidence,
                detection_thresholds$strength_variance,
                detection_thresholds$strength_spectral)
)

p_scores <- ggplot(results_long_scores, aes(x = factor(strength), y = Score)) +
  geom_boxplot(aes(fill = factor(strength)), alpha = 0.7, show.legend = FALSE) +
  geom_hline(data = threshold_lines, aes(yintercept = threshold),
             linetype = "dashed", color = "red") +
  facet_wrap(~ Method, scales = "free_y", ncol = 3) +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "Score Distributions by Seasonal Strength",
       x = "Seasonal Strength",
       y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation heatmap
cor_matrix <- cor(results[, c("aic_score", "fft_score", "acf_score",
                               "var_score", "spec_score")],
                  use = "pairwise.complete.obs")
rownames(cor_matrix) <- c("AIC", "FFT", "ACF", "Var", "Spec")
colnames(cor_matrix) <- c("AIC", "FFT", "ACF", "Var", "Spec")

cor_long <- as.data.frame(as.table(cor_matrix))
names(cor_long) <- c("Method1", "Method2", "Correlation")

p_cor <- ggplot(cor_long, aes(x = Method1, y = Method2, fill = Correlation)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 4) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Score Correlations Between Methods",
       x = "", y = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save detailed plots
ggsave("seasonality_detection_details.pdf",
       gridExtra::grid.arrange(p_scores, p_cor, ncol = 1, heights = c(2, 1)),
       width = 14, height = 12)

cat("Plots saved to:\n")
cat("  - seasonality_detection_comparison.pdf\n")
cat("  - seasonality_detection_details.pdf\n")

# --- Save results ---
saveRDS(results, "seasonality_detection_results.rds")
saveRDS(metrics, "seasonality_detection_metrics.rds")
cat("\nResults saved to:\n")
cat("  - seasonality_detection_results.rds\n")
cat("  - seasonality_detection_metrics.rds\n")

# --- Summary ---
cat("\n=== Summary ===\n")
best_method <- metrics$Method[which.max(metrics$F1)]
best_f1 <- max(metrics$F1, na.rm = TRUE)
cat(sprintf("Best performing method: %s (F1 = %.1f%%)\n", best_method, best_f1 * 100))

cat("\nKey observations:\n")
cat("  - AIC comparison (Fourier vs P-spline) provides a novel detection approach\n")
cat("  - Different methods have different sensitivity/specificity trade-offs\n")
cat("  - Optimal thresholds can improve performance over defaults\n")

cat("\nAnalysis complete!\n")
