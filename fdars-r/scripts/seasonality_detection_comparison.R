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

set.seed(42)

# --- Configuration ---
n_strengths <- 11           # Number of seasonal strength levels (0 to 1)
n_curves_per_strength <- 50 # Number of curves per strength level
n_years <- 5                # Number of years of monthly data
n_months <- n_years * 12    # Total number of monthly observations
noise_sd <- 0.3             # Standard deviation of noise

# Detection threshold for binary classification
# Methods return continuous scores; we threshold for detection rate
detection_thresholds <- list(
  aic_comparison = 0,           # Fourier AIC < P-spline AIC
  fft_confidence = 2.0,         # Power ratio threshold
  acf_confidence = 0.3,         # ACF peak threshold
  strength_variance = 0.3,      # Variance ratio threshold
  strength_spectral = 0.3,      # Spectral power threshold
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
detect_strength_variance <- function(fd_single, period = 12) {
  score <- tryCatch({
    seasonal_strength(fd_single, period = period, method = "variance", detrend = "linear")
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

  return(data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
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

cat(sprintf("%-20s %10s %10s %10s %10s %10s\n",
            "Method", "Accuracy", "Precision", "Recall", "Specific.", "F1"))
cat("-" , rep("-", 70), "\n", sep = "")
for (i in 1:nrow(metrics)) {
  cat(sprintf("%-20s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
              metrics$Method[i],
              metrics$Accuracy[i] * 100,
              metrics$Precision[i] * 100,
              metrics$Recall[i] * 100,
              metrics$Specificity[i] * 100,
              metrics$F1[i] * 100))
}
cat("-" , rep("-", 70), "\n", sep = "")

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

# --- Visualization ---
cat("\n=== Generating Plots ===\n")

pdf("seasonality_detection_comparison.pdf", width = 14, height = 10)

# Plot 1: Detection rates by seasonal strength
par(mfrow = c(2, 2))

# 1a: Detection rate curves
plot(detection_rates$strength, detection_rates$AIC * 100,
     type = "b", col = "blue", pch = 19, lwd = 2,
     xlab = "True Seasonal Strength", ylab = "Detection Rate (%)",
     main = "Detection Rates by Method",
     ylim = c(0, 100))
lines(detection_rates$strength, detection_rates$FFT * 100,
      type = "b", col = "red", pch = 17, lwd = 2)
lines(detection_rates$strength, detection_rates$ACF * 100,
      type = "b", col = "green", pch = 15, lwd = 2)
lines(detection_rates$strength, detection_rates$Var_Strength * 100,
      type = "b", col = "orange", pch = 18, lwd = 2)
lines(detection_rates$strength, detection_rates$Spec_Strength * 100,
      type = "b", col = "purple", pch = 8, lwd = 2)
lines(detection_rates$strength, detection_rates$Basis_Auto * 100,
      type = "b", col = "brown", pch = 4, lwd = 2)
abline(v = truth_threshold, lty = 2, col = "gray50")
text(truth_threshold, 50, "Ground truth\nthreshold", pos = 4, cex = 0.7)
legend("bottomright",
       legend = c("AIC", "FFT", "ACF", "Var Strength", "Spec Strength", "Basis Auto"),
       col = c("blue", "red", "green", "orange", "purple", "brown"),
       pch = c(19, 17, 15, 18, 8, 4), lwd = 2, cex = 0.7)

# 1b: F1 score comparison
barplot(metrics$F1 * 100,
        names.arg = c("AIC", "FFT", "ACF", "Var", "Spec", "Auto"),
        col = c("blue", "red", "green", "orange", "purple", "brown"),
        main = "F1 Score by Method",
        ylab = "F1 Score (%)",
        ylim = c(0, 100))

# 1c: Score distributions for seasonal vs non-seasonal
boxplot(aic_score ~ ground_truth, data = results,
        names = c("Non-seasonal", "Seasonal"),
        main = "AIC Score Distribution",
        ylab = "AIC Difference (P-spline - Fourier)",
        col = c("lightcoral", "lightgreen"))
abline(h = 0, lty = 2)

# 1d: Variance strength score distribution
boxplot(var_score ~ ground_truth, data = results,
        names = c("Non-seasonal", "Seasonal"),
        main = "Variance Strength Distribution",
        ylab = "Seasonal Strength",
        col = c("lightcoral", "lightgreen"))
abline(h = detection_thresholds$strength_variance, lty = 2, col = "red")

dev.off()

# Additional detailed plot
pdf("seasonality_detection_details.pdf", width = 14, height = 10)

par(mfrow = c(2, 3))

# Score distributions for all methods
boxplot(aic_score ~ strength, data = results,
        main = "AIC Score by Strength",
        xlab = "Seasonal Strength", ylab = "Score",
        col = "steelblue", las = 2)
abline(h = 0, lty = 2, col = "red")

boxplot(fft_score ~ strength, data = results,
        main = "FFT Confidence by Strength",
        xlab = "Seasonal Strength", ylab = "Confidence",
        col = "coral", las = 2)
abline(h = detection_thresholds$fft_confidence, lty = 2, col = "red")

boxplot(acf_score ~ strength, data = results,
        main = "ACF Confidence by Strength",
        xlab = "Seasonal Strength", ylab = "Confidence",
        col = "lightgreen", las = 2)
abline(h = detection_thresholds$acf_confidence, lty = 2, col = "red")

boxplot(var_score ~ strength, data = results,
        main = "Variance Strength by Strength",
        xlab = "Seasonal Strength", ylab = "Strength",
        col = "orange", las = 2)
abline(h = detection_thresholds$strength_variance, lty = 2, col = "red")

boxplot(spec_score ~ strength, data = results,
        main = "Spectral Strength by Strength",
        xlab = "Seasonal Strength", ylab = "Strength",
        col = "plum", las = 2)
abline(h = detection_thresholds$strength_spectral, lty = 2, col = "red")

# Correlation between scores
cor_matrix <- cor(results[, c("aic_score", "fft_score", "acf_score",
                               "var_score", "spec_score")],
                  use = "pairwise.complete.obs")
image(1:5, 1:5, cor_matrix,
      col = colorRampPalette(c("blue", "white", "red"))(100),
      axes = FALSE, main = "Score Correlations",
      xlab = "", ylab = "")
axis(1, at = 1:5, labels = c("AIC", "FFT", "ACF", "Var", "Spec"), las = 2)
axis(2, at = 1:5, labels = c("AIC", "FFT", "ACF", "Var", "Spec"), las = 2)
for (i in 1:5) {
  for (j in 1:5) {
    text(i, j, sprintf("%.2f", cor_matrix[i, j]), cex = 0.8)
  }
}

dev.off()

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
