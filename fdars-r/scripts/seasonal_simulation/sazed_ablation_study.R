#!/usr/bin/env Rscript
# SAZED Ablation Study
#
# This script analyzes the contribution of each component in the SAZED ensemble
# by simulating leave-one-out and minimal ensemble configurations.
#
# SAZED Components (5):
# 1. Spectral - FFT periodogram peak
# 2. ACF Peak - First significant ACF peak
# 3. ACF Average - Weighted mean of ACF peaks
# 4. Zero-crossing - ACF zero-crossing intervals
# 5. Spectral Diff - FFT on first-differenced signal

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(42)

cat("=== SAZED Ablation Study ===\n\n")

# --- Load existing results (or regenerate) ---
results_file <- "seasonality_detection_results.rds"

if (!file.exists(results_file)) {
  stop(paste0(
    "Results file '", results_file, "' not found.\n",
    "Please run 'seasonality_detection_comparison.R' first to generate the results."
  ))
}

results_base <- readRDS(results_file)
cat(sprintf("Loaded %d observations from '%s'\n", nrow(results_base), results_file))

# Ensure ground truth exists
if (!"ground_truth" %in% names(results_base)) {
  results_base$ground_truth <- results_base$strength >= 0.2
}

# --- Configuration ---
n_strengths <- 11
n_curves_per_strength <- 50
n_months <- 60
noise_sd <- 0.3

# Default SAZED threshold (minimum agreeing components for detection)
sazed_threshold <- 2

# Component names
component_names <- c("spectral", "acf_peak", "acf_average", "zero_crossing", "spectral_diff")
component_labels <- c("Spectral", "ACF Peak", "ACF Average", "Zero-crossing", "Spectral Diff")

# --- Helper functions ---

#' Generate seasonal curve (same as in main comparison)
generate_seasonal_curve <- function(t, strength, noise_sd = 0.3) {
  trend <- 0.5 * sin(2 * pi * t) * 0.3
  n_cycles <- length(t) / 12
  seasonal <- strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + strength * 0.3 * cos(4 * pi * n_cycles * t)
  noise <- rnorm(length(t), sd = noise_sd)
  return(trend + seasonal + noise)
}

#' Simulate SAZED voting with subset of components
#'
#' @param component_periods Named vector of period estimates from each component
#' @param include_components Character vector of component names to include
#' @param tolerance Tolerance for period agreement (fraction of period)
#' @return List with consensus period, agreeing count, and detection result
simulate_sazed_voting <- function(component_periods, include_components, tolerance = 0.1) {
  # Filter to included components
  periods <- component_periods[include_components]
  periods <- periods[!is.na(periods) & periods > 0]

  if (length(periods) == 0) {
    return(list(period = NA, agreeing = 0, detected = FALSE))
  }

  if (length(periods) == 1) {
    return(list(period = periods[1], agreeing = 1, detected = FALSE))
  }

  # Find consensus: periods within tolerance of each other
  n <- length(periods)
  best_count <- 0
  best_period <- NA

  for (i in 1:n) {
    ref_period <- periods[i]
    # Count how many periods are within tolerance
    agreeing <- sum(abs(periods - ref_period) <= tolerance * ref_period)
    if (agreeing > best_count) {
      best_count <- agreeing
      best_period <- mean(periods[abs(periods - ref_period) <= tolerance * ref_period])
    }
  }

  # Adjust threshold for smaller ensembles
  # For full SAZED (5): threshold = 3 (majority)
  # For 4 components: threshold = 3
  # For 3 components: threshold = 2
  # For 2 components: threshold = 2
  n_components <- length(include_components)
  adjusted_threshold <- if (n_components >= 5) 3
                        else if (n_components >= 3) 2
                        else 2

  detected <- best_count >= adjusted_threshold

  list(period = best_period, agreeing = best_count, detected = detected)
}

#' Calculate classification metrics
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

  data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    FPR = fpr
  )
}

# --- Generate data and collect SAZED component outputs ---
cat("Generating data and running SAZED analysis...\n")

seasonal_strengths <- seq(0, 1, length.out = n_strengths)
t <- seq(0, 1, length.out = n_months)

# Store component-level results
component_results <- data.frame(
  strength = numeric(0),
  curve_id = integer(0),
  spectral = numeric(0),
  acf_peak = numeric(0),
  acf_average = numeric(0),
  zero_crossing = numeric(0),
  spectral_diff = numeric(0),
  full_detected = logical(0),
  ground_truth = logical(0)
)

for (i in seq_along(seasonal_strengths)) {
  strength <- seasonal_strengths[i]
  if (i %% 3 == 0) cat(sprintf("Processing strength = %.1f...\n", strength))

  for (j in 1:n_curves_per_strength) {
    # Generate curve
    x <- generate_seasonal_curve(t, strength, noise_sd)
    fd <- fdata(matrix(x, nrow = 1), argvals = t, rangeval = c(0, 1))

    # Run SAZED to get component estimates
    sazed_result <- tryCatch({
      sazed(fd)
    }, error = function(e) NULL)

    if (!is.null(sazed_result)) {
      component_results <- rbind(component_results, data.frame(
        strength = strength,
        curve_id = j,
        spectral = sazed_result$components$spectral,
        acf_peak = sazed_result$components$acf_peak,
        acf_average = sazed_result$components$acf_average,
        zero_crossing = sazed_result$components$zero_crossing,
        spectral_diff = sazed_result$components$spectral_diff,
        full_detected = sazed_result$agreeing_components >= 3,
        ground_truth = strength >= 0.2
      ))
    }
  }
}

cat(sprintf("Collected %d observations with SAZED component data\n", nrow(component_results)))

# --- Define ablation configurations ---
ablation_configs <- list(
  # Full ensemble
  "Full (5)" = component_names,

  # Leave-one-out (5 configs)
  "-Spectral" = setdiff(component_names, "spectral"),
  "-ACF_Peak" = setdiff(component_names, "acf_peak"),
  "-ACF_Average" = setdiff(component_names, "acf_average"),
  "-Zero_cross" = setdiff(component_names, "zero_crossing"),
  "-Spec_Diff" = setdiff(component_names, "spectral_diff"),

  # Minimal ensembles (combinations of 2-3 components)
  "FFT-only (2)" = c("spectral", "spectral_diff"),
  "ACF-only (3)" = c("acf_peak", "acf_average", "zero_crossing"),
  "Spectral+ACF_Peak (2)" = c("spectral", "acf_peak"),
  "Best 3" = c("spectral", "acf_peak", "spectral_diff")  # Hypothesis: these may be sufficient
)

# --- Run ablation analysis ---
cat("\n=== Running Ablation Analysis ===\n\n")

ablation_results <- data.frame(
  Config = character(0),
  N_Components = integer(0),
  Accuracy = numeric(0),
  Precision = numeric(0),
  Recall = numeric(0),
  F1 = numeric(0),
  FPR = numeric(0),
  stringsAsFactors = FALSE
)

for (config_name in names(ablation_configs)) {
  include_components <- ablation_configs[[config_name]]

  # Simulate voting for each observation
  detected <- sapply(1:nrow(component_results), function(i) {
    periods <- c(
      spectral = component_results$spectral[i],
      acf_peak = component_results$acf_peak[i],
      acf_average = component_results$acf_average[i],
      zero_crossing = component_results$zero_crossing[i],
      spectral_diff = component_results$spectral_diff[i]
    )
    result <- simulate_sazed_voting(periods, include_components)
    result$detected
  })

  # Calculate metrics
  metrics <- calculate_metrics(detected, component_results$ground_truth)

  ablation_results <- rbind(ablation_results, data.frame(
    Config = config_name,
    N_Components = length(include_components),
    Accuracy = metrics$Accuracy,
    Precision = metrics$Precision,
    Recall = metrics$Recall,
    F1 = metrics$F1,
    FPR = metrics$FPR,
    stringsAsFactors = FALSE
  ))
}

# Sort by F1 score
ablation_results <- ablation_results %>% arrange(desc(F1))

# --- Print results ---
cat("Ablation Study Results:\n")
cat(sprintf("%-20s %5s %8s %8s %8s %8s %8s\n",
            "Config", "N", "Acc", "Prec", "Recall", "F1", "FPR"))
cat(paste(rep("-", 75), collapse = ""), "\n")

for (i in 1:nrow(ablation_results)) {
  r <- ablation_results[i, ]
  cat(sprintf("%-20s %5d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n",
              r$Config, r$N_Components,
              r$Accuracy * 100, r$Precision * 100, r$Recall * 100,
              r$F1 * 100, r$FPR * 100))
}

# --- Component contribution analysis ---
cat("\n=== Component Contribution Analysis ===\n\n")

# Calculate F1 drop when each component is removed
full_f1 <- ablation_results$F1[ablation_results$Config == "Full (5)"]

contribution <- data.frame(
  Component = c("Spectral", "ACF Peak", "ACF Average", "Zero-crossing", "Spectral Diff"),
  Config_Without = c("-Spectral", "-ACF_Peak", "-ACF_Average", "-Zero_cross", "-Spec_Diff")
) %>%
  left_join(ablation_results %>% select(Config, F1), by = c("Config_Without" = "Config")) %>%
  mutate(
    F1_Drop = (full_f1 - F1) * 100,
    Contribution_Pct = F1_Drop / sum(F1_Drop) * 100
  ) %>%
  arrange(desc(F1_Drop))

cat("Impact of removing each component (from Full ensemble):\n\n")
cat(sprintf("%-15s %10s %12s %15s\n", "Component", "F1 Without", "F1 Drop", "Contribution"))
cat(paste(rep("-", 55), collapse = ""), "\n")
for (i in 1:nrow(contribution)) {
  r <- contribution[i, ]
  cat(sprintf("%-15s %9.1f%% %11.2f%% %14.1f%%\n",
              r$Component, r$F1 * 100, r$F1_Drop, r$Contribution_Pct))
}

# --- Statistical comparison using McNemar's test ---
cat("\n=== Statistical Significance vs Full Ensemble ===\n\n")

# Get full ensemble detections
full_detected <- sapply(1:nrow(component_results), function(i) {
  periods <- c(
    spectral = component_results$spectral[i],
    acf_peak = component_results$acf_peak[i],
    acf_average = component_results$acf_average[i],
    zero_crossing = component_results$zero_crossing[i],
    spectral_diff = component_results$spectral_diff[i]
  )
  result <- simulate_sazed_voting(periods, component_names)
  result$detected
})

# Compare each config to full
full_correct <- (full_detected == component_results$ground_truth)

for (config_name in names(ablation_configs)) {
  if (config_name == "Full (5)") next

  include_components <- ablation_configs[[config_name]]
  config_detected <- sapply(1:nrow(component_results), function(i) {
    periods <- c(
      spectral = component_results$spectral[i],
      acf_peak = component_results$acf_peak[i],
      acf_average = component_results$acf_average[i],
      zero_crossing = component_results$zero_crossing[i],
      spectral_diff = component_results$spectral_diff[i]
    )
    result <- simulate_sazed_voting(periods, include_components)
    result$detected
  })

  config_correct <- (config_detected == component_results$ground_truth)

  # Build contingency table
  a <- sum(full_correct & config_correct)      # Both correct
  b <- sum(full_correct & !config_correct)     # Full correct, config wrong
  c <- sum(!full_correct & config_correct)     # Full wrong, config correct
  d <- sum(!full_correct & !config_correct)    # Both wrong

  # McNemar's test
  if (b + c > 0) {
    p_value <- binom.test(b, b + c, p = 0.5)$p.value
    sig <- if (p_value < 0.001) "***" else if (p_value < 0.01) "**" else if (p_value < 0.05) "*" else ""

    cat(sprintf("%-20s: Full better in %d, Config better in %d, p=%.4f%s\n",
                config_name, b, c, p_value, sig))
  }
}
cat("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001\n")

# --- Create plots ---
cat("\n=== Generating Plots ===\n")

# Plot 1: F1 scores by configuration
ablation_results$Config <- factor(ablation_results$Config,
                                   levels = ablation_results$Config[order(ablation_results$F1, decreasing = TRUE)])

p_f1 <- ggplot(ablation_results, aes(x = Config, y = F1 * 100, fill = factor(N_Components))) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = full_f1 * 100, linetype = "dashed", color = "red") +
  annotate("text", x = nrow(ablation_results) - 0.5, y = full_f1 * 100 + 1,
           label = "Full ensemble", color = "red", hjust = 1, size = 3) +
  scale_fill_brewer(palette = "Blues", name = "Components") +
  coord_flip() +
  labs(
    title = "SAZED Ablation Study: F1 Scores by Configuration",
    subtitle = "Dashed line = full 5-component ensemble",
    x = "",
    y = "F1 Score (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9),
    legend.position = "bottom"
  )

ggsave("plots/sazed_ablation_f1.pdf", p_f1, width = 10, height = 6)
cat("Saved: plots/sazed_ablation_f1.pdf\n")

# Plot 2: Component contribution bar chart
p_contribution <- ggplot(contribution, aes(x = reorder(Component, F1_Drop), y = F1_Drop)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_text(aes(label = sprintf("%.2f%%", F1_Drop)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  labs(
    title = "Component Contribution to SAZED Performance",
    subtitle = "F1 score drop when component is removed",
    x = "",
    y = "F1 Drop (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

ggsave("plots/sazed_component_contribution.pdf", p_contribution, width = 8, height = 5)
cat("Saved: plots/sazed_component_contribution.pdf\n")

# Plot 3: Precision-Recall trade-off
p_pr <- ggplot(ablation_results, aes(x = Recall * 100, y = Precision * 100,
                                      color = factor(N_Components), label = Config)) +
  geom_point(size = 4) +
  geom_text(vjust = -1, hjust = 0.5, size = 2.5, show.legend = FALSE) +
  scale_color_brewer(palette = "Set1", name = "Components") +
  coord_cartesian(xlim = c(80, 100), ylim = c(80, 100)) +
  labs(
    title = "Precision-Recall Trade-off by Configuration",
    x = "Recall (%)",
    y = "Precision (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

ggsave("plots/sazed_ablation_pr.pdf", p_pr, width = 8, height = 6)
cat("Saved: plots/sazed_ablation_pr.pdf\n")

# --- Save results ---
saveRDS(ablation_results, "sazed_ablation_results.rds")
cat("Saved: sazed_ablation_results.rds\n")

write.csv(ablation_results, "sazed_ablation_results.csv", row.names = FALSE)
cat("Saved: sazed_ablation_results.csv\n")

write.csv(contribution, "sazed_component_contribution.csv", row.names = FALSE)
cat("Saved: sazed_component_contribution.csv\n")

# --- Key Findings ---
cat("\n=== Key Findings ===\n\n")

# Best minimal ensemble
minimal_best <- ablation_results %>%
  filter(N_Components < 5) %>%
  slice(1)

cat(sprintf("1. Full ensemble F1: %.1f%%\n", full_f1 * 100))
cat(sprintf("2. Best minimal ensemble: %s (F1 = %.1f%%, %d components)\n",
            minimal_best$Config, minimal_best$F1 * 100, minimal_best$N_Components))

# Most important component
most_important <- contribution[1, ]
cat(sprintf("3. Most important component: %s (%.2f%% F1 drop when removed)\n",
            most_important$Component, most_important$F1_Drop))

# Least important component
least_important <- contribution[nrow(contribution), ]
cat(sprintf("4. Least important component: %s (%.2f%% F1 drop when removed)\n",
            least_important$Component, least_important$F1_Drop))

# Recommendation
cat("\n=== Recommendation ===\n\n")
if (minimal_best$F1 >= full_f1 - 0.01) {
  cat(sprintf("The '%s' configuration achieves near-equivalent performance\n", minimal_best$Config))
  cat(sprintf("to the full ensemble (F1 diff = %.2f%%) with only %d components.\n",
              (full_f1 - minimal_best$F1) * 100, minimal_best$N_Components))
  cat("This could reduce computational overhead for resource-constrained applications.\n")
} else {
  cat("The full 5-component ensemble provides the best performance.\n")
  cat("All components contribute meaningfully to the consensus.\n")
}

cat("\n=== SAZED Ablation Study Complete ===\n")
