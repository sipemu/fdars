#!/usr/bin/env Rscript
# Real-World Dataset Benchmarking
#
# This script benchmarks seasonality detection methods on real-world data
# from the M4 Competition to validate findings from the simulation study.
#
# M4 Competition: 100,000 time series across multiple frequencies
# - Hourly: 414 series (daily seasonality)
# - Daily: 4,227 series (weekly/annual seasonality)
# - Weekly: 359 series (annual seasonality)
# - Monthly: 48,000 series (annual seasonality)
# - Quarterly: 24,000 series (annual seasonality)
# - Yearly: 23,000 series (no seasonality expected)

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)

cat("=== Real-World Dataset Benchmarking ===\n\n")

# --- Check for M4 package ---
if (!requireNamespace("M4comp2018", quietly = TRUE)) {
  cat("M4comp2018 package not found. Attempting to install...\n")
  tryCatch({
    install.packages("M4comp2018", repos = "https://cloud.r-project.org")
  }, error = function(e) {
    cat("Could not install M4comp2018. Trying Mcomp (M3) instead...\n")
    if (!requireNamespace("Mcomp", quietly = TRUE)) {
      install.packages("Mcomp", repos = "https://cloud.r-project.org")
    }
  })
}

# Determine which package to use
use_m4 <- requireNamespace("M4comp2018", quietly = TRUE)
use_m3 <- requireNamespace("Mcomp", quietly = TRUE)

if (!use_m4 && !use_m3) {
  stop("Neither M4comp2018 nor Mcomp package is available. Please install one of them.")
}

# --- Configuration ---
set.seed(42)

# Detection thresholds (from simulation study)
detection_thresholds <- list(
  fft_confidence = 6.0,
  acf_confidence = 0.25,
  strength_variance = 0.2,
  strength_spectral = 0.3,
  strength_wavelet = 0.26,
  sazed_consensus = 3,
  autoperiod_conf = 0.3,
  cfd_conf = 0.25
)

# --- Helper functions ---

#' Convert ts object to fdata
ts_to_fdata <- function(ts_obj) {
  values <- as.numeric(ts_obj)
  n <- length(values)

  # Normalize time to [0, 1]
  t <- seq(0, 1, length.out = n)

  fdata(matrix(values, nrow = 1), argvals = t, rangeval = c(0, 1))
}

#' Get expected period in normalized units
#' @param frequency ts frequency (e.g., 12 for monthly, 4 for quarterly)
#' @param n Length of time series
get_expected_period <- function(frequency, n) {
  # One seasonal cycle = frequency observations
  # In normalized [0, 1], one cycle has period = frequency/n
  frequency / n
}

#' Detect seasonality using all methods
detect_all_methods <- function(fd, expected_period = NULL) {
  results <- list()

  # FFT
  results$fft <- tryCatch({
    r <- estimate.period(fd, method = "fft")
    list(score = r$confidence, detected = r$confidence > detection_thresholds$fft_confidence)
  }, error = function(e) list(score = NA, detected = NA))

  # ACF
  results$acf <- tryCatch({
    r <- estimate.period(fd, method = "acf")
    list(score = r$confidence, detected = r$confidence > detection_thresholds$acf_confidence)
  }, error = function(e) list(score = NA, detected = NA))

  # SAZED (period-free)
  results$sazed <- tryCatch({
    r <- sazed(fd)
    list(score = r$agreeing_components,
         detected = r$agreeing_components >= detection_thresholds$sazed_consensus,
         period = r$period)
  }, error = function(e) list(score = NA, detected = NA, period = NA))

  # Autoperiod
  results$autoperiod <- tryCatch({
    r <- autoperiod(fd)
    list(score = r$acf_validation,
         detected = r$acf_validation > detection_thresholds$autoperiod_conf,
         period = r$period)
  }, error = function(e) list(score = NA, detected = NA, period = NA))

  # CFDAutoperiod
  results$cfd <- tryCatch({
    r <- cfd.autoperiod(fd)
    list(score = r$acf_validation,
         detected = r$acf_validation > detection_thresholds$cfd_conf,
         period = r$period)
  }, error = function(e) list(score = NA, detected = NA, period = NA))

  # Period-aware methods (if period known)
  if (!is.null(expected_period)) {
    # Variance Strength
    results$variance <- tryCatch({
      score <- seasonal.strength(fd, period = expected_period, method = "variance")
      list(score = score, detected = score > detection_thresholds$strength_variance)
    }, error = function(e) list(score = NA, detected = NA))

    # Spectral Strength
    results$spectral <- tryCatch({
      score <- seasonal.strength(fd, period = expected_period, method = "spectral")
      list(score = score, detected = score > detection_thresholds$strength_spectral)
    }, error = function(e) list(score = NA, detected = NA))

    # Wavelet Strength
    results$wavelet <- tryCatch({
      score <- seasonal.strength(fd, period = expected_period, method = "wavelet")
      list(score = score, detected = score > detection_thresholds$strength_wavelet)
    }, error = function(e) list(score = NA, detected = NA))
  }

  results
}

#' Calculate metrics from results
calculate_metrics <- function(detected, ground_truth) {
  valid <- !is.na(detected) & !is.na(ground_truth)
  if (sum(valid) == 0) return(data.frame(Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, FPR = NA))

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

  data.frame(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1, FPR = fpr)
}

# --- Load and process data ---
cat("Loading competition data...\n")

if (use_m4) {
  library(M4comp2018)
  data(M4)

  # Extract different frequency subsets
  frequencies <- c("Monthly", "Quarterly", "Yearly")

  all_results <- list()

  for (freq in frequencies) {
    cat(sprintf("\nProcessing %s series...\n", freq))

    # Filter by frequency type
    freq_data <- M4[sapply(M4, function(x) x$type == freq)]

    # Sample subset for tractability (M4 is very large)
    n_sample <- min(200, length(freq_data))
    sample_idx <- sample(length(freq_data), n_sample)
    sampled_data <- freq_data[sample_idx]

    cat(sprintf("  Sampled %d of %d series\n", n_sample, length(freq_data)))

    # Process each series
    freq_results <- data.frame(
      id = character(0),
      frequency = character(0),
      ts_frequency = integer(0),
      length = integer(0),
      ground_truth = logical(0),
      fft_detected = logical(0),
      acf_detected = logical(0),
      sazed_detected = logical(0),
      autoperiod_detected = logical(0),
      cfd_detected = logical(0),
      variance_detected = logical(0),
      spectral_detected = logical(0),
      wavelet_detected = logical(0)
    )

    for (i in seq_along(sampled_data)) {
      if (i %% 50 == 0) cat(sprintf("  Processing %d/%d...\n", i, n_sample))

      item <- sampled_data[[i]]
      ts_obj <- item$x

      # Skip very short series
      if (length(ts_obj) < 24) next

      # Convert to fdata
      fd <- tryCatch(ts_to_fdata(ts_obj), error = function(e) NULL)
      if (is.null(fd)) next

      # Ground truth: series has seasonality if frequency > 1
      ts_freq <- frequency(ts_obj)
      ground_truth <- ts_freq > 1  # Monthly=12, Quarterly=4, Yearly=1

      # Calculate expected period
      expected_period <- get_expected_period(ts_freq, length(ts_obj))

      # Run detection methods
      detection <- detect_all_methods(fd, expected_period)

      freq_results <- rbind(freq_results, data.frame(
        id = item$st,
        frequency = freq,
        ts_frequency = ts_freq,
        length = length(ts_obj),
        ground_truth = ground_truth,
        fft_detected = detection$fft$detected,
        acf_detected = detection$acf$detected,
        sazed_detected = detection$sazed$detected,
        autoperiod_detected = detection$autoperiod$detected,
        cfd_detected = detection$cfd$detected,
        variance_detected = if (!is.null(detection$variance)) detection$variance$detected else NA,
        spectral_detected = if (!is.null(detection$spectral)) detection$spectral$detected else NA,
        wavelet_detected = if (!is.null(detection$wavelet)) detection$wavelet$detected else NA
      ))
    }

    all_results[[freq]] <- freq_results
    cat(sprintf("  Completed: %d series processed\n", nrow(freq_results)))
  }

} else {
  # Use M3 data (smaller, simpler)
  library(Mcomp)
  data(M3)

  frequencies <- c("MONTHLY", "QUARTERLY", "YEARLY")

  all_results <- list()

  for (freq in frequencies) {
    cat(sprintf("\nProcessing %s series...\n", freq))

    # Filter by period
    freq_data <- M3[sapply(M3, function(x) x$period == freq)]

    n_sample <- min(200, length(freq_data))
    sample_idx <- sample(length(freq_data), n_sample)
    sampled_data <- freq_data[sample_idx]

    cat(sprintf("  Sampled %d of %d series\n", n_sample, length(freq_data)))

    freq_results <- data.frame(
      id = character(0),
      frequency = character(0),
      ts_frequency = integer(0),
      length = integer(0),
      ground_truth = logical(0),
      fft_detected = logical(0),
      acf_detected = logical(0),
      sazed_detected = logical(0),
      autoperiod_detected = logical(0),
      cfd_detected = logical(0),
      variance_detected = logical(0),
      spectral_detected = logical(0),
      wavelet_detected = logical(0)
    )

    for (i in seq_along(sampled_data)) {
      if (i %% 50 == 0) cat(sprintf("  Processing %d/%d...\n", i, n_sample))

      item <- sampled_data[[i]]
      ts_obj <- item$x

      if (length(ts_obj) < 24) next

      fd <- tryCatch(ts_to_fdata(ts_obj), error = function(e) NULL)
      if (is.null(fd)) next

      ts_freq <- frequency(ts_obj)
      ground_truth <- ts_freq > 1

      expected_period <- get_expected_period(ts_freq, length(ts_obj))

      detection <- detect_all_methods(fd, expected_period)

      freq_results <- rbind(freq_results, data.frame(
        id = item$sn,
        frequency = freq,
        ts_frequency = ts_freq,
        length = length(ts_obj),
        ground_truth = ground_truth,
        fft_detected = detection$fft$detected,
        acf_detected = detection$acf$detected,
        sazed_detected = detection$sazed$detected,
        autoperiod_detected = detection$autoperiod$detected,
        cfd_detected = detection$cfd$detected,
        variance_detected = if (!is.null(detection$variance)) detection$variance$detected else NA,
        spectral_detected = if (!is.null(detection$spectral)) detection$spectral$detected else NA,
        wavelet_detected = if (!is.null(detection$wavelet)) detection$wavelet$detected else NA
      ))
    }

    all_results[[freq]] <- freq_results
  }
}

# --- Combine results ---
combined_results <- bind_rows(all_results)
cat(sprintf("\n=== Total: %d series processed ===\n", nrow(combined_results)))

# --- Calculate metrics by method ---
cat("\n=== Performance by Method (All Frequencies) ===\n\n")

method_cols <- c("fft_detected", "acf_detected", "sazed_detected",
                 "autoperiod_detected", "cfd_detected",
                 "variance_detected", "spectral_detected", "wavelet_detected")
method_names <- c("FFT", "ACF", "SAZED", "Autoperiod", "CFD",
                  "Variance", "Spectral", "Wavelet")

overall_metrics <- data.frame(
  Method = character(0),
  Accuracy = numeric(0),
  Precision = numeric(0),
  Recall = numeric(0),
  F1 = numeric(0),
  FPR = numeric(0)
)

for (i in seq_along(method_cols)) {
  m <- calculate_metrics(combined_results[[method_cols[i]]], combined_results$ground_truth)
  m$Method <- method_names[i]
  overall_metrics <- rbind(overall_metrics, m)
}

overall_metrics <- overall_metrics %>% arrange(desc(F1))

cat(sprintf("%-12s %8s %8s %8s %8s %8s\n", "Method", "Acc", "Prec", "Recall", "F1", "FPR"))
cat(paste(rep("-", 60), collapse = ""), "\n")
for (i in 1:nrow(overall_metrics)) {
  r <- overall_metrics[i, ]
  cat(sprintf("%-12s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n",
              r$Method,
              ifelse(is.na(r$Accuracy), NA, r$Accuracy * 100),
              ifelse(is.na(r$Precision), NA, r$Precision * 100),
              ifelse(is.na(r$Recall), NA, r$Recall * 100),
              ifelse(is.na(r$F1), NA, r$F1 * 100),
              ifelse(is.na(r$FPR), NA, r$FPR * 100)))
}

# --- Metrics by frequency ---
cat("\n=== Performance by Frequency ===\n\n")

freq_metrics <- list()
for (freq in unique(combined_results$frequency)) {
  freq_data <- combined_results %>% filter(frequency == freq)

  cat(sprintf("%s (n=%d):\n", freq, nrow(freq_data)))

  for (i in seq_along(method_cols)) {
    m <- calculate_metrics(freq_data[[method_cols[i]]], freq_data$ground_truth)
    if (!is.na(m$F1)) {
      cat(sprintf("  %-12s: F1=%.1f%%\n", method_names[i], m$F1 * 100))
    }
  }
  cat("\n")
}

# --- Comparison with simulation results ---
cat("\n=== Comparison: Simulation vs Real-World ===\n")

# Load simulation metrics if available
sim_file <- "seasonality_detection_metrics.rds"
if (file.exists(sim_file)) {
  sim_metrics <- readRDS(sim_file)

  # Map method names
  sim_name_map <- c(
    "FFT Confidence" = "FFT",
    "ACF Confidence" = "ACF",
    "Variance Strength" = "Variance",
    "Spectral Strength" = "Spectral",
    "Wavelet Strength" = "Wavelet",
    "SAZED" = "SAZED",
    "Autoperiod" = "Autoperiod",
    "CFDAutoperiod" = "CFD"
  )

  comparison <- sim_metrics %>%
    filter(Method %in% names(sim_name_map)) %>%
    mutate(Method_Short = sim_name_map[Method]) %>%
    select(Method_Short, F1) %>%
    rename(Sim_F1 = F1) %>%
    left_join(overall_metrics %>% select(Method, F1) %>% rename(Real_F1 = F1),
              by = c("Method_Short" = "Method")) %>%
    mutate(
      Difference = (Real_F1 - Sim_F1) * 100,
      Direction = ifelse(Difference > 0, "Real better", ifelse(Difference < 0, "Sim better", "Same"))
    )

  cat("\n")
  cat(sprintf("%-12s %10s %10s %10s\n", "Method", "Sim F1", "Real F1", "Diff"))
  cat(paste(rep("-", 45), collapse = ""), "\n")
  for (i in 1:nrow(comparison)) {
    r <- comparison[i, ]
    cat(sprintf("%-12s %9.1f%% %9.1f%% %+9.1f%%\n",
                r$Method_Short,
                ifelse(is.na(r$Sim_F1), NA, r$Sim_F1 * 100),
                ifelse(is.na(r$Real_F1), NA, r$Real_F1 * 100),
                ifelse(is.na(r$Difference), NA, r$Difference)))
  }
}

# --- Create plots ---
cat("\n=== Generating Plots ===\n")

# Plot 1: F1 scores comparison
overall_metrics$Method <- factor(overall_metrics$Method,
                                  levels = overall_metrics$Method[order(overall_metrics$F1, decreasing = TRUE)])

p_f1 <- ggplot(overall_metrics %>% filter(!is.na(F1)),
               aes(x = Method, y = F1 * 100, fill = Method)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Seasonality Detection on Real-World Data",
    subtitle = sprintf("M%s Competition - %d series",
                       ifelse(use_m4, "4", "3"), nrow(combined_results)),
    x = "",
    y = "F1 Score (%)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

ggsave("plots/real_world_f1.pdf", p_f1, width = 8, height = 5)
cat("Saved: plots/real_world_f1.pdf\n")

# Plot 2: Simulation vs Real comparison (if available)
if (exists("comparison") && nrow(comparison) > 0) {
  comparison_long <- comparison %>%
    select(Method_Short, Sim_F1, Real_F1) %>%
    pivot_longer(cols = c(Sim_F1, Real_F1),
                 names_to = "Source",
                 values_to = "F1") %>%
    mutate(Source = recode(Source, "Sim_F1" = "Simulation", "Real_F1" = "Real-World"))

  p_compare <- ggplot(comparison_long %>% filter(!is.na(F1)),
                      aes(x = Method_Short, y = F1 * 100, fill = Source)) +
    geom_col(position = "dodge", width = 0.7) +
    coord_flip() +
    scale_fill_manual(values = c("Simulation" = "steelblue", "Real-World" = "coral")) +
    labs(
      title = "Simulation vs Real-World Performance",
      x = "",
      y = "F1 Score (%)",
      fill = "Data Source"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )

  ggsave("plots/sim_vs_real_comparison.pdf", p_compare, width = 10, height = 6)
  cat("Saved: plots/sim_vs_real_comparison.pdf\n")
}

# --- Save results ---
saveRDS(combined_results, "real_world_benchmark_results.rds")
cat("Saved: real_world_benchmark_results.rds\n")

write.csv(overall_metrics, "real_world_benchmark_metrics.csv", row.names = FALSE)
cat("Saved: real_world_benchmark_metrics.csv\n")

# --- Key Findings ---
cat("\n=== Key Findings ===\n\n")

# Best method on real data
best_real <- overall_metrics[1, ]
cat(sprintf("1. Best method on real data: %s (F1 = %.1f%%)\n",
            best_real$Method, best_real$F1 * 100))

# Compare to simulation winner
if (exists("comparison")) {
  sim_winner <- comparison %>% arrange(desc(Sim_F1)) %>% slice(1)
  real_winner <- comparison %>% arrange(desc(Real_F1)) %>% slice(1)

  if (sim_winner$Method_Short != real_winner$Method_Short) {
    cat(sprintf("2. Simulation winner: %s (F1 = %.1f%%)\n",
                sim_winner$Method_Short, sim_winner$Sim_F1 * 100))
    cat(sprintf("   Real-world winner: %s (F1 = %.1f%%)\n",
                real_winner$Method_Short, real_winner$Real_F1 * 100))
    cat("   => Rankings differ between simulation and real-world data\n")
  } else {
    cat(sprintf("2. Same winner in both: %s\n", sim_winner$Method_Short))
    cat("   => Simulation findings validated on real-world data\n")
  }
}

cat("\n=== Real-World Benchmarking Complete ===\n")
