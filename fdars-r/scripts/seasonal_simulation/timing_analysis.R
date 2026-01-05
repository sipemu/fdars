#!/usr/bin/env Rscript
# Computational Complexity Analysis
#
# This script measures the execution time of each seasonality detection method
# across different series lengths to understand:
# - Scaling behavior (O(n), O(n log n), O(n^2), etc.)
# - Cost-benefit trade-offs (accuracy vs speed)
# - Practical recommendations for large datasets

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Check for microbenchmark package
if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  message("Installing microbenchmark package...")
  install.packages("microbenchmark", repos = "https://cloud.r-project.org")
}
library(microbenchmark)

set.seed(42)

cat("=== Computational Complexity Analysis ===\n\n")

# --- Configuration ---
series_lengths <- c(60, 120, 240, 480, 960, 1920)  # Number of time points
n_reps <- 10  # Repetitions per measurement for stability
warmup_reps <- 2  # Warmup runs before measurement

# --- Generate test signal function ---
generate_test_signal <- function(n, strength = 0.5, noise_sd = 0.3) {
  t <- seq(0, 1, length.out = n)
  n_cycles <- 5
  signal <- strength * sin(2 * pi * n_cycles * t) +
            strength * 0.3 * cos(4 * pi * n_cycles * t) +
            rnorm(n, sd = noise_sd)
  fdata(matrix(signal, nrow = 1), argvals = t, rangeval = c(0, 1))
}

# --- Method definitions ---
# Each method is wrapped in a function that takes an fdata object
methods <- list(
  "FFT" = function(fd) estimate.period(fd, method = "fft"),
  "ACF" = function(fd) estimate.period(fd, method = "acf"),
  "Variance" = function(fd) seasonal.strength(fd, period = 0.2, method = "variance"),
  "Spectral" = function(fd) seasonal.strength(fd, period = 0.2, method = "spectral"),
  "Wavelet" = function(fd) seasonal.strength(fd, period = 0.2, method = "wavelet"),
  "SAZED" = function(fd) sazed(fd),
  "Autoperiod" = function(fd) autoperiod(fd),
  "CFDAutoperiod" = function(fd) cfd.autoperiod(fd)
)

# Note: AIC comparison is omitted because it requires basis fitting which
# has different computational characteristics and is much slower

# --- Measure execution time ---
cat(sprintf("Measuring execution time for %d series lengths...\n", length(series_lengths)))
cat(sprintf("Methods: %s\n", paste(names(methods), collapse = ", ")))
cat(sprintf("Repetitions per measurement: %d\n\n", n_reps))

timing_results <- data.frame(
  Method = character(0),
  Length = integer(0),
  Time_ms = numeric(0),
  stringsAsFactors = FALSE
)

for (n in series_lengths) {
  cat(sprintf("Processing length = %d...\n", n))

  # Generate test signal
  fd <- generate_test_signal(n)

  for (method_name in names(methods)) {
    method_fn <- methods[[method_name]]

    # Warmup runs
    for (i in 1:warmup_reps) {
      tryCatch(method_fn(fd), error = function(e) NULL)
    }

    # Benchmark
    mb <- tryCatch({
      microbenchmark(
        method_fn(fd),
        times = n_reps,
        unit = "ms"
      )
    }, error = function(e) {
      message(sprintf("  Warning: %s failed at length %d: %s", method_name, n, e$message))
      NULL
    })

    if (!is.null(mb)) {
      # Store all individual measurements
      for (t in mb$time / 1e6) {  # Convert nanoseconds to milliseconds
        timing_results <- rbind(timing_results, data.frame(
          Method = method_name,
          Length = n,
          Time_ms = t,
          stringsAsFactors = FALSE
        ))
      }
      cat(sprintf("  %s: %.2f ms (median)\n", method_name, median(mb$time) / 1e6))
    }
  }
}

# --- Summary statistics ---
timing_summary <- timing_results %>%
  group_by(Method, Length) %>%
  summarise(
    Median = median(Time_ms),
    Mean = mean(Time_ms),
    SD = sd(Time_ms),
    Min = min(Time_ms),
    Max = max(Time_ms),
    .groups = "drop"
  )

cat("\n=== Timing Summary (median ms) ===\n\n")
timing_wide <- timing_summary %>%
  select(Method, Length, Median) %>%
  pivot_wider(names_from = Length, values_from = Median)
print(as.data.frame(timing_wide), row.names = FALSE)

# --- Estimate computational complexity ---
cat("\n=== Complexity Estimation ===\n\n")

# Fit log-log model to estimate complexity exponent
# Time ~ c * n^alpha  =>  log(Time) ~ log(c) + alpha * log(n)
complexity_estimates <- timing_summary %>%
  group_by(Method) %>%
  summarise(
    # Fit linear model on log scale
    alpha = {
      fit <- lm(log(Median) ~ log(Length))
      coef(fit)[2]
    },
    r_squared = {
      fit <- lm(log(Median) ~ log(Length))
      summary(fit)$r.squared
    },
    .groups = "drop"
  ) %>%
  mutate(
    Complexity = case_when(
      alpha < 0.8 ~ "Sub-linear",
      alpha < 1.2 ~ "O(n)",
      alpha < 1.5 ~ "O(n log n)",
      alpha < 2.2 ~ "O(n^2)",
      TRUE ~ "Super-quadratic"
    )
  ) %>%
  arrange(alpha)

cat(sprintf("%-15s %8s %8s %15s\n", "Method", "Alpha", "R^2", "Est. Complexity"))
cat(paste(rep("-", 50), collapse = ""), "\n")
for (i in 1:nrow(complexity_estimates)) {
  r <- complexity_estimates[i, ]
  cat(sprintf("%-15s %8.2f %8.3f %15s\n", r$Method, r$alpha, r$r_squared, r$Complexity))
}

# --- Load accuracy metrics for cost-benefit analysis ---
metrics_file <- "seasonality_detection_metrics.rds"
if (file.exists(metrics_file)) {
  metrics <- readRDS(metrics_file)
  cat("\n=== Cost-Benefit Analysis ===\n\n")

  # Map method names to match
  method_name_map <- c(
    "FFT Confidence" = "FFT",
    "ACF Confidence" = "ACF",
    "Variance Strength" = "Variance",
    "Spectral Strength" = "Spectral",
    "Wavelet Strength" = "Wavelet",
    "SAZED" = "SAZED",
    "Autoperiod" = "Autoperiod",
    "CFDAutoperiod" = "CFDAutoperiod"
  )

  # Get F1 scores
  f1_scores <- metrics %>%
    filter(Method %in% names(method_name_map)) %>%
    mutate(Method_Short = method_name_map[Method]) %>%
    select(Method_Short, F1)

  # Get median time at typical length (240 points = 4 years monthly)
  typical_length <- 240
  typical_times <- timing_summary %>%
    filter(Length == typical_length) %>%
    select(Method, Median)

  cost_benefit <- merge(f1_scores, typical_times, by.x = "Method_Short", by.y = "Method") %>%
    rename(Method = Method_Short, Time_ms = Median) %>%
    mutate(
      F1_pct = F1 * 100,
      Efficiency = F1 / log10(Time_ms + 1)  # F1 per log(time)
    ) %>%
    arrange(desc(F1))

  cat(sprintf("%-15s %8s %10s %12s\n", "Method", "F1 (%)", "Time (ms)", "Efficiency"))
  cat(paste(rep("-", 50), collapse = ""), "\n")
  for (i in 1:nrow(cost_benefit)) {
    r <- cost_benefit[i, ]
    cat(sprintf("%-15s %7.1f%% %10.2f %12.2f\n",
                r$Method, r$F1_pct, r$Time_ms, r$Efficiency))
  }
}

# --- Create plots ---
cat("\n=== Generating Plots ===\n")

# Color palette
method_colors <- c(
  "Variance" = "#E41A1C",
  "Wavelet" = "#377EB8",
  "SAZED" = "#4DAF4A",
  "Spectral" = "#984EA3",
  "Autoperiod" = "#FF7F00",
  "FFT" = "#FFFF33",
  "CFDAutoperiod" = "#A65628",
  "ACF" = "#999999"
)

# Plot 1: Log-log scaling plot
p_scaling <- ggplot(timing_summary, aes(x = Length, y = Median, color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_ribbon(aes(ymin = Min, ymax = Max, fill = Method), alpha = 0.1, color = NA) +
  scale_x_log10(breaks = series_lengths, labels = series_lengths) +
  scale_y_log10() +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  labs(
    title = "Computational Scaling: Execution Time vs Series Length",
    subtitle = "Log-log scale; slope indicates complexity exponent",
    x = "Series Length (log scale)",
    y = "Execution Time (ms, log scale)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

ggsave("plots/computational_scaling.pdf", p_scaling, width = 10, height = 6)
cat("Saved: plots/computational_scaling.pdf\n")

# Plot 2: Linear scale for typical range
p_linear <- ggplot(timing_summary %>% filter(Length <= 480),
                   aes(x = Length, y = Median, color = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = method_colors) +
  labs(
    title = "Execution Time for Typical Series Lengths",
    subtitle = "Up to 480 observations (8 years of monthly data)",
    x = "Series Length",
    y = "Execution Time (ms)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

ggsave("plots/computational_linear.pdf", p_linear, width = 10, height = 6)
cat("Saved: plots/computational_linear.pdf\n")

# Plot 3: Complexity exponent bar chart
p_complexity <- ggplot(complexity_estimates, aes(x = reorder(Method, alpha), y = alpha, fill = Method)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = 2, linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.5, y = 1.05, label = "O(n)", hjust = 0, size = 3, color = "gray40") +
  annotate("text", x = 0.5, y = 2.05, label = "O(n^2)", hjust = 0, size = 3, color = "gray40") +
  scale_fill_manual(values = method_colors) +
  coord_flip() +
  labs(
    title = "Estimated Computational Complexity",
    subtitle = sprintf("Exponent alpha from fit: Time ~ n^alpha (R^2 > %.2f for all)",
                       min(complexity_estimates$r_squared)),
    x = "",
    y = "Complexity Exponent (alpha)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

ggsave("plots/computational_complexity.pdf", p_complexity, width = 8, height = 5)
cat("Saved: plots/computational_complexity.pdf\n")

# Plot 4: Cost-benefit scatter (if metrics available)
if (exists("cost_benefit") && nrow(cost_benefit) > 0) {
  p_cost_benefit <- ggplot(cost_benefit, aes(x = Time_ms, y = F1_pct, color = Method)) +
    geom_point(size = 6) +
    geom_text(aes(label = Method), vjust = -1.5, hjust = 0.5, size = 3) +
    scale_color_manual(values = method_colors) +
    scale_x_log10() +
    coord_cartesian(ylim = c(80, 100)) +
    labs(
      title = "Cost-Benefit: Accuracy vs Speed",
      subtitle = sprintf("At %d observations; x-axis is log scale", typical_length),
      x = "Execution Time (ms, log scale)",
      y = "F1 Score (%)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 9)
    )

  ggsave("plots/cost_benefit.pdf", p_cost_benefit, width = 8, height = 6)
  cat("Saved: plots/cost_benefit.pdf\n")
}

# Plot 5: Combined figure
p_combined <- grid.arrange(
  p_scaling + theme(legend.position = "bottom"),
  p_complexity,
  ncol = 2,
  widths = c(1.5, 1)
)

ggsave("plots/computational_analysis_combined.pdf", p_combined, width = 14, height = 6)
cat("Saved: plots/computational_analysis_combined.pdf\n")

# --- Save results ---
saveRDS(timing_results, "timing_results.rds")
cat("Saved: timing_results.rds\n")

saveRDS(timing_summary, "timing_summary.rds")
cat("Saved: timing_summary.rds\n")

write.csv(timing_summary, "timing_summary.csv", row.names = FALSE)
cat("Saved: timing_summary.csv\n")

write.csv(complexity_estimates, "complexity_estimates.csv", row.names = FALSE)
cat("Saved: complexity_estimates.csv\n")

# --- Key Findings ---
cat("\n=== Key Findings ===\n\n")

# Fastest method
fastest <- complexity_estimates[1, ]
cat(sprintf("Fastest scaling: %s (alpha = %.2f, %s)\n",
            fastest$Method, fastest$alpha, fastest$Complexity))

# Slowest method
slowest <- complexity_estimates[nrow(complexity_estimates), ]
cat(sprintf("Slowest scaling: %s (alpha = %.2f, %s)\n",
            slowest$Method, slowest$alpha, slowest$Complexity))

# Best efficiency (if cost_benefit available)
if (exists("cost_benefit") && nrow(cost_benefit) > 0) {
  best_eff <- cost_benefit[which.max(cost_benefit$Efficiency), ]
  cat(sprintf("Best efficiency (F1/log(time)): %s (F1=%.1f%%, Time=%.2fms)\n",
              best_eff$Method, best_eff$F1_pct, best_eff$Time_ms))
}

# Practical recommendations
cat("\n=== Practical Recommendations ===\n\n")

cat("For real-time applications (< 10ms at n=240):\n")
fast_methods <- timing_summary %>%
  filter(Length == 240, Median < 10) %>%
  arrange(Median)
if (nrow(fast_methods) > 0) {
  for (i in 1:nrow(fast_methods)) {
    cat(sprintf("  - %s (%.2f ms)\n", fast_methods$Method[i], fast_methods$Median[i]))
  }
} else {
  cat("  No methods meet this criterion\n")
}

cat("\nFor large datasets (n > 1000):\n")
linear_methods <- complexity_estimates %>% filter(alpha < 1.5)
if (nrow(linear_methods) > 0) {
  for (i in 1:nrow(linear_methods)) {
    cat(sprintf("  - %s (alpha = %.2f)\n", linear_methods$Method[i], linear_methods$alpha[i]))
  }
} else {
  cat("  All methods have super-linear complexity\n")
}

cat("\n=== Computational Complexity Analysis Complete ===\n")
