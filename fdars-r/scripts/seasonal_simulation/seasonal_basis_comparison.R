#!/usr/bin/env Rscript
# Seasonal Basis Comparison: Fourier vs P-splines
#
# This script generates monthly seasonal time series with varying seasonal
# strengths (0 = no seasonality, 1 = strong seasonality), fits them with
# both Fourier basis and P-splines, and compares AIC to determine which
# method performs better at different seasonal strength levels.

library(fdars)
library(ggplot2)
library(tidyr)
library(dplyr)
library(gridExtra)

set.seed(42)

# --- Configuration ---
n_strengths <- 11           # Number of seasonal strength levels (0 to 1)
n_curves_per_strength <- 30 # Number of curves per strength level
n_years <- 5                # Number of years of monthly data
n_months <- n_years * 12    # Total number of monthly observations
noise_sd <- 0.3             # Standard deviation of noise

# Seasonal strengths to test (0 = no season, 1 = full season)
seasonal_strengths <- seq(0, 1, length.out = n_strengths)

# Time grid (monthly data normalized to [0, 1])
t <- seq(0, 1, length.out = n_months)

# --- Function to generate seasonal time series ---
generate_seasonal_curve <- function(t, strength, noise_sd = 0.3) {
  # Base trend (slow polynomial)
  trend <- 0.5 * sin(2 * pi * t) * 0.3

  # Seasonal component (annual cycle = 5 complete cycles over our time span)
  # Since we have 5 years of monthly data, annual seasonality means 5 cycles
  n_cycles <- length(t) / 12  # Number of annual cycles
  seasonal <- strength * sin(2 * pi * n_cycles * t)

  # Add some harmonic for realistic seasonality
  seasonal <- seasonal + strength * 0.3 * cos(4 * pi * n_cycles * t)

  # Noise
  noise <- rnorm(length(t), sd = noise_sd)

  return(trend + seasonal + noise)
}

# --- Generate data for all seasonal strengths ---
cat("Generating seasonal time series data...\n")
cat(sprintf("  - %d seasonal strength levels\n", n_strengths))
cat(sprintf("  - %d curves per strength level\n", n_curves_per_strength))
cat(sprintf("  - %d monthly observations per curve (%d years)\n", n_months, n_years))

all_data <- list()
strength_labels <- character(0)

for (i in seq_along(seasonal_strengths)) {
  strength <- seasonal_strengths[i]

  # Generate curves for this strength level
  X <- matrix(0, nrow = n_curves_per_strength, ncol = n_months)
  for (j in 1:n_curves_per_strength) {
    X[j, ] <- generate_seasonal_curve(t, strength, noise_sd)
  }

  # Create fdata object
  fd <- fdata(X, argvals = t, rangeval = c(0, 1))

  all_data[[i]] <- list(
    strength = strength,
    fdata = fd
  )

  strength_labels <- c(strength_labels, sprintf("%.1f", strength))
}

# --- Compare Fourier vs P-splines using AIC ---
cat("\nComparing Fourier basis vs P-splines...\n")

# Store results
results <- data.frame(
  strength = numeric(0),
  curve_id = integer(0),
  aic_fourier = numeric(0),
  aic_pspline = numeric(0),
  winner = character(0)
)

# Basis parameters
# For Fourier: use enough basis functions to capture seasonal patterns
# For monthly data with 5 years: need at least 2*5+1 = 11 for annual cycles
fourier_nbasis_range <- seq(5, 21, by = 2)  # Odd numbers for Fourier

# For P-splines
pspline_nbasis <- 20

for (i in seq_along(all_data)) {
  strength <- all_data[[i]]$strength
  fd <- all_data[[i]]$fdata

  cat(sprintf("  Processing strength = %.1f...\n", strength))

  # For each curve, find optimal Fourier nbasis and compare with P-spline
  for (j in 1:nrow(fd$data)) {
    fd_single <- fd[j]

    # --- Fourier: find optimal nbasis via AIC ---
    fourier_aics <- sapply(fourier_nbasis_range, function(k) {
      tryCatch({
        basis.aic(fd_single, nbasis = k, type = "fourier")
      }, error = function(e) NA)
    })

    best_fourier_idx <- which.min(fourier_aics)
    best_fourier_aic <- fourier_aics[best_fourier_idx]
    best_fourier_nbasis <- fourier_nbasis_range[best_fourier_idx]

    # --- P-spline: use automatic lambda selection ---
    pspline_result <- tryCatch({
      pspline(fd_single, nbasis = pspline_nbasis, lambda.select = TRUE, criterion = "AIC")
    }, error = function(e) NULL)

    if (!is.null(pspline_result)) {
      pspline_aic <- pspline_result$aic
    } else {
      pspline_aic <- NA
    }

    # Determine winner
    if (!is.na(best_fourier_aic) && !is.na(pspline_aic)) {
      winner <- ifelse(best_fourier_aic < pspline_aic, "fourier", "pspline")
    } else {
      winner <- NA
    }

    results <- rbind(results, data.frame(
      strength = strength,
      curve_id = j,
      aic_fourier = best_fourier_aic,
      aic_pspline = pspline_aic,
      winner = winner
    ))
  }
}

# --- Analyze results ---
cat("\n=== Results Summary ===\n\n")

# Aggregate by seasonal strength
summary_by_strength <- aggregate(
  cbind(aic_fourier, aic_pspline) ~ strength,
  data = results,
  FUN = mean,
  na.rm = TRUE
)

# Count winners by strength
winner_counts <- table(results$strength, results$winner)
winner_props <- prop.table(winner_counts, margin = 1)

cat("Mean AIC by Seasonal Strength:\n")
cat("-" , rep("-", 50), "\n", sep = "")
cat(sprintf("%-10s %15s %15s %15s\n", "Strength", "Fourier AIC", "P-spline AIC", "Winner"))
cat("-" , rep("-", 50), "\n", sep = "")

for (i in 1:nrow(summary_by_strength)) {
  s <- summary_by_strength$strength[i]
  f_aic <- summary_by_strength$aic_fourier[i]
  p_aic <- summary_by_strength$aic_pspline[i]
  winner <- ifelse(f_aic < p_aic, "Fourier", "P-spline")

  cat(sprintf("%-10.1f %15.2f %15.2f %15s\n", s, f_aic, p_aic, winner))
}
cat("-" , rep("-", 50), "\n", sep = "")

cat("\nProportion of Fourier wins by Seasonal Strength:\n")
cat("-" , rep("-", 40), "\n", sep = "")

strengths_in_table <- as.numeric(rownames(winner_props))
for (i in seq_along(strengths_in_table)) {
  s <- strengths_in_table[i]
  fourier_prop <- ifelse("fourier" %in% colnames(winner_props), winner_props[i, "fourier"], 0)
  pspline_prop <- ifelse("pspline" %in% colnames(winner_props), winner_props[i, "pspline"], 0)

  cat(sprintf("Strength %.1f: Fourier %.0f%% | P-spline %.0f%%\n",
              s, fourier_prop * 100, pspline_prop * 100))
}

# --- Visualization with ggplot2 ---
cat("\n=== Generating plots ===\n")

# Prepare data for ggplot
aic_diff <- summary_by_strength$aic_fourier - summary_by_strength$aic_pspline

# Reshape for mean AIC plot
aic_long <- summary_by_strength %>%
  pivot_longer(cols = c(aic_fourier, aic_pspline),
               names_to = "Method", values_to = "AIC") %>%
  mutate(Method = recode(Method,
                         "aic_fourier" = "Fourier",
                         "aic_pspline" = "P-spline"))

# Plot 1: Mean AIC by strength
p1 <- ggplot(aic_long, aes(x = strength, y = AIC, color = Method, shape = Method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Fourier" = "#2166AC", "P-spline" = "#B2182B")) +
  labs(title = "Mean AIC: Fourier vs P-splines",
       x = "Seasonal Strength",
       y = "Mean AIC") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 2: AIC difference
diff_data <- data.frame(strength = summary_by_strength$strength, aic_diff = aic_diff)
p2 <- ggplot(diff_data, aes(x = strength, y = aic_diff)) +
  geom_line(color = "#762A83", linewidth = 1) +
  geom_point(color = "#762A83", size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.5, y = max(aic_diff) * 0.3,
           label = "P-spline better above\nFourier better below",
           size = 3, hjust = 0.5) +
  labs(title = "AIC Difference by Seasonal Strength",
       x = "Seasonal Strength",
       y = "AIC Difference (Fourier - P-spline)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 3: Fourier win rate
fourier_win_prop <- sapply(strengths_in_table, function(s) {
  idx <- as.character(s)
  if (idx %in% rownames(winner_props) && "fourier" %in% colnames(winner_props)) {
    return(winner_props[idx, "fourier"])
  }
  return(0)
})

win_data <- data.frame(strength = strengths_in_table, win_rate = fourier_win_prop * 100)
p3 <- ggplot(win_data, aes(x = factor(strength), y = win_rate)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "red") +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Fourier Win Rate by Seasonal Strength",
       x = "Seasonal Strength",
       y = "Fourier Win Rate (%)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 4: Example curves
example_indices <- c(1, 6, 11)
example_data <- do.call(rbind, lapply(example_indices, function(idx) {
  data.frame(
    t = t,
    value = all_data[[idx]]$fdata$data[1, ],
    strength = sprintf("Strength = %.1f", seasonal_strengths[idx])
  )
}))

p4 <- ggplot(example_data, aes(x = t, y = value, color = strength)) +
  geom_line(linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Example Curves at Different Seasonal Strengths",
       x = "Time (normalized)",
       y = "Value",
       color = "") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, face = "bold"))

# Save combined plot
ggsave("plots/seasonal_basis_comparison.pdf",
       grid.arrange(p1, p2, p3, p4, ncol = 2),
       width = 12, height = 10)

cat("Plots saved to: plots/seasonal_basis_comparison.pdf\n")

# --- Save results ---
saveRDS(results, "seasonal_basis_results.rds")
saveRDS(all_data, "seasonal_fdata_objects.rds")
cat("Results saved to: seasonal_basis_results.rds\n")
cat("Fdata objects saved to: seasonal_fdata_objects.rds\n")

# --- Final summary ---
cat("\n=== Key Findings ===\n")

# Calculate correlation between seasonal strength and Fourier advantage
cor_test <- cor.test(summary_by_strength$strength, -aic_diff)
cat(sprintf("Correlation between seasonal strength and Fourier advantage: r = %.3f (p = %.4f)\n",
            cor_test$estimate, cor_test$p.value))

# Find crossover point (if any)
if (any(aic_diff > 0) && any(aic_diff < 0)) {
  # Linear interpolation to find crossover
  crossover_idx <- which(diff(sign(aic_diff)) != 0)[1]
  if (!is.na(crossover_idx)) {
    x1 <- summary_by_strength$strength[crossover_idx]
    x2 <- summary_by_strength$strength[crossover_idx + 1]
    y1 <- aic_diff[crossover_idx]
    y2 <- aic_diff[crossover_idx + 1]
    crossover <- x1 - y1 * (x2 - x1) / (y2 - y1)
    cat(sprintf("Crossover point (equal AIC): Seasonal strength ~ %.2f\n", crossover))
  }
} else if (all(aic_diff < 0)) {
  cat("Fourier consistently outperforms P-splines across all seasonal strengths\n")
} else if (all(aic_diff > 0)) {
  cat("P-splines consistently outperform Fourier across all seasonal strengths\n")
}

cat("\nAnalysis complete!\n")
