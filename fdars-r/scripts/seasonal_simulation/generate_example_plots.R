# Generate example plots for Simulations 1-3
# These complement the robustness example plots already created

library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(42)

# Ensure plots directory exists
if (!dir.exists("plots")) dir.create("plots")

# Common parameters
n_years <- 5
n_months <- n_years * 12
t <- seq(0, 1, length.out = n_months)
n_cycles <- n_months / 12  # 5 cycles
noise_sd <- 0.3

# ============================================================================
# Simulation 1: Varying Seasonal Strength
# ============================================================================

cat("Generating Simulation 1 example plot...\n")

generate_seasonal_curve <- function(t, strength, noise_sd = 0.3) {
  n_cycles <- length(t) / 12
  seasonal <- strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + strength * 0.3 * cos(4 * pi * n_cycles * t)
  noise <- rnorm(length(t), sd = noise_sd)
  return(seasonal + noise)
}

# Generate examples at key strength levels
strengths <- c(0, 0.2, 0.5, 1.0)
strength_labels <- c("s = 0 (No seasonality)",
                     "s = 0.2 (Threshold)",
                     "s = 0.5 (Moderate)",
                     "s = 1.0 (Strong)")

sim1_data <- data.frame()
for (i in seq_along(strengths)) {
  y <- generate_seasonal_curve(t, strengths[i], noise_sd)
  sim1_data <- rbind(sim1_data, data.frame(
    t = t,
    y = y,
    strength = strength_labels[i]
  ))
}

sim1_data$strength <- factor(sim1_data$strength, levels = strength_labels)

p_sim1 <- ggplot(sim1_data, aes(x = t, y = y)) +
  geom_line(color = "steelblue", linewidth = 0.5) +
  facet_wrap(~ strength, ncol = 2, scales = "free_y") +
  labs(
    title = "Simulation 1: Varying Seasonal Strength",
    subtitle = "Same noise (sd = 0.3), different seasonal amplitude",
    x = "Time (normalized to [0,1])",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40")
  )

ggsave("plots/example_sim1_seasonal_strength.pdf", p_sim1, width = 8, height = 6)
cat("  Saved: plots/example_sim1_seasonal_strength.pdf\n")

# ============================================================================
# Simulation 2: Non-linear Trend
# ============================================================================

cat("Generating Simulation 2 example plot...\n")

generate_nonlinear_trend <- function(t, trend_strength) {
  quadratic <- 2 * (t - 0.5)^2
  cubic <- 0.5 * (t - 0.3)^3
  sigmoid <- 1 / (1 + exp(-10 * (t - 0.6)))
  trend <- trend_strength * (quadratic + cubic + 0.3 * sigmoid - 0.5)
  return(trend)
}

generate_curve_with_trend <- function(t, seasonal_strength, trend_strength, noise_sd = 0.3) {
  trend <- generate_nonlinear_trend(t, trend_strength)
  n_cycles <- length(t) / 12
  seasonal <- seasonal_strength * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + seasonal_strength * 0.3 * cos(4 * pi * n_cycles * t)
  noise <- rnorm(length(t), sd = noise_sd)
  return(trend + seasonal + noise)
}

# Show: s=0.5 (seasonal) with different trend strengths
trend_strengths <- c(0, 0.5, 1.0, 2.0)
trend_labels <- c("trend = 0 (No trend)",
                  "trend = 0.5 (Weak)",
                  "trend = 1.0 (Moderate)",
                  "trend = 2.0 (Strong)")

sim2_data <- data.frame()
for (i in seq_along(trend_strengths)) {
  y <- generate_curve_with_trend(t, 0.5, trend_strengths[i], noise_sd)
  sim2_data <- rbind(sim2_data, data.frame(
    t = t,
    y = y,
    trend = trend_labels[i]
  ))
}

sim2_data$trend <- factor(sim2_data$trend, levels = trend_labels)

p_sim2 <- ggplot(sim2_data, aes(x = t, y = y)) +
  geom_line(color = "steelblue", linewidth = 0.5) +
  facet_wrap(~ trend, ncol = 2, scales = "free_y") +
  labs(
    title = "Simulation 2: Non-linear Trend + Seasonality",
    subtitle = "Fixed seasonality (s = 0.5), varying trend strength",
    x = "Time (normalized to [0,1])",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40")
  )

ggsave("plots/example_sim2_nonlinear_trend.pdf", p_sim2, width = 8, height = 6)
cat("  Saved: plots/example_sim2_nonlinear_trend.pdf\n")

# ============================================================================
# Simulation 3: Multiple Trend Types
# ============================================================================

cat("Generating Simulation 3 example plot...\n")

# Define trend functions
trend_functions <- list(
  none = function(t, strength) rep(0, length(t)),
  linear = function(t, strength) strength * (t - 0.5),
  quadratic = function(t, strength) strength * ((t - 0.5)^2 - 0.25),
  cubic = function(t, strength) strength * 2 * (t - 0.5)^3,
  exponential = function(t, strength) strength * (exp(2 * t) / exp(2) - 0.5),
  logarithmic = function(t, strength) {
    log_vals <- log(t + 0.1)
    normalized <- (log_vals - min(log_vals)) / (max(log_vals) - min(log_vals)) - 0.5
    strength * normalized
  },
  sigmoid = function(t, strength) strength * (1 / (1 + exp(-10 * (t - 0.5))) - 0.5),
  slow_sine = function(t, strength) strength * sin(2 * pi * t)
)

# Generate example for each trend type with s=0.5, trend_strength=1.0
trend_names <- c("None", "Linear", "Quadratic", "Cubic",
                 "Exponential", "Logarithmic", "Sigmoid", "Slow Sine")

sim3_data <- data.frame()
for (i in seq_along(trend_functions)) {
  trend_fn <- trend_functions[[i]]
  trend <- trend_fn(t, 1.0)
  n_cycles <- length(t) / 12
  seasonal <- 0.5 * sin(2 * pi * n_cycles * t)
  seasonal <- seasonal + 0.5 * 0.3 * cos(4 * pi * n_cycles * t)
  noise <- rnorm(length(t), sd = noise_sd)
  y <- trend + seasonal + noise

  sim3_data <- rbind(sim3_data, data.frame(
    t = t,
    y = y,
    trend_type = trend_names[i]
  ))
}

sim3_data$trend_type <- factor(sim3_data$trend_type, levels = trend_names)

p_sim3 <- ggplot(sim3_data, aes(x = t, y = y)) +
  geom_line(color = "steelblue", linewidth = 0.4) +
  facet_wrap(~ trend_type, ncol = 4, scales = "free_y") +
  labs(
    title = "Simulation 3: Multiple Trend Types + Seasonality",
    subtitle = "Fixed seasonality (s = 0.5), trend strength = 1.0",
    x = "Time",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 9, face = "bold"),
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    axis.text = element_text(size = 7)
  )

ggsave("plots/example_sim3_trend_types.pdf", p_sim3, width = 10, height = 5)
cat("  Saved: plots/example_sim3_trend_types.pdf\n")

cat("\nAll example plots generated!\n")
