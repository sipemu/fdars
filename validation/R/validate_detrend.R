#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Validating detrend methods ===\n")

results <- list()

# Load standard dataset for detrending tests
std_data <- load_data("standard_50x101")
n <- std_data$n
m <- std_data$m
argvals <- std_data$argvals
mat <- to_matrix(std_data$data, n, m)
curve1 <- mat[1, ]

# ---- (a) Linear detrend -------------------------------------------------------
message("  Computing linear detrend on first curve...")
results$linear_detrend <- tryCatch({
  fit <- lm(curve1 ~ argvals)
  trend <- as.numeric(fitted(fit))
  detrended <- as.numeric(residuals(fit))
  coefs <- as.numeric(coef(fit))  # intercept, slope

  message(sprintf("    Intercept: %.6f, Slope: %.6f", coefs[1], coefs[2]))

  list(
    trend = trend,
    detrended = detrended,
    intercept = coefs[1],
    slope = coefs[2]
  )
}, error = function(e) {
  warning(sprintf("Linear detrend failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Polynomial detrend (degree 2) ----------------------------------------
message("  Computing polynomial detrend (degree=2) on first curve...")
results$poly_detrend <- tryCatch({
  argvals_normalized <- (argvals - min(argvals)) / (max(argvals) - min(argvals))
  fit <- lm(curve1 ~ poly(argvals_normalized, 2, raw = TRUE))
  trend <- as.numeric(fitted(fit))
  detrended <- as.numeric(residuals(fit))
  coefs <- as.numeric(coef(fit))  # intercept, linear, quadratic

  message(sprintf("    Coefficients: %.6f, %.6f, %.6f", coefs[1], coefs[2], coefs[3]))

  list(
    trend = trend,
    detrended = detrended,
    coefficients = coefs
  )
}, error = function(e) {
  warning(sprintf("Polynomial detrend failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Differencing (order 1) -----------------------------------------------
message("  Computing first-order differencing on first curve...")
results$differencing <- tryCatch({
  differenced <- as.numeric(diff(curve1, differences = 1))

  message(sprintf("    Differenced length: %d (original: %d)", length(differenced), length(curve1)))

  list(
    differenced = differenced
  )
}, error = function(e) {
  warning(sprintf("Differencing failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) STL decomposition ----------------------------------------------------
message("  Computing STL decomposition on seasonal data...")
results$stl_decomposition <- tryCatch({
  seasonal_data <- load_data("seasonal_200")
  noisy_sine <- seasonal_data$noisy_sine
  t_vals <- seasonal_data$t
  n_ts <- seasonal_data$n
  period <- seasonal_data$period

  # Compute frequency: number of observations per period
  # t spans [0, 10] with 200 points, period = 2.0
  # dt = 10/199, so observations per period = period / dt = 2.0 * 199 / 10
  dt <- (max(t_vals) - min(t_vals)) / (n_ts - 1)
  freq <- round(period / dt)

  message(sprintf("    Using frequency = %d (period=%.1f, dt=%.4f)", freq, period, dt))

  ts_obj <- ts(noisy_sine, frequency = freq)
  stl_result <- stl(ts_obj, s.window = "periodic")

  trend_component <- as.numeric(stl_result$time.series[, "trend"])
  seasonal_component <- as.numeric(stl_result$time.series[, "seasonal"])
  remainder_component <- as.numeric(stl_result$time.series[, "remainder"])

  message(sprintf("    Components length: %d each", length(trend_component)))

  list(
    frequency = freq,
    trend = trend_component,
    seasonal = seasonal_component,
    remainder = remainder_component
  )
}, error = function(e) {
  warning(sprintf("STL decomposition failed: %s", conditionMessage(e)))
  NULL
})

# ---- (e) Additive decomposition -----------------------------------------------
message("  Computing additive decomposition on seasonal data...")
results$additive_decomposition <- tryCatch({
  seasonal_data <- load_data("seasonal_200")
  noisy_sine <- seasonal_data$noisy_sine
  t_vals <- seasonal_data$t
  n_ts <- seasonal_data$n
  period <- seasonal_data$period

  dt <- (max(t_vals) - min(t_vals)) / (n_ts - 1)
  freq <- round(period / dt)

  ts_obj <- ts(noisy_sine, frequency = freq)
  decomp <- stats::decompose(ts_obj, type = "additive")

  trend_component <- as.numeric(decomp$trend)
  seasonal_component <- as.numeric(decomp$seasonal)
  random_component <- as.numeric(decomp$random)

  # Note: decompose returns NA for first/last (freq/2) values in trend and random
  message(sprintf("    Trend NAs: %d, Random NAs: %d",
                  sum(is.na(trend_component)), sum(is.na(random_component))))

  list(
    frequency = freq,
    trend = trend_component,
    seasonal = seasonal_component,
    random = random_component
  )
}, error = function(e) {
  warning(sprintf("Additive decomposition failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results --------------------------------------------------------------
message("\n  Saving expected detrend values...")
save_expected(results, "detrend_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Detrend validation complete: %d/%d computations succeeded ===", computed, total))
