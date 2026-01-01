#' Generate Diverse Time Series for Seasonality Classifier Training
#'
#' This script provides functions to generate a comprehensive dataset of
#' synthetic time series with known properties for training ML classifiers.
#'
#' The generated data covers:
#' - Varying seasonal strengths (0 to 1)
#' - Multiple trend types (none, linear, quadratic, cubic, exponential, etc.)
#' - Different noise types (white, red/AR(1), outliers)
#' - Multiple seasonalities
#' - Amplitude modulation (time-varying seasonality)
#'
#' Usage:
#'   source("generate_training_data.R")
#'   data <- generate_training_dataset(n_per_config = 10, seed = 42)
#'   # data$series: list of time series vectors
#'   # data$metadata: data.frame with labels for each series

library(fdars)

# =============================================================================
# Component Functions
# =============================================================================

#' Generate white noise
#' @param n Length of series
#' @param sd Standard deviation
generate_white_noise <- function(n, sd = 0.3) {
rnorm(n, mean = 0, sd = sd)
}

#' Generate red noise (AR(1) process)
#' @param n Length of series
#' @param phi AR(1) coefficient (0 < phi < 1)
#' @param sd Innovation standard deviation
generate_red_noise <- function(n, phi = 0.7, sd = 0.3) {
  noise <- numeric(n)
  innovation_sd <- sd * sqrt(1 - phi^2)  # Scale to maintain marginal variance
  noise[1] <- rnorm(1, sd = sd)
  for (i in 2:n) {
    noise[i] <- phi * noise[i-1] + rnorm(1, sd = innovation_sd)
  }
  noise
}

#' Generate noise with outliers
#' @param n Length of series
#' @param sd Base standard deviation
#' @param outlier_prob Probability of outlier at each point
#' @param outlier_magnitude Multiplier for outlier magnitude
generate_outlier_noise <- function(n, sd = 0.3, outlier_prob = 0.05,
                                   outlier_magnitude = 5) {
  noise <- rnorm(n, sd = sd)
  outlier_idx <- runif(n) < outlier_prob
  noise[outlier_idx] <- noise[outlier_idx] * outlier_magnitude
  noise
}

#' Generate trend component
#' @param t Time vector (normalized to [0,1])
#' @param type Trend type
#' @param strength Trend strength multiplier
generate_trend <- function(t, type = "none", strength = 1.0) {
  trend <- switch(type,
    "none" = rep(0, length(t)),
    "linear" = t - 0.5,
    "quadratic" = (t - 0.5)^2 - 0.25,
    "cubic" = 2 * (t - 0.5)^3,
    "exponential" = exp(2 * t) / exp(2) - 0.5,
    "logarithmic" = {
      raw <- log(t + 0.1)
      (raw - min(raw)) / (max(raw) - min(raw)) - 0.5
    },
    "sigmoid" = 1 / (1 + exp(-10 * (t - 0.5))) - 0.5,
    "slow_sine" = sin(2 * pi * t),  # One full cycle
    rep(0, length(t))
  )
  strength * trend
}

#' Generate seasonal component
#' @param t Time vector (normalized to [0,1])
#' @param strength Seasonal strength (0 = none, 1 = full)
#' @param n_cycles Number of seasonal cycles
#' @param harmonics Include harmonics (more realistic seasonality)
generate_seasonal <- function(t, strength = 1.0, n_cycles = 5,
                              harmonics = TRUE) {
  seasonal <- sin(2 * pi * n_cycles * t)
  if (harmonics) {
    seasonal <- seasonal + 0.3 * cos(4 * pi * n_cycles * t)
  }
  strength * seasonal
}

#' Generate multiple seasonalities
#' @param t Time vector
#' @param strengths Vector of strengths for each seasonality
#' @param n_cycles Vector of cycle counts for each seasonality
generate_multi_seasonal <- function(t, strengths = c(1.0, 0.5),
                                    n_cycles = c(5, 20)) {
  seasonal <- rep(0, length(t))
  for (i in seq_along(strengths)) {
    seasonal <- seasonal + strengths[i] * sin(2 * pi * n_cycles[i] * t)
  }
  seasonal
}

#' Generate amplitude-modulated seasonality
#' @param t Time vector
#' @param base_strength Base seasonal strength
#' @param n_cycles Number of seasonal cycles
#' @param modulation_type Type of amplitude modulation
generate_amplitude_modulated <- function(t, base_strength = 1.0, n_cycles = 5,
                                         modulation_type = "linear_growth") {
  # Envelope function
envelope <- switch(modulation_type,
    "linear_growth" = 0.2 + 0.8 * t,           # Grows from 0.2 to 1.0
    "linear_decay" = 1.0 - 0.8 * t,            # Decays from 1.0 to 0.2
    "emergence" = ifelse(t < 0.5, 0.1, 1.0),   # Emerges halfway
    "periodic" = 0.5 + 0.5 * sin(2 * pi * t),  # Periodic modulation
    rep(1.0, length(t))
  )

  seasonal <- sin(2 * pi * n_cycles * t)
  base_strength * envelope * seasonal
}

# =============================================================================
# Main Generation Function
# =============================================================================

#' Generate a single time series with specified properties
#'
#' @param n_points Number of time points
#' @param seasonal_strength Strength of seasonality (0-1)
#' @param n_cycles Number of seasonal cycles
#' @param trend_type Type of trend
#' @param trend_strength Strength of trend
#' @param noise_type Type of noise ("white", "red", "outliers")
#' @param noise_sd Noise standard deviation
#' @param ar_phi AR(1) coefficient for red noise
#' @param multi_seasonal Use multiple seasonalities
#' @param secondary_strength Strength of secondary seasonality
#' @param secondary_cycles Cycles for secondary seasonality
#' @param amplitude_modulation Type of amplitude modulation (NULL for none)
#'
#' @return Numeric vector of time series values
generate_single_series <- function(
  n_points = 120,
  seasonal_strength = 0.5,
  n_cycles = 5,
  trend_type = "none",
  trend_strength = 1.0,
  noise_type = "white",
  noise_sd = 0.3,
  ar_phi = 0.7,
  multi_seasonal = FALSE,
  secondary_strength = 0.3,
  secondary_cycles = 20,
  amplitude_modulation = NULL
) {
  # Time vector normalized to [0, 1]
  t <- seq(0, 1, length.out = n_points)

  # Generate trend
  trend <- generate_trend(t, trend_type, trend_strength)

  # Generate seasonal component
  if (!is.null(amplitude_modulation)) {
    seasonal <- generate_amplitude_modulated(t, seasonal_strength, n_cycles,
                                             amplitude_modulation)
  } else if (multi_seasonal) {
    seasonal <- generate_multi_seasonal(
      t,
      strengths = c(seasonal_strength, secondary_strength),
      n_cycles = c(n_cycles, secondary_cycles)
    )
  } else {
    seasonal <- generate_seasonal(t, seasonal_strength, n_cycles)
  }

  # Generate noise
  noise <- switch(noise_type,
    "white" = generate_white_noise(n_points, noise_sd),
    "red" = generate_red_noise(n_points, ar_phi, noise_sd),
    "outliers" = generate_outlier_noise(n_points, noise_sd),
    generate_white_noise(n_points, noise_sd)
  )

  # Combine components
  trend + seasonal + noise
}

# =============================================================================
# Dataset Generation
# =============================================================================

#' Generate comprehensive training dataset
#'
#' Creates a diverse dataset of time series with known properties for
#' training seasonality detection classifiers.
#'
#' @param n_per_config Number of series per configuration
#' @param n_points Number of time points per series
#' @param seed Random seed for reproducibility
#' @param include_robustness Include robustness scenarios (red noise, etc.)
#'
#' @return List with:
#'   - series: List of numeric vectors (time series)
#'   - metadata: Data frame with labels for each series
#'   - t: Time vector used for generation
#'
#' @examples
#' data <- generate_training_dataset(n_per_config = 5, seed = 42)
#' dim(data$metadata)  # Number of series x number of features
#'
#' @export
generate_training_dataset <- function(
  n_per_config = 10,
  n_points = 120,
  seed = NULL,
  include_robustness = TRUE
) {
  if (!is.null(seed)) set.seed(seed)

  # Configuration space
  seasonal_strengths <- c(0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
  trend_types <- c("none", "linear", "quadratic", "cubic",
                   "exponential", "sigmoid", "slow_sine")
  trend_strengths <- c(0, 0.5, 1.0, 2.0)

  # Storage
  all_series <- list()
  all_metadata <- list()
  series_id <- 0

  cat("Generating training dataset...\n")

  # -------------------------------------------------------------------------
  # Part 1: Basic configurations (seasonal strength × trend type × trend strength)
  # -------------------------------------------------------------------------
  cat("Part 1: Basic configurations (white noise)...\n")

  for (s_str in seasonal_strengths) {
    for (t_type in trend_types) {
      for (t_str in trend_strengths) {
        # Skip redundant combinations
        if (t_type == "none" && t_str > 0) next
        if (t_type != "none" && t_str == 0) next

        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            trend_type = t_type,
            trend_strength = t_str,
            noise_type = "white"
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = s_str >= 0.2,
            trend_type = t_type,
            trend_strength = t_str,
            noise_type = "white",
            ar_phi = NA,
            multi_seasonal = FALSE,
            amplitude_modulation = "none",
            scenario = "basic"
          )
        }
      }
    }
  }

  cat(sprintf("  Generated %d basic series\n", series_id))

  if (include_robustness) {
    # -----------------------------------------------------------------------
    # Part 2: Red noise (AR(1)) scenarios
    # -----------------------------------------------------------------------
    cat("Part 2: Red noise (AR(1)) scenarios...\n")
    ar_phis <- c(0.3, 0.5, 0.7, 0.9)
    base_id <- series_id

    for (s_str in c(0, 0.3, 0.5, 1.0)) {
      for (phi in ar_phis) {
        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            noise_type = "red",
            ar_phi = phi
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = s_str >= 0.2,
            trend_type = "none",
            trend_strength = 0,
            noise_type = "red",
            ar_phi = phi,
            multi_seasonal = FALSE,
            amplitude_modulation = "none",
            scenario = "red_noise"
          )
        }
      }
    }

    cat(sprintf("  Generated %d red noise series\n", series_id - base_id))

    # -----------------------------------------------------------------------
    # Part 3: Multiple seasonalities
    # -----------------------------------------------------------------------
    cat("Part 3: Multiple seasonalities...\n")
    base_id <- series_id

    secondary_configs <- list(
      list(strength = 0.3, cycles = 20),
      list(strength = 0.5, cycles = 15),
      list(strength = 0.7, cycles = 25)
    )

    for (s_str in c(0.3, 0.5, 0.7, 1.0)) {
      for (sec_cfg in secondary_configs) {
        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            multi_seasonal = TRUE,
            secondary_strength = sec_cfg$strength,
            secondary_cycles = sec_cfg$cycles
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = TRUE,
            trend_type = "none",
            trend_strength = 0,
            noise_type = "white",
            ar_phi = NA,
            multi_seasonal = TRUE,
            amplitude_modulation = "none",
            scenario = "multi_seasonal"
          )
        }
      }
    }

    cat(sprintf("  Generated %d multi-seasonal series\n", series_id - base_id))

    # -----------------------------------------------------------------------
    # Part 4: Amplitude modulation
    # -----------------------------------------------------------------------
    cat("Part 4: Amplitude modulation...\n")
    base_id <- series_id

    modulation_types <- c("linear_growth", "linear_decay", "emergence", "periodic")

    for (s_str in c(0.5, 0.7, 1.0)) {
      for (mod_type in modulation_types) {
        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            amplitude_modulation = mod_type
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = TRUE,
            trend_type = "none",
            trend_strength = 0,
            noise_type = "white",
            ar_phi = NA,
            multi_seasonal = FALSE,
            amplitude_modulation = mod_type,
            scenario = "amplitude_modulated"
          )
        }
      }
    }

    cat(sprintf("  Generated %d amplitude-modulated series\n", series_id - base_id))

    # -----------------------------------------------------------------------
    # Part 5: Outliers
    # -----------------------------------------------------------------------
    cat("Part 5: Outlier scenarios...\n")
    base_id <- series_id

    for (s_str in c(0, 0.3, 0.5, 1.0)) {
      for (i in 1:n_per_config) {
        series_id <- series_id + 1

        series <- generate_single_series(
          n_points = n_points,
          seasonal_strength = s_str,
          noise_type = "outliers"
        )

        all_series[[series_id]] <- series
        all_metadata[[series_id]] <- data.frame(
          id = series_id,
          seasonal_strength = s_str,
          is_seasonal = s_str >= 0.2,
          trend_type = "none",
          trend_strength = 0,
          noise_type = "outliers",
          ar_phi = NA,
          multi_seasonal = FALSE,
          amplitude_modulation = "none",
          scenario = "outliers"
        )
      }
    }

    cat(sprintf("  Generated %d outlier series\n", series_id - base_id))

    # -----------------------------------------------------------------------
    # Part 6: Combined challenges (trend + red noise, trend + outliers)
    # -----------------------------------------------------------------------
    cat("Part 6: Combined challenges...\n")
    base_id <- series_id

    for (s_str in c(0, 0.5, 1.0)) {
      for (t_type in c("linear", "quadratic", "slow_sine")) {
        # Trend + red noise
        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            trend_type = t_type,
            trend_strength = 1.0,
            noise_type = "red",
            ar_phi = 0.7
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = s_str >= 0.2,
            trend_type = t_type,
            trend_strength = 1.0,
            noise_type = "red",
            ar_phi = 0.7,
            multi_seasonal = FALSE,
            amplitude_modulation = "none",
            scenario = "combined_trend_rednoise"
          )
        }

        # Trend + outliers
        for (i in 1:n_per_config) {
          series_id <- series_id + 1

          series <- generate_single_series(
            n_points = n_points,
            seasonal_strength = s_str,
            trend_type = t_type,
            trend_strength = 1.0,
            noise_type = "outliers"
          )

          all_series[[series_id]] <- series
          all_metadata[[series_id]] <- data.frame(
            id = series_id,
            seasonal_strength = s_str,
            is_seasonal = s_str >= 0.2,
            trend_type = t_type,
            trend_strength = 1.0,
            noise_type = "outliers",
            ar_phi = NA,
            multi_seasonal = FALSE,
            amplitude_modulation = "none",
            scenario = "combined_trend_outliers"
          )
        }
      }
    }

    cat(sprintf("  Generated %d combined challenge series\n", series_id - base_id))
  }

  # Combine metadata
  metadata <- do.call(rbind, all_metadata)
  rownames(metadata) <- NULL

  # Convert logical to factor for classification
  metadata$is_seasonal <- factor(metadata$is_seasonal, levels = c(FALSE, TRUE),
                                 labels = c("non_seasonal", "seasonal"))
  metadata$trend_type <- factor(metadata$trend_type)
  metadata$noise_type <- factor(metadata$noise_type)
  metadata$scenario <- factor(metadata$scenario)

  cat(sprintf("\nTotal: %d time series generated\n", series_id))
  cat(sprintf("  Seasonal: %d (%.1f%%)\n",
              sum(metadata$is_seasonal == "seasonal"),
              100 * mean(metadata$is_seasonal == "seasonal")))
  cat(sprintf("  Non-seasonal: %d (%.1f%%)\n",
              sum(metadata$is_seasonal == "non_seasonal"),
              100 * mean(metadata$is_seasonal == "non_seasonal")))

  list(
    series = all_series,
    metadata = metadata,
    t = seq(0, 1, length.out = n_points),
    n_points = n_points
  )
}

# =============================================================================
# Feature Extraction (for use with classifiers)
# =============================================================================

#' Extract time series features for classification
#'
#' Extracts a comprehensive set of features from each time series that can
#' be used for training ML classifiers.
#'
#' @param data Output from generate_training_dataset()
#' @param period Expected seasonal period (in time units, e.g., 0.2 for 5 cycles)
#'
#' @return Data frame with features for each time series
#'
#' @export
extract_features <- function(data, period = 0.2) {
  n_series <- length(data$series)
  t <- data$t
  n_points <- data$n_points

  cat(sprintf("Extracting features from %d series...\n", n_series))

  # Pre-allocate feature matrix
  features <- data.frame(
    id = integer(n_series),
    # Variance-based features
    variance = numeric(n_series),
    seasonal_strength_var = numeric(n_series),
    # Spectral features
    spectral_entropy = numeric(n_series),
    spectral_strength = numeric(n_series),
    peak_frequency = numeric(n_series),
    fft_confidence = numeric(n_series),
    # ACF features
    acf_lag1 = numeric(n_series),
    acf_seasonal = numeric(n_series),
    acf_decay_rate = numeric(n_series),
    # Trend features
    linear_trend_coef = numeric(n_series),
    linearity = numeric(n_series),
    # Stability features
    stability = numeric(n_series),
    lumpiness = numeric(n_series),
    # Crossing features
    zero_crossings = numeric(n_series),
    mean_crossings = numeric(n_series),
    # Shape features
    skewness = numeric(n_series),
    kurtosis = numeric(n_series),
    # Outlier features
    outlier_score = numeric(n_series)
  )

  for (i in seq_len(n_series)) {
    if (i %% 500 == 0) cat(sprintf("  Processing series %d/%d\n", i, n_series))

    y <- data$series[[i]]
    n <- length(y)

    features$id[i] <- i

    # --- Variance-based features ---
    features$variance[i] <- var(y)

    # Detrend for seasonal strength
    y_detrended <- y - predict(lm(y ~ t))
    period_points <- round(period * n)

    if (period_points > 1 && period_points < n/2) {
      # Compute seasonal component via moving average
      seasonal_idx <- rep(1:period_points, length.out = n)
      seasonal_means <- tapply(y_detrended, seasonal_idx, mean)
      seasonal_component <- seasonal_means[seasonal_idx]
      residual <- y_detrended - seasonal_component

      var_detrended <- var(y_detrended)
      var_residual <- var(residual)

      if (var_detrended > 0) {
        features$seasonal_strength_var[i] <- max(0, 1 - var_residual / var_detrended)
      }
    }

    # --- Spectral features ---
    # FFT
    fft_result <- fft(y - mean(y))
    power <- Mod(fft_result[1:(n %/% 2)])^2
    freqs <- (0:(n %/% 2 - 1)) / n

    # Normalize power to probability distribution
    power_norm <- power / sum(power)
    power_norm[power_norm == 0] <- 1e-10

    # Spectral entropy (higher = more uniform spectrum = less seasonal)
    features$spectral_entropy[i] <- -sum(power_norm * log(power_norm)) / log(length(power_norm))

    # Peak frequency and confidence
    peak_idx <- which.max(power[-1]) + 1  # Exclude DC
    features$peak_frequency[i] <- freqs[peak_idx]
    features$fft_confidence[i] <- max(power[-1]) / mean(power[-1])

    # Spectral strength at expected seasonal frequency
    expected_freq <- 1 / period
    freq_idx <- which.min(abs(freqs - expected_freq))
    freq_window <- max(1, freq_idx - 2):min(length(power), freq_idx + 2)
    features$spectral_strength[i] <- sum(power[freq_window]) / sum(power)

    # --- ACF features ---
    acf_vals <- acf(y, lag.max = min(n - 1, period_points * 3), plot = FALSE)$acf[-1]

    features$acf_lag1[i] <- acf_vals[1]

    if (period_points <= length(acf_vals)) {
      features$acf_seasonal[i] <- acf_vals[period_points]
    }

    # ACF decay rate (how fast autocorrelation decays)
    positive_acf <- which(acf_vals > 0.1)
    if (length(positive_acf) > 0) {
      features$acf_decay_rate[i] <- 1 / max(positive_acf)
    } else {
      features$acf_decay_rate[i] <- 1
    }

    # --- Trend features ---
    lm_fit <- lm(y ~ t)
    features$linear_trend_coef[i] <- coef(lm_fit)[2]

    # Trend strength (R-squared of linear fit)
    features$linearity[i] <- summary(lm_fit)$r.squared

    # --- Stability and lumpiness ---
    # Split into windows and compute variance of variances
    n_windows <- 10
    window_size <- n %/% n_windows
    if (window_size > 2) {
      window_vars <- sapply(1:n_windows, function(w) {
        idx <- ((w-1) * window_size + 1):min(w * window_size, n)
        var(y[idx])
      })
      features$stability[i] <- var(window_vars) / mean(window_vars)^2
      features$lumpiness[i] <- var(window_vars)
    }

    # --- Crossing features ---
    features$zero_crossings[i] <- sum(diff(sign(y)) != 0) / n
    features$mean_crossings[i] <- sum(diff(sign(y - mean(y))) != 0) / n

    # --- Shape features ---
    y_centered <- y - mean(y)
    y_sd <- sd(y)
    if (y_sd > 0) {
      features$skewness[i] <- mean((y_centered / y_sd)^3)
      features$kurtosis[i] <- mean((y_centered / y_sd)^4) - 3
    }

    # --- Outlier score ---
    # MAD-based outlier detection
    y_mad <- mad(y, constant = 1)
    if (y_mad > 0) {
      features$outlier_score[i] <- mean(abs(y - median(y)) / y_mad > 3)
    }
  }

  cat("Feature extraction complete.\n")
  features
}

# =============================================================================
# Convenience function to generate full training data
# =============================================================================

#' Generate training data with features
#'
#' Combines data generation and feature extraction into one call.
#'
#' @param n_per_config Number of series per configuration
#' @param n_points Number of time points
#' @param seed Random seed
#' @param period Expected seasonal period
#'
#' @return List with series, metadata, features, and combined data frame
#'
#' @export
generate_training_data_with_features <- function(
  n_per_config = 10,
  n_points = 120,
  seed = 42,
  period = 0.2
) {
  # Generate series
  data <- generate_training_dataset(
    n_per_config = n_per_config,
    n_points = n_points,
    seed = seed
  )

  # Extract features
  features <- extract_features(data, period = period)

  # Combine features with metadata for ML
  training_df <- merge(features, data$metadata, by = "id")

  list(
    series = data$series,
    metadata = data$metadata,
    features = features,
    training_df = training_df,
    t = data$t
  )
}

# =============================================================================
# Example usage
# =============================================================================

if (FALSE) {
  # Generate small dataset for testing
  data <- generate_training_dataset(n_per_config = 5, seed = 42)

  # Check distribution of scenarios
  table(data$metadata$scenario)

  # Extract features
  features <- extract_features(data)

  # Combine for ML
  training_df <- merge(features, data$metadata, by = "id")

  # Train a simple classifier (requires randomForest package)
  if (requireNamespace("randomForest", quietly = TRUE)) {
    library(randomForest)

    # Select feature columns
    feature_cols <- setdiff(names(features), "id")

    # Train
    rf_model <- randomForest(
      x = training_df[, feature_cols],
      y = training_df$is_seasonal,
      ntree = 100
    )

    print(rf_model)
    importance(rf_model)
  }
}
