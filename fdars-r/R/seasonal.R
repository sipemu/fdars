#' Seasonal Analysis Functions for Functional Data
#'
#' Functions for analyzing seasonal patterns in functional data including
#' period estimation, peak detection, seasonal strength measurement, and
#' detection of seasonality changes.

# ==============================================================================
# Period Estimation
# ==============================================================================

#' Estimate Seasonal Period using FFT
#'
#' Estimates the dominant period in functional data using Fast Fourier Transform
#' and periodogram analysis.
#'
#' @param fdataobj An fdata object.
#' @param method Method for period estimation: "fft" (Fast Fourier Transform,
#'   default) or "acf" (autocorrelation function).
#' @param max_lag Maximum lag for ACF method. Default: half the series length.
#' @param detrend_method Detrending method to apply before period estimation:
#' \describe{
#'   \item{"none"}{No detrending (default)}
#'   \item{"linear"}{Remove linear trend}
#'   \item{"auto"}{Automatic AIC-based selection of detrending method}
#' }
#'
#' @return A list with components:
#' \describe{
#'   \item{period}{Estimated period}
#'   \item{frequency}{Dominant frequency (1/period)}
#'   \item{power}{Power at the dominant frequency}
#'   \item{confidence}{Confidence measure (ratio of peak power to mean power)}
#' }
#'
#' @details
#' The function computes the periodogram of the mean curve and finds the
#' frequency with maximum power. The confidence measure indicates how
#' pronounced the dominant frequency is relative to the background.
#'
#' For data with trends, the detrend_method parameter can significantly
#' improve period estimation accuracy. Strong trends can mask the true
#' seasonal period.
#'
#' @export
#' @examples
#' # Generate seasonal data with period = 2
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2) + rnorm(200, sd = 0.1), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Estimate period
#' result <- estimate.period(fd, method = "fft")
#' print(result$period)  # Should be close to 2
#'
#' # With trend - detrending improves estimation
#' X_trend <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
#' fd_trend <- fdata(X_trend, argvals = t)
#' result <- estimate.period(fd_trend, detrend_method = "linear")
estimate.period <- function(fdataobj, method = c("fft", "acf"),
                            max_lag = NULL,
                            detrend_method = c("none", "linear", "auto")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("estimate.period not yet implemented for 2D functional data")
  }

  method <- match.arg(method)
  detrend_method <- match.arg(detrend_method)

  # Apply detrending if requested
  if (detrend_method != "none") {
    fdataobj <- detrend(fdataobj, method = detrend_method)
  }

  if (method == "fft") {
    result <- .Call("wrap__seasonal_estimate_period_fft",
                    fdataobj$data, fdataobj$argvals)
  } else {
    if (is.null(max_lag)) {
      max_lag <- as.integer(ncol(fdataobj$data) / 2)
    }
    result <- .Call("wrap__seasonal_estimate_period_acf",
                    fdataobj$data, fdataobj$argvals, as.integer(max_lag))
  }

  class(result) <- "period_estimate"
  result
}

#' @export
print.period_estimate <- function(x, ...) {
  cat("Period Estimate\n")
  cat("---------------\n")
  cat(sprintf("Period:     %.4f\n", x$period))
  cat(sprintf("Frequency:  %.4f\n", x$frequency))
  cat(sprintf("Power:      %.4f\n", x$power))
  cat(sprintf("Confidence: %.4f\n", x$confidence))
  invisible(x)
}

# ==============================================================================
# Multiple Period Detection
# ==============================================================================

#' Detect Multiple Concurrent Periods
#'
#' Detects multiple periodicities in functional data using iterative residual
#' subtraction. At each iteration, the dominant period is detected using FFT,
#' its sinusoidal component is subtracted, and the process repeats on the
#' residual until stopping criteria are met.
#'
#' @param fdataobj An fdata object.
#' @param max_periods Maximum number of periods to detect. Default: 3.
#' @param min_confidence Minimum FFT confidence to continue detection.
#'   Default: 0.5.
#' @param min_strength Minimum seasonal strength to continue detection.
#'   Default: 0.2.
#' @param detrend_method Detrending method to apply before period detection:
#' \describe{
#'   \item{"auto"}{Automatic AIC-based selection of detrending method (default)}
#'   \item{"none"}{No detrending}
#'   \item{"linear"}{Remove linear trend}
#' }
#'
#' @return A list with components:
#' \describe{
#'   \item{periods}{Numeric vector of detected periods}
#'   \item{confidence}{FFT confidence for each period}
#'   \item{strength}{Seasonal strength for each period}
#'   \item{amplitude}{Amplitude of the sinusoidal component}
#'   \item{phase}{Phase of the sinusoidal component (radians)}
#'   \item{n_periods}{Number of periods detected}
#' }
#'
#' @details
#' The function uses two stopping criteria:
#' \itemize{
#'   \item FFT confidence: How prominent the dominant frequency is
#'   \item Seasonal strength: How much variance is explained by the periodicity
#' }
#'
#' Both must exceed their thresholds for detection to continue. Higher thresholds
#' result in fewer (but more reliable) detected periods.
#'
#' Periods are detected in order of amplitude (FFT power), not period length.
#' A weak yearly cycle will be detected after a strong weekly cycle.
#'
#' Trends can interfere with period detection. The default "auto" detrending
#' automatically selects an appropriate method to remove trends.
#'
#' @seealso \code{\link{estimate.period}} for single period estimation,
#'   \code{\link{detrend}} for standalone detrending
#'
#' @export
#' @examples
#' # Signal with two periods: 2 and 7
#' t <- seq(0, 20, length.out = 400)
#' X <- sin(2 * pi * t / 2) + 0.6 * sin(2 * pi * t / 7)
#' fd <- fdata(matrix(X, nrow = 1), argvals = t)
#'
#' # Detect multiple periods
#' result <- detect.periods(fd, max_periods = 3)
#' print(result$periods)  # Should find approximately 2 and 7
#' print(result$n_periods)  # Should be 2
detect.periods <- function(fdataobj, max_periods = 3,
                                     min_confidence = 0.5,
                                     min_strength = 0.2,
                                     detrend_method = c("auto", "none", "linear")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect.periods not yet implemented for 2D functional data")
  }

  detrend_method <- match.arg(detrend_method)

  # Apply detrending if requested
  if (detrend_method != "none") {
    fdataobj <- detrend(fdataobj, method = detrend_method)
  }

  # Pure R implementation using iterative residual subtraction
  periods <- numeric(0)
  confidence <- numeric(0)
  strength <- numeric(0)
  amplitude <- numeric(0)

  residual <- fdataobj
  tt <- fdataobj$argvals

  for (i in seq_len(max_periods)) {
    # Estimate dominant period
    est <- estimate.period(residual, method = "fft")
    if (est$confidence < min_confidence) break

    # Check seasonal strength
    ss <- seasonal.strength(residual, period = est$period)
    if (ss < min_strength) break

    # Compute amplitude via Fourier coefficients
    omega <- 2 * pi / est$period
    cos_comp <- cos(omega * tt)
    sin_comp <- sin(omega * tt)
    y <- residual$data[1, ]
    a <- 2 * mean(y * cos_comp)
    b <- 2 * mean(y * sin_comp)
    amp <- sqrt(a^2 + b^2)

    # Store results
    periods <- c(periods, est$period)
    confidence <- c(confidence, est$confidence)
    strength <- c(strength, ss)
    amplitude <- c(amplitude, amp)

    # Subtract fitted sinusoid from residual
    fitted <- a * cos_comp + b * sin_comp
    residual$data <- residual$data - matrix(fitted, nrow = 1)
  }

  result <- list(
    periods = periods,
    confidence = confidence,
    strength = strength,
    amplitude = amplitude,
    n_periods = length(periods)
  )

  class(result) <- "multiple_periods"
  result
}

#' @export
print.multiple_periods <- function(x, ...) {
  cat("Multiple Period Detection\n")
  cat("-------------------------\n")
  cat(sprintf("Periods detected: %d\n\n", x$n_periods))

  if (x$n_periods > 0) {
    for (i in seq_len(x$n_periods)) {
      cat(sprintf("Period %d: %.3f (confidence=%.3f, strength=%.3f, amplitude=%.3f)\n",
                  i, x$periods[i], x$confidence[i], x$strength[i], x$amplitude[i]))
    }
  } else {
    cat("No significant periods detected.\n")
  }
  invisible(x)
}

# ==============================================================================
# Peak Detection
# ==============================================================================

#' Detect Peaks in Functional Data
#'
#' Detects local maxima (peaks) in functional data using derivative
#' zero-crossings. Returns peak times, values, and prominence measures.
#'
#' @param fdataobj An fdata object.
#' @param min_distance Minimum time between peaks. Default: NULL (no constraint).
#' @param min_prominence Minimum prominence for a peak (0-1 scale). Peaks with
#'   lower prominence are filtered out. Default: NULL (no filter).
#' @param smooth_first Logical. If TRUE, apply Fourier basis smoothing before
#'   peak detection. Recommended for noisy data. Default: FALSE.
#' @param smooth_nbasis Number of Fourier basis functions for smoothing.
#'   If NULL and smooth_first=TRUE, uses GCV to automatically select
#'   optimal nbasis (range 5-25). Default: NULL (auto).
#' @param detrend_method Detrending method to apply before peak detection:
#' \describe{
#'   \item{"none"}{No detrending (default)}
#'   \item{"linear"}{Remove linear trend}
#'   \item{"auto"}{Automatic AIC-based selection of detrending method}
#' }
#'
#' @return A list with components:
#' \describe{
#'   \item{peaks}{List of data frames, one per curve, with columns: time, value, prominence}
#'   \item{inter_peak_distances}{List of numeric vectors with distances between consecutive peaks}
#'   \item{mean_period}{Mean inter-peak distance across all curves (estimates period)}
#' }
#'
#' @details
#' Peak prominence measures how much a peak stands out from its surroundings.
#' It is computed as the height difference between the peak and the highest
#' of the two minimum values on either side, normalized by the data range.
#'
#' Fourier basis smoothing is ideal for seasonal signals because it naturally
#' captures periodic patterns without introducing boundary artifacts.
#'
#' For data with trends, use detrend_method to remove the trend before
#' detecting peaks. This prevents the trend from affecting peak prominence
#' calculations.
#'
#' @export
#' @examples
#' # Generate data with clear peaks
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Detect peaks
#' peaks <- detect.peaks(fd, min_distance = 1.5)
#' print(peaks$mean_period)  # Should be close to 2
#'
#' # With automatic Fourier smoothing (GCV selects nbasis)
#' peaks_smooth <- detect.peaks(fd, min_distance = 1.5, smooth_first = TRUE)
#'
#' # With detrending for trending data
#' X_trend <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
#' fd_trend <- fdata(X_trend, argvals = t)
#' peaks_det <- detect.peaks(fd_trend, detrend_method = "linear")
detect.peaks <- function(fdataobj, min_distance = NULL, min_prominence = NULL,
                         smooth_first = FALSE, smooth_nbasis = NULL,
                         detrend_method = c("none", "linear", "auto")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect.peaks not yet implemented for 2D functional data")
  }

  detrend_method <- match.arg(detrend_method)

  # Apply detrending if requested
  if (detrend_method != "none") {
    fdataobj <- detrend(fdataobj, method = detrend_method)
  }

  # Handle NULL values - pass as NULL Robj
  min_dist_arg <- if (is.null(min_distance)) NULL else as.double(min_distance)
  min_prom_arg <- if (is.null(min_prominence)) NULL else as.double(min_prominence)
  # NULL smooth_nbasis triggers automatic GCV selection
  nbasis_arg <- if (is.null(smooth_nbasis)) NULL else as.integer(smooth_nbasis)

  result <- .Call("wrap__seasonal_detect_peaks",
                  fdataobj$data, fdataobj$argvals,
                  min_dist_arg, min_prom_arg,
                  as.logical(smooth_first), nbasis_arg)

  # Convert peaks list to data frames
  result$peaks <- lapply(result$peaks, function(p) {
    if (length(p$time) == 0) {
      data.frame(time = numeric(0), value = numeric(0), prominence = numeric(0))
    } else {
      data.frame(time = p$time, value = p$value, prominence = p$prominence)
    }
  })

  class(result) <- "peak_detection"
  result
}

#' @export
print.peak_detection <- function(x, ...) {
  n_curves <- length(x$peaks)
  total_peaks <- sum(sapply(x$peaks, nrow))
  cat("Peak Detection Result\n")
  cat("---------------------\n")
  cat(sprintf("Number of curves:  %d\n", n_curves))
  cat(sprintf("Total peaks found: %d\n", total_peaks))
  cat(sprintf("Mean period:       %.4f\n", x$mean_period))
  invisible(x)
}

# ==============================================================================
# Seasonal Strength
# ==============================================================================

#' Measure Seasonal Strength
#'
#' Computes the strength of seasonality in functional data. Values range from
#' 0 (no seasonality) to 1 (pure seasonal signal).
#'
#' @param fdataobj An fdata object.
#' @param period Known or estimated period. If NULL, period is estimated
#'   automatically using FFT.
#' @param method Method for computing strength:
#'   \describe{
#'     \item{"variance"}{Variance decomposition: Var(seasonal) / Var(total)}
#'     \item{"spectral"}{Spectral: power at seasonal frequencies / total power}
#'   }
#' @param n_harmonics Number of Fourier harmonics to use (for variance method).
#'   Default: 3.
#' @param detrend_method Detrending method to apply before computing strength:
#' \describe{
#'   \item{"none"}{No detrending (default)}
#'   \item{"linear"}{Remove linear trend}
#'   \item{"auto"}{Automatic AIC-based selection of detrending method}
#' }
#'
#' @return A numeric value between 0 and 1 representing seasonal strength.
#'
#' @details
#' The variance method decomposes the signal into a seasonal component
#' (using Fourier basis with the specified period) and computes the
#' proportion of variance explained by the seasonal component.
#'
#' The spectral method computes the proportion of total spectral power
#' that falls at the seasonal frequency and its harmonics.
#'
#' Trends can artificially lower the seasonal strength measure by
#' contributing non-seasonal variance. Use detrend_method to remove
#' trends before computing strength.
#'
#' @export
#' @examples
#' # Pure seasonal signal
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2), nrow = 1)
#' fd_seasonal <- fdata(X, argvals = t)
#' seasonal.strength(fd_seasonal, period = 2)  # Should be close to 1
#'
#' # Pure noise
#' X_noise <- matrix(rnorm(200), nrow = 1)
#' fd_noise <- fdata(X_noise, argvals = t)
#' seasonal.strength(fd_noise, period = 2)  # Should be close to 0
#'
#' # Trending data - detrending improves strength estimate
#' X_trend <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
#' fd_trend <- fdata(X_trend, argvals = t)
#' seasonal.strength(fd_trend, period = 2, detrend_method = "linear")
seasonal.strength <- function(fdataobj, period = NULL,
                              method = c("variance", "spectral"),
                              n_harmonics = 3,
                              detrend_method = c("none", "linear", "auto")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("seasonal.strength not yet implemented for 2D functional data")
  }

  method <- match.arg(method)
  detrend_method <- match.arg(detrend_method)

  # Apply detrending if requested
  if (detrend_method != "none") {
    fdataobj <- detrend(fdataobj, method = detrend_method)
  }

  if (is.null(period)) {
    period <- estimate.period(fdataobj)$period
  }

  if (method == "variance") {
    result <- .Call("wrap__seasonal_strength_variance",
                    fdataobj$data, fdataobj$argvals,
                    as.double(period), as.integer(n_harmonics))
  } else {
    result <- .Call("wrap__seasonal_strength_spectral",
                    fdataobj$data, fdataobj$argvals, as.double(period))
  }

  result
}

#' Time-Varying Seasonal Strength
#'
#' Computes seasonal strength at each time point using a sliding window,
#' allowing detection of how seasonality changes over time.
#'
#' @param fdataobj An fdata object.
#' @param period Known or estimated period.
#' @param window_size Width of the sliding window. Recommended: 2 * period.
#' @param method Method for computing strength: "variance" or "spectral".
#'
#' @return An fdata object containing the time-varying seasonal strength curve.
#'
#' @export
#' @examples
#' # Signal that transitions from seasonal to non-seasonal
#' t <- seq(0, 20, length.out = 400)
#' X <- ifelse(t < 10, sin(2 * pi * t / 2), rnorm(length(t[t >= 10]), sd = 0.5))
#' X <- matrix(X, nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute time-varying strength
#' ss <- seasonal.strength.curve(fd, period = 2, window_size = 4)
#' # plot(ss)  # Shows strength declining around t = 10
seasonal.strength.curve <- function(fdataobj, period, window_size = NULL,
                                    method = c("variance", "spectral")) {
 if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("seasonal.strength.curve not yet implemented for 2D functional data")
  }

  method <- match.arg(method)

  if (is.null(window_size)) {
    window_size <- 2 * period
  }

  strength <- .Call("wrap__seasonal_strength_windowed",
                    fdataobj$data, fdataobj$argvals,
                    as.double(period), as.double(window_size), method)

  fdata(matrix(strength, nrow = 1), argvals = fdataobj$argvals,
        rangeval = fdataobj$rangeval)
}

# ==============================================================================
# Seasonality Change Detection
# ==============================================================================

#' Detect Changes in Seasonality
#'
#' Detects points in time where seasonality starts (onset) or stops (cessation)
#' by monitoring time-varying seasonal strength.
#'
#' @param fdataobj An fdata object.
#' @param period Known or estimated period.
#' @param threshold Seasonal strength threshold for classification (0-1).
#'   Above threshold = seasonal, below = non-seasonal. Default: 0.3.
#' @param window_size Width of sliding window for strength estimation.
#'   Default: 2 * period.
#' @param min_duration Minimum duration to confirm a change. Prevents
#'   detection of spurious short-term fluctuations. Default: period.
#'
#' @return A list with components:
#' \describe{
#'   \item{change_points}{Data frame with columns: time, type ("onset" or "cessation"),
#'     strength_before, strength_after}
#'   \item{strength_curve}{Time-varying seasonal strength used for detection}
#' }
#'
#' @export
#' @examples
#' # Signal that starts non-seasonal, becomes seasonal, then stops
#' t <- seq(0, 30, length.out = 600)
#' X <- ifelse(t < 10, rnorm(sum(t < 10), sd = 0.3),
#'             ifelse(t < 20, sin(2 * pi * t[t >= 10 & t < 20] / 2),
#'                    rnorm(sum(t >= 20), sd = 0.3)))
#' X <- matrix(X, nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Detect changes
#' changes <- detect.seasonality.changes(fd, period = 2)
#' print(changes$change_points)  # Should show onset ~10, cessation ~20
detect.seasonality.changes <- function(fdataobj, period, threshold = 0.3,
                                       window_size = NULL, min_duration = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect.seasonality.changes not yet implemented for 2D functional data")
  }

  if (is.null(window_size)) {
    window_size <- 2 * period
  }

  if (is.null(min_duration)) {
    min_duration <- period
  }

  result <- .Call("wrap__seasonal_detect_changes",
                  fdataobj$data, fdataobj$argvals,
                  as.double(period), as.double(threshold),
                  as.double(window_size), as.double(min_duration))

  # Convert to data frame
  if (length(result$change_times) > 0) {
    result$change_points <- data.frame(
      time = result$change_times,
      type = result$change_types,
      strength_before = result$strength_before,
      strength_after = result$strength_after
    )
  } else {
    result$change_points <- data.frame(
      time = numeric(0),
      type = character(0),
      strength_before = numeric(0),
      strength_after = numeric(0)
    )
  }

  # Remove intermediate vectors
  result$change_times <- NULL
  result$change_types <- NULL
  result$strength_before <- NULL
  result$strength_after <- NULL

  class(result) <- "seasonality_changes"
  result
}

#' @export
print.seasonality_changes <- function(x, ...) {
  cat("Seasonality Change Detection\n")
  cat("----------------------------\n")
  if (nrow(x$change_points) == 0) {
    cat("No changes detected.\n")
  } else {
    cat(sprintf("Number of changes: %d\n\n", nrow(x$change_points)))
    print(x$change_points)
  }
  invisible(x)
}

# ==============================================================================
# Instantaneous Period
# ==============================================================================

#' Estimate Instantaneous Period
#'
#' For signals with time-varying frequency (drifting period), estimates the
#' instantaneous period at each time point using the Hilbert transform.
#'
#' @param fdataobj An fdata object.
#'
#' @return A list with fdata objects:
#' \describe{
#'   \item{period}{Instantaneous period at each time point}
#'   \item{frequency}{Instantaneous frequency at each time point}
#'   \item{amplitude}{Instantaneous amplitude (envelope) at each time point}
#' }
#'
#' @details
#' The Hilbert transform is used to compute the analytic signal, from which
#' the instantaneous phase is extracted. The derivative of the phase gives
#' the instantaneous frequency, and 1/frequency gives the period.
#'
#' This is particularly useful for signals where the period is not constant,
#' such as circadian rhythms with drift or frequency-modulated signals.
#'
#' @export
#' @examples
#' # Chirp signal with increasing frequency
#' t <- seq(0, 10, length.out = 500)
#' freq <- 0.5 + 0.1 * t  # Frequency increases from 0.5 to 1.5
#' X <- matrix(sin(2 * pi * cumsum(freq) * diff(c(0, t))), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Estimate instantaneous period
#' inst <- instantaneous.period(fd)
#' # plot(inst$period)  # Shows decreasing period over time
instantaneous.period <- function(fdataobj) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("instantaneous.period not yet implemented for 2D functional data")
  }

  result <- .Call("wrap__seasonal_instantaneous_period",
                  fdataobj$data, fdataobj$argvals)

  # Convert to fdata objects
  list(
    period = fdata(matrix(result$period, nrow = 1),
                   argvals = fdataobj$argvals, rangeval = fdataobj$rangeval),
    frequency = fdata(matrix(result$frequency, nrow = 1),
                      argvals = fdataobj$argvals, rangeval = fdataobj$rangeval),
    amplitude = fdata(matrix(result$amplitude, nrow = 1),
                      argvals = fdataobj$argvals, rangeval = fdataobj$rangeval)
  )
}

# ==============================================================================
# Peak Timing Variability Analysis
# ==============================================================================

#' Analyze Peak Timing Variability
#'
#' For short series (e.g., 3-5 years of yearly data), this function detects
#' one peak per cycle and analyzes how peak timing varies between cycles.
#' Uses Fourier basis smoothing for peak detection.
#'
#' @param fdataobj An fdata object.
#' @param period Known period (e.g., 365 for daily data with yearly seasonality).
#' @param smooth_nbasis Number of Fourier basis functions for smoothing.
#'   If NULL, uses GCV for automatic selection (range 5-25). Default: NULL.
#'
#' @return A list with components:
#' \describe{
#'   \item{peak_times}{Vector of peak times}
#'   \item{peak_values}{Vector of peak values}
#'   \item{normalized_timing}{Position within cycle (0-1 scale)}
#'   \item{mean_timing}{Mean normalized timing}
#'   \item{std_timing}{Standard deviation of normalized timing}
#'   \item{range_timing}{Range of normalized timing (max - min)}
#'   \item{variability_score}{Variability score (0 = stable, 1 = highly variable)}
#'   \item{timing_trend}{Linear trend in timing (positive = peaks getting later)}
#'   \item{cycle_indices}{Cycle indices (1-indexed)}
#' }
#'
#' @details
#' The variability score is computed as std_timing / 0.1, capped at 1.
#' A score > 0.5 suggests peaks are shifting substantially between cycles.
#' The timing_trend indicates if peaks are systematically moving earlier
#' or later over time.
#'
#' Fourier basis smoothing is ideal for seasonal signals because it naturally
#' captures periodic patterns.
#'
#' @export
#' @examples
#' # 5 years of yearly data where peak shifts
#' t <- seq(0, 5, length.out = 365 * 5)
#' periods <- c(1, 1, 1, 1, 1)  # 5 complete years
#' # Peaks shift: March (0.2), April (0.3), May (0.4), April (0.3), March (0.2)
#' peak_phases <- c(0.2, 0.3, 0.4, 0.3, 0.2)
#' X <- sin(2 * pi * t + rep(peak_phases, each = 365))
#' fd <- fdata(matrix(X, nrow = 1), argvals = t)
#'
#' result <- analyze.peak.timing(fd, period = 1)
#' print(result$variability_score)  # Shows timing variability
analyze.peak.timing <- function(fdataobj, period, smooth_nbasis = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("analyze.peak.timing not yet implemented for 2D functional data")
  }

  nbasis_arg <- if (is.null(smooth_nbasis)) NULL else as.integer(smooth_nbasis)

  result <- .Call("wrap__seasonal_analyze_peak_timing",
                  fdataobj$data, fdataobj$argvals,
                  as.double(period), nbasis_arg)

  class(result) <- "peak_timing"
  result
}

#' @export
print.peak_timing <- function(x, ...) {
  cat("Peak Timing Variability Analysis\n")
  cat("---------------------------------\n")
  cat(sprintf("Number of peaks: %d\n", length(x$peak_times)))
  cat(sprintf("Mean timing:     %.4f\n", x$mean_timing))
  cat(sprintf("Std timing:      %.4f\n", x$std_timing))
  cat(sprintf("Range timing:    %.4f\n", x$range_timing))
  cat(sprintf("Variability:     %.4f", x$variability_score))
  if (x$variability_score > 0.5) {
    cat(" (HIGH)\n")
  } else if (x$variability_score > 0.2) {
    cat(" (moderate)\n")
  } else {
    cat(" (low)\n")
  }
  cat(sprintf("Timing trend:    %.4f\n", x$timing_trend))
  invisible(x)
}

# ==============================================================================
# Seasonality Classification
# ==============================================================================

#' Classify Seasonality Type
#'
#' Classifies the type of seasonality in functional data. Particularly useful
#' for short series (3-5 years) to identify stable vs variable timing patterns.
#'
#' @param fdataobj An fdata object.
#' @param period Known seasonal period.
#' @param strength_threshold Threshold for seasonal/non-seasonal (default: 0.3).
#' @param timing_threshold Max std of normalized timing for "stable" (default: 0.05).
#'
#' @return A list with components:
#' \describe{
#'   \item{is_seasonal}{Logical: is the series seasonal overall?}
#'   \item{has_stable_timing}{Logical: is peak timing stable across cycles?}
#'   \item{timing_variability}{Timing variability score (0-1)}
#'   \item{seasonal_strength}{Overall seasonal strength}
#'   \item{cycle_strengths}{Per-cycle seasonal strength}
#'   \item{weak_seasons}{Indices of weak/missing seasons (0-indexed)}
#'   \item{classification}{One of: "StableSeasonal", "VariableTiming",
#'     "IntermittentSeasonal", "NonSeasonal"}
#'   \item{peak_timing}{Peak timing analysis (if peaks detected)}
#' }
#'
#' @details
#' Classification types:
#' \itemize{
#'   \item StableSeasonal: Regular peaks with consistent timing
#'   \item VariableTiming: Regular peaks but timing shifts between cycles
#'   \item IntermittentSeasonal: Some cycles seasonal, some not
#'   \item NonSeasonal: No clear seasonality
#' }
#'
#' @export
#' @examples
#' # Pure seasonal signal
#' t <- seq(0, 10, length.out = 500)
#' X <- matrix(sin(2 * pi * t / 2), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' result <- classify.seasonality(fd, period = 2)
#' print(result$classification)  # "StableSeasonal"
classify.seasonality <- function(fdataobj, period,
                                  strength_threshold = NULL,
                                  timing_threshold = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("classify.seasonality not yet implemented for 2D functional data")
  }

  str_thresh <- if (is.null(strength_threshold)) NULL else as.double(strength_threshold)
  tim_thresh <- if (is.null(timing_threshold)) NULL else as.double(timing_threshold)

  result <- .Call("wrap__seasonal_classify_seasonality",
                  fdataobj$data, fdataobj$argvals,
                  as.double(period), str_thresh, tim_thresh)

  class(result) <- "seasonality_classification"
  result
}

#' @export
print.seasonality_classification <- function(x, ...) {
  cat("Seasonality Classification\n")
  cat("--------------------------\n")
  cat(sprintf("Classification:   %s\n", x$classification))
  cat(sprintf("Is seasonal:      %s\n", x$is_seasonal))
  cat(sprintf("Stable timing:    %s\n", x$has_stable_timing))
  cat(sprintf("Timing variability: %.4f\n", x$timing_variability))
  cat(sprintf("Seasonal strength:  %.4f\n", x$seasonal_strength))
  if (length(x$weak_seasons) > 0) {
    cat(sprintf("Weak seasons:     %s\n", paste(x$weak_seasons, collapse = ", ")))
  }
  invisible(x)
}

# ==============================================================================
# Automatic Threshold Detection
# ==============================================================================

#' Detect Seasonality Changes with Automatic Threshold
#'
#' Detects points where seasonality starts or stops, using automatic
#' threshold selection instead of a fixed value.
#'
#' @param fdataobj An fdata object.
#' @param period Known seasonal period.
#' @param threshold_method Method for threshold selection:
#' \describe{
#'   \item{"fixed"}{Use threshold_value as fixed threshold}
#'   \item{"percentile"}{Use threshold_value as percentile of strength distribution}
#'   \item{"otsu"}{Use Otsu's method for bimodal separation (default)}
#' }
#' @param threshold_value Value for "fixed" or "percentile" methods.
#' @param window_size Width of sliding window for strength estimation.
#'   Default: 2 * period.
#' @param min_duration Minimum duration to confirm a change.
#'   Default: period.
#'
#' @return A list with components:
#' \describe{
#'   \item{change_points}{Data frame with time, type, strength_before, strength_after}
#'   \item{strength_curve}{Time-varying seasonal strength used for detection}
#'   \item{computed_threshold}{The threshold that was computed/used}
#' }
#'
#' @details
#' Otsu's method automatically finds the optimal threshold for separating
#' seasonal from non-seasonal regions based on the strength distribution.
#' This is particularly useful when you don't know the appropriate threshold
#' for your data.
#'
#' @export
#' @examples
#' # Signal that transitions from seasonal to non-seasonal
#' t <- seq(0, 20, length.out = 400)
#' X <- ifelse(t < 10, sin(2 * pi * t / 2), rnorm(sum(t >= 10), sd = 0.3))
#' X <- matrix(X, nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Detect changes with Otsu threshold
#' changes <- detect.seasonality.changes.auto(fd, period = 2)
#' print(changes$computed_threshold)
detect.seasonality.changes.auto <- function(fdataobj, period,
                                             threshold_method = "otsu",
                                             threshold_value = NULL,
                                             window_size = NULL,
                                             min_duration = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect.seasonality.changes.auto not yet implemented for 2D functional data")
  }

  if (is.null(window_size)) {
    window_size <- 2 * period
  }

  if (is.null(min_duration)) {
    min_duration <- period
  }

  if (is.null(threshold_value)) {
    threshold_value <- switch(threshold_method,
                               "fixed" = 0.3,
                               "percentile" = 20,
                               0.0)  # Not used for Otsu
  }

  result <- .Call("wrap__seasonal_detect_changes_auto",
                  fdataobj$data, fdataobj$argvals,
                  as.double(period), as.character(threshold_method),
                  as.double(threshold_value), as.double(window_size),
                  as.double(min_duration))

  # Convert to data frame
  if (length(result$change_times) > 0) {
    result$change_points <- data.frame(
      time = result$change_times,
      type = result$change_types,
      strength_before = result$strength_before,
      strength_after = result$strength_after
    )
  } else {
    result$change_points <- data.frame(
      time = numeric(0),
      type = character(0),
      strength_before = numeric(0),
      strength_after = numeric(0)
    )
  }

  # Remove intermediate vectors
  result$change_times <- NULL
  result$change_types <- NULL
  result$strength_before <- NULL
  result$strength_after <- NULL

  class(result) <- "seasonality_changes_auto"
  result
}

#' @export
print.seasonality_changes_auto <- function(x, ...) {
  cat("Seasonality Change Detection (Auto Threshold)\n")
  cat("----------------------------------------------\n")
  cat(sprintf("Computed threshold: %.4f\n", x$computed_threshold))
  if (nrow(x$change_points) == 0) {
    cat("No changes detected.\n")
  } else {
    cat(sprintf("Number of changes: %d\n\n", nrow(x$change_points)))
    print(x$change_points)
  }
  invisible(x)
}

# ==============================================================================
# Detrending Functions
# ==============================================================================

#' Remove Trend from Functional Data
#'
#' Removes trend from functional data using various methods. This is useful
#' for preprocessing data before seasonal analysis when the data has a
#' significant trend component.
#'
#' @param fdataobj An fdata object.
#' @param method Detrending method:
#' \describe{
#'   \item{"linear"}{Least squares linear fit (default)}
#'   \item{"polynomial"}{Polynomial regression of specified degree}
#'   \item{"diff1"}{First-order differencing}
#'   \item{"diff2"}{Second-order differencing}
#'   \item{"loess"}{Local polynomial regression (LOESS)}
#'   \item{"auto"}{Automatic selection via AIC}
#' }
#' @param degree Polynomial degree for "polynomial" method. Default: 2.
#' @param bandwidth Bandwidth as fraction of data range for "loess" method.
#'   Default: 0.3.
#' @param return_trend Logical. If TRUE, return both trend and detrended data.
#'   Default: FALSE.
#'
#' @return If return_trend = FALSE, an fdata object with detrended data.
#'   If return_trend = TRUE, a list with components:
#' \describe{
#'   \item{detrended}{fdata object with detrended data}
#'   \item{trend}{fdata object with estimated trend}
#'   \item{method}{Method used for detrending}
#'   \item{rss}{Residual sum of squares per curve}
#' }
#'
#' @details
#' For series with polynomial trends, "linear" or "polynomial" methods are
#' appropriate. For more complex trends, "loess" provides flexibility.
#' The "auto" method compares linear, polynomial (degree 2 and 3), and LOESS,
#' selecting the method with lowest AIC.
#'
#' Differencing methods ("diff1", "diff2") reduce the series length by 1 or 2
#' points respectively. The resulting fdata has correspondingly shorter argvals.
#'
#' @seealso \code{\link{decompose}} for full seasonal decomposition
#'
#' @export
#' @examples
#' # Generate data with linear trend and seasonal component
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Detrend with linear method
#' fd_detrended <- detrend(fd, method = "linear")
#'
#' # Now estimate period on detrended data
#' period <- estimate.period(fd_detrended)
#' print(period$period)  # Should be close to 2
#'
#' # Get both trend and detrended data
#' result <- detrend(fd, method = "linear", return_trend = TRUE)
#' # plot(result$trend)  # Shows the linear trend
detrend <- function(fdataobj,
                    method = c("linear", "polynomial", "diff1", "diff2", "loess", "auto"),
                    degree = 2,
                    bandwidth = 0.3,
                    return_trend = FALSE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detrend not yet implemented for 2D functional data")
  }

  method <- match.arg(method)

  result <- .Call("wrap__seasonal_detrend",
                  fdataobj$data, fdataobj$argvals,
                  as.character(method), as.integer(degree),
                  as.double(bandwidth))

  # Handle differencing methods which reduce series length
  if (method %in% c("diff1", "diff2")) {
    order <- if (method == "diff1") 1 else 2
    new_m <- ncol(fdataobj$data) - order
    new_argvals <- fdataobj$argvals[1:new_m]
    new_rangeval <- c(new_argvals[1], new_argvals[new_m])
  } else {
    new_argvals <- fdataobj$argvals
    new_rangeval <- fdataobj$rangeval
  }

  detrended_fd <- fdata(result$detrended, argvals = new_argvals,
                        rangeval = new_rangeval)

  if (return_trend) {
    trend_fd <- fdata(result$trend, argvals = new_argvals,
                      rangeval = new_rangeval)
    list(
      detrended = detrended_fd,
      trend = trend_fd,
      method = result$method,
      rss = result$rss
    )
  } else {
    detrended_fd
  }
}

#' Seasonal-Trend Decomposition
#'
#' Decomposes functional data into trend, seasonal, and remainder components.
#' Similar to STL (Seasonal-Trend decomposition using LOESS).
#'
#' @param fdataobj An fdata object.
#' @param period Seasonal period. If NULL, estimated automatically using FFT.
#' @param method Decomposition method:
#' \describe{
#'   \item{"additive"}{data = trend + seasonal + remainder (default)}
#'   \item{"multiplicative"}{data = trend * seasonal * remainder}
#' }
#' @param trend_method Method for trend extraction: "loess" or "spline".
#'   Default: "loess".
#' @param bandwidth Bandwidth for trend extraction (fraction of range).
#'   Default: 0.3.
#' @param n_harmonics Number of Fourier harmonics for seasonal component.
#'   Default: 3.
#'
#' @return A list with components:
#' \describe{
#'   \item{trend}{fdata object with trend component}
#'   \item{seasonal}{fdata object with seasonal component}
#'   \item{remainder}{fdata object with remainder/residual}
#'   \item{period}{Period used for decomposition}
#'   \item{method}{Decomposition method ("additive" or "multiplicative")}
#' }
#'
#' @details
#' For additive decomposition: data = trend + seasonal + remainder.
#' The trend is extracted using LOESS or spline smoothing, then the seasonal
#' component is estimated by fitting Fourier harmonics to the detrended data.
#'
#' For multiplicative decomposition: data = trend * seasonal * remainder.
#' This is achieved by log-transforming the data, applying additive
#' decomposition, and back-transforming. Use this when the seasonal amplitude
#' grows with the trend level.
#'
#' @seealso \code{\link{detrend}} for simple trend removal,
#'   \code{\link{seasonal.strength}} for measuring seasonality
#'
#' @export
#' @examples
#' # Additive seasonal pattern
#' t <- seq(0, 20, length.out = 400)
#' X <- matrix(2 + 0.3 * t + sin(2 * pi * t / 2.5), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' result <- decompose(fd, period = 2.5, method = "additive")
#' # plot(result$trend)      # Linear trend
#' # plot(result$seasonal)   # Sinusoidal seasonal
#' # plot(result$remainder)  # Residual noise
#'
#' # Multiplicative pattern (amplitude grows with level)
#' X_mult <- matrix((2 + 0.3 * t) * (1 + 0.3 * sin(2 * pi * t / 2.5)), nrow = 1)
#' fd_mult <- fdata(X_mult, argvals = t)
#'
#' result_mult <- decompose(fd_mult, period = 2.5, method = "multiplicative")
decompose <- function(fdataobj,
                      period = NULL,
                      method = c("additive", "multiplicative"),
                      trend_method = c("loess", "spline"),
                      bandwidth = 0.3,
                      n_harmonics = 3) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("decompose not yet implemented for 2D functional data")
  }

  method <- match.arg(method)
  trend_method <- match.arg(trend_method)

  # Estimate period if not provided
  if (is.null(period)) {
    period <- estimate.period(fdataobj)$period
  }

  result <- .Call("wrap__seasonal_decompose",
                  fdataobj$data, fdataobj$argvals,
                  as.double(period), as.character(method),
                  as.character(trend_method), as.double(bandwidth),
                  as.integer(n_harmonics))

  list(
    trend = fdata(result$trend, argvals = fdataobj$argvals,
                  rangeval = fdataobj$rangeval),
    seasonal = fdata(result$seasonal, argvals = fdataobj$argvals,
                     rangeval = fdataobj$rangeval),
    remainder = fdata(result$remainder, argvals = fdataobj$argvals,
                      rangeval = fdataobj$rangeval),
    period = result$period,
    method = result$method
  )
}

#' @export
print.decomposition <- function(x, ...) {
  cat("Seasonal Decomposition\n")
  cat("----------------------\n")
  cat(sprintf("Method:  %s\n", x$method))
  cat(sprintf("Period:  %.4f\n", x$period))
  cat("\nComponents: trend, seasonal, remainder\n")
  invisible(x)
}
