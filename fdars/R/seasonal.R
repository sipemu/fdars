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
#' @export
#' @examples
#' # Generate seasonal data with period = 2
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2) + rnorm(200, sd = 0.1), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Estimate period
#' result <- estimate_period(fd, method = "fft")
#' print(result$period)  # Should be close to 2
estimate_period <- function(fdataobj, method = c("fft", "acf"),
                            max_lag = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("estimate_period not yet implemented for 2D functional data")
  }

  method <- match.arg(method)

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
#' @param smooth_first Logical. If TRUE, apply P-spline smoothing before
#'   peak detection. Default: FALSE.
#' @param smooth_lambda Smoothing parameter for P-splines. Default: 10.
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
#' @export
#' @examples
#' # Generate data with clear peaks
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Detect peaks
#' peaks <- detect_peaks(fd, min_distance = 1.5)
#' print(peaks$mean_period)  # Should be close to 2
detect_peaks <- function(fdataobj, min_distance = NULL, min_prominence = NULL,
                         smooth_first = FALSE, smooth_lambda = 10) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect_peaks not yet implemented for 2D functional data")
  }

  # Handle NULL values - pass as NULL Robj
  min_dist_arg <- if (is.null(min_distance)) NULL else as.double(min_distance)
  min_prom_arg <- if (is.null(min_prominence)) NULL else as.double(min_prominence)

  result <- .Call("wrap__seasonal_detect_peaks",
                  fdataobj$data, fdataobj$argvals,
                  min_dist_arg, min_prom_arg,
                  as.logical(smooth_first), as.double(smooth_lambda))

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
#' @export
#' @examples
#' # Pure seasonal signal
#' t <- seq(0, 10, length.out = 200)
#' X <- matrix(sin(2 * pi * t / 2), nrow = 1)
#' fd_seasonal <- fdata(X, argvals = t)
#' seasonal_strength(fd_seasonal, period = 2)  # Should be close to 1
#'
#' # Pure noise
#' X_noise <- matrix(rnorm(200), nrow = 1)
#' fd_noise <- fdata(X_noise, argvals = t)
#' seasonal_strength(fd_noise, period = 2)  # Should be close to 0
seasonal_strength <- function(fdataobj, period = NULL,
                              method = c("variance", "spectral"),
                              n_harmonics = 3) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("seasonal_strength not yet implemented for 2D functional data")
  }

  method <- match.arg(method)

  if (is.null(period)) {
    period <- estimate_period(fdataobj)$period
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
#' ss <- seasonal_strength_curve(fd, period = 2, window_size = 4)
#' # plot(ss)  # Shows strength declining around t = 10
seasonal_strength_curve <- function(fdataobj, period, window_size = NULL,
                                    method = c("variance", "spectral")) {
 if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("seasonal_strength_curve not yet implemented for 2D functional data")
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
#' changes <- detect_seasonality_changes(fd, period = 2)
#' print(changes$change_points)  # Should show onset ~10, cessation ~20
detect_seasonality_changes <- function(fdataobj, period, threshold = 0.3,
                                       window_size = NULL, min_duration = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("detect_seasonality_changes not yet implemented for 2D functional data")
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
#' inst <- instantaneous_period(fd)
#' # plot(inst$period)  # Shows decreasing period over time
instantaneous_period <- function(fdataobj) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("instantaneous_period not yet implemented for 2D functional data")
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
