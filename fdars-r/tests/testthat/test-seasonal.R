# Tests for seasonal analysis functions

test_that("detect_peaks finds known peaks in sine wave", {
  # Pure sine wave with known peak locations
  # sin(2*pi*t/period) has peaks at t = period/4 + k*period
  t <- seq(0, 10, length.out = 200)
  period <- 2
  X <- matrix(sin(2 * pi * t / period), nrow = 1)
  fd <- fdata(X, argvals = t)

  # Expected peaks at t = 0.5, 2.5, 4.5, 6.5, 8.5
  expected_peaks <- c(0.5, 2.5, 4.5, 6.5, 8.5)

  result <- detect_peaks(fd)
  peaks <- result$peaks[[1]]

  expect_equal(nrow(peaks), 5)
  expect_equal(peaks$time, expected_peaks, tolerance = 0.15)
  expect_equal(result$mean_period, period, tolerance = 0.1)
})

test_that("detect_peaks works with different periods", {
  # Higher frequency: period = 1
  t <- seq(0, 10, length.out = 400)
  X <- matrix(sin(2 * pi * t / 1), nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- detect_peaks(fd)
  peaks <- result$peaks[[1]]

  # Should find 10 peaks for period=1 over [0,10]
  expect_equal(nrow(peaks), 10)
  expect_equal(result$mean_period, 1.0, tolerance = 0.1)
})

test_that("detect_peaks respects min_distance", {
  t <- seq(0, 10, length.out = 200)
  X <- matrix(sin(2 * pi * t / 2), nrow = 1)
  fd <- fdata(X, argvals = t)

  # With min_distance = 1.5, should find all 5 peaks (spacing = 2)
  result1 <- detect_peaks(fd, min_distance = 1.5)
  expect_equal(nrow(result1$peaks[[1]]), 5)

  # With min_distance = 2.5, should find fewer peaks
  result2 <- detect_peaks(fd, min_distance = 2.5)
  expect_lt(nrow(result2$peaks[[1]]), 5)
})

test_that("detect_peaks handles shifted sine wave", {
  # Shifted sine: sin(2*pi*t/2) + 1
  # Same peak locations, just shifted up
  t <- seq(0, 10, length.out = 200)
  X <- matrix(sin(2 * pi * t / 2) + 1, nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- detect_peaks(fd)
  peaks <- result$peaks[[1]]

  # Should still find 5 peaks
  expect_equal(nrow(peaks), 5)

  # Peak values should be around 2.0 (max of sin + 1)
  expect_equal(peaks$value, rep(2.0, 5), tolerance = 0.05)
})

test_that("detect_peaks handles smoothing for noisy data", {
  set.seed(123)
  t <- seq(0, 10, length.out = 200)
  X_noisy <- matrix(sin(2 * pi * t / 2) + rnorm(200, sd = 0.3), nrow = 1)
  fd <- fdata(X_noisy, argvals = t)

  # With smoothing, should find approximately 5 peaks
  result_smooth <- detect_peaks(fd, min_distance = 1.5, smooth_first = TRUE)

  # Allow some tolerance due to noise

  expect_gte(nrow(result_smooth$peaks[[1]]), 4)
  expect_lte(nrow(result_smooth$peaks[[1]]), 6)
})

test_that("detect_peaks prominence filtering works", {
  # Create signal with main peaks and small ripples
  t <- seq(0, 10, length.out = 200)
  base <- sin(2 * pi * t / 2)
  ripple <- 0.1 * sin(2 * pi * t * 4)
  X <- matrix(base + ripple, nrow = 1)
  fd <- fdata(X, argvals = t)

  # Without prominence filter, may find extra peaks from ripples
  result_no_filter <- detect_peaks(fd)

  # With prominence filter, should only find major peaks
  result_filtered <- detect_peaks(fd, min_prominence = 0.5)

  # Filtered should have fewer or equal peaks
  expect_lte(nrow(result_filtered$peaks[[1]]), nrow(result_no_filter$peaks[[1]]))
})

test_that("detect_peaks handles multiple curves", {
  t <- seq(0, 10, length.out = 200)
  # Two curves with same period but different phases
  X <- rbind(
    sin(2 * pi * t / 2),
    sin(2 * pi * t / 2 + pi / 4)
  )
  fd <- fdata(X, argvals = t)

  result <- detect_peaks(fd)

  # Should have results for both curves
  expect_equal(length(result$peaks), 2)
  expect_equal(nrow(result$peaks[[1]]), 5)
  expect_equal(nrow(result$peaks[[2]]), 5)
})

test_that("analyze_peak_timing works for periodic data", {
  t <- seq(0, 10, length.out = 500)
  X <- matrix(sin(2 * pi * t / 2), nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- analyze_peak_timing(fd, period = 2)

  # Should find peaks
  expect_gt(length(result$peak_times), 0)

  # Pure sine should have low timing variability
  expect_true(is.finite(result$mean_timing))
  expect_true(result$std_timing < 0.1 || is.nan(result$std_timing))
})

test_that("estimate_period correctly estimates sine wave period", {
  t <- seq(0, 20, length.out = 400)
  period_true <- 2.5
  X <- matrix(sin(2 * pi * t / period_true), nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- estimate_period(fd)

  expect_equal(result$period, period_true, tolerance = 0.2)
  expect_gt(result$confidence, 1.0)
})

test_that("seasonal_strength is high for pure sine", {
  t <- seq(0, 20, length.out = 400)
  X <- matrix(sin(2 * pi * t / 2), nrow = 1)
  fd <- fdata(X, argvals = t)

  strength <- seasonal_strength(fd, period = 2)

  # Pure sine should have very high seasonal strength
  expect_gt(strength, 0.8)
})

test_that("detect_peaks works with different amplitudes", {
  t <- seq(0, 10, length.out = 200)

  for (amplitude in c(0.5, 1.0, 2.0, 5.0)) {
    X <- matrix(amplitude * sin(2 * pi * t / 2), nrow = 1)
    fd <- fdata(X, argvals = t)

    result <- detect_peaks(fd)
    peaks <- result$peaks[[1]]

    expect_equal(nrow(peaks), 5,
                 info = paste("Amplitude", amplitude, "should find 5 peaks"))

    # Peak values should be close to amplitude
    expect_equal(peaks$value, rep(amplitude, 5), tolerance = 0.1,
                 info = paste("Peak values should equal amplitude", amplitude))
  }
})

test_that("detect_peaks handles varying frequency (chirp)", {
  t <- seq(0, 10, length.out = 400)

  # Chirp signal: frequency increases with time
  # Phase = 2*pi * (0.5*t + 0.05*t^2)
  phase <- 2 * pi * (0.5 * t + 0.05 * t^2)
  X <- matrix(sin(phase), nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- detect_peaks(fd)
  peaks <- result$peaks[[1]]

  # Should find multiple peaks
  expect_gte(nrow(peaks), 5)

  # Inter-peak distances should decrease over time
  distances <- result$inter_peak_distances[[1]]
  if (length(distances) >= 4) {
    early_avg <- mean(distances[1:2])
    late_avg <- mean(distances[(length(distances)-1):length(distances)])
    expect_lt(late_avg, early_avg)
  }
})

test_that("detect_peaks handles sum of sines with different periods", {
  t <- seq(0, 12, length.out = 300)

  # Sum of two sines: period 2 and period 3
  X <- matrix(sin(2 * pi * t / 2) + 0.5 * sin(2 * pi * t / 3), nrow = 1)
  fd <- fdata(X, argvals = t)

  result <- detect_peaks(fd, min_distance = 1.0)
  peaks <- result$peaks[[1]]

  # Should find peaks
  expect_gte(nrow(peaks), 4)

  # Inter-peak distances should vary (not all equal)
  distances <- result$inter_peak_distances[[1]]
  if (length(distances) >= 2) {
    expect_gt(max(distances), min(distances) * 1.1)
  }
})
