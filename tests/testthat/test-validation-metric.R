# Validation tests for metric functions
# Compare fdars results with fda.usc reference implementation

test_that("metric.lp matches fda.usc implementation", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid * (1 + 0.1 * i / n)) + rnorm(m, sd = 0.1)
  }

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  # L2 metric
  D_orig <- fda.usc::metric.lp(fd_orig, lp = 2)
  D_rust <- fdars::metric.lp(fd_rust, lp = 2)

  # Compare values ignoring attributes (fda.usc adds metadata attributes)
  expect_equal(as.matrix(D_orig), D_rust, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("metric.hausdorff matches fda.usc implementation", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  D_orig <- fda.usc::metric.hausdorff(fd_orig)
  D_rust <- fdars::metric.hausdorff(fd_rust)

  # Compare values ignoring attributes (fda.usc adds metadata attributes)
  expect_equal(as.matrix(D_orig), D_rust, tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("metric.DTW produces valid distances", {
  set.seed(42)
  n <- 15
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid + 0.1 * i) + rnorm(m, sd = 0.05)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::metric.DTW(fd)

  # DTW distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(diag(D) == 0))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("semimetric.pca produces valid distances", {
  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::semimetric.pca(fd, ncomp = 3)

  # Distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(diag(D) == 0))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("semimetric.deriv produces valid distances", {
  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::semimetric.deriv(fd, nderiv = 1)

  # Distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(diag(D) == 0))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("semimetric.basis produces valid distances", {
  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::semimetric.basis(fd, nbasis = 10)

  # Distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero (or very small)
  expect_true(all(abs(diag(D)) < 1e-10))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("semimetric.fourier produces valid distances", {
  set.seed(42)
  n <- 20
  m <- 64  # Power of 2 for FFT
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::semimetric.fourier(fd, nfreq = 5)

  # Distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(abs(diag(D)) < 1e-10))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("semimetric.hshift produces valid distances", {
  set.seed(42)
  n <- 15
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid + 0.1 * i) + rnorm(m, sd = 0.05)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::semimetric.hshift(fd, max_shift = 5)

  # Distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(diag(D) == 0))
  # Symmetric
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("metric.kl produces valid distances", {
  set.seed(42)
  n <- 20
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  # Create positive curves for KL
  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- abs(sin(2 * pi * t_grid) + 0.5 + rnorm(m, sd = 0.1))
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::metric.kl(fd)

  # KL distances should be non-negative
  expect_true(all(D >= 0))
  # Diagonal should be zero
  expect_true(all(abs(diag(D)) < 1e-10))
  # Symmetric (due to symmetrization)
  expect_equal(D, t(D), tolerance = 1e-10)
})

test_that("metric.dist dispatches correctly", {
  set.seed(42)
  n <- 10
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)

  # Test various methods
  D_lp <- fdars::metric.dist(fd, method = "lp")
  D_hausdorff <- fdars::metric.dist(fd, method = "hausdorff")
  D_dtw <- fdars::metric.dist(fd, method = "dtw")
  D_pca <- fdars::metric.dist(fd, method = "pca")

  # All should be matrices of correct size
  expect_equal(dim(D_lp), c(n, n))
  expect_equal(dim(D_hausdorff), c(n, n))
  expect_equal(dim(D_dtw), c(n, n))
  expect_equal(dim(D_pca), c(n, n))
})
