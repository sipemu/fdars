# Unit tests for fdata functions validation against fda.usc
# These tests compare fdars (Rust) vs fda.usc (R) implementations

test_that("fdata creation matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  # Check that numerical values are identical (ignoring attributes like dimnames)
  expect_equal(as.vector(fd_orig$data), as.vector(fd_rust$data))
  expect_equal(as.vector(fd_orig$argvals), as.vector(fd_rust$argvals))
})

test_that("mean matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  mean_orig <- mean(fd_orig)
  mean_rust <- mean(fd_rust)

  # fda.usc returns an fdata object
  if (inherits(mean_orig, "fdata")) {
    mean_orig_vec <- as.vector(mean_orig$data)
  } else {
    mean_orig_vec <- mean_orig
  }

  expect_equal(mean_orig_vec, mean_rust, tolerance = 1e-10)
})

test_that("fdata.cen matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  fd_cen_orig <- fda.usc::fdata.cen(fd_orig)
  fd_cen_rust <- fdars::fdata.cen(fd_rust)

  # fda.usc returns a list with $Xcen$data, fdars returns fdata directly
  orig_cen_data <- if (!is.null(fd_cen_orig$Xcen)) fd_cen_orig$Xcen$data else fd_cen_orig$data
  rust_cen_data <- fd_cen_rust$data

  expect_equal(as.vector(orig_cen_data), as.vector(rust_cen_data), tolerance = 1e-10)

  # Check centered data has zero mean
  mean_cen <- colMeans(rust_cen_data)
  expect_equal(mean_cen, rep(0, m), tolerance = 1e-10)
})

test_that("norm.fdata L2 matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  norm_orig <- fda.usc::norm.fdata(fd_orig)
  norm_rust <- fdars::norm.fdata(fd_rust)

  # Allow small differences due to integration method
  expect_equal(norm_orig, norm_rust, tolerance = 1e-4)
})

test_that("norm.fdata L1 matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  norm_orig <- fda.usc::norm.fdata(fd_orig, lp = 1)
  norm_rust <- fdars::norm.fdata(fd_rust, lp = 1)

  expect_equal(norm_orig, norm_rust, tolerance = 1e-4)
})

test_that("fdata subsetting matches fda.usc", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 20
  m <- 30
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  fd_sub_orig <- fd_orig[1:5, ]
  fd_sub_rust <- fd_rust[1:5, ]

  # Compare numerical values (ignoring attributes like dimnames)
  expect_equal(as.vector(fd_sub_orig$data), as.vector(fd_sub_rust$data))
})
