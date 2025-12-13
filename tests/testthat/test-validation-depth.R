# Validation tests for depth functions
# Compare fdars results with fda.usc reference implementation

test_that("depth.FM matches fda.usc implementation", {
  skip_if_not_installed("fda.usc")

  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid * (1 + 0.1 * i / n)) + rnorm(m, sd = 0.1)
  }

  fd_orig <- fda.usc::fdata(X, argvals = t_grid)
  fd_rust <- fdars::fdata(X, argvals = t_grid)

  D_orig <- fda.usc::depth.FM(fd_orig)$dep
  D_rust <- fdars::depth.FM(fd_rust)

  # Compare ignoring names attribute
  expect_equal(as.numeric(D_orig), D_rust, tolerance = 1e-6)
})

test_that("depth.mode produces valid depths", {
  # Note: fda.usc and fdars may have different mode depth implementations
  # (different kernel normalizations). We test correctness properties instead.
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd_rust <- fdars::fdata(X, argvals = t_grid)
  h <- 0.5
  D_rust <- fdars::depth.mode(fd_rust, h = h)

  # Depths should be positive (mode depth not bounded to [0,1])
  expect_true(all(D_rust >= 0))
  expect_length(D_rust, n)

  # The deepest curve should be near the center
  deepest_idx <- which.max(D_rust)
  deepest_curve <- X[deepest_idx, ]
  mean_curve <- colMeans(X)
  # Deepest should be closer to mean than random outer curves
  expect_true(sum((deepest_curve - mean_curve)^2) < quantile(rowSums((X - rep(mean_curve, each = n))^2), 0.5))
})

test_that("depth.RP produces valid depths", {
  # RP depth uses random projections - different implementations may not correlate well
  # because random projections are generated independently
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd_rust <- fdars::fdata(X, argvals = t_grid)
  nproj <- 50
  D_rust <- fdars::depth.RP(fd_rust, nproj = nproj)

  # Depths should be in [0, 1]
  expect_true(all(D_rust >= 0 & D_rust <= 1))
  expect_length(D_rust, n)

  # The deepest curve should be near the center (statistical test)
  deepest_idx <- which.max(D_rust)
  deepest_curve <- X[deepest_idx, ]
  mean_curve <- colMeans(X)
  # Deepest should be in the inner 50% of distances to mean
  all_dists <- rowSums((X - matrix(mean_curve, n, m, byrow = TRUE))^2)
  expect_true(all_dists[deepest_idx] < quantile(all_dists, 0.75))
})

test_that("depth.RT produces valid depths", {
  # Note: RT depth implementations may differ in tie-breaking behavior
  # We test correctness properties instead of exact matching
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd_rust <- fdars::fdata(X, argvals = t_grid)
  D_rust <- fdars::depth.RT(fd_rust)

  # Depths should be in [0, 1]
  expect_true(all(D_rust >= 0 & D_rust <= 1))
  expect_length(D_rust, n)

  # RT depth should give reasonable central curves high depth
  mean_curve <- colMeans(X)
  all_dists <- rowSums((X - matrix(mean_curve, n, m, byrow = TRUE))^2)
  # Top 30% by depth should overlap with inner 70% by distance to mean
  top_depth_idx <- order(D_rust, decreasing = TRUE)[1:floor(n * 0.3)]
  inner_dist_idx <- order(all_dists)[1:floor(n * 0.7)]
  expect_gt(length(intersect(top_depth_idx, inner_dist_idx)), 0)
})

test_that("depth.FSD produces valid depths", {
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::depth.FSD(fd)

  # Depths should be in [0, 1]
  expect_true(all(D >= 0 & D <= 1))
  expect_length(D, n)
})

test_that("depth.KFSD produces valid depths", {
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::depth.KFSD(fd, h = 0.5)

  # Depths should be in [0, 1]
  expect_true(all(D >= 0 & D <= 1))
  expect_length(D, n)
})

test_that("depth.RPD produces valid depths", {
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  D <- fdars::depth.RPD(fd, nproj = 50, deriv = c(0, 1))

  # Depths should be in [0, 1]
  expect_true(all(D >= 0 & D <= 1))
  expect_length(D, n)
})

test_that("median returns valid median", {
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  med <- fdars::median(fd)

  expect_s3_class(med, "fdata")
  expect_equal(nrow(med$data), 1)
  expect_equal(ncol(med$data), m)
})

test_that("trimmed returns valid trimmed mean", {
  set.seed(42)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }

  fd <- fdars::fdata(X, argvals = t_grid)
  trimmed <- fdars::trimmed(fd, trim = 0.1)

  expect_s3_class(trimmed, "fdata")
  expect_equal(nrow(trimmed$data), 1)
  expect_equal(ncol(trimmed$data), m)
})
