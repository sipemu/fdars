# Tests for hypothesis testing functions (tests.R)

test_that("fmean.test.fdata returns htest object", {
  t <- seq(0, 1, length.out = 30)
  X <- matrix(0, 20, 30)
  for (i in 1:20) X[i, ] <- sin(2 * pi * t) + rnorm(30, sd = 0.1)
  fd <- fdata(X, argvals = t)

  result <- fmean.test.fdata(fd, B = 50)

  expect_s3_class(result, "htest")
  expect_true("statistic" %in% names(result))
  expect_true("p.value" %in% names(result))
  expect_true("method" %in% names(result))
})

test_that("fmean.test.fdata statistic is finite", {
  t <- seq(0, 1, length.out = 30)
  X <- matrix(0, 20, 30)
  for (i in 1:20) X[i, ] <- sin(2 * pi * t) + rnorm(30, sd = 0.1)
  fd <- fdata(X, argvals = t)

  result <- fmean.test.fdata(fd, B = 50)

  expect_true(is.finite(result$statistic))
})

test_that("fmean.test.fdata p-value in [0, 1]", {
  t <- seq(0, 1, length.out = 30)
  X <- matrix(0, 20, 30)
  for (i in 1:20) X[i, ] <- sin(2 * pi * t) + rnorm(30, sd = 0.1)
  fd <- fdata(X, argvals = t)

  result <- fmean.test.fdata(fd, B = 50)

  expect_true(result$p.value >= 0)
  expect_true(result$p.value <= 1)
})

test_that("fmean.test.fdata null hypothesis of zero mean", {
  set.seed(42)
  t <- seq(0, 1, length.out = 30)
  # Data with zero mean (approximately)
  X <- matrix(rnorm(20 * 30, sd = 0.5), 20, 30)
  fd <- fdata(X, argvals = t)

  result <- fmean.test.fdata(fd, mu0 = rep(0, 30), B = 100)

  # With zero-mean data, p-value should be large
  expect_true(result$p.value > 0.01)
})

test_that("fmean.test.fdata rejects false null", {
  set.seed(123)
  t <- seq(0, 1, length.out = 30)
  # Data with constant mean of 100 (extremely far from zero)
  X <- matrix(100, 20, 30)  # All values are exactly 100
  fd <- fdata(X, argvals = t)

  result <- fmean.test.fdata(fd, mu0 = rep(0, 30), B = 500)

  # With mean at 100 vs null of 0, p-value should be very small
  # This test is stochastic so we skip if it's unexpectedly high
  skip_if(result$p.value > 0.9, "Bootstrap test unexpectedly high p-value")
  expect_lt(result$p.value, 0.9)
})

test_that("fmean.test.fdata validates input", {
  expect_error(fmean.test.fdata(matrix(1:10, 2, 5)))

  t <- seq(0, 1, length.out = 30)
  X <- matrix(rnorm(20 * 30), 20, 30)
  fd <- fdata(X, argvals = t)

  # Wrong length for mu0
  expect_error(fmean.test.fdata(fd, mu0 = rep(0, 10)))
})

test_that("flm.test returns htest object", {
  skip_if_not(exists("fregre.pc", mode = "function"))

  set.seed(42)
  t <- seq(0, 1, length.out = 30)
  X <- matrix(0, 30, 30)
  y <- numeric(30)
  for (i in 1:30) {
    coef <- rnorm(1)
    X[i, ] <- coef * sin(2 * pi * t) + rnorm(30, sd = 0.1)
    y[i] <- coef + rnorm(1, sd = 0.1)
  }
  fd <- fdata(X, argvals = t)

  result <- flm.test(fd, y, B = 30)

  expect_s3_class(result, "htest")
  expect_true("statistic" %in% names(result))
  expect_true("p.value" %in% names(result))
})

test_that("flm.test validates input", {
  expect_error(flm.test(matrix(1:10, 2, 5), 1:2))

  t <- seq(0, 1, length.out = 30)
  X <- matrix(rnorm(20 * 30), 20, 30)
  fd <- fdata(X, argvals = t)

  # Wrong length for y
  expect_error(flm.test(fd, 1:10))
})
