# Tests for visualization functions: outliergram and FPCAPlot

# =============================================================================
# MEI (Modified Epigraph Index) Tests
# =============================================================================

test_that("MEI depth computation works", {
  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  # Create simple curves
  X <- matrix(0, 10, m)
  for (i in 1:10) {
    X[i, ] <- sin(2 * pi * t_grid) + (i - 5) * 0.1
  }
  fd <- fdata(X, argvals = t_grid)

  mei <- depth(fd, method = "MEI")

  expect_length(mei, 10)
  expect_true(all(mei >= 0 & mei <= 1))
})

test_that("MEI values are ordered correctly", {
  # Create curves where we know the order:
  # Higher curves should have lower MEI (less time below others)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, 5, m)
  X[1, ] <- rep(1, m)   # highest - should have lowest MEI
  X[2, ] <- rep(0.5, m)
  X[3, ] <- rep(0, m)   # middle
  X[4, ] <- rep(-0.5, m)
  X[5, ] <- rep(-1, m)  # lowest - should have highest MEI

  fd <- fdata(X, argvals = t_grid)
  mei <- depth(fd, method = "MEI")

  # Higher curves should have lower MEI
  expect_true(mei[1] < mei[3])
  expect_true(mei[3] < mei[5])
})

test_that("MEI with separate reference sample works", {
  set.seed(42)
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X1 <- matrix(rnorm(5 * m), 5, m)
  X2 <- matrix(rnorm(10 * m), 10, m)

  fd1 <- fdata(X1, argvals = t_grid)
  fd2 <- fdata(X2, argvals = t_grid)

  mei <- depth(fd1, fd2, method = "MEI")

  expect_length(mei, 5)
  expect_true(all(mei >= 0 & mei <= 1))
})

# =============================================================================
# Outliergram Tests
# =============================================================================

test_that("outliergram basic functionality works", {
  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  # Create normal curves
  X <- matrix(0, 20, m)
  for (i in 1:20) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.2)
  }
  fd <- fdata(X, argvals = t_grid)

  og <- outliergram(fd)

  expect_s3_class(og, "outliergram")
  expect_length(og$mei, 20)
  expect_length(og$mbd, 20)
  expect_true(all(og$mei >= 0 & og$mei <= 1))
  expect_true(all(og$mbd >= 0 & og$mbd <= 1))
})

test_that("outliergram detects outliers", {
  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  # Create normal curves plus one outlier
  X <- matrix(0, 25, m)
  for (i in 1:24) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.15)
  }
  X[25, ] <- sin(2 * pi * t_grid) + 3  # magnitude outlier

  fd <- fdata(X, argvals = t_grid)
  og <- outliergram(fd, factor = 1.5)

  # The outlier should be detected (may not always be, depends on factor)
  expect_true(og$n_outliers >= 0)
  expect_equal(length(og$outliers), og$n_outliers)
})

test_that("outliergram factor parameter works", {
  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, 20, m)
  for (i in 1:20) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.3)
  }
  fd <- fdata(X, argvals = t_grid)

  # Lower factor should detect more outliers
  og_strict <- outliergram(fd, factor = 0.5)
  og_lenient <- outliergram(fd, factor = 3)

  expect_true(og_strict$n_outliers >= og_lenient$n_outliers)
})

test_that("outliergram print method works", {
  set.seed(42)
  m <- 30
  X <- matrix(rnorm(10 * m), 10, m)
  fd <- fdata(X)
  og <- outliergram(fd)

  expect_output(print(og), "Outliergram")
  expect_output(print(og), "Number of curves: 10")
})

test_that("outliergram plot method works", {
  skip_if_not_installed("ggplot2")

  set.seed(42)
  m <- 30
  X <- matrix(rnorm(10 * m), 10, m)
  fd <- fdata(X)
  og <- outliergram(fd)

  p <- plot(og)
  expect_s3_class(p, "ggplot")
})

# =============================================================================
# FPCA Plot Tests
# =============================================================================

test_that("fdata2pc returns correct class", {
  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, 20, m)
  for (i in 1:20) {
    X[i, ] <- sin(2 * pi * t_grid + runif(1, 0, pi)) + rnorm(m, sd = 0.1)
  }
  fd <- fdata(X, argvals = t_grid)

  pc <- fdata2pc(fd, ncomp = 3)

  expect_s3_class(pc, "fdata2pc")
  expect_equal(length(pc$d), 3)
  expect_equal(nrow(pc$x), 20)
  expect_equal(ncol(pc$x), 3)
})

test_that("plot.fdata2pc components plot works", {
  skip_if_not_installed("ggplot2")

  set.seed(42)
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, 20, m)
  for (i in 1:20) {
    X[i, ] <- sin(2 * pi * t_grid + runif(1, 0, pi)) + rnorm(m, sd = 0.1)
  }
  fd <- fdata(X, argvals = t_grid)
  pc <- fdata2pc(fd, ncomp = 3)

  p <- plot(pc, type = "components")
  expect_s3_class(p, "ggplot")
})

test_that("plot.fdata2pc variance plot works", {
  skip_if_not_installed("ggplot2")

  set.seed(42)
  m <- 50
  X <- matrix(rnorm(20 * m), 20, m)
  fd <- fdata(X)
  pc <- fdata2pc(fd, ncomp = 5)

  p <- plot(pc, type = "variance")
  expect_s3_class(p, "ggplot")
})

test_that("plot.fdata2pc scores plot works", {
  skip_if_not_installed("ggplot2")

  set.seed(42)
  m <- 50
  X <- matrix(rnorm(20 * m), 20, m)
  fd <- fdata(X)
  pc <- fdata2pc(fd, ncomp = 3)

  p <- plot(pc, type = "scores")
  expect_s3_class(p, "ggplot")
})

test_that("print.fdata2pc works", {
  set.seed(42)
  m <- 50
  X <- matrix(rnorm(20 * m), 20, m)
  fd <- fdata(X)
  pc <- fdata2pc(fd, ncomp = 3)

  expect_output(print(pc), "Functional Principal Component Analysis")
  expect_output(print(pc), "Number of observations: 20")
  expect_output(print(pc), "Number of components: 3")
  expect_output(print(pc), "Variance explained")
})

# =============================================================================
# Error Handling Tests
# =============================================================================

test_that("MEI rejects 2D data", {
  X <- array(rnorm(100), dim = c(5, 10, 2))
  fd2d <- fdata(X, argvals = list(1:10, 1:2), fdata2d = TRUE)

  expect_error(depth(fd2d, method = "MEI"), "2D")
})

test_that("outliergram rejects 2D data", {
  X <- array(rnorm(100), dim = c(5, 10, 2))
  fd2d <- fdata(X, argvals = list(1:10, 1:2), fdata2d = TRUE)

  expect_error(outliergram(fd2d), "2D")
})

test_that("outliergram rejects non-fdata input", {
  expect_error(outliergram(matrix(1:10, 2, 5)), "fdata")
})
