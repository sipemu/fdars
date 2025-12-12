# Validation tests for regression functions

test_that("fregre.pc produces valid predictions", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid * (1 + 0.2 * i/n)) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  model <- fdars::fregre.pc(fd, y, ncomp = 3)

  expect_s3_class(model, "fregre.fd")
  expect_length(model$fitted.values, n)
  expect_length(model$residuals, n)

  # Fitted + residuals should equal y
  expect_equal(model$fitted.values + model$residuals, y, tolerance = 1e-10)
})

test_that("fregre.basis produces valid predictions", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  model <- fdars::fregre.basis(fd, y, nbasis = 10)

  expect_s3_class(model, "fregre.fd")
  expect_length(model$fitted.values, n)
  expect_length(model$residuals, n)
})

test_that("fregre.np produces valid predictions", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  model <- fdars::fregre.np(fd, y, h = 0.5)

  expect_s3_class(model, "fregre.np")
  expect_length(model$fitted.values, n)
  expect_length(model$residuals, n)
})

test_that("fregre.pc.cv selects optimal ncomp", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid * (1 + 0.2 * i/n)) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  cv_result <- fdars::fregre.pc.cv(fd, y, kfold = 5, ncomp.range = 1:5, seed = 123)

  expect_true("optimal.ncomp" %in% names(cv_result))
  expect_true("cv.errors" %in% names(cv_result))
  expect_true("model" %in% names(cv_result))

  expect_gte(cv_result$optimal.ncomp, 1)
  expect_lte(cv_result$optimal.ncomp, 5)
  expect_length(cv_result$cv.errors, 5)
})

test_that("fregre.basis.cv selects optimal lambda", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  lambdas <- c(0, 0.01, 0.1, 1)
  cv_result <- fdars::fregre.basis.cv(fd, y, kfold = 5, lambda.range = lambdas, seed = 123)

  expect_true("optimal.lambda" %in% names(cv_result))
  expect_true("cv.errors" %in% names(cv_result))
  expect_true("model" %in% names(cv_result))

  expect_true(cv_result$optimal.lambda %in% lambdas)
  expect_length(cv_result$cv.errors, length(lambdas))
})

test_that("fregre.np.cv selects optimal bandwidth", {
  set.seed(42)
  n <- 50
  m <- 30
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  h_range <- c(0.1, 0.5, 1, 2)
  cv_result <- fdars::fregre.np.cv(fd, y, kfold = 5, h.range = h_range, seed = 123)

  expect_true("optimal.h" %in% names(cv_result))
  expect_true("cv.errors" %in% names(cv_result))
  expect_true("model" %in% names(cv_result))

  expect_true(cv_result$optimal.h %in% h_range)
  expect_length(cv_result$cv.errors, length(h_range))
})

# =============================================================================
# Predict Method Tests
# =============================================================================

test_that("predict.fregre.fd works for fregre.pc", {
  set.seed(123)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  # Split into train/test
  train_idx <- 1:20
  test_idx <- 21:30
  fd_train <- fdars::fdata(X[train_idx, ], argvals = t_grid)
  fd_test <- fdars::fdata(X[test_idx, ], argvals = t_grid)
  y_train <- y[train_idx]
  y_test <- y[test_idx]

  # Fit model
  fit <- fdars::fregre.pc(fd_train, y_train, ncomp = 3)

  # Test predict with new data
  pred <- predict(fit, fd_test)
  expect_length(pred, length(test_idx))
  expect_true(is.numeric(pred))

  # Test predict without new data returns fitted values
  pred_fitted <- predict(fit)
  expect_equal(pred_fitted, fit$fitted.values)
})

test_that("predict.fregre.fd works for fregre.basis", {
  set.seed(123)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  # Split into train/test
  train_idx <- 1:20
  test_idx <- 21:30
  fd_train <- fdars::fdata(X[train_idx, ], argvals = t_grid)
  fd_test <- fdars::fdata(X[test_idx, ], argvals = t_grid)
  y_train <- y[train_idx]

  # Fit model
  fit <- fdars::fregre.basis(fd_train, y_train, lambda = 0.1)

  # Test predict with new data
  pred <- predict(fit, fd_test)
  expect_length(pred, length(test_idx))
  expect_true(is.numeric(pred))

  # Test predict without new data returns fitted values
  pred_fitted <- predict(fit)
  expect_equal(pred_fitted, fit$fitted.values)
})

test_that("predict.fregre.np works", {
  set.seed(123)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  # Split into train/test
  train_idx <- 1:20
  test_idx <- 21:30
  fd_train <- fdars::fdata(X[train_idx, ], argvals = t_grid)
  fd_test <- fdars::fdata(X[test_idx, ], argvals = t_grid)
  y_train <- y[train_idx]

  # Fit model
  fit <- fdars::fregre.np(fd_train, y_train)

  # Test predict with new data
  pred <- predict(fit, fd_test)
  expect_length(pred, length(test_idx))
  expect_true(is.numeric(pred))

  # Test predict without new data returns fitted values
  pred_fitted <- predict(fit)
  expect_equal(pred_fitted, fit$fitted.values)
})

test_that("predict accepts matrix input and converts to fdata", {
  set.seed(123)
  n <- 30
  m <- 50
  t_grid <- seq(0, 1, length.out = m)

  X <- matrix(0, n, m)
  for (i in 1:n) {
    X[i, ] <- sin(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
  }
  y <- rowMeans(X) + rnorm(n, sd = 0.1)

  fd <- fdars::fdata(X, argvals = t_grid)
  fit <- fdars::fregre.pc(fd, y, ncomp = 3)

  # Predict with raw matrix (should be converted to fdata internally)
  X_new <- matrix(sin(2 * pi * t_grid) * 0.5, nrow = 1)
  pred <- predict(fit, X_new)

  expect_length(pred, 1)
  expect_true(is.numeric(pred))
})
