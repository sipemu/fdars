# Test Phase 3: Kernels, Smoothing, Clustering, and Utilities

# =============================================================================
# Kernel Function Tests
# =============================================================================

test_that("Ker.norm is a proper kernel", {
  u <- seq(-3, 3, length.out = 100)
  k <- Ker.norm(u)

  # Non-negative
  expect_true(all(k >= 0))

  # Symmetric
  expect_equal(k[1:50], rev(k[51:100]), tolerance = 1e-10)

  # Integrates to approximately 1 (for standard normal)
  integral <- sum(k * diff(c(u[1], u))) # simple approximation
  expect_lt(abs(integral - 1), 0.1)
})

test_that("Ker.epa has correct support", {
  u_inside <- c(-0.5, 0, 0.5)
  u_outside <- c(-1.5, 1.5)

  expect_true(all(Ker.epa(u_inside) > 0))
  expect_equal(Ker.epa(u_outside), c(0, 0))
})

test_that("Ker.tri has correct support", {
  u_inside <- c(-0.5, 0, 0.5)
  u_outside <- c(-1.5, 1.5)

  expect_true(all(Ker.tri(u_inside) > 0))
  expect_equal(Ker.tri(u_outside), c(0, 0))
})

test_that("Ker.quar has correct support", {
  u_inside <- c(-0.5, 0, 0.5)
  u_outside <- c(-1.5, 1.5)

  expect_true(all(Ker.quar(u_inside) > 0))
  expect_equal(Ker.quar(u_outside), c(0, 0))
})

test_that("Ker.cos has correct support", {
  u_inside <- c(-0.5, 0, 0.5)
  u_outside <- c(-1.5, 1.5)

  expect_true(all(Ker.cos(u_inside) > 0))
  expect_equal(Ker.cos(u_outside), c(0, 0))
})

test_that("Ker.unif has correct support", {
  u_inside <- c(-0.5, 0, 0.5)
  u_outside <- c(-1.5, 1.5)

  expect_equal(Ker.unif(u_inside), c(0.5, 0.5, 0.5))
  expect_equal(Ker.unif(u_outside), c(0, 0))
})

test_that("Asymmetric kernels are one-sided", {
  u_positive <- c(0, 0.5, 0.9)
  u_negative <- c(-0.5, -0.1)

  expect_true(all(AKer.norm(u_positive) > 0))
  expect_equal(AKer.norm(u_negative), c(0, 0))

  expect_true(all(AKer.epa(u_positive) > 0))
  expect_equal(AKer.epa(u_negative), c(0, 0))
})

test_that("Integrated kernels range from 0 to 1", {
  u <- seq(-1.5, 1.5, length.out = 100)

  for (ker in c("norm", "epa", "tri", "quar", "cos", "unif")) {
    ik <- Kernel.integrate(u, ker)
    # All values should be between 0 and 1
    expect_true(all(ik >= 0 & ik <= 1))
    # Should be monotonically increasing
    expect_true(all(diff(ik) >= -1e-10))
  }

  # Test boundary values specifically at -1 and 1
  u_bounds <- c(-1, 0, 1)
  for (ker in c("epa", "tri", "quar", "cos", "unif")) {
    ik <- Kernel.integrate(u_bounds, ker)
    expect_equal(ik[1], 0, tolerance = 0.01)  # At -1
    expect_equal(ik[3], 1, tolerance = 0.01)  # At 1
  }
})

test_that("Kernel wrapper works for all types", {
  u <- c(-0.5, 0, 0.5)

  expect_equal(Kernel(u, "norm"), Ker.norm(u))
  expect_equal(Kernel(u, "epa"), Ker.epa(u))
  expect_equal(Kernel(u, "tri"), Ker.tri(u))
  expect_equal(Kernel(u, "quar"), Ker.quar(u))
  expect_equal(Kernel(u, "cos"), Ker.cos(u))
  expect_equal(Kernel(u, "unif"), Ker.unif(u))
})

# =============================================================================
# Smoothing Function Tests
# =============================================================================

test_that("S.NW produces valid smoother matrix", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.1
  S <- S.NW(tt, h)

  # Square matrix
  expect_equal(dim(S), c(20, 20))

  # Rows sum to 1
  row_sums <- rowSums(S)
  expect_equal(row_sums, rep(1, 20), tolerance = 1e-10)

  # Non-negative
  expect_true(all(S >= 0))
})

test_that("S.LLR produces valid smoother matrix", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.15
  S <- S.LLR(tt, h)

  # Square matrix
  expect_equal(dim(S), c(20, 20))

  # Rows sum to 1 (linear regression preserves means)
  row_sums <- rowSums(S)
  expect_equal(row_sums, rep(1, 20), tolerance = 1e-8)
})

test_that("S.LPR with p=0 equals S.NW", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.15

  S_nw <- S.NW(tt, h, Ker = "norm")
  S_lpr0 <- S.LPR(tt, h, p = 0, Ker = "norm")

  expect_equal(S_nw, S_lpr0, tolerance = 1e-10)
})

test_that("S.LPR with p=1 equals S.LLR", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.15

  S_llr <- S.LLR(tt, h, Ker = "norm")
  S_lpr1 <- S.LPR(tt, h, p = 1, Ker = "norm")

  expect_equal(S_llr, S_lpr1, tolerance = 1e-10)
})

test_that("S.LCR calls S.LPR with p=3", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.2

  S_lcr <- S.LCR(tt, h)
  S_lpr3 <- S.LPR(tt, h, p = 3)

  expect_equal(S_lcr, S_lpr3)
})

test_that("S.KNN produces valid smoother matrix", {
  tt <- seq(0, 1, length.out = 20)
  S <- S.KNN(tt, knn = 5)

  expect_equal(dim(S), c(20, 20))
  row_sums <- rowSums(S)
  expect_equal(row_sums, rep(1, 20), tolerance = 1e-10)
})

test_that("CV smoother has zero diagonal", {
  tt <- seq(0, 1, length.out = 20)
  h <- 0.15
  S_cv <- S.NW(tt, h, cv = TRUE)

  expect_equal(diag(S_cv), rep(0, 20))
})

test_that("h.default returns reasonable bandwidth", {
  tt <- seq(0, 1, length.out = 50)
  h <- h.default(tt)

  expect_true(h > 0)
  expect_true(h < 1)  # Should be smaller than the domain range
})

test_that("CV.S computes cross-validation score", {
  tt <- seq(0, 1, length.out = 50)
  y <- sin(2 * pi * tt) + rnorm(50, sd = 0.1)

  cv <- CV.S(S.NW, tt, h = 0.1, y = y)
  expect_true(is.numeric(cv))
  expect_true(cv > 0)
})

test_that("GCV.S computes GCV score", {
  tt <- seq(0, 1, length.out = 50)
  y <- sin(2 * pi * tt) + rnorm(50, sd = 0.1)

  gcv <- GCV.S(S.NW, tt, h = 0.1, y = y)
  expect_true(is.numeric(gcv))
  expect_true(gcv > 0)
})

# =============================================================================
# Clustering Tests
# =============================================================================

test_that("cluster.kmeans clusters correctly", {
  set.seed(123)
  t <- seq(0, 1, length.out = 50)
  n <- 30

  # Create two distinct groups
  X <- matrix(0, n, 50)
  true_cluster <- rep(1:2, each = 15)
  for (i in 1:n) {
    if (true_cluster[i] == 1) {
      X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
    } else {
      X[i, ] <- cos(2*pi*t) + rnorm(50, sd = 0.1)
    }
  }
  fd <- fdata(X, argvals = t)

  result <- cluster.kmeans(fd, ncl = 2, seed = 42)

  # Check structure
  expect_s3_class(result, "cluster.kmeans")
  expect_length(result$cluster, n)
  expect_equal(nrow(result$centers$data), 2)
  expect_length(result$size, 2)
  expect_equal(sum(result$size), n)

  # Clusters should be 1 or 2
  expect_true(all(result$cluster %in% c(1, 2)))

  # Should separate the groups reasonably well
  # (allowing for some misclassification due to randomness)
  tab <- table(result$cluster, true_cluster)
  accuracy <- max(sum(diag(tab)), sum(diag(tab[2:1, ]))) / n
  expect_gt(accuracy, 0.7)
})

test_that("cluster.kmeans handles single cluster", {
  t <- seq(0, 1, length.out = 50)
  X <- matrix(rnorm(20 * 50), 20, 50)
  fd <- fdata(X, argvals = t)

  result <- cluster.kmeans(fd, ncl = 1, seed = 123)

  expect_equal(unique(result$cluster), 1)
  expect_equal(result$size, 20)
})

test_that("cluster.init returns correct number of centers", {
  t <- seq(0, 1, length.out = 50)
  X <- matrix(rnorm(20 * 50), 20, 50)
  fd <- fdata(X, argvals = t)

  centers <- cluster.init(fd, ncl = 3, seed = 123)

  expect_equal(nrow(centers$data), 3)
})

# =============================================================================
# Utility Function Tests
# =============================================================================

test_that("int.simpson integrates correctly", {
  t <- seq(0, 2*pi, length.out = 100)

  # Integral of sin from 0 to 2*pi is 0
  X1 <- matrix(sin(t), nrow = 1)
  fd1 <- fdata(X1, argvals = t)
  expect_equal(int.simpson(fd1)[1], 0, tolerance = 0.01)

  # Integral of cos^2 from 0 to 2*pi is pi
  X2 <- matrix(cos(t)^2, nrow = 1)
  fd2 <- fdata(X2, argvals = t)
  expect_equal(int.simpson(fd2)[1], pi, tolerance = 0.05)
})

test_that("inprod.fdata computes inner products", {
  t <- seq(0, 2*pi, length.out = 100)

  # sin and cos are orthogonal on [0, 2*pi]
  X1 <- matrix(sin(t), nrow = 1)
  X2 <- matrix(cos(t), nrow = 1)
  fd1 <- fdata(X1, argvals = t)
  fd2 <- fdata(X2, argvals = t)

  ip <- inprod.fdata(fd1, fd2)
  expect_equal(ip[1, 1], 0, tolerance = 0.1)

  # sin with itself: integral of sin^2 from 0 to 2*pi is pi
  ip_self <- inprod.fdata(fd1, fd1)
  expect_equal(ip_self[1, 1], pi, tolerance = 0.05)
})

test_that("inprod.fdata handles multiple curves", {
  t <- seq(0, 1, length.out = 50)
  X1 <- matrix(rnorm(10 * 50), 10, 50)
  X2 <- matrix(rnorm(5 * 50), 5, 50)
  fd1 <- fdata(X1, argvals = t)
  fd2 <- fdata(X2, argvals = t)

  ip <- inprod.fdata(fd1, fd2)
  expect_equal(dim(ip), c(10, 5))
})

test_that("r.ou generates valid functional data", {
  t <- seq(0, 1, length.out = 50)
  fd <- r.ou(n = 10, t = t, seed = 123)

  expect_s3_class(fd, "fdata")
  expect_equal(nrow(fd$data), 10)
  expect_equal(ncol(fd$data), 50)
})

test_that("r.brownian generates valid functional data", {
  t <- seq(0, 1, length.out = 50)
  fd <- r.brownian(n = 10, t = t, seed = 123)

  expect_s3_class(fd, "fdata")
  expect_equal(fd$data[, 1], rep(0, 10))  # Starts at 0
})

test_that("r.bridge returns to zero at end", {
  t <- seq(0, 1, length.out = 50)
  fd <- r.bridge(n = 10, t = t, seed = 123)

  expect_s3_class(fd, "fdata")
  # Brownian bridge should end near 0
  expect_true(all(abs(fd$data[, 50]) < 0.5))
})

test_that("Prediction metrics work correctly", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1, 2, 3, 4, 5)  # Perfect prediction

  expect_equal(pred.MAE(y_true, y_pred), 0)
  expect_equal(pred.MSE(y_true, y_pred), 0)
  expect_equal(pred.RMSE(y_true, y_pred), 0)
  expect_equal(pred.R2(y_true, y_pred), 1)
})

test_that("Prediction metrics handle imperfect predictions", {
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.2, 2.9, 4.1, 4.8)

  expect_gt(pred.MAE(y_true, y_pred), 0)
  expect_gt(pred.MSE(y_true, y_pred), 0)
  expect_gt(pred.RMSE(y_true, y_pred), 0)
  expect_lt(pred.R2(y_true, y_pred), 1)
  expect_gt(pred.R2(y_true, y_pred), 0.9)  # Still a good fit
})

# =============================================================================
# Integration Tests - Smoothing with fdata
# =============================================================================

test_that("Smoothing can be applied to fdata", {
  set.seed(123)
  t <- seq(0, 1, length.out = 50)
  y <- sin(2 * pi * t) + rnorm(50, sd = 0.2)
  fd <- fdata(matrix(y, nrow = 1), argvals = t)

  # Compute smoother and apply
  S <- S.NW(t, h = 0.1)
  y_smooth <- as.vector(S %*% y)

  # Smoothed curve should be closer to true curve
  true_y <- sin(2 * pi * t)
  error_original <- mean((y - true_y)^2)
  error_smooth <- mean((y_smooth - true_y)^2)

  expect_lt(error_smooth, error_original)
})

test_that("optim.np finds reasonable bandwidth", {
  set.seed(123)
  t <- seq(0, 1, length.out = 50)
  y <- sin(2 * pi * t) + rnorm(50, sd = 0.1)
  fd <- fdata(matrix(y, nrow = 1), argvals = t)

  result <- optim.np(fd, S.NW)

  expect_true(result$h.opt > 0)
  expect_true(result$h.opt < 1)
})
