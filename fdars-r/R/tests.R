#' Statistical Tests for Functional Data
#'
#' Functions for hypothesis testing with functional data.

#' Test for Functional Linear Model
#'
#' Tests the goodness-of-fit for a functional linear model using
#' the projected Cramer-von Mises statistic.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param B Number of bootstrap samples for p-value computation.
#' @param ... Additional arguments.
#'
#' @return A list of class 'htest' with components:
#' \describe{
#'   \item{statistic}{The test statistic}
#'   \item{p.value}{Bootstrap p-value}
#'   \item{method}{Name of the test}
#' }
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' y <- rnorm(20)
#' # test_result <- flm.test(fd, y, B = 100)
flm.test <- function(fdataobj, y, B = 500, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  if (length(y) != n) {
    stop("Length of y must equal number of curves")
  }

  # Fit functional PC regression
  fit <- fregre.pc(fdataobj, y, ...)
  residuals <- fit$residuals

  # Compute inner product matrix for Adot
  X <- fdataobj$data
  X_centered <- scale(X, center = TRUE, scale = FALSE)

  # Lower triangle of X'X (including diagonal)
  m <- ncol(X)
  inprod <- numeric((n * n + n) / 2)
  idx <- 1
  for (j in 1:n) {
    for (i in j:n) {
      inprod[idx] <- sum(X_centered[i, ] * X_centered[j, ]) / m
      idx <- idx + 1
    }
  }

  # Compute Adot matrix
  adot_vec <- .Call("wrap__compute_adot", as.integer(n), as.numeric(inprod))

  # Compute test statistic
  stat_obs <- .Call("wrap__pcvm_statistic", as.numeric(adot_vec), as.numeric(residuals))

  # Bootstrap for p-value
  boot_stats <- numeric(B)
  for (b in seq_len(B)) {
    # Permute residuals
    perm_residuals <- sample(residuals)
    boot_stats[b] <- .Call("wrap__pcvm_statistic", as.numeric(adot_vec), as.numeric(perm_residuals))
  }

  p_value <- mean(boot_stats >= stat_obs)

  structure(
    list(
      statistic = stat_obs,
      p.value = p_value,
      boot.stats = boot_stats,
      method = "Projected Cramer-von Mises test for FLM"
    ),
    class = "htest"
  )
}

#' Test for Equality of Functional Means
#'
#' Tests whether the mean function equals a specified value.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param mu0 Hypothesized mean function (vector). If NULL, tests against zero.
#' @param B Number of bootstrap samples for p-value computation.
#' @param ... Additional arguments.
#'
#' @return A list of class 'htest' with components:
#' \describe{
#'   \item{statistic}{The test statistic}
#'   \item{p.value}{Bootstrap p-value}
#'   \item{method}{Name of the test}
#' }
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' # test_result <- fmean.test.fdata(fd, B = 100)
fmean.test.fdata <- function(fdataobj, mu0 = NULL, B = 500, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  # Default null hypothesis: mean = 0
  if (is.null(mu0)) {
    mu0 <- rep(0, m)
  }

  if (length(mu0) != m) {
    stop("mu0 must have the same length as number of evaluation points")
  }

  # Center data around hypothesized mean
  X_centered <- sweep(fdataobj$data, 2, mu0)

  # Compute sample mean
  X_mean <- colMeans(X_centered)

  # Test statistic: L2 norm of sample mean
  stat_obs <- sqrt(sum(X_mean^2) / m) * sqrt(n)

  # Bootstrap for p-value (resample and compute statistic)
  boot_stats <- numeric(B)
  for (b in seq_len(B)) {
    # Resample indices
    idx <- sample(n, n, replace = TRUE)
    X_boot <- X_centered[idx, ]
    X_mean_boot <- colMeans(X_boot)
    boot_stats[b] <- sqrt(sum(X_mean_boot^2) / m) * sqrt(n)
  }

  # Two-sided p-value
  p_value <- mean(abs(boot_stats) >= abs(stat_obs))

  structure(
    list(
      statistic = stat_obs,
      p.value = p_value,
      boot.stats = boot_stats,
      method = "Test for functional mean"
    ),
    class = "htest"
  )
}
