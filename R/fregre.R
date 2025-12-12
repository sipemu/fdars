#' Functional Regression
#'
#' Functions for functional regression models.

#' Functional Principal Component Regression
#'
#' Fits a functional linear model using principal component regression.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param ncomp Number of principal components to use.
#' @param ... Additional arguments.
#'
#' @return A fitted regression object.
#'
#' @export
fregre.pc <- function(fdataobj, y, ncomp = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (fdataobj$fdata2d) {
    stop("fregre.pc not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Center the functional data
  X_centered <- scale(fdataobj$data, center = TRUE, scale = FALSE)
  y_centered <- y - mean(y)

  # Compute SVD
  svd_result <- svd(X_centered)

  # Determine number of components
  if (is.null(ncomp)) {
    # Use enough components to explain 95% variance
    var_explained <- cumsum(svd_result$d^2) / sum(svd_result$d^2)
    ncomp <- min(which(var_explained >= 0.95), n - 1)
    ncomp <- max(ncomp, 1)
  }

  ncomp <- min(ncomp, length(svd_result$d))

  # Get PC scores
  scores <- X_centered %*% svd_result$v[, seq_len(ncomp), drop = FALSE]

  # Fit OLS on scores
  fit <- lm(y_centered ~ scores - 1)

  # Compute beta coefficient function
  beta_coef <- svd_result$v[, seq_len(ncomp), drop = FALSE] %*% coef(fit)

  # Compute intercept and fitted values
  X_mean <- colMeans(fdataobj$data)
  intercept <- as.numeric(mean(y) - sum(X_mean * beta_coef))
  fitted_values <- as.vector(fdataobj$data %*% beta_coef) + intercept

  structure(
    list(
      coefficients = beta_coef,
      intercept = intercept,
      fitted.values = fitted_values,
      residuals = y - fitted_values,
      ncomp = ncomp,
      svd = svd_result,
      fdataobj = fdataobj,
      y = y
    ),
    class = "fregre.fd"
  )
}

#' Functional Basis Regression
#'
#' Fits a functional linear model using basis expansion.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param basis.x Basis for the functional covariate (currently ignored).
#' @param basis.b Basis for the coefficient function (currently ignored).
#' @param lambda Smoothing parameter.
#' @param ... Additional arguments.
#'
#' @return A fitted regression object.
#'
#' @export
fregre.basis <- function(fdataobj, y, basis.x = NULL, basis.b = NULL,
                         lambda = 0, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  # For now, use a simple ridge regression approach
  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  X <- fdataobj$data
  y <- as.vector(y)

  # Ridge regression: beta = (X'X + lambda*I)^-1 X'y
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)

  if (lambda > 0) {
    XtX <- XtX + lambda * diag(m)
  }

  beta <- solve(XtX, Xty)
  fitted <- as.vector(X %*% beta)
  residuals <- y - fitted

  structure(
    list(
      coefficients = beta,
      fitted.values = fitted,
      residuals = residuals,
      lambda = lambda,
      fdataobj = fdataobj,
      y = y
    ),
    class = "fregre.fd"
  )
}

#' Nonparametric Functional Regression
#'
#' Fits a functional regression model using kernel smoothing.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param h Bandwidth parameter. If NULL, computed automatically.
#' @param metric Distance metric function. Default is metric.lp.
#' @param ... Additional arguments passed to metric function.
#'
#' @return A fitted regression object.
#'
#' @export
fregre.np <- function(fdataobj, y, h = NULL, metric = metric.lp, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Compute distance matrix
  D <- metric(fdataobj, ...)

  # Auto-select bandwidth if not provided
  if (is.null(h)) {
    # Use median of non-zero distances
    d_vec <- D[lower.tri(D)]
    h <- median(d_vec[d_vec > 0])
  }

  # Nadaraya-Watson estimator
  fitted <- numeric(n)
  for (i in seq_len(n)) {
    # Gaussian kernel weights
    weights <- exp(-0.5 * (D[i, ] / h)^2)
    weights[i] <- 0  # Leave-one-out
    weights <- weights / sum(weights)
    fitted[i] <- sum(weights * y)
  }

  structure(
    list(
      fitted.values = fitted,
      residuals = y - fitted,
      h = h,
      fdataobj = fdataobj,
      y = y,
      D = D
    ),
    class = "fregre.np"
  )
}

#' Print method for fregre objects
#' @export
print.fregre.fd <- function(x, ...) {
  cat("Functional regression model\n")
  cat("  Number of observations:", length(x$y), "\n")
  cat("  R-squared:", 1 - sum(x$residuals^2) / sum((x$y - mean(x$y))^2), "\n")
  invisible(x)
}

#' Print method for fregre.np objects
#' @export
print.fregre.np <- function(x, ...) {
  cat("Nonparametric functional regression model\n")
  cat("  Number of observations:", length(x$y), "\n")
  cat("  Bandwidth:", x$h, "\n")
  cat("  R-squared:", 1 - sum(x$residuals^2) / sum((x$y - mean(x$y))^2), "\n")
  invisible(x)
}

#' Cross-Validation for Functional PC Regression
#'
#' Performs k-fold cross-validation to select the optimal number of
#' principal components for functional PC regression.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param kfold Number of folds for cross-validation (default 10).
#' @param ncomp.range Range of number of components to try.
#'   Default is 1 to min(n-1, ncol(data)).
#' @param seed Random seed for fold assignment.
#' @param ... Additional arguments passed to fregre.pc.
#'
#' @return A list containing:
#'   \item{optimal.ncomp}{Optimal number of components}
#'   \item{cv.errors}{Mean squared prediction error for each ncomp}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal ncomp}
#'
#' @export
#' @examples
#' # Create functional data with a linear relationship
#' set.seed(42)
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 100, 50)
#' for (i in 1:100) X[i, ] <- sin(2*pi*t) * i/100 + rnorm(50, sd = 0.1)
#' beta_true <- cos(2*pi*t)
#' y <- X %*% beta_true + rnorm(100, sd = 0.5)
#' fd <- fdata(X, argvals = t)
#'
#' # Cross-validate to find optimal number of PCs
#' cv_result <- fregre.pc.cv(fd, y, ncomp.range = 1:10)
fregre.pc.cv <- function(fdataobj, y, kfold = 10, ncomp.range = NULL,
                         seed = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Set default ncomp.range
  if (is.null(ncomp.range)) {
    max_comp <- min(n - 1, m)
    ncomp.range <- seq_len(min(max_comp, 15))
  }
  ncomp.range <- ncomp.range[ncomp.range < n & ncomp.range <= m]

  if (length(ncomp.range) == 0) {
    stop("No valid values in ncomp.range")
  }

  # Create fold assignments
  if (!is.null(seed)) set.seed(seed)
  folds <- sample(rep(seq_len(kfold), length.out = n))

  # Initialize storage for CV errors
  cv_errors <- matrix(0, nrow = length(ncomp.range), ncol = kfold)

  for (k in seq_len(kfold)) {
    # Split data
    test_idx <- which(folds == k)
    train_idx <- which(folds != k)

    # Create train/test fdata objects
    fd_train <- fdataobj[train_idx, ]
    fd_test <- fdataobj[test_idx, ]
    y_train <- y[train_idx]
    y_test <- y[test_idx]

    for (j in seq_along(ncomp.range)) {
      ncomp <- ncomp.range[j]

      # Fit model on training data
      tryCatch({
        fit <- fregre.pc(fd_train, y_train, ncomp = ncomp, ...)

        # Predict on test data
        y_pred <- as.vector(fd_test$data %*% fit$coefficients) + fit$intercept

        # Compute MSE
        cv_errors[j, k] <- mean((y_test - y_pred)^2)
      }, error = function(e) {
        cv_errors[j, k] <<- NA
      })
    }
  }

  # Compute mean and SE of CV errors
  mean_cv <- rowMeans(cv_errors, na.rm = TRUE)
  se_cv <- apply(cv_errors, 1, sd, na.rm = TRUE) / sqrt(kfold)
  names(mean_cv) <- names(se_cv) <- ncomp.range

  # Find optimal ncomp (minimum CV error)
  optimal_idx <- which.min(mean_cv)
  optimal_ncomp <- ncomp.range[optimal_idx]

  # Fit final model with optimal ncomp
  final_model <- fregre.pc(fdataobj, y, ncomp = optimal_ncomp, ...)

  list(
    optimal.ncomp = optimal_ncomp,
    cv.errors = mean_cv,
    cv.se = se_cv,
    model = final_model
  )
}

#' Cross-Validation for Functional Basis Regression
#'
#' Performs k-fold cross-validation to select the optimal regularization
#' parameter (lambda) for functional basis regression.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param kfold Number of folds for cross-validation (default 10).
#' @param lambda.range Range of lambda values to try.
#'   Default is 10^seq(-4, 4, length.out = 20).
#' @param seed Random seed for fold assignment.
#' @param ... Additional arguments passed to fregre.basis.
#'
#' @return A list containing:
#'   \item{optimal.lambda}{Optimal regularization parameter}
#'   \item{cv.errors}{Mean squared prediction error for each lambda}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal lambda}
#'
#' @export
fregre.basis.cv <- function(fdataobj, y, kfold = 10, lambda.range = NULL,
                            seed = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)

  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Set default lambda.range
  if (is.null(lambda.range)) {
    lambda.range <- 10^seq(-4, 4, length.out = 20)
  }

  # Create fold assignments
  if (!is.null(seed)) set.seed(seed)
  folds <- sample(rep(seq_len(kfold), length.out = n))

  # Initialize storage for CV errors
  cv_errors <- matrix(0, nrow = length(lambda.range), ncol = kfold)

  for (k in seq_len(kfold)) {
    # Split data
    test_idx <- which(folds == k)
    train_idx <- which(folds != k)

    # Create train/test fdata objects
    fd_train <- fdataobj[train_idx, ]
    fd_test <- fdataobj[test_idx, ]
    y_train <- y[train_idx]
    y_test <- y[test_idx]

    for (j in seq_along(lambda.range)) {
      lambda <- lambda.range[j]

      # Fit model on training data
      tryCatch({
        fit <- fregre.basis(fd_train, y_train, lambda = lambda, ...)

        # Predict on test data
        y_pred <- as.vector(fd_test$data %*% fit$coefficients)

        # Compute MSE
        cv_errors[j, k] <- mean((y_test - y_pred)^2)
      }, error = function(e) {
        cv_errors[j, k] <<- NA
      })
    }
  }

  # Compute mean and SE of CV errors
  mean_cv <- rowMeans(cv_errors, na.rm = TRUE)
  se_cv <- apply(cv_errors, 1, sd, na.rm = TRUE) / sqrt(kfold)
  names(mean_cv) <- names(se_cv) <- lambda.range

  # Find optimal lambda (minimum CV error)
  optimal_idx <- which.min(mean_cv)
  optimal_lambda <- lambda.range[optimal_idx]

  # Fit final model with optimal lambda
  final_model <- fregre.basis(fdataobj, y, lambda = optimal_lambda, ...)

  list(
    optimal.lambda = optimal_lambda,
    cv.errors = mean_cv,
    cv.se = se_cv,
    model = final_model
  )
}

#' Cross-Validation for Nonparametric Functional Regression
#'
#' Performs k-fold cross-validation to select the optimal bandwidth
#' parameter (h) for nonparametric functional regression.
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param kfold Number of folds for cross-validation (default 10).
#' @param h.range Range of bandwidth values to try. If NULL, automatically
#'   determined from the distance matrix.
#' @param metric Distance metric function. Default is metric.lp.
#' @param seed Random seed for fold assignment.
#' @param ... Additional arguments passed to the metric function.
#'
#' @return A list containing:
#'   \item{optimal.h}{Optimal bandwidth parameter}
#'   \item{cv.errors}{Mean squared prediction error for each h}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal h}
#'
#' @export
fregre.np.cv <- function(fdataobj, y, kfold = 10, h.range = NULL,
                         metric = metric.lp, seed = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)

  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Compute full distance matrix once
  D <- metric(fdataobj, ...)

  # Set default h.range based on distances
  if (is.null(h.range)) {
    d_vec <- D[lower.tri(D)]
    d_vec <- d_vec[d_vec > 0]
    h.range <- quantile(d_vec, probs = seq(0.05, 0.95, length.out = 20))
  }

  # Create fold assignments
  if (!is.null(seed)) set.seed(seed)
  folds <- sample(rep(seq_len(kfold), length.out = n))

  # Initialize storage for CV errors
  cv_errors <- matrix(0, nrow = length(h.range), ncol = kfold)

  for (k in seq_len(kfold)) {
    # Split data
    test_idx <- which(folds == k)
    train_idx <- which(folds != k)

    y_train <- y[train_idx]
    y_test <- y[test_idx]

    # Extract sub-distance matrices
    D_train <- D[train_idx, train_idx]
    D_test_train <- D[test_idx, train_idx, drop = FALSE]

    for (j in seq_along(h.range)) {
      h <- h.range[j]

      tryCatch({
        # Nadaraya-Watson prediction for test set
        n_test <- length(test_idx)
        y_pred <- numeric(n_test)

        for (i in seq_len(n_test)) {
          # Gaussian kernel weights
          weights <- exp(-0.5 * (D_test_train[i, ] / h)^2)
          weights <- weights / sum(weights)
          y_pred[i] <- sum(weights * y_train)
        }

        # Compute MSE
        cv_errors[j, k] <- mean((y_test - y_pred)^2)
      }, error = function(e) {
        cv_errors[j, k] <<- NA
      })
    }
  }

  # Compute mean and SE of CV errors
  mean_cv <- rowMeans(cv_errors, na.rm = TRUE)
  se_cv <- apply(cv_errors, 1, sd, na.rm = TRUE) / sqrt(kfold)
  names(mean_cv) <- names(se_cv) <- h.range

  # Find optimal h (minimum CV error)
  optimal_idx <- which.min(mean_cv)
  optimal_h <- h.range[optimal_idx]

  # Fit final model with optimal h
  final_model <- fregre.np(fdataobj, y, h = optimal_h, metric = metric, ...)

  list(
    optimal.h = optimal_h,
    cv.errors = mean_cv,
    cv.se = se_cv,
    model = final_model
  )
}
