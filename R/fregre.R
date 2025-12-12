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
#' @return A fitted regression object of class 'fregre.fd' with components:
#'   \item{coefficients}{Beta coefficient function values}
#'   \item{intercept}{Intercept term}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{ncomp}{Number of components used}
#'   \item{mean.X}{Mean of functional covariate (for prediction)}
#'   \item{mean.y}{Mean of response (for prediction)}
#'   \item{rotation}{PC loadings (for prediction)}
#'   \item{l}{Indices of selected components}
#'   \item{lm}{Underlying linear model}
#'   \item{sr2}{Residual variance}
#'   \item{fdataobj}{Original functional data}
#'   \item{y}{Response vector}
#'   \item{call}{The function call}
#'
#' @export
fregre.pc <- function(fdataobj, y, ncomp = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fregre.pc not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  if (length(y) != n) {
    stop("Length of y must equal number of curves in fdataobj")
  }

  # Store means for prediction
  X_mean <- colMeans(fdataobj$data)
  y_mean <- mean(y)

  # Center the functional data
  X_centered <- scale(fdataobj$data, center = TRUE, scale = FALSE)
  y_centered <- y - y_mean

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
  l <- seq_len(ncomp)

  # Get PC scores (rotation/loadings)
  rotation <- svd_result$v[, l, drop = FALSE]
  scores <- X_centered %*% rotation
  colnames(scores) <- paste0("PC", l)

  # Fit OLS on scores
  scores_df <- as.data.frame(scores)
  lm_fit <- lm(y_centered ~ ., data = scores_df)

  # Compute beta coefficient function
  beta_coef <- rotation %*% coef(lm_fit)[-1]  # Exclude intercept

  # Compute intercept and fitted values
  intercept <- as.numeric(y_mean - sum(X_mean * beta_coef))
  fitted_values <- as.vector(fdataobj$data %*% beta_coef) + intercept

  # Residual variance
  residuals <- y - fitted_values
  sr2 <- sum(residuals^2) / (n - ncomp - 1)

  structure(
    list(
      coefficients = beta_coef,
      intercept = intercept,
      fitted.values = fitted_values,
      residuals = residuals,
      ncomp = ncomp,
      mean.X = X_mean,
      mean.y = y_mean,
      rotation = rotation,
      l = l,
      lm = lm_fit,
      sr2 = sr2,
      svd = svd_result,
      fdataobj = fdataobj,
      y = y,
      call = match.call()
    ),
    class = "fregre.fd"
  )
}

#' Functional Basis Regression
#'
#' Fits a functional linear model using basis expansion (ridge regression).
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param basis.x Basis for the functional covariate (currently ignored).
#' @param basis.b Basis for the coefficient function (currently ignored).
#' @param lambda Smoothing/regularization parameter.
#' @param ... Additional arguments.
#'
#' @return A fitted regression object of class 'fregre.fd' with components:
#'   \item{coefficients}{Beta coefficient function values}
#'   \item{intercept}{Intercept term}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{lambda}{Regularization parameter used}
#'   \item{mean.X}{Mean of functional covariate (for prediction)}
#'   \item{mean.y}{Mean of response (for prediction)}
#'   \item{sr2}{Residual variance}
#'   \item{fdataobj}{Original functional data}
#'   \item{y}{Response vector}
#'   \item{call}{The function call}
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

  # Store means for prediction
  X_mean <- colMeans(fdataobj$data)
  y_mean <- mean(y)

  # Center the data
  X_centered <- scale(fdataobj$data, center = TRUE, scale = FALSE)
  y_centered <- y - y_mean

  # Ridge regression on centered data: beta = (X'X + lambda*I)^-1 X'y
  XtX <- crossprod(X_centered)
  Xty <- crossprod(X_centered, y_centered)

  if (lambda > 0) {
    XtX <- XtX + lambda * diag(m)
  }

  beta <- solve(XtX, Xty)

  # Compute intercept and fitted values
  intercept <- as.numeric(y_mean - sum(X_mean * beta))
  fitted <- as.vector(fdataobj$data %*% beta) + intercept
  residuals <- y - fitted

  # Residual variance
  df <- n - m - 1
  if (df <= 0) df <- 1
  sr2 <- sum(residuals^2) / df

  structure(
    list(
      coefficients = beta,
      intercept = intercept,
      fitted.values = fitted,
      residuals = residuals,
      lambda = lambda,
      mean.X = X_mean,
      mean.y = y_mean,
      sr2 = sr2,
      fdataobj = fdataobj,
      y = y,
      call = match.call()
    ),
    class = "fregre.fd"
  )
}

#' Nonparametric Functional Regression
#'
#' Fits a functional regression model using kernel smoothing (Nadaraya-Watson).
#'
#' @param fdataobj An object of class 'fdata' (functional covariate).
#' @param y Response vector.
#' @param h Bandwidth parameter. If NULL, computed automatically.
#' @param Ker Kernel type for smoothing. Default is "norm" (Gaussian).
#' @param metric Distance metric function. Default is metric.lp.
#' @param ... Additional arguments passed to metric function.
#'
#' @return A fitted regression object of class 'fregre.np' with components:
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{h.opt}{Optimal/used bandwidth}
#'   \item{Ker}{Kernel type used}
#'   \item{fdataobj}{Original functional data}
#'   \item{y}{Response vector}
#'   \item{mdist}{Distance matrix}
#'   \item{H}{Hat/smoother matrix}
#'   \item{sr2}{Residual variance}
#'   \item{metric}{Metric function used}
#'   \item{call}{The function call}
#'
#' @export
fregre.np <- function(fdataobj, y, h = NULL, Ker = "norm", metric = metric.lp, ...) {
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

  # Compute hat matrix H and fitted values
  H <- matrix(0, n, n)
  fitted <- numeric(n)

  for (i in seq_len(n)) {
    # Gaussian kernel weights
    weights <- exp(-0.5 * (D[i, ] / h)^2)
    weights[i] <- 0  # Leave-one-out for fitting
    if (sum(weights) > 0) {
      weights <- weights / sum(weights)
    }
    H[i, ] <- weights
    fitted[i] <- sum(weights * y)
  }

  residuals <- y - fitted

  # Residual variance
  df <- n - sum(diag(H))
  if (df <= 0) df <- 1
  sr2 <- sum(residuals^2) / df

  structure(
    list(
      fitted.values = fitted,
      residuals = residuals,
      h.opt = h,
      Ker = Ker,
      fdataobj = fdataobj,
      y = y,
      mdist = D,
      H = H,
      sr2 = sr2,
      metric = metric,
      call = match.call()
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
#' @return A list with components:
#' \describe{
#'   \item{optimal.ncomp}{Optimal number of components}
#'   \item{cv.errors}{Mean squared prediction error for each ncomp}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal ncomp}
#' }
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
#' @return A list with components:
#' \describe{
#'   \item{optimal.lambda}{Optimal regularization parameter}
#'   \item{cv.errors}{Mean squared prediction error for each lambda}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal lambda}
#' }
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
#' @return A list with components:
#' \describe{
#'   \item{optimal.h}{Optimal bandwidth parameter}
#'   \item{cv.errors}{Mean squared prediction error for each h}
#'   \item{cv.se}{Standard error of cv.errors}
#'   \item{model}{Fitted model with optimal h}
#' }
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

# =============================================================================
# Predict Methods
# =============================================================================

#' Predict Method for Functional Regression (fregre.fd)
#'
#' Predictions from a fitted functional regression model (fregre.pc or fregre.basis).
#'
#' @param object A fitted model object of class 'fregre.fd'.
#' @param newdata An fdata object containing new functional data for prediction.
#'   If NULL, returns fitted values from training data.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of predicted values.
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:30) X[i, ] <- sin(2*pi*t) * i/30 + rnorm(50, sd = 0.1)
#' y <- rowMeans(X) + rnorm(30, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Fit model
#' fit <- fregre.pc(fd, y, ncomp = 3)
#'
#' # Predict on new data
#' X_new <- matrix(sin(2*pi*t) * 0.5, nrow = 1)
#' fd_new <- fdata(X_new, argvals = t)
#' predict(fit, fd_new)
predict.fregre.fd <- function(object, newdata = NULL, ...) {
  if (is.null(object)) {
    stop("No fregre.fd object provided")
  }

  # If no new data, return fitted values
  if (is.null(newdata)) {
    return(object$fitted.values)
  }

  # Convert to fdata if needed
  if (!inherits(newdata, "fdata")) {
    newdata <- fdata(newdata,
                     argvals = object$fdataobj$argvals,
                     rangeval = object$fdataobj$rangeval)
  }

  # Get new data matrix
  X_new <- newdata$data
  nn <- nrow(X_new)

  # Check dimensions match
  if (ncol(X_new) != ncol(object$fdataobj$data)) {
    stop("Number of evaluation points in newdata must match training data")
  }

  # Compute predictions using stored coefficients and intercept
  if (!is.null(object$intercept)) {
    y_pred <- as.vector(X_new %*% object$coefficients) + object$intercept
  } else {
    y_pred <- as.vector(X_new %*% object$coefficients)
  }

  names(y_pred) <- rownames(X_new)
  y_pred
}

#' Predict Method for Nonparametric Functional Regression (fregre.np)
#'
#' Predictions from a fitted nonparametric functional regression model.
#'
#' @param object A fitted model object of class 'fregre.np'.
#' @param newdata An fdata object containing new functional data for prediction.
#'   If NULL, returns fitted values from training data.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of predicted values.
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:30) X[i, ] <- sin(2*pi*t) * i/30 + rnorm(50, sd = 0.1)
#' y <- rowMeans(X) + rnorm(30, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Fit model
#' fit <- fregre.np(fd, y)
#'
#' # Predict on new data
#' X_new <- matrix(sin(2*pi*t) * 0.5, nrow = 1)
#' fd_new <- fdata(X_new, argvals = t)
#' predict(fit, fd_new)
predict.fregre.np <- function(object, newdata = NULL, ...) {
  if (is.null(object)) {
    stop("No fregre.np object provided")
  }

  # If no new data, return fitted values
  if (is.null(newdata)) {
    return(object$fitted.values)
  }

  # Convert to fdata if needed
  if (!inherits(newdata, "fdata")) {
    newdata <- fdata(newdata,
                     argvals = object$fdataobj$argvals,
                     rangeval = object$fdataobj$rangeval)
  }

  # Check dimensions match
  if (ncol(newdata$data) != ncol(object$fdataobj$data)) {
    stop("Number of evaluation points in newdata must match training data")
  }

  nn <- nrow(newdata$data)
  n_train <- nrow(object$fdataobj$data)
  h <- object$h.opt
  y_train <- object$y

  # Compute distances from new data to training data
  # Use the same metric that was used for fitting
  metric_fn <- object$metric
  D_new <- metric_fn(newdata, object$fdataobj)

  # Nadaraya-Watson prediction
  y_pred <- numeric(nn)
  for (i in seq_len(nn)) {
    # Gaussian kernel weights
    weights <- exp(-0.5 * (D_new[i, ] / h)^2)
    if (sum(weights) > 0) {
      weights <- weights / sum(weights)
    } else {
      # If all weights are zero, use uniform weights
      weights <- rep(1/n_train, n_train)
    }
    y_pred[i] <- sum(weights * y_train)
  }

  names(y_pred) <- rownames(newdata$data)
  y_pred
}
