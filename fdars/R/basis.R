#' Basis Representation Functions for Functional Data
#'
#' Functions for representing functional data using basis expansions,
#' including B-splines, Fourier bases, and P-splines with penalization.

# ==============================================================================
# Reconstruction: basis2fdata
# ==============================================================================

#' Reconstruct Functional Data from Basis Coefficients
#'
#' Given basis coefficients, reconstruct the functional data by evaluating
#' the basis expansion at specified argument values.
#'
#' @param coefs Coefficient matrix \code{[n x nbasis]} where n is the number
#'   of curves and nbasis is the number of basis functions. Can also be a
#'   vector for a single curve.
#' @param argvals Numeric vector of evaluation points for reconstruction.
#' @param nbasis Number of basis functions. If NULL, inferred from ncol(coefs).
#' @param type Basis type: "bspline" (default) or "fourier".
#' @param rangeval Range of argvals. Default: \code{range(argvals)}.
#'
#' @return An fdata object with reconstructed curves.
#'
#' @details
#' The reconstruction computes \code{X(t) = sum(coef_k * B_k(t))} where
#' \code{B_k} are the basis functions evaluated at \code{argvals}.
#'
#' @export
#' @examples
#' # Create some functional data
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(sin(2 * pi * t), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Project to basis and reconstruct
#' coefs <- fdata2basis(fd, nbasis = 15, type = "fourier")
#' fd_recon <- basis2fdata(coefs, argvals = t, type = "fourier")
basis2fdata <- function(coefs, argvals, nbasis = NULL, type = c("bspline", "fourier"),
                        rangeval = NULL) {
  type <- match.arg(type)

  # Handle vector input (single curve)
  if (is.vector(coefs)) {
    coefs <- matrix(coefs, nrow = 1)
  }

  n <- nrow(coefs)
  if (is.null(nbasis)) {
    nbasis <- ncol(coefs)
  }

  if (ncol(coefs) != nbasis) {
    stop("Number of columns in coefs must equal nbasis")
  }

  if (is.null(rangeval)) {
    rangeval <- range(argvals)
  }

  # Call Rust backend for reconstruction
  basis_type <- if (type == "fourier") 1L else 0L
  data_mat <- .Call("wrap__basis2fdata_1d", coefs, argvals, as.integer(nbasis), basis_type)

  # Create fdata object
  fdata(data_mat, argvals = argvals, rangeval = rangeval)
}

# ==============================================================================
# Goodness-of-fit metrics: GCV, AIC, BIC
# ==============================================================================

#' GCV Score for Basis Representation
#'
#' Computes the Generalized Cross-Validation score for a basis representation.
#' Lower GCV indicates better fit with appropriate complexity.
#'
#' @param fdataobj An fdata object.
#' @param nbasis Number of basis functions.
#' @param type Basis type: "bspline" (default) or "fourier".
#' @param lambda Smoothing/penalty parameter (default 0, no penalty).
#' @param pooled Logical. If TRUE (default), compute a single GCV across all
#'   curves. If FALSE, compute GCV for each curve and return the mean.
#'
#' @return The GCV score (scalar).
#'
#' @details
#' GCV is computed as:
#' \deqn{GCV = \frac{RSS/n}{(1 - edf/n)^2}}
#' where RSS is the residual sum of squares and edf is the effective
#' degrees of freedom (trace of the hat matrix).
#'
#' When \code{pooled = TRUE}, the criterion is computed globally across all
#' curves. When \code{pooled = FALSE}, the criterion is computed for each
#' curve separately and the mean is returned. Use \code{pooled = FALSE} when
#' curves have heterogeneous noise levels.
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(sin(2 * pi * t) + rnorm(50, sd = 0.1), nrow = 1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compare GCV for different nbasis
#' gcv_5 <- basis.gcv(fd, nbasis = 5)
#' gcv_10 <- basis.gcv(fd, nbasis = 10)
#' gcv_20 <- basis.gcv(fd, nbasis = 20)
basis.gcv <- function(fdataobj, nbasis, type = c("bspline", "fourier"),
                      lambda = 0, pooled = TRUE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("basis.gcv not yet implemented for 2D functional data")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  .Call("wrap__basis_gcv_1d", fdataobj$data, fdataobj$argvals,
        as.integer(nbasis), basis_type, as.double(lambda), as.logical(pooled))
}

#' AIC for Basis Representation
#'
#' Computes the Akaike Information Criterion for a basis representation.
#' Lower AIC indicates better model (balancing fit and complexity).
#'
#' @param fdataobj An fdata object.
#' @param nbasis Number of basis functions.
#' @param type Basis type: "bspline" (default) or "fourier".
#' @param lambda Smoothing/penalty parameter (default 0).
#' @param pooled Logical. If TRUE (default), compute a single AIC across all
#'   curves. If FALSE, compute AIC for each curve and return the mean.
#'
#' @return The AIC value (scalar).
#'
#' @details
#' AIC is computed as:
#' \deqn{AIC = n \log(RSS/n) + 2 \cdot edf}
#'
#' When \code{pooled = TRUE}, the criterion uses total observations and total
#' effective degrees of freedom (n_curves * edf). When \code{pooled = FALSE},
#' the criterion is computed for each curve separately and the mean is returned.
#'
#' @export
basis.aic <- function(fdataobj, nbasis, type = c("bspline", "fourier"),
                      lambda = 0, pooled = TRUE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("basis.aic not yet implemented for 2D functional data")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  .Call("wrap__basis_aic_1d", fdataobj$data, fdataobj$argvals,
        as.integer(nbasis), basis_type, as.double(lambda), as.logical(pooled))
}

#' BIC for Basis Representation
#'
#' Computes the Bayesian Information Criterion for a basis representation.
#' BIC penalizes complexity more strongly than AIC for larger samples.
#'
#' @param fdataobj An fdata object.
#' @param nbasis Number of basis functions.
#' @param type Basis type: "bspline" (default) or "fourier".
#' @param lambda Smoothing/penalty parameter (default 0).
#' @param pooled Logical. If TRUE (default), compute a single BIC across all
#'   curves. If FALSE, compute BIC for each curve and return the mean.
#'
#' @return The BIC value (scalar).
#'
#' @details
#' BIC is computed as:
#' \deqn{BIC = n \log(RSS/n) + \log(n) \cdot edf}
#'
#' When \code{pooled = TRUE}, the criterion uses total observations and total
#' effective degrees of freedom (n_curves * edf). When \code{pooled = FALSE},
#' the criterion is computed for each curve separately and the mean is returned.
#'
#' @export
basis.bic <- function(fdataobj, nbasis, type = c("bspline", "fourier"),
                      lambda = 0, pooled = TRUE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("basis.bic not yet implemented for 2D functional data")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  .Call("wrap__basis_bic_1d", fdataobj$data, fdataobj$argvals,
        as.integer(nbasis), basis_type, as.double(lambda), as.logical(pooled))
}

# ==============================================================================
# Cross-validation for nbasis selection
# ==============================================================================

#' Cross-Validation for Basis Function Number Selection
#'
#' Selects the optimal number of basis functions using k-fold cross-validation
#' or generalized cross-validation.
#'
#' @param fdataobj An fdata object.
#' @param nbasis.range Vector of nbasis values to evaluate (default: 4:20).
#' @param type Basis type: "bspline" (default) or "fourier".
#' @param criterion Selection criterion: "GCV" (default), "CV", "AIC", or "BIC".
#' @param kfold Number of folds for k-fold CV (default 10). Ignored if
#'   criterion is "GCV", "AIC", or "BIC".
#' @param lambda Smoothing parameter (default 0).
#'
#' @return A list with:
#'   \describe{
#'     \item{optimal.nbasis}{Optimal number of basis functions}
#'     \item{scores}{Score for each nbasis value}
#'     \item{nbasis.range}{The tested nbasis values}
#'     \item{criterion}{The criterion used}
#'     \item{fitted}{fdata object fitted with optimal nbasis}
#'   }
#'
#' @export
#' @examples
#' set.seed(42)
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(4 * pi * t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Find optimal nbasis
#' cv_result <- fdata2basis.cv(fd, nbasis.range = 5:15, type = "fourier")
#' print(cv_result$optimal.nbasis)
fdata2basis.cv <- function(fdataobj, nbasis.range = 4:20,
                           type = c("bspline", "fourier"),
                           criterion = c("GCV", "CV", "AIC", "BIC"),
                           kfold = 10, lambda = 0) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fdata2basis.cv not yet implemented for 2D functional data")
  }

  type <- match.arg(type)
  criterion <- match.arg(criterion)

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  # Compute scores for each nbasis
  scores <- numeric(length(nbasis.range))

  for (i in seq_along(nbasis.range)) {
    nb <- nbasis.range[i]

    if (criterion == "GCV") {
      scores[i] <- basis.gcv(fdataobj, nbasis = nb, type = type, lambda = lambda)
    } else if (criterion == "AIC") {
      scores[i] <- basis.aic(fdataobj, nbasis = nb, type = type, lambda = lambda)
    } else if (criterion == "BIC") {
      scores[i] <- basis.bic(fdataobj, nbasis = nb, type = type, lambda = lambda)
    } else {
      # k-fold CV
      scores[i] <- .kfold_cv_basis(fdataobj, nbasis = nb, type = type,
                                   kfold = kfold, lambda = lambda)
    }
  }

  # Find optimal
  optimal_idx <- which.min(scores)
  optimal_nbasis <- nbasis.range[optimal_idx]

  # Fit with optimal nbasis
  coefs <- fdata2basis(fdataobj, nbasis = optimal_nbasis, type = type)
  fitted <- basis2fdata(coefs, argvals = fdataobj$argvals, type = type)

  structure(
    list(
      optimal.nbasis = optimal_nbasis,
      scores = scores,
      nbasis.range = nbasis.range,
      criterion = criterion,
      fitted = fitted,
      coefs = coefs
    ),
    class = "basis.cv"
  )
}

#' @noRd
.kfold_cv_basis <- function(fdataobj, nbasis, type, kfold, lambda) {
  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  # Create fold assignments
  folds <- sample(rep(1:kfold, length.out = n))

  cv_errors <- numeric(kfold)

  for (k in 1:kfold) {
    # Training and test sets
    train_idx <- which(folds != k)
    test_idx <- which(folds == k)

    if (length(test_idx) == 0) next

    train_fd <- fdataobj[train_idx, ]
    test_fd <- fdataobj[test_idx, ]

    # Fit on training, predict on test
    train_coefs <- fdata2basis(train_fd, nbasis = nbasis, type = type)
    train_fitted <- basis2fdata(train_coefs, argvals = fdataobj$argvals, type = type)

    # For test: project to basis using training basis matrix, then reconstruct
    test_coefs <- fdata2basis(test_fd, nbasis = nbasis, type = type)
    test_fitted <- basis2fdata(test_coefs, argvals = fdataobj$argvals, type = type)

    # Compute MSE
    cv_errors[k] <- mean((test_fd$data - test_fitted$data)^2)
  }

  mean(cv_errors, na.rm = TRUE)
}

#' Print method for basis.cv objects
#' @param x A basis.cv object.
#' @param ... Additional arguments (ignored).
#' @export
print.basis.cv <- function(x, ...) {
  cat("Basis Cross-Validation Results\n")
  cat("==============================\n")
  cat("Criterion:", x$criterion, "\n")
  cat("Optimal nbasis:", x$optimal.nbasis, "\n")
  cat("Score at optimal:", x$scores[which(x$nbasis.range == x$optimal.nbasis)], "\n")
  cat("Range tested:", min(x$nbasis.range), "-", max(x$nbasis.range), "\n")
  invisible(x)
}

#' Plot method for basis.cv objects
#' @param x A basis.cv object.
#' @param ... Additional arguments (ignored).
#' @export
plot.basis.cv <- function(x, ...) {
  df <- data.frame(
    nbasis = x$nbasis.range,
    score = x$scores
  )

  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$nbasis, y = .data$score)) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::geom_vline(xintercept = x$optimal.nbasis, linetype = "dashed", color = "red") +
    ggplot2::labs(
      title = paste(x$criterion, "Score vs Number of Basis Functions"),
      x = "Number of Basis Functions",
      y = paste(x$criterion, "Score")
    ) +
    ggplot2::annotate("text", x = x$optimal.nbasis, y = max(df$score),
                      label = paste("optimal =", x$optimal.nbasis),
                      hjust = -0.1, color = "red")

  p
}

# ==============================================================================
# P-splines (Penalized B-splines)
# ==============================================================================

#' P-spline Smoothing for Functional Data
#'
#' Fits penalized B-splines (P-splines) to functional data with automatic
#' or manual selection of the smoothing parameter.
#'
#' @param fdataobj An fdata object.
#' @param nbasis Number of B-spline basis functions (default 20).
#' @param lambda Smoothing parameter. Higher values give smoother curves.
#'   If NULL and lambda.select = TRUE, selected automatically.
#' @param order Order of the difference penalty (default 2, for second
#'   derivative penalty).
#' @param lambda.select Logical. If TRUE, select lambda automatically
#'   using the specified criterion.
#' @param criterion Criterion for lambda selection: "GCV" (default),
#'   "AIC", or "BIC".
#' @param lambda.range Range of lambda values to search (log10 scale).
#'   Default: \code{10^seq(-4, 4, length.out = 50)}.
#'
#' @return A list of class "pspline" with:
#'   \describe{
#'     \item{fdata}{Smoothed fdata object}
#'     \item{coefs}{Coefficient matrix}
#'     \item{lambda}{Used or selected lambda value}
#'     \item{edf}{Effective degrees of freedom}
#'     \item{gcv/aic/bic}{Criterion values}
#'     \item{nbasis}{Number of basis functions used}
#'   }
#'
#' @details
#' P-splines minimize:
#' \deqn{||y - B c||^2 + \lambda c' D' D c}
#' where B is the B-spline basis matrix, c are coefficients, and D is the
#' difference matrix of the specified order.
#'
#' @references
#' Eilers, P.H.C. and Marx, B.D. (1996). Flexible smoothing with B-splines
#' and penalties. \emph{Statistical Science}, 11(2), 89-121.
#'
#' @export
#' @examples
#' # Create noisy data
#' t <- seq(0, 1, length.out = 100)
#' true_signal <- sin(2 * pi * t)
#' noisy <- true_signal + rnorm(100, sd = 0.3)
#' fd <- fdata(matrix(noisy, nrow = 1), argvals = t)
#'
#' # Smooth with P-splines
#' result <- pspline(fd, nbasis = 20, lambda = 10)
#' plot(fd)
#' lines(t, result$fdata$data[1, ], col = "red", lwd = 2)
#'
#' # Automatic lambda selection
#' result_auto <- pspline(fd, nbasis = 20, lambda.select = TRUE)
pspline <- function(fdataobj, nbasis = 20, lambda = 1, order = 2,
                    lambda.select = FALSE, criterion = c("GCV", "AIC", "BIC"),
                    lambda.range = 10^seq(-4, 4, length.out = 50)) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("pspline not yet implemented for 2D functional data. Use pspline.2d()")
  }

  criterion <- match.arg(criterion)

  # Automatic lambda selection
  if (lambda.select || is.null(lambda)) {
    best_score <- Inf
    best_lambda <- lambda.range[1]

    for (lam in lambda.range) {
      result <- .Call("wrap__pspline_fit_1d", fdataobj$data, fdataobj$argvals,
                      as.integer(nbasis), as.double(lam), as.integer(order))

      if (criterion == "GCV") {
        score <- result$gcv
      } else if (criterion == "AIC") {
        score <- result$aic
      } else {
        score <- result$bic
      }

      if (score < best_score) {
        best_score <- score
        best_lambda <- lam
      }
    }

    lambda <- best_lambda
  }

  # Fit with selected/specified lambda
  result <- .Call("wrap__pspline_fit_1d", fdataobj$data, fdataobj$argvals,
                  as.integer(nbasis), as.double(lambda), as.integer(order))

  # Create smoothed fdata object
  smoothed_fdata <- fdata(result$fitted, argvals = fdataobj$argvals,
                          rangeval = fdataobj$rangeval, names = fdataobj$names)

  # Preserve metadata
  if (!is.null(fdataobj$id)) {
    smoothed_fdata$id <- fdataobj$id
  }
  if (!is.null(fdataobj$metadata)) {
    smoothed_fdata$metadata <- fdataobj$metadata
  }

  structure(
    list(
      fdata = smoothed_fdata,
      coefs = result$coefs,
      lambda = lambda,
      edf = result$edf,
      gcv = result$gcv,
      aic = result$aic,
      bic = result$bic,
      rss = result$rss,
      nbasis = nbasis,
      order = order
    ),
    class = "pspline"
  )
}

#' Print method for pspline objects
#' @param x A pspline object.
#' @param ... Additional arguments (ignored).
#' @export
print.pspline <- function(x, ...) {
  cat("P-spline Smoothing Results\n")
  cat("==========================\n")
  cat("Number of curves:", nrow(x$fdata$data), "\n")
  cat("Number of basis functions:", x$nbasis, "\n")
  cat("Penalty order:", x$order, "\n")
  cat("Lambda:", format(x$lambda, scientific = TRUE, digits = 3), "\n")
  cat("Effective df:", round(x$edf, 2), "\n")
  cat("GCV:", format(x$gcv, scientific = TRUE, digits = 4), "\n")
  invisible(x)
}

#' Plot method for pspline objects
#' @param x A pspline object.
#' @param ... Additional arguments passed to plot.fdata.
#' @export
plot.pspline <- function(x, ...) {
  plot(x$fdata, ...)
}

# ==============================================================================
# 2D Tensor Product Basis Functions
# ==============================================================================

#' Convert 2D Functional Data to Tensor Product Basis Coefficients
#'
#' Projects 2D functional data (surfaces) onto a tensor product basis,
#' which is the Kronecker product of two 1D bases.
#'
#' @param fdataobj A 2D fdata object (surfaces).
#' @param nbasis.s Number of basis functions in the s (first) direction.
#' @param nbasis.t Number of basis functions in the t (second) direction.
#' @param type Basis type: "bspline" (default) or "fourier".
#'
#' @return A matrix of coefficients \code{[n x (nbasis.s * nbasis.t)]}.
#'
#' @details
#' The tensor product basis is defined as:
#' \deqn{B_{2d}(s, t) = B_s(s) \otimes B_t(t)}
#'
#' @export
#' @examples
#' # Create 2D surface data
#' s <- seq(0, 1, length.out = 20)
#' t <- seq(0, 1, length.out = 20)
#' surface <- outer(sin(2*pi*s), cos(2*pi*t))
#' fd2d <- fdata(array(surface, dim = c(1, 20, 20)))
#'
#' # Project to tensor product basis
#' coefs <- fdata2basis.2d(fd2d, nbasis.s = 7, nbasis.t = 7, type = "fourier")
fdata2basis.2d <- function(fdataobj, nbasis.s = 10, nbasis.t = 10,
                           type = c("bspline", "fourier")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (!isTRUE(fdataobj$fdata2d)) {
    stop("fdataobj must be 2D functional data. Use fdata2basis() for 1D data.")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  argvals_s <- as.numeric(fdataobj$argvals[[1]])
  argvals_t <- as.numeric(fdataobj$argvals[[2]])

  # Ensure data is numeric (Rust expects doubles)
  data_mat <- as.matrix(fdataobj$data)
  storage.mode(data_mat) <- "double"

  .Call("wrap__fdata2basis_2d", data_mat, argvals_s, argvals_t,
        as.integer(nbasis.s), as.integer(nbasis.t), basis_type)
}

#' Reconstruct 2D Functional Data from Tensor Product Basis Coefficients
#'
#' Reconstructs 2D surfaces from tensor product basis coefficients.
#'
#' @param coefs Coefficient matrix \code{[n x (nbasis.s * nbasis.t)]}.
#' @param argvals List with two numeric vectors for s and t coordinates.
#' @param nbasis.s Number of basis functions in s direction.
#' @param nbasis.t Number of basis functions in t direction.
#' @param type Basis type: "bspline" (default) or "fourier".
#'
#' @return A 2D fdata object.
#'
#' @export
basis2fdata.2d <- function(coefs, argvals, nbasis.s, nbasis.t,
                           type = c("bspline", "fourier")) {
  if (!is.list(argvals) || length(argvals) != 2) {
    stop("argvals must be a list with two components for 2D data")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  # Handle vector input
  if (is.vector(coefs)) {
    coefs <- matrix(coefs, nrow = 1)
  }

  data_mat <- .Call("wrap__basis2fdata_2d", coefs, argvals[[1]], argvals[[2]],
                    as.integer(nbasis.s), as.integer(nbasis.t), basis_type)

  m1 <- length(argvals[[1]])
  m2 <- length(argvals[[2]])

  # Create 2D fdata object
  result <- list(
    data = data_mat,
    argvals = argvals,
    rangeval = list(
      s = range(argvals[[1]]),
      t = range(argvals[[2]])
    ),
    names = list(main = "", xlab = "s", ylab = "t", zlab = "X(s,t)"),
    fdata2d = TRUE,
    dims = c(m1, m2)
  )
  class(result) <- "fdata"
  result
}

#' P-spline Smoothing for 2D Functional Data
#'
#' Fits 2D P-splines with anisotropic penalties in both directions.
#'
#' @param fdataobj A 2D fdata object.
#' @param nbasis.s Number of B-spline basis functions in s direction.
#' @param nbasis.t Number of B-spline basis functions in t direction.
#' @param lambda.s Smoothing parameter in s direction.
#' @param lambda.t Smoothing parameter in t direction.
#' @param order Order of the difference penalty (default 2).
#' @param lambda.select Logical. If TRUE, select lambdas automatically.
#' @param criterion Criterion for selection: "GCV", "AIC", or "BIC".
#'
#' @return A list of class "pspline.2d" similar to \code{pspline()}.
#'
#' @details
#' The 2D penalty uses Kronecker product structure:
#' \deqn{P = \lambda_s (I_t \otimes P_s) + \lambda_t (P_t \otimes I_s)}
#'
#' @export
pspline.2d <- function(fdataobj, nbasis.s = 10, nbasis.t = 10,
                       lambda.s = 1, lambda.t = 1, order = 2,
                       lambda.select = FALSE, criterion = c("GCV", "AIC", "BIC")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (!isTRUE(fdataobj$fdata2d)) {
    stop("fdataobj must be 2D functional data. Use pspline() for 1D data.")
  }

  criterion <- match.arg(criterion)

  argvals_s <- fdataobj$argvals[[1]]
  argvals_t <- fdataobj$argvals[[2]]

  # Automatic lambda selection (grid search)
  if (lambda.select) {
    lambda_grid <- 10^seq(-2, 2, length.out = 10)
    best_score <- Inf
    best_lambda_s <- lambda.s
    best_lambda_t <- lambda.t

    for (ls in lambda_grid) {
      for (lt in lambda_grid) {
        result <- .Call("wrap__pspline_fit_2d", fdataobj$data,
                        argvals_s, argvals_t,
                        as.integer(nbasis.s), as.integer(nbasis.t),
                        as.double(ls), as.double(lt), as.integer(order))

        score <- switch(criterion,
                        GCV = result$gcv,
                        AIC = result$aic,
                        BIC = result$bic)

        if (score < best_score) {
          best_score <- score
          best_lambda_s <- ls
          best_lambda_t <- lt
        }
      }
    }

    lambda.s <- best_lambda_s
    lambda.t <- best_lambda_t
  }

  # Fit with selected/specified lambda
  result <- .Call("wrap__pspline_fit_2d", fdataobj$data,
                  argvals_s, argvals_t,
                  as.integer(nbasis.s), as.integer(nbasis.t),
                  as.double(lambda.s), as.double(lambda.t), as.integer(order))

  # Create smoothed fdata object
  m1 <- length(argvals_s)
  m2 <- length(argvals_t)

  smoothed_fdata <- list(
    data = result$fitted,
    argvals = fdataobj$argvals,
    rangeval = fdataobj$rangeval,
    names = fdataobj$names,
    fdata2d = TRUE,
    dims = c(m1, m2)
  )
  class(smoothed_fdata) <- "fdata"

  structure(
    list(
      fdata = smoothed_fdata,
      coefs = result$coefs,
      lambda.s = lambda.s,
      lambda.t = lambda.t,
      edf = result$edf,
      gcv = result$gcv,
      aic = result$aic,
      bic = result$bic,
      nbasis.s = nbasis.s,
      nbasis.t = nbasis.t,
      order = order
    ),
    class = "pspline.2d"
  )
}

#' Print method for pspline.2d objects
#' @param x A pspline.2d object.
#' @param ... Additional arguments (ignored).
#' @export
print.pspline.2d <- function(x, ...) {
  cat("2D P-spline Smoothing Results\n")
  cat("=============================\n")
  cat("Number of surfaces:", nrow(x$fdata$data), "\n")
  cat("Basis functions (s x t):", x$nbasis.s, "x", x$nbasis.t, "\n")
  cat("Penalty order:", x$order, "\n")
  cat("Lambda (s):", format(x$lambda.s, scientific = TRUE, digits = 3), "\n")
  cat("Lambda (t):", format(x$lambda.t, scientific = TRUE, digits = 3), "\n")
  cat("Effective df:", round(x$edf, 2), "\n")
  cat("GCV:", format(x$gcv, scientific = TRUE, digits = 4), "\n")
  invisible(x)
}

#' Plot method for pspline.2d objects
#' @param x A pspline.2d object.
#' @param ... Additional arguments passed to plot.fdata.
#' @export
plot.pspline.2d <- function(x, ...) {
  plot(x$fdata, ...)
}
