#' Distance Metrics for Functional Data
#'
#' Functions for computing various distance metrics between functional data.

#' Generic Distance Function for Functional Data
#'
#' Unified interface for computing various distance metrics between functional
#' data objects. This function dispatches to the appropriate specialized
#' distance function based on the method parameter.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, computes self-distances.
#' @param method Distance method to use. One of:
#'   \itemize{
#'     \item "lp" - Lp metric (default)
#'     \item "hausdorff" - Hausdorff distance
#'     \item "dtw" - Dynamic Time Warping
#'     \item "pca" - Semi-metric based on PCA scores
#'     \item "deriv" - Semi-metric based on derivatives
#'     \item "basis" - Semi-metric based on basis coefficients
#'     \item "fourier" - Semi-metric based on FFT coefficients
#'     \item "hshift" - Semi-metric with horizontal shift
#'     \item "kl" - Symmetric Kullback-Leibler divergence
#'   }
#' @param ... Additional arguments passed to the specific distance function.
#'
#' @return A distance matrix.
#'
#' @details
#' This function provides a convenient unified interface for all distance
#' computations in fdars. The additional arguments in \code{...} are passed
#' to the underlying distance function:
#'
#' \itemize{
#'   \item \code{metric.lp}: lp, w
#'   \item \code{metric.hausdorff}: (none)
#'   \item \code{metric.DTW}: p, w
#'   \item \code{semimetric.pca}: ncomp
#'   \item \code{semimetric.deriv}: nderiv, lp
#'   \item \code{semimetric.basis}: nbasis, basis, nderiv
#'   \item \code{semimetric.fourier}: nfreq
#'   \item \code{semimetric.hshift}: max_shift
#'   \item \code{metric.kl}: eps, normalize
#' }
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#'
#' # Using different distance methods
#' D_lp <- metric.dist(fd, method = "lp")
#' D_hausdorff <- metric.dist(fd, method = "hausdorff")
#' D_pca <- metric.dist(fd, method = "pca", ncomp = 3)
#'
#' # Cross-distances
#' fd2 <- fdata(matrix(rnorm(100), 10, 10))
#' D_cross <- metric.dist(fd, fd2, method = "lp")
metric.dist <- function(fdata1, fdata2 = NULL, method = "lp", ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  method <- match.arg(method, c("lp", "hausdorff", "dtw", "pca", "deriv",
                                 "basis", "fourier", "hshift", "kl"))

  switch(method,
    "lp" = metric.lp(fdata1, fdata2, ...),
    "hausdorff" = metric.hausdorff(fdata1, fdata2, ...),
    "dtw" = metric.DTW(fdata1, fdata2, ...),
    "pca" = semimetric.pca(fdata1, fdata2, ...),
    "deriv" = semimetric.deriv(fdata1, fdata2, ...),
    "basis" = semimetric.basis(fdata1, fdata2, ...),
    "fourier" = semimetric.fourier(fdata1, fdata2, ...),
    "hshift" = semimetric.hshift(fdata1, fdata2, ...),
    "kl" = metric.kl(fdata1, fdata2, ...)
  )
}

#' Lp Metric for Functional Data
#'
#' Computes the Lp distance between functional data objects using
#' numerical integration (Simpson's rule).
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, computes self-distances
#'   for fdata1 (more efficient symmetric computation).
#' @param lp The p in Lp metric. Default is 2 (L2 distance).
#' @param w Optional weight vector of length equal to number of evaluation
#'   points. Default is uniform weighting.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix of dimensions n1 x n2 (or n x n if fdata2 is NULL).
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' D <- metric.lp(fd)  # Self-distances
#'
#' fd2 <- fdata(matrix(rnorm(50), 5, 10))
#' D2 <- metric.lp(fd, fd2)  # Cross-distances
metric.lp <- function(fdata1, fdata2 = NULL, lp = 2, w = 1, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (fdata1$fdata2d) {
    # 2D functional data (surfaces)
    dims <- fdata1$dims
    m1 <- dims[1]
    m2 <- dims[2]
    m <- m1 * m2

    # Handle weight matrix for 2D
    if (length(w) == 1) {
      w <- matrix(w, m1, m2)
    }
    if (!is.matrix(w) || nrow(w) != m1 || ncol(w) != m2) {
      if (length(w) == m) {
        w <- matrix(w, m1, m2)
      } else {
        stop("Weight must be scalar, vector of length m1*m2, or m1 x m2 matrix")
      }
    }

    argvals_s <- as.numeric(fdata1$argvals[[1]])
    argvals_t <- as.numeric(fdata1$argvals[[2]])

    if (is.null(fdata2)) {
      D <- .Call("wrap__metric_lp_self_2d", fdata1$data, argvals_s, argvals_t,
                 as.numeric(lp), as.numeric(as.vector(t(w))))
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }
      if (!fdata2$fdata2d) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }
      if (!identical(fdata1$dims, fdata2$dims)) {
        stop("fdata1 and fdata2 must have the same grid dimensions")
      }

      D <- .Call("wrap__metric_lp_2d", fdata1$data, fdata2$data, argvals_s, argvals_t,
                 as.numeric(lp), as.numeric(as.vector(t(w))))
    }
  } else {
    # 1D functional data (curves)
    m <- ncol(fdata1$data)

    # Handle weight vector
    if (length(w) == 1) {
      w <- rep(w, m)
    }
    if (length(w) != m) {
      stop("Weight vector must have length equal to number of evaluation points")
    }

    if (is.null(fdata2)) {
      # Self-distances (symmetric, more efficient)
      D <- .Call("wrap__metric_lp_self_1d", fdata1$data, as.numeric(fdata1$argvals), as.numeric(lp), as.numeric(w))
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }

      if (fdata2$fdata2d) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }

      if (ncol(fdata1$data) != ncol(fdata2$data)) {
        stop("fdata1 and fdata2 must have the same number of evaluation points")
      }

      D <- .Call("wrap__metric_lp_1d", fdata1$data, fdata2$data, as.numeric(fdata1$argvals), as.numeric(lp), as.numeric(w))
    }
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}

#' Hausdorff Metric for Functional Data
#'
#' Computes the Hausdorff distance between functional data objects.
#' The Hausdorff distance treats each curve as a set of points (t, f(t))
#' in 2D space and computes the maximum of the minimum distances.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' D <- metric.hausdorff(fd)
metric.hausdorff <- function(fdata1, fdata2 = NULL, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (fdata1$fdata2d) {
    # 2D functional data (surfaces)
    dims <- fdata1$dims
    m1 <- dims[1]
    m2 <- dims[2]

    argvals_s <- as.numeric(fdata1$argvals[[1]])
    argvals_t <- as.numeric(fdata1$argvals[[2]])

    if (is.null(fdata2)) {
      # Self-distances (symmetric)
      D <- .Call("wrap__metric_hausdorff_2d", fdata1$data, argvals_s, argvals_t)
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }
      if (!fdata2$fdata2d) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }
      if (!identical(fdata1$dims, fdata2$dims)) {
        stop("fdata1 and fdata2 must have the same grid dimensions")
      }

      D <- .Call("wrap__metric_hausdorff_cross_2d", fdata1$data, fdata2$data, argvals_s, argvals_t)
    }
  } else {
    # 1D functional data (curves)
    if (is.null(fdata2)) {
      # Self-distances (symmetric) - use optimized Rust implementation
      D <- .Call("wrap__metric_hausdorff_1d", fdata1$data, as.numeric(fdata1$argvals))
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }

      if (fdata2$fdata2d) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }

      if (ncol(fdata1$data) != ncol(fdata2$data)) {
        stop("fdata1 and fdata2 must have the same number of evaluation points")
      }

      # Cross-distances - use Rust implementation
      D <- .Call("wrap__metric_hausdorff_cross_1d", fdata1$data, fdata2$data, as.numeric(fdata1$argvals))
    }
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}

#' Dynamic Time Warping for Functional Data
#'
#' Computes the Dynamic Time Warping distance between functional data.
#' DTW allows for non-linear alignment of curves.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, computes self-distances.
#' @param p The p in Lp distance (default 2 for L2/Euclidean).
#' @param w Sakoe-Chiba window constraint. Default is min(ncol(fdata1), ncol(fdata2)).
#'   Use -1 for no window constraint.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' D <- metric.DTW(fd)
metric.DTW <- function(fdata1, fdata2 = NULL, p = 2, w = NULL, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (fdata1$fdata2d) {
    stop("metric.DTW not yet implemented for 2D functional data")
  }

  m <- ncol(fdata1$data)

  if (is.null(fdata2)) {
    # Self-distances - use optimized Rust implementation
    if (is.null(w)) w <- m
    D <- .Call("wrap__metric_dtw_self_1d", fdata1$data, as.numeric(p), as.integer(w))
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }

    if (fdata2$fdata2d) {
      stop("metric.DTW not yet implemented for 2D functional data")
    }

    # Cross-distances - use Rust implementation
    m2 <- ncol(fdata2$data)
    if (is.null(w)) w <- min(m, m2)
    D <- .Call("wrap__metric_dtw_cross_1d", fdata1$data, fdata2$data, as.numeric(p), as.integer(w))
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}

#' Semi-metric based on Principal Components
#'
#' Computes a semi-metric based on the first ncomp principal component scores.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param ncomp Number of principal components to use.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix based on PC scores.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' D <- semimetric.pca(fd, ncomp = 3)
semimetric.pca <- function(fdata1, fdata2 = NULL, ncomp = 2, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  # Compute PCA on combined data (works for both 1D and 2D - data is flattened)
  if (is.null(fdata2)) {
    combined <- fdata1$data
    n1 <- nrow(fdata1$data)
    n2 <- n1
    same_data <- TRUE
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }
    if (isTRUE(fdata1$fdata2d) != isTRUE(fdata2$fdata2d)) {
      stop("Cannot compute distances between 1D and 2D functional data")
    }
    if (ncol(fdata1$data) != ncol(fdata2$data)) {
      stop("fdata1 and fdata2 must have the same number of evaluation points")
    }
    combined <- rbind(fdata1$data, fdata2$data)
    n1 <- nrow(fdata1$data)
    n2 <- nrow(fdata2$data)
    same_data <- FALSE
  }

  # Center the data
  centered <- scale(combined, center = TRUE, scale = FALSE)

  # Compute SVD
  svd_result <- svd(centered, nu = ncomp, nv = ncomp)

  # Get PC scores
  scores <- centered %*% svd_result$v[, seq_len(ncomp), drop = FALSE]

  # Compute distances
  scores1 <- scores[seq_len(n1), , drop = FALSE]
  if (same_data) {
    scores2 <- scores1
  } else {
    scores2 <- scores[(n1 + 1):(n1 + n2), , drop = FALSE]
  }

  # Euclidean distance in PC space
  D <- matrix(0, n1, n2)
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      D[i, j] <- sqrt(sum((scores1[i, ] - scores2[j, ])^2))
    }
  }

  D
}

#' Semi-metric based on Derivatives
#'
#' Computes a semi-metric based on the Lp distance of the nderiv-th derivative
#' of functional data.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param nderiv Derivative order (1, 2, ...). Default is 1.
#' @param lp The p in Lp metric. Default is 2 (L2 distance).
#' @param ... Additional arguments passed to fdata.deriv.
#'
#' @return A distance matrix based on derivative distances.
#'
#' @export
#' @examples
#' # Create smooth curves
#' t <- seq(0, 2*pi, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) X[i, ] <- sin(t + i/5)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute distance based on first derivative
#' D <- semimetric.deriv(fd, nderiv = 1)
semimetric.deriv <- function(fdata1, fdata2 = NULL, nderiv = 1, lp = 2, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  is_2d <- isTRUE(fdata1$fdata2d)

  if (is_2d) {
    # 2D case: fdata.deriv returns a list of derivatives (ds, dt, dsdt)
    # Use the sum of all derivative Lp distances
    fdata1_derivs <- fdata.deriv(fdata1, nderiv = nderiv, ...)

    if (is.null(fdata2)) {
      # Self-distances - combine derivative distances
      D_ds <- metric.lp(fdata1_derivs$ds, lp = lp)
      D_dt <- metric.lp(fdata1_derivs$dt, lp = lp)
      D <- sqrt(D_ds^2 + D_dt^2)
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }
      if (!isTRUE(fdata2$fdata2d)) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }

      fdata2_derivs <- fdata.deriv(fdata2, nderiv = nderiv, ...)
      D_ds <- metric.lp(fdata1_derivs$ds, fdata2_derivs$ds, lp = lp)
      D_dt <- metric.lp(fdata1_derivs$dt, fdata2_derivs$dt, lp = lp)
      D <- sqrt(D_ds^2 + D_dt^2)
    }
  } else {
    # 1D case
    # Compute derivative of fdata1
    fdata1_deriv <- fdata.deriv(fdata1, nderiv = nderiv, ...)

    if (is.null(fdata2)) {
      # Self-distances
      D <- metric.lp(fdata1_deriv, lp = lp)
    } else {
      if (!inherits(fdata2, "fdata")) {
        stop("fdata2 must be of class 'fdata'")
      }

      if (isTRUE(fdata2$fdata2d)) {
        stop("Cannot compute distances between 1D and 2D functional data")
      }

      # Compute derivative of fdata2
      fdata2_deriv <- fdata.deriv(fdata2, nderiv = nderiv, ...)

      D <- metric.lp(fdata1_deriv, fdata2_deriv, lp = lp)
    }
  }

  D
}

#' Semi-metric based on Basis Expansion
#'
#' Computes a semi-metric based on the L2 distance of basis expansion
#' coefficients. Supports B-spline and Fourier basis.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param nbasis Number of basis functions. Default is 5.
#' @param basis Type of basis: "bspline" (default) or "fourier".
#' @param nderiv Derivative order to compute distance on (default 0).
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix based on basis coefficients.
#'
#' @export
#' @examples
#' # Create curves
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) X[i, ] <- sin(2*pi*t + i/5) + rnorm(100, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute distance based on B-spline coefficients
#' D <- semimetric.basis(fd, nbasis = 7)
semimetric.basis <- function(fdata1, fdata2 = NULL, nbasis = 5, basis = "bspline",
                             nderiv = 0, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (isTRUE(fdata1$fdata2d)) {
    stop("semimetric.basis not yet implemented for 2D functional data")
  }

  basis <- match.arg(basis, c("bspline", "fourier"))
  argvals <- fdata1$argvals
  rangeval <- fdata1$rangeval
  m <- length(argvals)

  # Create basis matrix
  if (basis == "bspline") {
    # B-spline basis using polynomial representation
    degree <- 3
    knots <- seq(rangeval[1], rangeval[2], length.out = nbasis - degree + 1)
    # Use R's built-in splineDesign for B-splines
    B <- splines::bs(argvals, knots = knots[2:(length(knots)-1)],
                     degree = degree, intercept = TRUE, Boundary.knots = rangeval)
    B <- as.matrix(B)
  } else {
    # Fourier basis
    B <- matrix(0, m, nbasis)
    t_scaled <- (argvals - rangeval[1]) / (rangeval[2] - rangeval[1])

    B[, 1] <- 1  # constant term
    k <- 2
    for (freq in 1:((nbasis - 1) %/% 2)) {
      if (k <= nbasis) {
        B[, k] <- sin(2 * pi * freq * t_scaled)
        k <- k + 1
      }
      if (k <= nbasis) {
        B[, k] <- cos(2 * pi * freq * t_scaled)
        k <- k + 1
      }
    }
  }

  # Compute basis coefficients for fdata1 via least squares
  # data[n x m], B[m x nbasis], coef[n x nbasis]
  # coef = data %*% B %*% (B'B)^-1
  BtB_inv <- solve(crossprod(B))
  coef1 <- fdata1$data %*% B %*% BtB_inv

  if (is.null(fdata2)) {
    coef2 <- coef1
    same_data <- TRUE
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }
    if (isTRUE(fdata2$fdata2d)) {
      stop("semimetric.basis not yet implemented for 2D functional data")
    }
    coef2 <- fdata2$data %*% B %*% BtB_inv
    same_data <- FALSE
  }

  # Compute L2 distance in coefficient space
  n1 <- nrow(coef1)
  n2 <- nrow(coef2)

  D <- matrix(0, n1, n2)
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      D[i, j] <- sqrt(sum((coef1[i, ] - coef2[j, ])^2))
    }
  }

  D
}

#' Semi-metric based on Fourier Coefficients (FFT)
#'
#' Computes a semi-metric based on the L2 distance of Fourier coefficients
#' computed via Fast Fourier Transform (FFT). This is more efficient than
#' the Fourier basis option in semimetric.basis for large nfreq.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param nfreq Number of Fourier frequencies to use (excluding DC). Default is 5.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix based on Fourier coefficients.
#'
#' @details
#' The Fourier coefficients are computed using FFT and normalized by the
#' number of points. The distance is the L2 distance between the magnitude
#' of the first nfreq+1 coefficients (DC + nfreq frequencies).
#'
#' This function uses Rust's rustfft library for efficient FFT computation,
#' making it faster than R's base fft for large datasets.
#'
#' @export
#' @examples
#' # Create curves with different frequency content
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) X[i, ] <- sin(2*pi*i*t) + rnorm(100, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute distance based on Fourier coefficients
#' D <- semimetric.fourier(fd, nfreq = 10)
semimetric.fourier <- function(fdata1, fdata2 = NULL, nfreq = 5, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (isTRUE(fdata1$fdata2d)) {
    stop("semimetric.fourier not yet implemented for 2D functional data")
  }

  m <- ncol(fdata1$data)
  nfreq <- min(nfreq, m %/% 2 - 1)  # Can't have more than m/2 meaningful frequencies

  if (is.null(fdata2)) {
    # Self-distances (symmetric)
    D <- .Call("wrap__semimetric_fourier_self_1d", fdata1$data, as.integer(nfreq))
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }

    if (isTRUE(fdata2$fdata2d)) {
      stop("semimetric.fourier not yet implemented for 2D functional data")
    }

    D <- .Call("wrap__semimetric_fourier_cross_1d", fdata1$data, fdata2$data, as.integer(nfreq))
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}

#' Semi-metric based on Horizontal Shift (Time Warping)
#'
#' Computes a semi-metric based on the minimum L2 distance after optimal
#' horizontal shifting of curves. This is useful for comparing curves that
#' may have phase differences.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, uses fdata1.
#' @param max_shift Maximum shift in number of grid points. Default is m/4
#'   where m is the number of evaluation points. Use -1 for automatic.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix based on minimum L2 distance after shift.
#'
#' @details
#' For each pair of curves, this function computes:
#' \deqn{d(f, g) = \min_{|h| \le h_{max}} ||f(t) - g(t+h)||}
#' where h is the horizontal shift in discrete units.
#'
#' This semi-metric is useful when comparing curves with phase shifts,
#' such as ECG signals with different heart rates or periodic signals
#' with different phases.
#'
#' @export
#' @examples
#' # Create curves with phase shifts
#' t <- seq(0, 2*pi, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) X[i, ] <- sin(t + i*0.2) + rnorm(100, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute distance accounting for phase shifts
#' D <- semimetric.hshift(fd, max_shift = 10)
semimetric.hshift <- function(fdata1, fdata2 = NULL, max_shift = -1, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (isTRUE(fdata1$fdata2d)) {
    stop("semimetric.hshift not yet implemented for 2D functional data")
  }

  if (is.null(fdata2)) {
    # Self-distances (symmetric)
    D <- .Call("wrap__semimetric_hshift_self_1d", fdata1$data,
               as.numeric(fdata1$argvals), as.integer(max_shift))
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }

    if (isTRUE(fdata2$fdata2d)) {
      stop("semimetric.hshift not yet implemented for 2D functional data")
    }

    if (ncol(fdata1$data) != ncol(fdata2$data)) {
      stop("fdata1 and fdata2 must have the same number of evaluation points")
    }

    D <- .Call("wrap__semimetric_hshift_cross_1d", fdata1$data, fdata2$data,
               as.numeric(fdata1$argvals), as.integer(max_shift))
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}

#' Kullback-Leibler Divergence Metric for Functional Data
#'
#' Computes the symmetric Kullback-Leibler divergence between functional data
#' treated as probability distributions. Curves are first normalized to be
#' valid probability density functions.
#'
#' @param fdata1 An object of class 'fdata'.
#' @param fdata2 An object of class 'fdata'. If NULL, computes self-distances.
#' @param eps Small value for numerical stability (default 1e-10).
#' @param normalize Logical. If TRUE (default), curves are shifted to be
#'   non-negative and normalized to integrate to 1.
#' @param ... Additional arguments (ignored).
#'
#' @return A distance matrix based on symmetric KL divergence.
#'
#' @details
#' The symmetric KL divergence is computed as:
#' \deqn{D_{KL}(f, g) = \frac{1}{2}[KL(f||g) + KL(g||f)]}
#' where
#' \deqn{KL(f||g) = \int f(t) \log\frac{f(t)}{g(t)} dt}
#'
#' When \code{normalize = TRUE}, curves are first shifted to be non-negative
#' (by subtracting the minimum and adding eps), then normalized to integrate
#' to 1. This makes them valid probability density functions.
#'
#' The symmetric KL divergence is always non-negative and equals zero only
#' when the two distributions are identical. However, it does not satisfy
#' the triangle inequality.
#'
#' @export
#' @examples
#' # Create curves that look like probability densities
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) {
#'   # Shifted Gaussian-like curves
#'   X[i, ] <- exp(-(t - 0.3 - i/50)^2 / 0.02) + rnorm(100, sd = 0.01)
#' }
#' fd <- fdata(X, argvals = t)
#'
#' # Compute KL divergence
#' D <- metric.kl(fd)
metric.kl <- function(fdata1, fdata2 = NULL, eps = 1e-10, normalize = TRUE, ...) {
  if (!inherits(fdata1, "fdata")) {
    stop("fdata1 must be of class 'fdata'")
  }

  if (isTRUE(fdata1$fdata2d)) {
    stop("metric.kl not yet implemented for 2D functional data")
  }

  if (is.null(fdata2)) {
    # Self-distances (symmetric)
    D <- .Call("wrap__metric_kl_self_1d", fdata1$data,
               as.numeric(fdata1$argvals), as.numeric(eps), as.logical(normalize))
  } else {
    if (!inherits(fdata2, "fdata")) {
      stop("fdata2 must be of class 'fdata'")
    }

    if (isTRUE(fdata2$fdata2d)) {
      stop("metric.kl not yet implemented for 2D functional data")
    }

    if (ncol(fdata1$data) != ncol(fdata2$data)) {
      stop("fdata1 and fdata2 must have the same number of evaluation points")
    }

    D <- .Call("wrap__metric_kl_cross_1d", fdata1$data, fdata2$data,
               as.numeric(fdata1$argvals), as.numeric(eps), as.logical(normalize))
  }

  # Convert to proper matrix with dimnames
  D <- as.matrix(D)

  # Add row and column names if available
  if (!is.null(rownames(fdata1$data))) {
    rownames(D) <- rownames(fdata1$data)
  }

  if (is.null(fdata2)) {
    if (!is.null(rownames(fdata1$data))) {
      colnames(D) <- rownames(fdata1$data)
    }
  } else if (!is.null(rownames(fdata2$data))) {
    colnames(D) <- rownames(fdata2$data)
  }

  D
}
