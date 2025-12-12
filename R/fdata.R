#' Functional Data Class and Operations
#'
#' Functions for creating and manipulating functional data objects.
#' Supports both 1D functional data (curves) and 2D functional data (surfaces).

#' Create a functional data object
#'
#' @param mdata A matrix (for 1D) or 3D array (for 2D) of observations.
#'   Rows are samples, columns are evaluation points.
#' @param argvals Evaluation points. For 1D: a numeric vector.
#'   For 2D: a list with two numeric vectors.
#' @param rangeval Range of the argument values. For 1D: a numeric vector of
#'   length 2. For 2D: a list with two numeric vectors of length 2.
#' @param names List with components 'main', 'xlab', 'ylab' for plot titles.
#' @param fdata2d Logical. If TRUE, create 2D functional data (surface).
#'
#' @return An object of class 'fdata' containing:
#'   \item{data}{The data matrix}
#'   \item{argvals}{Evaluation points}
#'   \item{rangeval}{Range of arguments}
#'   \item{names}{Plot labels}
#'   \item{fdata2d}{Logical indicating if 2D}
#'
#' @export
#' @examples
#' # Create 1D functional data (curves)
#' x <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' fd <- fdata(x, argvals = seq(0, 1, length.out = 10))
#'
#' # Create 2D functional data (surfaces) - future
#' # x2d <- array(rnorm(1000), dim = c(10, 10, 10))
#' # fd2d <- fdata(x2d, fdata2d = TRUE)
fdata <- function(mdata, argvals = NULL, rangeval = NULL,
                  names = NULL, fdata2d = FALSE) {

  # Detect 2D data from input
  if (is.array(mdata) && length(dim(mdata)) == 3) {
    fdata2d <- TRUE
  }

  if (fdata2d) {
    return(.fdata2d(mdata, argvals, rangeval, names))
  } else {
    return(.fdata1d(mdata, argvals, rangeval, names))
  }
}

#' Internal: Create 1D functional data
#' @noRd
.fdata1d <- function(mdata, argvals = NULL, rangeval = NULL, names = NULL) {

  # Convert vector to matrix
  if (is.vector(mdata)) {
    mdata <- matrix(mdata, nrow = 1)
  }

  if (!is.matrix(mdata)) {
    mdata <- as.matrix(mdata)
  }

  n <- nrow(mdata)
  m <- ncol(mdata)

  # Set default argvals
  if (is.null(argvals)) {
    argvals <- seq_len(m)
  }

  if (length(argvals) != m) {
    stop("Length of argvals must equal number of columns in mdata")
  }

  # Set default rangeval
  if (is.null(rangeval)) {
    rangeval <- range(argvals)
  }

  # Set default names
  if (is.null(names)) {
    names <- list(main = "", xlab = "t", ylab = "X(t)")
  }

  structure(
    list(
      data = mdata,
      argvals = argvals,
      rangeval = rangeval,
      names = names,
      fdata2d = FALSE
    ),
    class = "fdata"
  )
}

#' Internal: Create 2D functional data (surfaces)
#' @noRd
.fdata2d <- function(mdata, argvals = NULL, rangeval = NULL, names = NULL) {

  if (is.array(mdata) && length(dim(mdata)) == 3) {
    dims <- dim(mdata)
    n <- dims[1]
    m1 <- dims[2]
    m2 <- dims[3]

    # Flatten to matrix: n x (m1*m2)
    data_mat <- matrix(mdata, nrow = n, ncol = m1 * m2)
  } else if (is.matrix(mdata)) {
    # Assume already flattened, need argvals to determine shape
    data_mat <- mdata
    n <- nrow(mdata)
  } else {
    stop("For 2D fdata, mdata must be a 3D array or matrix")
  }

  # Set default argvals
  if (is.null(argvals)) {
    if (exists("m1") && exists("m2")) {
      argvals <- list(
        s = seq_len(m1),
        t = seq_len(m2)
      )
    } else {
      stop("argvals must be provided for 2D fdata from matrix input")
    }
  }

  if (!is.list(argvals) || length(argvals) != 2) {
    stop("argvals must be a list with two components for 2D fdata")
  }

  # Set default rangeval
  if (is.null(rangeval)) {
    rangeval <- list(
      s = range(argvals[[1]]),
      t = range(argvals[[2]])
    )
  }

  # Set default names
  if (is.null(names)) {
    names <- list(main = "", xlab = "s", ylab = "t", zlab = "X(s,t)")
  }

  structure(
    list(
      data = data_mat,
      argvals = argvals,
      rangeval = rangeval,
      names = names,
      fdata2d = TRUE,
      dims = c(length(argvals[[1]]), length(argvals[[2]]))
    ),
    class = "fdata"
  )
}

#' Center functional data
#'
#' Subtract the mean function from each curve.
#'
#' @param fdataobj An object of class 'fdata'.
#'
#' @return A centered 'fdata' object.
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' fd_centered <- fdata.cen(fd)
fdata.cen <- function(fdataobj) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (fdataobj$fdata2d) {
    stop("fdata.cen not yet implemented for 2D functional data")
  }

  centered_data <- .Call("wrap__fdata_center_1d", fdataobj$data)

  fdataobj$data <- centered_data
  fdataobj
}

#' Compute functional mean
#'
#' @param fdataobj An object of class 'fdata'.
#'
#' @return A numeric vector containing the mean function values.
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' fm <- func.mean(fd)
func.mean <- function(fdataobj) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (fdataobj$fdata2d) {
    stop("func.mean not yet implemented for 2D functional data")
  }

  .Call("wrap__fdata_mean_1d", fdataobj$data)
}

#' Compute Lp norm of functional data
#'
#' @param fdataobj An object of class 'fdata'.
#' @param lp The p in Lp norm (default 2 for L2 norm).
#'
#' @return A numeric vector of norms, one per curve.
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' norms <- norm.fdata(fd)
norm.fdata <- function(fdataobj, lp = 2) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (fdataobj$fdata2d) {
    stop("norm.fdata not yet implemented for 2D functional data")
  }

  .Call("wrap__fdata_norm_lp_1d", fdataobj$data, as.numeric(fdataobj$argvals), as.numeric(lp))
}

#' Print method for fdata objects
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.fdata <- function(x, ...) {
  cat("Functional data object\n")
  cat("  Type:", if (isTRUE(x$fdata2d)) "2D (surface)" else "1D (curve)", "\n")
  cat("  Number of curves:", nrow(x$data), "\n")

  if (isTRUE(x$fdata2d)) {
    cat("  Grid dimensions:", x$dims[1], "x", x$dims[2], "\n")
    cat("  Range s:", x$rangeval$s[1], "-", x$rangeval$s[2], "\n")
    cat("  Range t:", x$rangeval$t[1], "-", x$rangeval$t[2], "\n")
  } else {
    cat("  Number of points:", ncol(x$data), "\n")
    cat("  Range:", x$rangeval[1], "-", x$rangeval[2], "\n")
  }

  invisible(x)
}

#' Summary method for fdata objects
#'
#' @param object An object of class 'fdata'.
#' @param ... Additional arguments (ignored).
#'
#' @export
summary.fdata <- function(object, ...) {
  cat("Functional data summary\n")
  cat("=======================\n")
  cat("Type:", if (isTRUE(object$fdata2d)) "2D (surface)" else "1D (curve)", "\n")
  cat("Number of observations:", nrow(object$data), "\n")

  if (isTRUE(object$fdata2d)) {
    cat("Grid dimensions:", object$dims[1], "x", object$dims[2], "\n")
    cat("Total evaluation points:", prod(object$dims), "\n")
  } else {
    cat("Number of evaluation points:", ncol(object$data), "\n")
  }

  cat("\nData range:\n")
  cat("  Min:", min(object$data), "\n")
  cat("  Max:", max(object$data), "\n")
  cat("  Mean:", mean(object$data), "\n")
  cat("  SD:", sd(object$data), "\n")

  invisible(object)
}

#' Plot method for fdata objects
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments passed to matplot.
#'
#' @export
plot.fdata <- function(x, ...) {
  if (x$fdata2d) {
    warning("2D fdata plotting not yet implemented")
    return(invisible(x))
  }

  args <- list(...)
  if (is.null(args$type)) args$type <- "l"
  if (is.null(args$xlab)) args$xlab <- x$names$xlab
  if (is.null(args$ylab)) args$ylab <- x$names$ylab
  if (is.null(args$main)) args$main <- x$names$main

  do.call(matplot, c(list(x = x$argvals, y = t(x$data)), args))
  invisible(x)
}

#' Compute functional derivative
#'
#' Compute the numerical derivative of functional data. Uses finite differences
#' for fast computation via Rust.
#'
#' For 1D functional data (curves), computes the nth derivative.
#' For 2D functional data (surfaces), computes partial derivatives:
#' \itemize{
#'   \item \code{ds}: partial derivative with respect to s (first argument)
#'   \item \code{dt}: partial derivative with respect to t (second argument)
#'   \item \code{dsdt}: mixed partial derivative
#' }
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nderiv Derivative order (1, 2, ...). Default is 1. For 2D data,
#'   only first-order derivatives are currently supported.
#' @param method Method for computing derivatives. Currently only "diff"
#'   (finite differences) is supported.
#' @param class.out Output class, either "fdata" or "fd". Default is "fdata".
#' @param nbasis Not used (for compatibility with fda.usc).
#' @param ... Additional arguments (ignored).
#'
#' @return For 1D data: an 'fdata' object containing the derivative values.
#'   For 2D data: a list with components \code{ds}, \code{dt}, and \code{dsdt},
#'   each an 'fdata' object containing the respective partial derivative.
#'
#' @export
#' @examples
#' # Create smooth curves
#' t <- seq(0, 2*pi, length.out = 100)
#' X <- matrix(0, 10, 100)
#' for (i in 1:10) X[i, ] <- sin(t + i/5)
#' fd <- fdata(X, argvals = t)
#'
#' # First derivative (should be approximately cos)
#' fd_deriv <- fdata.deriv(fd, nderiv = 1)
fdata.deriv <- function(fdataobj, nderiv = 1, method = "diff",
                        class.out = "fdata", nbasis = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (nderiv < 1) {
    return(fdataobj)
  }

  if (method != "diff") {
    warning("Only method='diff' is currently supported, using finite differences")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    # 2D functional data: compute partial derivatives
    if (nderiv > 1) {
      warning("For 2D data, only first-order derivatives are currently supported")
    }

    m1 <- fdataobj$dims[1]
    m2 <- fdataobj$dims[2]
    argvals_s <- fdataobj$argvals[[1]]
    argvals_t <- fdataobj$argvals[[2]]

    # Call Rust implementation for 2D derivatives
    result <- .Call("wrap__fdata_deriv_2d", fdataobj$data,
                    as.numeric(argvals_s), as.numeric(argvals_t),
                    as.integer(m1), as.integer(m2))

    # Create fdata objects for each derivative type
    make_deriv_fdata <- function(deriv_data, deriv_name) {
      new_names <- fdataobj$names
      if (!is.null(new_names$zlab)) {
        new_names$zlab <- paste0(deriv_name, "[", new_names$zlab, "]")
      }
      structure(
        list(
          data = deriv_data,
          argvals = fdataobj$argvals,
          rangeval = fdataobj$rangeval,
          names = new_names,
          fdata2d = TRUE,
          dims = fdataobj$dims
        ),
        class = "fdata"
      )
    }

    return(list(
      ds = make_deriv_fdata(result$ds, "d/ds"),
      dt = make_deriv_fdata(result$dt, "d/dt"),
      dsdt = make_deriv_fdata(result$dsdt, "d2/dsdt")
    ))
  }

  # 1D functional data
  m <- ncol(fdataobj$data)
  if (nderiv >= m) {
    stop("nderiv must be less than the number of evaluation points")
  }

  # Call Rust implementation
  deriv_data <- .Call("wrap__fdata_deriv_1d", fdataobj$data,
                      as.numeric(fdataobj$argvals), as.integer(nderiv))

  # Update argvals - derivative reduces number of points
  # For central differences, we keep interior points
  new_argvals <- fdataobj$argvals

  # Update names
  deriv_suffix <- if (nderiv == 1) "'" else paste0("^(", nderiv, ")")
  new_names <- fdataobj$names
  if (!is.null(new_names$ylab)) {
    new_names$ylab <- paste0("D", nderiv, "[", new_names$ylab, "]")
  }
  if (!is.null(new_names$main) && nchar(new_names$main) > 0) {
    new_names$main <- paste0(new_names$main, deriv_suffix)
  }

  structure(
    list(
      data = deriv_data,
      argvals = new_argvals,
      rangeval = fdataobj$rangeval,
      names = new_names,
      fdata2d = FALSE
    ),
    class = "fdata"
  )
}

#' Subset method for fdata objects
#'
#' @param x An object of class 'fdata'.
#' @param i Row indices (which curves to keep).
#' @param j Column indices (which time points to keep).
#' @param drop Logical. If TRUE and only one curve selected, return vector.
#'
#' @export
`[.fdata` <- function(x, i, j, drop = FALSE) {
  if (missing(i)) i <- seq_len(nrow(x$data))
  if (missing(j)) j <- seq_len(ncol(x$data))

  new_data <- x$data[i, j, drop = FALSE]

  # Check for 2D fdata - handle NULL or missing fdata2d (e.g., from fda.usc objects)
  is_2d <- isTRUE(x$fdata2d)

  if (is_2d) {
    # For 2D fdata, subsetting only rows (surfaces) preserves argvals and dims
    # Subsetting columns (grid points) is not well-defined for 2D
    if (!missing(j) && !identical(j, seq_len(ncol(x$data)))) {
      stop("Column subsetting is not supported for 2D fdata")
    }
    new_argvals <- x$argvals
    new_rangeval <- x$rangeval
    new_dims <- x$dims
  } else {
    new_argvals <- x$argvals[j]
    new_rangeval <- range(new_argvals)
    new_dims <- NULL
  }

  if (drop && nrow(new_data) == 1) {
    return(as.vector(new_data))
  }

  structure(
    list(
      data = new_data,
      argvals = new_argvals,
      rangeval = new_rangeval,
      names = x$names,
      fdata2d = is_2d,
      dims = new_dims
    ),
    class = "fdata"
  )
}

#' Bootstrap Functional Data
#'
#' Generate bootstrap samples from functional data. Supports naive bootstrap
#' (resampling curves with replacement) and smooth bootstrap (adding noise
#' based on estimated covariance structure).
#'
#' @param fdataobj An object of class 'fdata'.
#' @param n.boot Number of bootstrap replications (default 200).
#' @param method Bootstrap method: "naive" for resampling with replacement,
#'   "smooth" for adding Gaussian noise (default "naive").
#' @param variance For method="smooth", the variance of the added noise.
#'   If NULL, estimated from the data.
#' @param seed Optional seed for reproducibility.
#'
#' @return A list with:
#'   \item{boot.samples}{List of n.boot fdata objects, each a bootstrap sample}
#'   \item{original}{The original fdata object}
#'   \item{method}{The bootstrap method used}
#'   \item{n.boot}{Number of bootstrap replications}
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Naive bootstrap
#' boot_naive <- fdata.bootstrap(fd, n.boot = 100, method = "naive")
#'
#' # Smooth bootstrap
#' boot_smooth <- fdata.bootstrap(fd, n.boot = 100, method = "smooth")
fdata.bootstrap <- function(fdataobj, n.boot = 200, method = c("naive", "smooth"),
                            variance = NULL, seed = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fdata.bootstrap for 2D functional data not yet implemented")
  }

  method <- match.arg(method)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  boot.samples <- vector("list", n.boot)

  if (method == "naive") {
    # Naive bootstrap: resample curves with replacement
    for (b in seq_len(n.boot)) {
      idx <- sample(n, n, replace = TRUE)
      boot_data <- fdataobj$data[idx, , drop = FALSE]

      boot.samples[[b]] <- structure(
        list(
          data = boot_data,
          argvals = fdataobj$argvals,
          rangeval = fdataobj$rangeval,
          names = fdataobj$names,
          fdata2d = FALSE
        ),
        class = "fdata"
      )
    }
  } else if (method == "smooth") {
    # Smooth bootstrap: add noise based on estimated covariance
    # Estimate pointwise variance if not provided
    if (is.null(variance)) {
      # Use pooled residual variance from mean function
      mean_func <- colMeans(fdataobj$data)
      residuals <- sweep(fdataobj$data, 2, mean_func)
      variance <- mean(residuals^2)
    }

    for (b in seq_len(n.boot)) {
      # Resample with replacement
      idx <- sample(n, n, replace = TRUE)
      boot_data <- fdataobj$data[idx, , drop = FALSE]

      # Add Gaussian noise
      noise <- matrix(rnorm(n * m, mean = 0, sd = sqrt(variance)), n, m)
      boot_data <- boot_data + noise

      boot.samples[[b]] <- structure(
        list(
          data = boot_data,
          argvals = fdataobj$argvals,
          rangeval = fdataobj$rangeval,
          names = fdataobj$names,
          fdata2d = FALSE
        ),
        class = "fdata"
      )
    }
  }

  structure(
    list(
      boot.samples = boot.samples,
      original = fdataobj,
      method = method,
      n.boot = n.boot
    ),
    class = "fdata.bootstrap"
  )
}

#' Bootstrap Confidence Intervals for Functional Statistics
#'
#' Compute bootstrap confidence intervals for functional statistics such as
#' the mean function, depth values, or regression coefficients.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param statistic A function that computes the statistic of interest.
#'   Must take an fdata object and return a numeric vector.
#' @param n.boot Number of bootstrap replications (default 200).
#' @param alpha Significance level for confidence intervals (default 0.05
#'   for 95\% CI).
#' @param method CI method: "percentile" for simple percentile method,
#'   "basic" for basic bootstrap, "normal" for normal approximation
#'   (default "percentile").
#' @param seed Optional seed for reproducibility.
#'
#' @return A list with:
#'   \item{estimate}{The statistic computed on the original data}
#'   \item{ci.lower}{Lower confidence bound}
#'   \item{ci.upper}{Upper confidence bound}
#'   \item{boot.stats}{Matrix of bootstrap statistics (n.boot x length(statistic))}
#'   \item{alpha}{The significance level used}
#'   \item{method}{The CI method used}
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Bootstrap CI for the mean function
#' ci_mean <- fdata.bootstrap.ci(fd, statistic = func.mean, n.boot = 100)
#'
#' # Bootstrap CI for depth values
#' ci_depth <- fdata.bootstrap.ci(fd,
#'   statistic = function(x) depth.FM(x),
#'   n.boot = 100)
fdata.bootstrap.ci <- function(fdataobj, statistic, n.boot = 200,
                               alpha = 0.05,
                               method = c("percentile", "basic", "normal"),
                               seed = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (!is.function(statistic)) {
    stop("statistic must be a function")
  }

  method <- match.arg(method)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Compute statistic on original data
  estimate <- statistic(fdataobj)
  n_stat <- length(estimate)

  # Generate bootstrap samples and compute statistics
  boot_obj <- fdata.bootstrap(fdataobj, n.boot = n.boot, method = "naive")

  boot.stats <- matrix(NA, n.boot, n_stat)
  for (b in seq_len(n.boot)) {
    boot.stats[b, ] <- statistic(boot_obj$boot.samples[[b]])
  }

  # Compute confidence intervals
  if (method == "percentile") {
    # Simple percentile method
    ci.lower <- apply(boot.stats, 2, quantile, probs = alpha / 2, na.rm = TRUE)
    ci.upper <- apply(boot.stats, 2, quantile, probs = 1 - alpha / 2, na.rm = TRUE)

  } else if (method == "basic") {
    # Basic bootstrap: 2*theta_hat - theta*_(1-alpha/2), 2*theta_hat - theta*_(alpha/2)
    q_lower <- apply(boot.stats, 2, quantile, probs = alpha / 2, na.rm = TRUE)
    q_upper <- apply(boot.stats, 2, quantile, probs = 1 - alpha / 2, na.rm = TRUE)
    ci.lower <- 2 * estimate - q_upper
    ci.upper <- 2 * estimate - q_lower

  } else if (method == "normal") {
    # Normal approximation
    boot.se <- apply(boot.stats, 2, sd, na.rm = TRUE)
    z <- qnorm(1 - alpha / 2)
    ci.lower <- estimate - z * boot.se
    ci.upper <- estimate + z * boot.se
  }

  structure(
    list(
      estimate = estimate,
      ci.lower = ci.lower,
      ci.upper = ci.upper,
      boot.stats = boot.stats,
      alpha = alpha,
      method = method
    ),
    class = "fdata.bootstrap.ci"
  )
}

#' Print method for bootstrap CI
#' @export
print.fdata.bootstrap.ci <- function(x, ...) {
  cat("Bootstrap Confidence Intervals\n")
  cat("==============================\n")
  cat("Method:", x$method, "\n")
  cat("Confidence level:", (1 - x$alpha) * 100, "%\n")
  cat("Number of bootstrap replications:", nrow(x$boot.stats), "\n\n")

  n_show <- min(10, length(x$estimate))
  cat("First", n_show, "values:\n")
  df <- data.frame(
    Estimate = x$estimate[1:n_show],
    Lower = x$ci.lower[1:n_show],
    Upper = x$ci.upper[1:n_show]
  )
  print(df, digits = 4)

  if (length(x$estimate) > n_show) {
    cat("... (", length(x$estimate) - n_show, " more values)\n")
  }

  invisible(x)
}

#' Convert Functional Data to Principal Component Scores
#'
#' Performs functional PCA and returns principal component scores for
#' functional data. Uses SVD on centered data.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ncomp Number of principal components to extract (default 2).
#' @param lambda Regularization parameter (default 0, not currently used).
#' @param norm Logical. If TRUE (default), normalize the scores.
#'
#' @return A list with:
#'   \item{d}{Singular values (proportional to sqrt of eigenvalues)}
#'   \item{rotation}{fdata object containing PC loadings}
#'   \item{x}{Matrix of PC scores (n x ncomp)}
#'   \item{mean}{Mean function (numeric vector)}
#'   \item{fdataobj.cen}{Centered fdata object}
#'   \item{call}{The function call}
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' pc <- fdata2pc(fd, ncomp = 3)
fdata2pc <- function(fdataobj, ncomp = 2, lambda = 0, norm = TRUE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fdata2pc for 2D functional data not yet implemented")
  }

  result <- .Call("wrap__fdata2pc_1d", fdataobj$data,
                  as.integer(ncomp), as.numeric(lambda))

  # Construct fdata for rotation (loadings)
  # Rotation from Rust is m x ncomp, transpose to ncomp x m for fdata
  rotation <- fdata(t(result$rotation), argvals = fdataobj$argvals,
                    names = list(main = "PC Loadings",
                                 xlab = fdataobj$names$xlab,
                                 ylab = "Loading"))

  # Centered fdata
  fdataobj.cen <- fdata(result$centered, argvals = fdataobj$argvals,
                        names = list(main = "Centered Data",
                                     xlab = fdataobj$names$xlab,
                                     ylab = fdataobj$names$ylab))

  list(
    d = result$d,
    rotation = rotation,
    x = result$scores,
    mean = result$mean,
    fdataobj.cen = fdataobj.cen,
    call = match.call()
  )
}

#' Convert Functional Data to PLS Scores
#'
#' Performs Partial Least Squares regression and returns component scores
#' for functional data using the NIPALS algorithm.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param y Response vector (numeric).
#' @param ncomp Number of PLS components to extract (default 2).
#' @param lambda Regularization parameter (default 0, not currently used).
#' @param norm Logical. If TRUE (default), normalize the scores.
#'
#' @return A list with:
#'   \item{weights}{Matrix of PLS weights (m x ncomp)}
#'   \item{scores}{Matrix of PLS scores (n x ncomp)}
#'   \item{loadings}{Matrix of PLS loadings (m x ncomp)}
#'   \item{call}{The function call}
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' y <- rowMeans(X) + rnorm(20, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' pls <- fdata2pls(fd, y, ncomp = 3)
fdata2pls <- function(fdataobj, y, ncomp = 2, lambda = 0, norm = TRUE) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fdata2pls for 2D functional data not yet implemented")
  }

  if (length(y) != nrow(fdataobj$data)) {
    stop("Length of y must equal number of curves")
  }

  result <- .Call("wrap__fdata2pls_1d", fdataobj$data,
                  as.numeric(y), as.integer(ncomp), as.numeric(lambda))

  list(
    rotation = result$weights,
    x = result$scores,
    loadings = result$loadings,
    call = match.call()
  )
}

#' Convert Functional Data to Basis Coefficients
#'
#' Project functional data onto a basis system and return coefficients.
#' Supports B-spline and Fourier basis.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nbasis Number of basis functions (default 10).
#' @param type Type of basis: "bspline" (default) or "fourier".
#'
#' @return A matrix of coefficients (n x nbasis).
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' coefs <- fdata2basis(fd, nbasis = 10, type = "bspline")
fdata2basis <- function(fdataobj, nbasis = 10, type = c("bspline", "fourier")) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("fdata2basis for 2D functional data not yet implemented")
  }

  type <- match.arg(type)
  basis_type <- if (type == "fourier") 1L else 0L

  .Call("wrap__fdata2basis_1d", fdataobj$data,
        as.numeric(fdataobj$argvals), as.integer(nbasis), basis_type)
}

#' Convert Functional Data to fd class
#'
#' Converts an fdata object to an fd object from the fda package.
#' Requires the fda package to be installed.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nbasis Number of basis functions (default 10).
#' @param type Type of basis: "bspline" (default) or "fourier".
#'
#' @return An object of class 'fd' from the fda package.
#'
#' @export
#' @examples
#' \dontrun{
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' fd_obj <- fdata2fd(fd, nbasis = 10)
#' }
fdata2fd <- function(fdataobj, nbasis = 10, type = c("bspline", "fourier")) {
  if (!requireNamespace("fda", quietly = TRUE)) {
    stop("Package 'fda' is required for fdata2fd. Please install it.")
  }

  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  type <- match.arg(type)

  # Get coefficients from our implementation
  coefs <- fdata2basis(fdataobj, nbasis = nbasis, type = type)

  # Create basis object
  rangeval <- fdataobj$rangeval
  if (type == "fourier") {
    basis <- fda::create.fourier.basis(rangeval = rangeval, nbasis = nbasis)
  } else {
    basis <- fda::create.bspline.basis(rangeval = rangeval, nbasis = nbasis)
  }

  # Create fd object
  # Note: fda::fd expects coefs as (nbasis x n), we have (n x nbasis)
  fda::fd(coef = t(coefs), basisobj = basis)
}
