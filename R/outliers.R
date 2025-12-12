#' Outlier Detection for Functional Data
#'
#' Functions for detecting outliers in functional data using depth measures.

#' Outlier Detection using Weighted Depth
#'
#' Detects outliers based on depth with bootstrap resampling.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nb Number of bootstrap samples. Default is 200.
#' @param dfunc Depth function to use. Default is depth.mode.
#' @param quan Quantile for outlier cutoff. Default is 0.5.
#' @param ... Additional arguments passed to depth function.
#'
#' @return A list of class 'outliers.fdata' with components:
#' \describe{
#'   \item{outliers}{Indices of detected outliers}
#'   \item{depths}{Depth values for all curves}
#'   \item{cutoff}{Depth cutoff used}
#' }
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' out <- outliers.depth.pond(fd, nb = 50)
outliers.depth.pond <- function(fdataobj, nb = 200, dfunc = depth.mode,
                                 quan = 0.5, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)

  # Compute depths
  depths <- dfunc(fdataobj, fdataobj, ...)

  # Bootstrap to estimate null distribution of depths
  boot_depths <- matrix(0, nb, n)
  for (b in seq_len(nb)) {
    # Resample indices
    idx <- sample(n, n, replace = TRUE)
    fd_boot <- fdataobj[idx, ]

    # Compute depths in bootstrap sample
    boot_depths[b, ] <- dfunc(fdataobj, fd_boot, ...)
  }

  # Compute weighted depth (average over bootstrap samples)
  weighted_depths <- colMeans(boot_depths)

  # Determine cutoff
  cutoff <- quantile(weighted_depths, quan)

  # Identify outliers
  outliers <- which(depths < cutoff)

  structure(
    list(
      outliers = outliers,
      depths = depths,
      weighted_depths = weighted_depths,
      cutoff = cutoff,
      fdataobj = fdataobj
    ),
    class = "outliers.fdata"
  )
}

#' Outlier Detection using Trimmed Depth
#'
#' Detects outliers based on depth trimming.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to consider as potential outliers.
#'   Default is 0.1 (curves with depth in bottom 10%).
#' @param dfunc Depth function to use. Default is depth.mode.
#' @param ... Additional arguments passed to depth function.
#'
#' @return A list of class 'outliers.fdata' with components:
#' \describe{
#'   \item{outliers}{Indices of detected outliers}
#'   \item{depths}{Depth values for all curves}
#'   \item{cutoff}{Depth cutoff used}
#' }
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' out <- outliers.depth.trim(fd, trim = 0.1)
outliers.depth.trim <- function(fdataobj, trim = 0.1, dfunc = depth.mode, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (trim <= 0 || trim >= 1) {
    stop("trim must be between 0 and 1")
  }

  n <- nrow(fdataobj$data)

  # Compute depths
  depths <- dfunc(fdataobj, fdataobj, ...)

  # Determine cutoff based on trim proportion
  cutoff <- quantile(depths, trim)

  # Identify outliers
  outliers <- which(depths <= cutoff)

  structure(
    list(
      outliers = outliers,
      depths = depths,
      cutoff = cutoff,
      trim = trim,
      fdataobj = fdataobj
    ),
    class = "outliers.fdata"
  )
}

#' Print method for outliers.fdata objects
#' @export
print.outliers.fdata <- function(x, ...) {
  cat("Functional data outlier detection\n")
  cat("  Number of observations:", length(x$depths), "\n")
  cat("  Number of outliers:", length(x$outliers), "\n")
  if (length(x$outliers) > 0) {
    cat("  Outlier indices:", head(x$outliers, 10))
    if (length(x$outliers) > 10) cat(" ...")
    cat("\n")
  }
  cat("  Depth cutoff:", x$cutoff, "\n")
  invisible(x)
}

#' Plot method for outliers.fdata objects
#'
#' @param x An object of class 'outliers.fdata'.
#' @param col.outliers Color for outlier curves (default "red").
#' @param ... Additional arguments (currently ignored).
#'
#' @return A ggplot object.
#'
#' @export
plot.outliers.fdata <- function(x, col.outliers = "red", ...) {
  fd <- x$fdataobj
  n <- nrow(fd$data)
  m <- ncol(fd$data)

  # Create status factor
  status <- rep("Normal", n)
  status[x$outliers] <- "Outlier"

  # Reshape to long format
  df <- data.frame(
    curve_id = rep(seq_len(n), each = m),
    argval = rep(fd$argvals, n),
    value = as.vector(t(fd$data)),
    status = factor(rep(status, each = m), levels = c("Normal", "Outlier"))
  )

  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$argval, y = .data$value,
                                         group = .data$curve_id,
                                         color = .data$status)) +
    ggplot2::geom_line(alpha = 0.7) +
    ggplot2::scale_color_manual(values = c("Normal" = "gray60",
                                           "Outlier" = col.outliers)) +
    ggplot2::labs(
      x = fd$names$xlab %||% "t",
      y = fd$names$ylab %||% "X(t)",
      title = paste("Outliers detected:", length(x$outliers)),
      color = "Status"
    ) +
    ggplot2::theme_minimal()

  print(p)
  invisible(p)
}

#' LRT Outlier Detection Threshold
#'
#' Computes the bootstrap threshold for LRT-based outlier detection.
#' This is a highly parallelized Rust implementation providing significant
#' speedup over pure R implementations.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nb Number of bootstrap replications (default 200).
#' @param smo Smoothing parameter for bootstrap noise (default 0.05).
#' @param trim Proportion of curves to trim for robust estimation (default 0.1).
#' @param seed Random seed for reproducibility.
#'
#' @return The 99th percentile threshold value.
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:30) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' thresh <- outliers.thres.lrt(fd, nb = 100)
outliers.thres.lrt <- function(fdataobj, nb = 200, smo = 0.05, trim = 0.1, seed = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("outliers.thres.lrt for 2D functional data not yet implemented")
  }

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  .Call("wrap__outliers_thres_lrt", fdataobj$data,
        as.numeric(fdataobj$argvals), as.integer(nb),
        as.numeric(smo), as.numeric(trim), as.numeric(seed))
}

#' LRT-based Outlier Detection for Functional Data
#'
#' Detects outliers using the Likelihood Ratio Test approach based on
#' Febrero-Bande et al. Uses bootstrap to estimate a threshold and
#' iteratively removes curves exceeding this threshold.
#' Implemented in Rust for high performance with parallelized bootstrap.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nb Number of bootstrap replications for threshold estimation (default 200).
#' @param smo Smoothing parameter for bootstrap noise (default 0.05).
#' @param trim Proportion of curves to trim for robust estimation (default 0.1).
#' @param seed Random seed for reproducibility.
#'
#' @return A list of class 'outliers.fdata' with components:
#' \describe{
#'   \item{outliers}{Indices of detected outliers}
#'   \item{distances}{Normalized distances for all curves}
#'   \item{threshold}{Bootstrap threshold used}
#'   \item{fdataobj}{Original fdata object}
#' }
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:30) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' # Add an outlier
#' X[1, ] <- X[1, ] + 3
#' fd <- fdata(X, argvals = t)
#' out <- outliers.lrt(fd, nb = 100)
outliers.lrt <- function(fdataobj, nb = 200, smo = 0.05, trim = 0.1, seed = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("outliers.lrt for 2D functional data not yet implemented")
  }

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  result <- .Call("wrap__outliers_lrt", fdataobj$data,
                  as.numeric(fdataobj$argvals), as.integer(nb),
                  as.numeric(smo), as.numeric(trim), as.numeric(seed))

  structure(
    list(
      outliers = result$outliers,
      distances = result$distances,
      threshold = result$threshold,
      fdataobj = fdataobj
    ),
    class = "outliers.fdata"
  )
}
