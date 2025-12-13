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

#' Outlier Detection using Functional Boxplot
#'
#' Detects outliers based on the functional boxplot method.
#' Curves that exceed the fence (1.5 times the central envelope width)
#' at any point are flagged as outliers.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param prob Proportion of curves for the central region (default 0.5).
#' @param factor Factor for fence calculation (default 1.5).
#' @param depth.func Depth function to use. Default is depth.MBD.
#' @param ... Additional arguments passed to depth function.
#'
#' @return A list of class 'outliers.fdata' with components:
#' \describe{
#'   \item{outliers}{Indices of detected outliers}
#'   \item{depths}{Depth values for all curves}
#'   \item{cutoff}{Not used (for compatibility)}
#'   \item{fdataobj}{Original fdata object}
#' }
#'
#' @seealso \code{\link{boxplot.fdata}} for functional boxplot visualization
#'
#' @export
#' @examples
#' # Create functional data with outliers
#' set.seed(42)
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:28) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
#' X[29, ] <- sin(2*pi*t) + 2  # Magnitude outlier
#' X[30, ] <- cos(2*pi*t)       # Shape outlier
#' fd <- fdata(X, argvals = t)
#'
#' # Detect outliers
#' out <- outliers.boxplot(fd)
#' print(out)
outliers.boxplot <- function(fdataobj, prob = 0.5, factor = 1.5,
                             depth.func = depth.MBD, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  # Compute depths
  depths <- depth.func(fdataobj, fdataobj, ...)

  # Order curves by depth
  depth_order <- order(depths, decreasing = TRUE)

  # Central region: top prob proportion of curves
  n_central <- max(1, ceiling(n * prob))
  central_idx <- depth_order[seq_len(n_central)]

  # Compute central envelope
  central_data <- fdataobj$data[central_idx, , drop = FALSE]
  env_min <- apply(central_data, 2, min)
  env_max <- apply(central_data, 2, max)

  # Compute fence
  env_width <- env_max - env_min
  fence_min <- env_min - factor * env_width
  fence_max <- env_max + factor * env_width

  # Identify outliers: curves that exceed the fence at any point
  outlier_idx <- integer(0)
  for (i in seq_len(n)) {
    curve <- fdataobj$data[i, ]
    if (any(curve < fence_min) || any(curve > fence_max)) {
      outlier_idx <- c(outlier_idx, i)
    }
  }

  structure(
    list(
      outliers = outlier_idx,
      depths = depths,
      cutoff = NA,
      envelope = list(min = env_min, max = env_max),
      fence = list(min = fence_min, max = fence_max),
      fdataobj = fdataobj
    ),
    class = "outliers.fdata"
  )
}

#' Magnitude-Shape Plot for Functional Data
#'
#' Creates a Magnitude-Shape (MS) plot for functional outlier detection.
#' The MS plot displays each curve as a point in 2D space where the x-axis
#' represents magnitude outlyingness and the y-axis represents shape outlyingness.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param depth.func Depth function to use for computing outlyingness.
#'   Default is depth.MBD.
#' @param cutoff.quantile Quantile for outlier cutoff (default 0.993).
#' @param col.normal Color for normal curves (default "black").
#' @param col.outliers Color for outlier curves (default "red").
#' @param ... Additional arguments passed to depth function.
#' @importFrom stats qchisq
#'
#' @return A list of class 'ms.plot' with components:
#' \describe{
#'   \item{MO}{Magnitude outlyingness values}
#'   \item{VO}{Shape (variability) outlyingness values}
#'   \item{outliers}{Indices of detected outliers}
#'   \item{cutoff}{Chi-squared cutoff value used}
#'   \item{plot}{The ggplot object}
#' }
#'
#' @details
#' The MS plot (Dai & Genton, 2019) decomposes functional outlyingness into:
#' \itemize{
#'   \item \strong{Magnitude Outlyingness (MO)}: Based on pointwise median of
#'     directional outlyingness - captures shift outliers
#'   \item \strong{Shape Outlyingness (VO)}: Based on variability of directional
#'     outlyingness - captures shape outliers
#' }
#'
#' Outliers are detected using the chi-squared distribution with cutoff at
#' the specified quantile.
#'
#' @references
#' Dai, W. and Genton, M.G. (2019). Directional outlyingness for multivariate
#' functional data. \emph{Computational Statistics & Data Analysis}, 131, 50-65.
#'
#' @export
#' @examples
#' # Create functional data with outliers
#' set.seed(42)
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:28) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
#' X[29, ] <- sin(2*pi*t) + 2  # Magnitude outlier
#' X[30, ] <- sin(4*pi*t)       # Shape outlier
#' fd <- fdata(X, argvals = t)
#'
#' # Create MS plot
#' ms <- MS.plot(fd)
MS.plot <- function(fdataobj, depth.func = depth.MBD,
                    cutoff.quantile = 0.993,
                    col.normal = "black", col.outliers = "red", ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("MS.plot not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)
  argvals <- fdataobj$argvals

  # Compute median curve using depth
  depths <- depth.func(fdataobj, fdataobj, ...)
  median_idx <- which.max(depths)
  median_curve <- fdataobj$data[median_idx, ]

  # Compute pointwise MAD for robust scale
  centered <- sweep(fdataobj$data, 2, median_curve)
  pointwise_mad <- apply(abs(centered), 2, median) * 1.4826  # scale factor for consistency

  # Avoid division by zero
  pointwise_mad[pointwise_mad < 1e-10] <- 1e-10

  # Compute directional outlyingness at each time point
  # O(t) = (X(t) - median(t)) / MAD(t)
  outlyingness <- sweep(centered, 2, pointwise_mad, "/")

  # Magnitude Outlyingness (MO): pointwise median of outlyingness for each curve
  MO <- apply(outlyingness, 1, median)

  # Variability Outlyingness (VO): MAD of outlyingness for each curve
  # This captures shape deviation
  VO <- apply(outlyingness, 1, function(x) median(abs(x - median(x))) * 1.4826)

  # Chi-squared cutoff for outlier detection
  # Using 2 degrees of freedom for (MO, VO)
  cutoff <- qchisq(cutoff.quantile, df = 2)

  # Compute squared Mahalanobis-like distance
  # Using robust estimates of center and scale
  MO_center <- median(MO)
  VO_center <- median(VO)
  MO_scale <- median(abs(MO - MO_center)) * 1.4826
  VO_scale <- median(abs(VO - VO_center)) * 1.4826

  # Avoid division by zero
  if (MO_scale < 1e-10) MO_scale <- 1
  if (VO_scale < 1e-10) VO_scale <- 1

  # Squared standardized distance
  dist_sq <- ((MO - MO_center) / MO_scale)^2 + ((VO - VO_center) / VO_scale)^2

  # Identify outliers
  outliers <- which(dist_sq > cutoff)

  # Create status for plotting
  status <- rep("Normal", n)
  status[outliers] <- "Outlier"

  # Create data frame for plotting
  df <- data.frame(
    MO = MO,
    VO = VO,
    status = factor(status, levels = c("Normal", "Outlier")),
    curve_id = seq_len(n)
  )

  # Create ggplot
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$MO, y = .data$VO,
                                         color = .data$status)) +
    ggplot2::geom_point(size = 2) +
    ggplot2::scale_color_manual(values = c("Normal" = col.normal,
                                           "Outlier" = col.outliers)) +
    ggplot2::labs(
      x = "Magnitude Outlyingness (MO)",
      y = "Shape Outlyingness (VO)",
      title = "Magnitude-Shape Plot",
      color = "Status"
    ) +
    ggplot2::theme_minimal()

  # Add chi-squared contour at cutoff
  theta <- seq(0, 2*pi, length.out = 100)
  r <- sqrt(cutoff)
  ellipse_df <- data.frame(
    x = MO_center + r * MO_scale * cos(theta),
    y = VO_center + r * VO_scale * sin(theta)
  )
  p <- p + ggplot2::geom_path(data = ellipse_df,
                              ggplot2::aes(x = .data$x, y = .data$y),
                              color = "blue", linetype = "dashed",
                              inherit.aes = FALSE)

  print(p)

  # Return result
  result <- structure(
    list(
      MO = MO,
      VO = VO,
      outliers = outliers,
      cutoff = cutoff,
      dist_sq = dist_sq,
      fdataobj = fdataobj,
      plot = p
    ),
    class = "ms.plot"
  )

  invisible(result)
}

#' Print Method for ms.plot Objects
#'
#' @param x An object of class 'ms.plot'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.ms.plot <- function(x, ...) {
  cat("Magnitude-Shape Plot\n")
  cat("====================\n")
  cat("Number of curves:", length(x$MO), "\n")
  cat("Outliers detected:", length(x$outliers), "\n")
  if (length(x$outliers) > 0) {
    cat("Outlier indices:", paste(x$outliers, collapse = ", "), "\n")
  }
  cat("Chi-squared cutoff:", round(x$cutoff, 3), "\n")
  invisible(x)
}
