#' Functional Data Class and Operations
#'
#' Functions for creating and manipulating functional data objects.
#' Supports both 1D functional data (curves) and 2D functional data (surfaces).

# Null-coalescing operator (use left if not NULL, otherwise right)
`%||%` <- function(x, y) if (is.null(x)) y else x

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
#' \describe{
#'   \item{data}{The data matrix}
#'   \item{argvals}{Evaluation points}
#'   \item{rangeval}{Range of arguments}
#'   \item{names}{Plot labels}
#'   \item{fdata2d}{Logical indicating if 2D}
#' }
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
#' Computes the pointwise mean function across all observations.
#' This is an S3 method for the generic \code{mean} function.
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments (currently ignored).
#'
#' @return For 1D fdata: a numeric vector containing the mean function values.
#'   For 2D fdata: an fdata object containing the mean surface.
#' @export
#' @examples
#' # 1D functional data
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' fm <- mean(fd)
#'
#' # 2D functional data
#' X <- array(rnorm(500), dim = c(5, 10, 10))
#' fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
#' fm2d <- mean(fd2d)
mean.fdata <- function(x, ...) {
  if (!inherits(x, "fdata")) {
    stop("x must be of class 'fdata'")
  }

  if (isTRUE(x$fdata2d)) {
    # 2D case
    mean_vals <- .Call("wrap__fdata_mean_2d", x$data)
    # Return as fdata2d object
    result <- list(
      data = matrix(mean_vals, nrow = 1),
      argvals = x$argvals,
      rangeval = x$rangeval,
      names = list(
        main = "Mean surface",
        xlab = x$names$xlab,
        ylab = x$names$ylab,
        zlab = x$names$zlab
      ),
      fdata2d = TRUE,
      dims = x$dims
    )
    class(result) <- "fdata"
    return(result)
  }

  # 1D case
  mean_vals <- .Call("wrap__fdata_mean_1d", x$data)
  result <- list(
    data = matrix(mean_vals, nrow = 1),
    argvals = x$argvals,
    rangeval = x$rangeval,
    names = list(
      main = "Mean curve",
      xlab = x$names$xlab,
      ylab = x$names$ylab
    ),
    fdata2d = FALSE
  )
  class(result) <- "fdata"
  result
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
#' For 1D functional data, plots curves as lines with optional coloring by
#' external variables. For 2D functional data, plots surfaces as heatmaps
#' with contour lines.
#'
#' @param x An object of class 'fdata'.
#' @param color Optional vector for coloring curves. Can be:
#'   \itemize{
#'     \item Numeric vector: curves colored by continuous scale (viridis)
#'     \item Factor/character: curves colored by discrete groups
#'   }
#'   Must have length equal to number of curves.
#' @param alpha Transparency of curve lines (default 0.7).
#' @param show.mean Logical. If TRUE and color is categorical, overlay group
#'   mean curves with thicker lines (default FALSE).
#' @param show.ci Logical. If TRUE and color is categorical, show pointwise
#'   confidence interval ribbons per group (default FALSE).
#' @param ci.level Confidence level for CI ribbons (default 0.90 for 90\%).
#' @param palette Optional named vector of colors for categorical coloring,
#'   e.g., c("A" = "blue", "B" = "red").
#' @param ... Additional arguments (currently ignored).
#'
#' @return A ggplot object (invisibly).
#'
#' @export
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal geom_tile geom_contour scale_fill_viridis_c scale_color_viridis_c facet_wrap geom_ribbon scale_color_manual scale_fill_manual geom_text coord_equal
#' @examples
#' # Basic plot
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' plot(fd)
#'
#' # Color by numeric variable
#' y <- rnorm(20)
#' plot(fd, color = y)
#'
#' # Color by category with mean and CI
#' groups <- factor(rep(c("A", "B"), each = 10))
#' plot(fd, color = groups, show.mean = TRUE, show.ci = TRUE)
plot.fdata <- function(x, color = NULL, alpha = 0.7, show.mean = FALSE,
                       show.ci = FALSE, ci.level = 0.90, palette = NULL, ...) {
  if (isTRUE(x$fdata2d)) {
    # 2D surface plotting (color parameters not supported for 2D)
    n <- nrow(x$data)
    m1 <- x$dims[1]
    m2 <- x$dims[2]
    s <- x$argvals[[1]]
    t <- x$argvals[[2]]

    # Create grid for plotting
    grid <- expand.grid(s = s, t = t)

    # Build long-format data frame
    df_list <- lapply(seq_len(n), function(i) {
      data.frame(
        surface_id = i,
        s = grid$s,
        t = grid$t,
        value = as.vector(matrix(x$data[i, ], m1, m2))
      )
    })
    df <- do.call(rbind, df_list)

    # Plot with facets if multiple surfaces
    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$s, y = .data$t, fill = .data$value)) +
      ggplot2::geom_tile() +
      ggplot2::geom_contour(ggplot2::aes(z = .data$value), color = "black", alpha = 0.5) +
      ggplot2::scale_fill_viridis_c() +
      ggplot2::labs(
        x = x$names$xlab %||% "s",
        y = x$names$ylab %||% "t",
        fill = x$names$zlab %||% "value",
        title = x$names$main
      ) +
      ggplot2::theme_minimal()

    if (n > 1) {
      p <- p + ggplot2::facet_wrap(~ surface_id)
    }

    print(p)
    return(invisible(p))
  }

  # 1D curve plotting
  n <- nrow(x$data)
  m <- ncol(x$data)

  # Validate color parameter
 if (!is.null(color)) {
    if (length(color) != n) {
      stop("length(color) must equal the number of curves (", n, ")")
    }
  }

  # Reshape to long format
  df <- data.frame(
    curve_id = rep(seq_len(n), each = m),
    argval = rep(x$argvals, n),
    value = as.vector(t(x$data))
  )

  # Determine coloring type
  is_categorical <- !is.null(color) && (is.factor(color) || is.character(color))
  is_numeric <- !is.null(color) && is.numeric(color)

  if (is_categorical) {
    # Categorical coloring
    df$group <- factor(rep(color, each = m))

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$argval, y = .data$value,
                                           group = .data$curve_id,
                                           color = .data$group)) +
      ggplot2::geom_line(alpha = alpha)

    # Add confidence interval ribbons if requested
    if (show.ci) {
      ci_df <- .compute_group_ci(df, ci.level)
      p <- p + ggplot2::geom_ribbon(
        data = ci_df,
        ggplot2::aes(x = .data$argval, ymin = .data$lower, ymax = .data$upper,
                     fill = .data$group, group = .data$group),
        alpha = 0.2, inherit.aes = FALSE
      )
    }

    # Add group means if requested
    if (show.mean) {
      mean_df <- .compute_group_mean(df)
      p <- p + ggplot2::geom_line(
        data = mean_df,
        ggplot2::aes(x = .data$argval, y = .data$mean_val, color = .data$group,
                     group = .data$group),
        linewidth = 1.2, inherit.aes = FALSE
      )
    }

    # Apply custom palette if provided
    if (!is.null(palette)) {
      p <- p + ggplot2::scale_color_manual(values = palette)
      if (show.ci) {
        p <- p + ggplot2::scale_fill_manual(values = palette)
      }
    }

  } else if (is_numeric) {
    # Numeric coloring with continuous scale
    df$color_var <- rep(color, each = m)

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$argval, y = .data$value,
                                           group = .data$curve_id,
                                           color = .data$color_var)) +
      ggplot2::geom_line(alpha = alpha) +
      ggplot2::scale_color_viridis_c()

  } else {
    # No coloring (default behavior)
    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$argval, y = .data$value,
                                           group = .data$curve_id)) +
      ggplot2::geom_line(alpha = alpha)
  }

  p <- p +
    ggplot2::labs(
      x = x$names$xlab %||% "t",
      y = x$names$ylab %||% "X(t)",
      title = x$names$main
    ) +
    ggplot2::theme_minimal()

  print(p)
  invisible(p)
}

# Helper function to compute pointwise group means
.compute_group_mean <- function(df) {
  groups <- unique(df$group)
  argvals <- unique(df$argval)

  result_list <- lapply(groups, function(g) {
    group_data <- df[df$group == g, ]
    means <- tapply(group_data$value, group_data$argval, mean, na.rm = TRUE)
    data.frame(
      group = g,
      argval = as.numeric(names(means)),
      mean_val = as.numeric(means)
    )
  })
  do.call(rbind, result_list)
}

# Helper function to compute pointwise group confidence intervals
.compute_group_ci <- function(df, ci.level) {
  groups <- unique(df$group)
  argvals <- unique(df$argval)

  result_list <- lapply(groups, function(g) {
    group_data <- df[df$group == g, ]

    # Compute stats for each argval
    stats <- tapply(seq_len(nrow(group_data)), group_data$argval, function(idx) {
      vals <- group_data$value[idx]
      n_obs <- sum(!is.na(vals))
      if (n_obs < 2) {
        return(c(mean = mean(vals, na.rm = TRUE), lower = NA, upper = NA))
      }
      m <- mean(vals, na.rm = TRUE)
      se <- sd(vals, na.rm = TRUE) / sqrt(n_obs)
      t_crit <- qt((1 + ci.level) / 2, n_obs - 1)
      c(mean = m, lower = m - t_crit * se, upper = m + t_crit * se)
    })

    data.frame(
      group = g,
      argval = as.numeric(names(stats)),
      mean_val = sapply(stats, function(s) s["mean"]),
      lower = sapply(stats, function(s) s["lower"]),
      upper = sapply(stats, function(s) s["upper"])
    )
  })
  do.call(rbind, result_list)
}

#' Functional Boxplot
#'
#' Creates a functional boxplot for visualizing the distribution of functional
#' data. The boxplot shows the median curve, central 50\% envelope, fence
#' (equivalent to whiskers), and outliers.
#'
#' @param x An object of class 'fdata'.
#' @importFrom graphics boxplot
#' @param prob Proportion of curves for the central region (default 0.5 for 50\%).
#' @param factor Factor for fence calculation (default 1.5, as in standard boxplots).
#' @param depth.func Depth function to use. Default is depth.MBD.
#' @param show.outliers Logical. If TRUE (default), show outlier curves.
#' @param col.median Color for median curve (default "black").
#' @param col.envelope Color for central envelope (default "magenta").
#' @param col.fence Color for fence region (default "pink").
#' @param col.outliers Color for outlier curves (default "red").
#' @param ... Additional arguments passed to depth function.
#'
#' @return A list of class 'fbplot' with components:
#' \describe{
#'   \item{median}{Index of the median curve}
#'   \item{central}{Indices of curves in the central region}
#'   \item{outliers}{Indices of outlier curves}
#'   \item{depth}{Depth values for all curves}
#'   \item{plot}{The ggplot object}
#' }
#'
#' @details
#' The functional boxplot (Sun & Genton, 2011) generalizes the standard boxplot
#' to functional data using depth ordering:
#'
#' \itemize{
#'   \item \strong{Median}: The curve with maximum depth
#'   \item \strong{Central region}: Envelope of curves with top 50\% depth
#'   \item \strong{Fence}: 1.5 times the envelope width beyond the central region
#'   \item \strong{Outliers}: Curves that exceed the fence at any point
#' }
#'
#' @references
#' Sun, Y. and Genton, M.G. (2011). Functional boxplots.
#' \emph{Journal of Computational and Graphical Statistics}, 20(2), 316-334.
#'
#' @seealso \code{\link{depth.MBD}} for the default depth function,
#'   \code{\link{outliers.boxplot}} for outlier detection using functional boxplots
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
#' # Create functional boxplot
#' fbp <- boxplot.fdata(fd)
boxplot.fdata <- function(x, prob = 0.5, factor = 1.5,
                          depth.func = depth.MBD,
                          show.outliers = TRUE,
                          col.median = "black",
                          col.envelope = "magenta",
                          col.fence = "pink",
                          col.outliers = "red", ...) {
  if (!inherits(x, "fdata")) {
    stop("x must be of class 'fdata'")
  }

  if (isTRUE(x$fdata2d)) {
    stop("boxplot.fdata not yet implemented for 2D functional data")
  }

  n <- nrow(x$data)
  m <- ncol(x$data)
  argvals <- x$argvals

  # Compute depths
  depths <- depth.func(x, x, ...)

  # Order curves by depth
  depth_order <- order(depths, decreasing = TRUE)

  # Median: curve with maximum depth
  median_idx <- depth_order[1]

  # Central region: top prob proportion of curves
  n_central <- max(1, ceiling(n * prob))
  central_idx <- depth_order[seq_len(n_central)]

  # Compute central envelope (pointwise min/max of central curves)
  central_data <- x$data[central_idx, , drop = FALSE]
  env_min <- apply(central_data, 2, min)
  env_max <- apply(central_data, 2, max)

  # Compute fence: envelope expanded by factor * envelope width
  env_width <- env_max - env_min
  fence_min <- env_min - factor * env_width
  fence_max <- env_max + factor * env_width

  # Identify outliers: curves that exceed the fence at any point
  outlier_idx <- integer(0)
  for (i in seq_len(n)) {
    curve <- x$data[i, ]
    if (any(curve < fence_min) || any(curve > fence_max)) {
      outlier_idx <- c(outlier_idx, i)
    }
  }

  # Non-outlier curves
  normal_idx <- setdiff(seq_len(n), outlier_idx)

  # Create ggplot visualization
  # Build data frames for plotting

  # Fence region (ribbon)
  df_fence <- data.frame(
    argval = argvals,
    ymin = fence_min,
    ymax = fence_max
  )

  # Central envelope
  df_envelope <- data.frame(
    argval = argvals,
    ymin = env_min,
    ymax = env_max
  )

  # Median curve
  df_median <- data.frame(
    argval = argvals,
    value = x$data[median_idx, ]
  )

  # Start building plot
  p <- ggplot2::ggplot()

  # Add fence region
  p <- p + ggplot2::geom_ribbon(
    data = df_fence,
    ggplot2::aes(x = .data$argval, ymin = .data$ymin, ymax = .data$ymax),
    fill = col.fence, alpha = 0.5
  )

  # Add central envelope
  p <- p + ggplot2::geom_ribbon(
    data = df_envelope,
    ggplot2::aes(x = .data$argval, ymin = .data$ymin, ymax = .data$ymax),
    fill = col.envelope, alpha = 0.5
  )

  # Add outlier curves if requested
  if (show.outliers && length(outlier_idx) > 0) {
    df_outliers <- data.frame(
      curve_id = rep(outlier_idx, each = m),
      argval = rep(argvals, length(outlier_idx)),
      value = as.vector(t(x$data[outlier_idx, , drop = FALSE]))
    )
    p <- p + ggplot2::geom_line(
      data = df_outliers,
      ggplot2::aes(x = .data$argval, y = .data$value, group = .data$curve_id),
      color = col.outliers, alpha = 0.7
    )
  }

  # Add median curve
  p <- p + ggplot2::geom_line(
    data = df_median,
    ggplot2::aes(x = .data$argval, y = .data$value),
    color = col.median, linewidth = 1.2
  )

  # Add labels and theme
  p <- p + ggplot2::labs(
    x = x$names$xlab %||% "t",
    y = x$names$ylab %||% "X(t)",
    title = "Functional Boxplot"
  ) + ggplot2::theme_minimal()

  print(p)

  # Return result invisibly
  result <- structure(
    list(
      median = median_idx,
      central = central_idx,
      outliers = outlier_idx,
      depth = depths,
      envelope = list(min = env_min, max = env_max),
      fence = list(min = fence_min, max = fence_max),
      fdataobj = x,
      plot = p
    ),
    class = "fbplot"
  )

  invisible(result)
}

#' Print Method for fbplot Objects
#'
#' @param x An object of class 'fbplot'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.fbplot <- function(x, ...) {
  cat("Functional Boxplot\n")
  cat("==================\n")
  cat("Number of curves:", nrow(x$fdataobj$data), "\n")
  cat("Median curve index:", x$median, "\n")
  cat("Central region curves:", length(x$central), "\n")
  cat("Outliers detected:", length(x$outliers), "\n")
  if (length(x$outliers) > 0) {
    cat("Outlier indices:", paste(x$outliers, collapse = ", "), "\n")
  }
  invisible(x)
}

#' Curve Registration (Alignment)
#'
#' Aligns functional data by horizontal shifting to a target curve.
#' This reduces phase variation in the sample.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param target Target curve to align to. If NULL (default), uses the mean.
#' @param max.shift Maximum allowed shift as proportion of domain (default 0.2).
#'
#' @return A list of class 'register.fd' with components:
#' \describe{
#'   \item{registered}{An fdata object with registered (aligned) curves.}
#'   \item{shifts}{Numeric vector of shift amounts for each curve.}
#'   \item{target}{The target curve used for alignment.}
#'   \item{fdataobj}{Original (unregistered) functional data.}
#' }
#'
#' @details
#' Shift registration finds the horizontal translation that maximizes the
#' cross-correlation between each curve and the target. This is appropriate
#' when curves have similar shapes but differ mainly in timing.
#'
#' For more complex warping, consider DTW-based methods.
#'
#' @seealso \code{\link{metric.DTW}} for dynamic time warping distance
#'
#' @export
#' @examples
#' # Create phase-shifted curves
#' set.seed(42)
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(0, 20, 100)
#' for (i in 1:20) {
#'   phase <- runif(1, -0.1, 0.1)
#'   X[i, ] <- sin(2*pi*(t + phase)) + rnorm(100, sd = 0.1)
#' }
#' fd <- fdata(X, argvals = t)
#'
#' # Register curves
#' reg <- register.fd(fd)
#' print(reg)
#'
#' # Compare original vs registered
#' par(mfrow = c(1, 2))
#' plot(fd)
#' plot(reg$registered)
register.fd <- function(fdataobj, target = NULL, max.shift = 0.2) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("register.fd not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)
  argvals <- fdataobj$argvals

  # Determine target curve
  if (is.null(target)) {
    target <- colMeans(fdataobj$data)
  } else if (inherits(target, "fdata")) {
    if (ncol(target$data) != m) {
      stop("target must have same number of evaluation points as fdataobj")
    }
    target <- as.vector(target$data[1, ])
  } else if (!is.numeric(target) || length(target) != m) {
    stop("target must be NULL, an fdata object, or a numeric vector of length m")
  }

  # Compute max shift in domain units
  domain_range <- max(argvals) - min(argvals)
  max_shift_val <- max.shift * domain_range

  # Call Rust function
  result <- .Call("wrap__register_shift_1d", fdataobj$data,
                  as.numeric(target), as.numeric(argvals), as.numeric(max_shift_val))

  # Create registered fdata object
  registered <- fdata(result$registered, argvals = argvals,
                      names = list(main = "Registered Curves",
                                   xlab = fdataobj$names$xlab,
                                   ylab = fdataobj$names$ylab))

  structure(
    list(
      registered = registered,
      shifts = result$shifts,
      target = target,
      fdataobj = fdataobj
    ),
    class = "register.fd"
  )
}

#' Print Method for register.fd Objects
#'
#' @param x An object of class 'register.fd'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.register.fd <- function(x, ...) {
  cat("Curve Registration\n")
  cat("==================\n")
  cat("Number of curves:", nrow(x$fdataobj$data), "\n")
  cat("Shift statistics:\n")
  cat("  Min:", round(min(x$shifts), 4), "\n")
  cat("  Max:", round(max(x$shifts), 4), "\n")
  cat("  Mean:", round(mean(x$shifts), 4), "\n")
  cat("  SD:", round(sd(x$shifts), 4), "\n")
  invisible(x)
}

#' Plot Method for register.fd Objects
#'
#' @param x An object of class 'register.fd'.
#' @param type Type of plot: "registered" (default), "original", or "both".
#' @param ... Additional arguments (currently ignored).
#'
#' @return A ggplot object.
#'
#' @export
plot.register.fd <- function(x, type = c("registered", "original", "both"), ...) {
  type <- match.arg(type)

  if (type == "original") {
    plot(x$fdataobj)
  } else if (type == "registered") {
    plot(x$registered)
  } else {
    # Side-by-side comparison
    fd_orig <- x$fdataobj
    fd_reg <- x$registered
    n <- nrow(fd_orig$data)
    m <- ncol(fd_orig$data)

    df <- data.frame(
      curve_id = rep(rep(seq_len(n), each = m), 2),
      argval = rep(fd_orig$argvals, n * 2),
      value = c(as.vector(t(fd_orig$data)), as.vector(t(fd_reg$data))),
      type = factor(rep(c("Original", "Registered"), each = n * m),
                    levels = c("Original", "Registered"))
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$argval, y = .data$value,
                                           group = .data$curve_id)) +
      ggplot2::geom_line(alpha = 0.5) +
      ggplot2::facet_wrap(~ .data$type) +
      ggplot2::labs(
        x = fd_orig$names$xlab %||% "t",
        y = fd_orig$names$ylab %||% "X(t)",
        title = "Curve Registration: Before vs After"
      ) +
      ggplot2::theme_minimal()

    print(p)
    invisible(p)
  }
}

#' Local Averages Feature Extraction
#'
#' Extracts features from functional data by computing local averages over
#' specified intervals. This is a simple but effective dimension reduction
#' technique for functional data.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param n.intervals Number of equal-width intervals (default 10).
#' @param intervals Optional matrix of custom intervals (2 columns: start, end).
#'   If provided, \code{n.intervals} is ignored.
#'
#' @return A matrix with n rows (curves) and one column per interval, containing
#'   the local average for each curve in each interval.
#'
#' @details
#' Local averages provide a simple way to convert functional data to
#' multivariate data while preserving local structure. Each curve is
#' summarized by its average value over each interval.
#'
#' This can be useful as a preprocessing step for classification or
#' clustering methods that require fixed-dimensional input.
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 100)
#' X <- matrix(0, 20, 100)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(100, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Extract 5 local average features
#' features <- localavg.fdata(fd, n.intervals = 5)
#' dim(features)  # 20 x 5
#'
#' # Use custom intervals
#' intervals <- cbind(c(0, 0.25, 0.5), c(0.25, 0.5, 1))
#' features2 <- localavg.fdata(fd, intervals = intervals)
localavg.fdata <- function(fdataobj, n.intervals = 10, intervals = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("localavg.fdata not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)
  argvals <- fdataobj$argvals
  range_val <- range(argvals)

  # Create intervals if not provided
  if (is.null(intervals)) {
    breaks <- seq(range_val[1], range_val[2], length.out = n.intervals + 1)
    intervals <- cbind(breaks[-length(breaks)], breaks[-1])
  } else {
    intervals <- as.matrix(intervals)
    if (ncol(intervals) != 2) {
      stop("intervals must be a matrix with 2 columns (start, end)")
    }
  }

  n_int <- nrow(intervals)

  # Compute local averages for each curve and interval
  features <- matrix(0, n, n_int)

  for (k in seq_len(n_int)) {
    int_start <- intervals[k, 1]
    int_end <- intervals[k, 2]

    # Find indices within this interval
    idx <- which(argvals >= int_start & argvals <= int_end)

    if (length(idx) > 0) {
      # Compute mean within interval for each curve
      features[, k] <- rowMeans(fdataobj$data[, idx, drop = FALSE])
    }
  }

  # Add column names
  colnames(features) <- paste0("int_", seq_len(n_int))

  features
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
#' fd_deriv <- deriv(fd, nderiv = 1)
deriv <- function(fdataobj, nderiv = 1, method = "diff",
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
#' @return A list of class 'fdata.bootstrap' with components:
#' \describe{
#'   \item{boot.samples}{List of n.boot fdata objects, each a bootstrap sample}
#'   \item{original}{The original fdata object}
#'   \item{method}{The bootstrap method used}
#'   \item{n.boot}{Number of bootstrap replications}
#' }
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
#' @return A list of class 'fdata.bootstrap.ci' with components:
#' \describe{
#'   \item{estimate}{The statistic computed on the original data}
#'   \item{ci.lower}{Lower confidence bound}
#'   \item{ci.upper}{Upper confidence bound}
#'   \item{boot.stats}{Matrix of bootstrap statistics (n.boot x length(statistic))}
#'   \item{alpha}{The significance level used}
#'   \item{method}{The CI method used}
#' }
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
#' ci_mean <- fdata.bootstrap.ci(fd, statistic = mean, n.boot = 100)
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
#' @return A list with components:
#' \describe{
#'   \item{d}{Singular values (proportional to sqrt of eigenvalues)}
#'   \item{rotation}{fdata object containing PC loadings}
#'   \item{x}{Matrix of PC scores (n x ncomp)}
#'   \item{mean}{Mean function (numeric vector)}
#'   \item{fdataobj.cen}{Centered fdata object}
#'   \item{call}{The function call}
#' }
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

  structure(
    list(
      d = result$d,
      rotation = rotation,
      x = result$scores,
      mean = result$mean,
      fdataobj.cen = fdataobj.cen,
      argvals = fdataobj$argvals,
      call = match.call()
    ),
    class = "fdata2pc"
  )
}

#' Plot FPCA Results
#'
#' Visualize functional principal component analysis results with multiple
#' plot types: component perturbation plots, variance explained (scree plot),
#' or score plots.
#'
#' @param x An object of class 'fdata2pc' from \code{\link{fdata2pc}}.
#' @param type Type of plot: "components" (default) shows mean +/- scaled PC loadings,
#'   "variance" shows a scree plot of variance explained, "scores" shows PC1 vs PC2
#'   scatter plot of observations.
#' @param ncomp Number of components to display (default 3 or fewer if not available).
#' @param multiple Factor for scaling PC perturbations. Default is 2 (shows +/- 2*sqrt(eigenvalue)*PC).
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return A ggplot object (invisibly).
#'
#' @details
#' The "components" plot shows the mean function (black) with perturbations
#' in the direction of each principal component. The perturbation is computed as:
#' mean +/- multiple * sqrt(variance_explained) * PC_loading
#'
#' The "variance" plot shows a scree plot with the proportion of variance
#' explained by each component as a bar chart.
#'
#' The "scores" plot shows a scatter plot of observations in PC space,
#' typically PC1 vs PC2.
#'
#' @seealso \code{\link{fdata2pc}} for computing FPCA.
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 30, 50)
#' for (i in 1:30) X[i, ] <- sin(2*pi*t + runif(1, 0, pi)) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#' pc <- fdata2pc(fd, ncomp = 3)
#'
#' # Plot PC components (mean +/- perturbations)
#' plot(pc, type = "components")
#'
#' # Scree plot
#' plot(pc, type = "variance")
#'
#' # Score plot
#' plot(pc, type = "scores")
plot.fdata2pc <- function(x, type = c("components", "variance", "scores"),
                          ncomp = 3, multiple = 2, ...) {
  type <- match.arg(type)

  ncomp <- min(ncomp, length(x$d), ncol(x$x))

  switch(type,
    "components" = .plot_fpca_components(x, ncomp, multiple),
    "variance" = .plot_fpca_variance(x, ncomp),
    "scores" = .plot_fpca_scores(x, ncomp)
  )
}

# Internal: Plot FPCA components (mean +/- perturbations)
# @noRd
.plot_fpca_components <- function(x, ncomp, multiple) {
  m <- length(x$argvals)

  # Compute variance explained (proportional to d^2)
  var_explained <- x$d^2
  total_var <- sum(var_explained)
  prop_var <- var_explained / total_var

  # Build data frame for plotting
  plot_data <- list()

  # Add mean function
  plot_data[[1]] <- data.frame(
    t = x$argvals,
    value = x$mean,
    type = "Mean",
    component = "Mean",
    direction = "mean"
  )

  # Add PC perturbations
  for (k in seq_len(ncomp)) {
    loading <- x$rotation$data[k, ]
    scale_factor <- multiple * sqrt(var_explained[k])

    # Plus direction
    plot_data[[length(plot_data) + 1]] <- data.frame(
      t = x$argvals,
      value = x$mean + scale_factor * loading,
      type = paste0("PC", k),
      component = paste0("PC", k, " (", round(100 * prop_var[k], 1), "%)"),
      direction = "plus"
    )

    # Minus direction
    plot_data[[length(plot_data) + 1]] <- data.frame(
      t = x$argvals,
      value = x$mean - scale_factor * loading,
      type = paste0("PC", k),
      component = paste0("PC", k, " (", round(100 * prop_var[k], 1), "%)"),
      direction = "minus"
    )
  }

  df <- do.call(rbind, plot_data)
  df$component <- factor(df$component, levels = unique(df$component))

  # Create plot
  p <- ggplot2::ggplot(df, ggplot2::aes(x = t, y = value, color = component,
                                         linetype = direction)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::scale_linetype_manual(
      values = c("mean" = "solid", "plus" = "dashed", "minus" = "dotted"),
      guide = "none"
    ) +
    ggplot2::labs(
      title = "FPCA: Principal Component Perturbations",
      subtitle = paste0("Mean \u00B1 ", multiple, " \u00D7 sqrt(eigenvalue) \u00D7 PC"),
      x = "t",
      y = "X(t)",
      color = ""
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "bottom")

  print(p)
  invisible(p)
}

# Internal: Plot FPCA variance explained (scree plot)
# @noRd
.plot_fpca_variance <- function(x, ncomp) {
  # Compute variance explained
  var_explained <- x$d^2
  total_var <- sum(var_explained)
  prop_var <- var_explained / total_var
  cum_var <- cumsum(prop_var)

  # Limit to ncomp
  ncomp <- min(ncomp, length(var_explained))

  df <- data.frame(
    component = factor(seq_len(ncomp), levels = seq_len(ncomp)),
    prop = prop_var[seq_len(ncomp)] * 100,
    cumulative = cum_var[seq_len(ncomp)] * 100
  )

  p <- ggplot2::ggplot(df, ggplot2::aes(x = component, y = prop)) +
    ggplot2::geom_col(fill = "steelblue", alpha = 0.7) +
    ggplot2::geom_line(ggplot2::aes(y = cumulative, group = 1),
                        color = "darkred", linewidth = 1) +
    ggplot2::geom_point(ggplot2::aes(y = cumulative), color = "darkred", size = 2) +
    ggplot2::geom_text(ggplot2::aes(label = paste0(round(prop, 1), "%")),
                       vjust = -0.5, size = 3) +
    ggplot2::scale_y_continuous(
      name = "Variance Explained (%)",
      sec.axis = ggplot2::sec_axis(~., name = "Cumulative (%)")
    ) +
    ggplot2::labs(
      title = "FPCA: Variance Explained (Scree Plot)",
      x = "Principal Component"
    ) +
    ggplot2::theme_minimal()

  print(p)
  invisible(p)
}

# Internal: Plot FPCA scores
# @noRd
.plot_fpca_scores <- function(x, ncomp) {
  scores <- x$x
  n <- nrow(scores)

  # Compute variance explained for axis labels
  var_explained <- x$d^2
  total_var <- sum(var_explained)
  prop_var <- var_explained / total_var * 100

  if (ncol(scores) >= 2) {
    # 2D scatter plot: PC1 vs PC2
    df <- data.frame(
      PC1 = scores[, 1],
      PC2 = scores[, 2],
      id = seq_len(n)
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = PC1, y = PC2)) +
      ggplot2::geom_point(color = "steelblue", size = 2, alpha = 0.7) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = "FPCA: Score Plot",
        x = paste0("PC1 (", round(prop_var[1], 1), "%)"),
        y = paste0("PC2 (", round(prop_var[2], 1), "%)")
      ) +
      ggplot2::theme_minimal()
  } else {
    # Only 1 PC: plot scores as bar chart
    df <- data.frame(
      id = factor(seq_len(n)),
      PC1 = scores[, 1]
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = id, y = PC1)) +
      ggplot2::geom_col(fill = "steelblue", alpha = 0.7) +
      ggplot2::labs(
        title = "FPCA: Score Plot",
        x = "Observation",
        y = paste0("PC1 (", round(prop_var[1], 1), "%)")
      ) +
      ggplot2::theme_minimal()
  }

  print(p)
  invisible(p)
}

#' Print Method for FPCA Results
#'
#' @param x An object of class 'fdata2pc'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.fdata2pc <- function(x, ...) {
  cat("Functional Principal Component Analysis\n")
  cat("========================================\n")
  cat("Number of observations:", nrow(x$x), "\n")
  cat("Number of components:", length(x$d), "\n\n")

  # Compute variance explained
  var_explained <- x$d^2
  total_var <- sum(var_explained)
  prop_var <- var_explained / total_var * 100
  cum_var <- cumsum(prop_var)

  cat("Variance explained:\n")
  for (k in seq_along(x$d)) {
    cat(sprintf("  PC%d: %.1f%% (cumulative: %.1f%%)\n",
                k, prop_var[k], cum_var[k]))
  }
  invisible(x)
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
#' @return A list with components:
#' \describe{
#'   \item{weights}{Matrix of PLS weights (m x ncomp)}
#'   \item{scores}{Matrix of PLS scores (n x ncomp)}
#'   \item{loadings}{Matrix of PLS loadings (m x ncomp)}
#'   \item{call}{The function call}
#' }
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

#' Compute Distance/Similarity Between Groups of Functional Data
#'
#' Computes various distance and similarity measures between pre-defined groups
#' of functional curves.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param groups A factor or character vector specifying group membership for each curve.
#'   Must have length equal to the number of curves.
#' @param method Distance/similarity method:
#'   \itemize{
#'     \item "centroid": L2 distance between group mean curves
#'     \item "hausdorff": Hausdorff-style distance between groups
#'     \item "depth": Depth-based overlap (similarity, not distance)
#'     \item "all": Compute all methods
#'   }
#' @param metric Distance metric for centroid method (default "lp").
#' @param p Power for Lp metric (default 2 for L2).
#' @param depth.method Depth method for depth-based overlap (default "FM").
#' @param ... Additional arguments passed to metric functions.
#'
#' @return An object of class 'group.distance' containing:
#' \describe{
#'   \item{centroid}{Centroid distance matrix (if method includes centroid)}
#'   \item{hausdorff}{Hausdorff distance matrix (if method includes hausdorff)}
#'   \item{depth}{Depth-based similarity matrix (if method includes depth)}
#'   \item{groups}{Unique group labels}
#'   \item{group.sizes}{Number of curves per group}
#'   \item{method}{Methods used}
#' }
#'
#' @export
#' @examples
#' # Create grouped functional data
#' set.seed(42)
#' n <- 30
#' m <- 50
#' t_grid <- seq(0, 1, length.out = m)
#' X <- matrix(0, n, m)
#' for (i in 1:15) X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
#' for (i in 16:30) X[i, ] <- cos(2 * pi * t_grid) + rnorm(m, sd = 0.1)
#' fd <- fdata(X, argvals = t_grid)
#' groups <- factor(rep(c("A", "B"), each = 15))
#'
#' # Compute all distance measures
#' gd <- group.distance(fd, groups, method = "all")
#' print(gd)
group.distance <- function(fdataobj, groups,
                           method = c("centroid", "hausdorff", "depth", "all"),
                           metric = "lp", p = 2, depth.method = "FM", ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  if (length(groups) != n) {
    stop("length(groups) must equal the number of curves (", n, ")")
  }

  groups <- as.factor(groups)
  group_levels <- levels(groups)
  n_groups <- length(group_levels)

  if (n_groups < 2) {
    stop("Need at least 2 groups to compute distances")
  }

  method <- match.arg(method)
  compute_all <- method == "all"

  result <- list(
    groups = group_levels,
    group.sizes = table(groups),
    method = if (compute_all) c("centroid", "hausdorff", "depth") else method
  )

  # Compute centroid distances
  if (method == "centroid" || compute_all) {
    result$centroid <- .group_centroid_distance(fdataobj, groups, group_levels, metric, p, ...)
  }

  # Compute Hausdorff distances
  if (method == "hausdorff" || compute_all) {
    result$hausdorff <- .group_hausdorff_distance(fdataobj, groups, group_levels, metric, p, ...)
  }

  # Compute depth-based overlap
  if (method == "depth" || compute_all) {
    result$depth <- .group_depth_overlap(fdataobj, groups, group_levels, depth.method)
  }

  class(result) <- "group.distance"
  result
}

# Internal: Compute centroid (mean curve) distances between groups
.group_centroid_distance <- function(fdataobj, groups, group_levels, metric, p, ...) {
  n_groups <- length(group_levels)

  # Compute group means
  group_means <- lapply(group_levels, function(g) {
    idx <- which(groups == g)
    mean(fdataobj[idx])
  })

  # Compute pairwise distances between means
  dist_mat <- matrix(0, n_groups, n_groups)
  rownames(dist_mat) <- colnames(dist_mat) <- group_levels

  for (i in seq_len(n_groups)) {
    for (j in seq_len(n_groups)) {
      if (i < j) {
        # Combine means into single fdata for distance calculation
        combined <- fdata(
          rbind(group_means[[i]]$data, group_means[[j]]$data),
          argvals = fdataobj$argvals
        )
        d <- metric.lp(combined, p = p, ...)[1, 2]
        dist_mat[i, j] <- d
        dist_mat[j, i] <- d
      }
    }
  }

  dist_mat
}

# Internal: Compute Hausdorff-style distances between groups
.group_hausdorff_distance <- function(fdataobj, groups, group_levels, metric, p, ...) {
  n_groups <- length(group_levels)

  # Pre-compute full distance matrix
  full_dist <- metric.lp(fdataobj, p = p, ...)

  dist_mat <- matrix(0, n_groups, n_groups)
  rownames(dist_mat) <- colnames(dist_mat) <- group_levels

  for (i in seq_len(n_groups)) {
    for (j in seq_len(n_groups)) {
      if (i < j) {
        idx_i <- which(groups == group_levels[i])
        idx_j <- which(groups == group_levels[j])

        # Hausdorff: max(max_a min_b d(a,b), max_b min_a d(a,b))
        # For each curve in group i, find minimum distance to group j
        min_dists_i_to_j <- apply(full_dist[idx_i, idx_j, drop = FALSE], 1, min)
        # For each curve in group j, find minimum distance to group i
        min_dists_j_to_i <- apply(full_dist[idx_j, idx_i, drop = FALSE], 1, min)

        hausdorff_dist <- max(max(min_dists_i_to_j), max(min_dists_j_to_i))
        dist_mat[i, j] <- hausdorff_dist
        dist_mat[j, i] <- hausdorff_dist
      }
    }
  }

  dist_mat
}

# Internal: Compute depth-based overlap (similarity) between groups
.group_depth_overlap <- function(fdataobj, groups, group_levels, depth.method) {
  n_groups <- length(group_levels)

  sim_mat <- matrix(0, n_groups, n_groups)
  rownames(sim_mat) <- colnames(sim_mat) <- group_levels

  for (i in seq_len(n_groups)) {
    for (j in seq_len(n_groups)) {
      idx_i <- which(groups == group_levels[i])
      idx_j <- which(groups == group_levels[j])

      if (i == j) {
        # Self-overlap is 1 (curves in group have depth w.r.t. themselves)
        sim_mat[i, j] <- 1
      } else {
        # Compute mean depth of curves in group i w.r.t. group j
        depth_i_in_j <- depth(fdataobj[idx_i], fdataobj[idx_j], method = depth.method)
        sim_mat[i, j] <- mean(depth_i_in_j)
      }
    }
  }

  sim_mat
}

#' Print method for group.distance
#' @export
print.group.distance <- function(x, digits = 3, ...) {
  cat("Group Distance Analysis\n")
  cat("=======================\n")
  cat("Groups:", paste(x$groups, collapse = ", "), "\n")
  cat("Group sizes:", paste(paste0(names(x$group.sizes), "=", x$group.sizes), collapse = ", "), "\n\n")

  if (!is.null(x$centroid)) {
    cat("Centroid Distance (L2 between group means):\n")
    print(round(x$centroid, digits))
    cat("\n")
  }

  if (!is.null(x$hausdorff)) {
    cat("Hausdorff Distance (worst-case between groups):\n")
    print(round(x$hausdorff, digits))
    cat("\n")
  }

  if (!is.null(x$depth)) {
    cat("Depth Overlap (similarity, higher = more similar):\n")
    print(round(x$depth, digits))
    cat("\n")
  }

  invisible(x)
}

#' Plot method for group.distance
#'
#' @param x An object of class 'group.distance'.
#' @param type Plot type: "heatmap" or "dendrogram".
#' @param which Which distance matrix to plot (default "centroid").
#' @param ... Additional arguments.
#'
#' @return A ggplot object (for heatmap) or NULL (for dendrogram, uses base graphics).
#' @export
plot.group.distance <- function(x, type = c("heatmap", "dendrogram"),
                                which = c("centroid", "hausdorff", "depth"), ...) {
  type <- match.arg(type)
  which <- match.arg(which)

  # Get the appropriate matrix
  mat <- x[[which]]
  if (is.null(mat)) {
    stop("Distance matrix '", which, "' not available. Run group.distance with method='all' or method='", which, "'")
  }

  if (type == "heatmap") {
    # Convert to long format for ggplot
    n <- nrow(mat)
    df <- data.frame(
      group1 = rep(rownames(mat), n),
      group2 = rep(colnames(mat), each = n),
      value = as.vector(mat)
    )
    df$group1 <- factor(df$group1, levels = rownames(mat))
    df$group2 <- factor(df$group2, levels = colnames(mat))

    title <- switch(which,
      centroid = "Centroid Distance",
      hausdorff = "Hausdorff Distance",
      depth = "Depth Overlap (Similarity)"
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$group1, y = .data$group2, fill = .data$value)) +
      ggplot2::geom_tile() +
      ggplot2::geom_text(ggplot2::aes(label = round(.data$value, 2)), color = "white", size = 4) +
      ggplot2::scale_fill_viridis_c() +
      ggplot2::labs(x = "Group", y = "Group", fill = "Value", title = title) +
      ggplot2::theme_minimal() +
      ggplot2::coord_equal()

    print(p)
    invisible(p)

  } else {
    # Dendrogram using base R
    if (which == "depth") {
      # Convert similarity to distance for clustering
      dist_mat <- 1 - mat
    } else {
      dist_mat <- mat
    }

    hc <- hclust(as.dist(dist_mat), method = "complete")
    plot(hc, main = paste("Hierarchical Clustering -", which),
         xlab = "Group", ylab = "Distance")
    invisible(NULL)
  }
}

#' Permutation Test for Group Differences
#'
#' Tests whether groups of functional data are significantly different using
#' permutation testing.
#'
#' @details
#' **Null Hypothesis (H0):** All groups come from the same distribution. That is,
#' the group labels are exchangeable and there is no systematic difference between
#' the functional curves in different groups.
#'
#' **Alternative Hypothesis (H1):** At least one group differs from the others in
#' terms of location (mean function) or dispersion.
#'
#' The test works by:
#' 1. Computing a test statistic on the observed data
#' 2. Repeatedly permuting the group labels and recomputing the statistic
#' 3. Calculating the p-value as the proportion of permuted statistics >= observed
#'
#' Two test statistics are available:
#' \itemize{
#'   \item \code{"centroid"}: Sum of pairwise L2 distances between group mean
#'     functions. Sensitive to differences in group locations (means).
#'   \item \code{"ratio"}: Ratio of between-group to within-group variance,
#'     similar to an F-statistic. Sensitive to both location and dispersion.
#' }
#'
#' A small p-value (e.g., < 0.05) indicates evidence against H0, suggesting
#' that the groups are significantly different.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param groups A factor or character vector specifying group membership.
#' @param n.perm Number of permutations (default 1000).
#' @param statistic Test statistic: "centroid" (distance between group means) or
#'   "ratio" (between/within group variance ratio).
#' @param ... Additional arguments passed to distance functions.
#'
#' @return An object of class 'group.test' containing:
#' \describe{
#'   \item{statistic}{Observed test statistic}
#'   \item{p.value}{Permutation p-value}
#'   \item{perm.dist}{Permutation distribution of test statistic}
#'   \item{n.perm}{Number of permutations used}
#' }
#'
#' @export
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 30
#' m <- 50
#' t_grid <- seq(0, 1, length.out = m)
#' X <- matrix(0, n, m)
#' for (i in 1:15) X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
#' for (i in 16:30) X[i, ] <- cos(2 * pi * t_grid) + rnorm(m, sd = 0.1)
#' fd <- fdata(X, argvals = t_grid)
#' groups <- factor(rep(c("A", "B"), each = 15))
#'
#' # Test for significant difference
#' gt <- group.test(fd, groups, n.perm = 500)
#' print(gt)
#' }
group.test <- function(fdataobj, groups, n.perm = 1000,
                       statistic = c("centroid", "ratio"), ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  if (length(groups) != n) {
    stop("length(groups) must equal the number of curves (", n, ")")
  }

  groups <- as.factor(groups)
  statistic <- match.arg(statistic)

  # Compute observed test statistic
  obs_stat <- .compute_group_stat(fdataobj, groups, statistic, ...)

  # Permutation distribution
  perm_stats <- numeric(n.perm)
  for (i in seq_len(n.perm)) {
    perm_groups <- sample(groups)
    perm_stats[i] <- .compute_group_stat(fdataobj, perm_groups, statistic, ...)
  }

  # Compute p-value (proportion of permuted stats >= observed)
  p_value <- mean(perm_stats >= obs_stat)

  result <- list(
    statistic = obs_stat,
    p.value = p_value,
    perm.dist = perm_stats,
    n.perm = n.perm,
    stat.type = statistic
  )
  class(result) <- "group.test"
  result
}

# Internal: Compute test statistic for group comparison
.compute_group_stat <- function(fdataobj, groups, statistic, ...) {
  group_levels <- levels(groups)

  if (statistic == "centroid") {
    # Sum of pairwise centroid distances
    gd <- group.distance(fdataobj, groups, method = "centroid", ...)
    # Return sum of upper triangle (total between-group distance)
    return(sum(gd$centroid[upper.tri(gd$centroid)]))

  } else {
    # Between/within variance ratio
    # Between: sum of distances from group means to overall mean
    # Within: sum of distances from curves to their group means

    overall_mean <- mean(fdataobj)
    n_groups <- length(group_levels)

    between_var <- 0
    within_var <- 0

    for (g in group_levels) {
      idx <- which(groups == g)
      n_g <- length(idx)
      group_data <- fdataobj[idx]
      group_mean <- mean(group_data)

      # Between: distance from group mean to overall mean, weighted by group size
      combined <- fdata(rbind(group_mean$data, overall_mean$data), argvals = fdataobj$argvals)
      between_var <- between_var + n_g * metric.lp(combined, p = 2)[1, 2]^2

      # Within: distances from curves to group mean
      for (i in seq_len(n_g)) {
        curve_mean <- fdata(rbind(group_data$data[i, ], group_mean$data), argvals = fdataobj$argvals)
        within_var <- within_var + metric.lp(curve_mean, p = 2)[1, 2]^2
      }
    }

    # F-like ratio (higher = more separation)
    return(between_var / max(within_var, 1e-10))
  }
}

#' Print method for group.test
#' @export
print.group.test <- function(x, ...) {
  cat("Permutation Test for Group Differences\n")
  cat("======================================\n")
  cat("Test statistic type:", x$stat.type, "\n")
  cat("Observed statistic:", round(x$statistic, 4), "\n")
  cat("Number of permutations:", x$n.perm, "\n")
  cat("P-value:", format.pval(x$p.value, digits = 3))

  if (x$p.value < 0.001) {
    cat(" ***\n")
  } else if (x$p.value < 0.01) {
    cat(" **\n")
  } else if (x$p.value < 0.05) {
    cat(" *\n")
  } else {
    cat("\n")
  }

  invisible(x)
}
