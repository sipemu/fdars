#' Depth Functions for Functional Data
#'
#' Functions for computing various depth measures for functional data.

#' Fraiman-Muniz Depth
#'
#' Computes the Fraiman-Muniz depth for functional data. The FM depth
#' integrates the univariate depth over the domain.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param trim Trimming proportion (0 to 0.5). Default is 0.25.
#' @param scale Logical. If TRUE (default), scales depth to \[0, 1\] range
#'   using the FM1 formula from fda.usc. If FALSE, returns unscaled values
#'   in \[0, 0.5\] range.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.FM(fd)
depth.FM <- function(fdataobj, fdataori = NULL, trim = 0.25, scale = TRUE, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_fm_2d", fdataobj$data, fdataori$data, m1, m2, as.logical(scale)))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_fm_1d", fdataobj$data, fdataori$data, as.numeric(trim), as.logical(scale))
}

#' Band Depth
#'
#' Computes the band depth for functional data. The band depth measures
#' how often a curve lies completely within the band formed by pairs of
#' other curves.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'   Values range from 0 to 1, with higher values indicating more central curves.
#'
#' @details
#' For a curve x and reference sample Y_1, ..., Y_n, the band depth is:
#' \deqn{BD(x) = \binom{n}{2}^{-1} \sum_{1 \le i < j \le n} I(x \in B(Y_i, Y_j))}
#' where B(Y_i, Y_j) is the band formed by Y_i and Y_j, and I() is the indicator
#' function that equals 1 if x lies completely within the band at all time points.
#'
#' Band depth is computationally efficient but can be 0 for curves that cross
#' the reference curves at any point. For a more robust alternative, see
#' \code{\link{depth.MBD}}.
#'
#' @seealso \code{\link{depth.MBD}} for modified band depth
#'
#' @export
#' @examples
#' # Create functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' fd <- fdata(X, argvals = t)
#'
#' # Compute band depth
#' depths <- depth.BD(fd)
#' print(depths)
depth.BD <- function(fdataobj, fdataori = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d) || isTRUE(fdataori$fdata2d)) {
    stop("depth.BD not yet implemented for 2D functional data")
  }

  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  if (nrow(fdataori$data) < 2) {
    stop("fdataori must have at least 2 curves")
  }

  .Call("wrap__depth_bd_1d", fdataobj$data, fdataori$data)
}

#' Modified Band Depth
#'
#' Computes the modified band depth for functional data. Unlike standard band
#' depth, MBD measures the proportion of the domain where each curve lies
#' within the band, averaged over all pairs.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'   Values range from 0 to 1, with higher values indicating more central curves.
#'
#' @details
#' For a curve x and reference sample Y_1, ..., Y_n, the modified band depth is:
#' \deqn{MBD(x) = \binom{n}{2}^{-1} \sum_{1 \le i < j \le n} \lambda_r(A(x; Y_i, Y_j))}
#' where A(x; Y_i, Y_j) is the set of time points where x lies within the band
#' formed by Y_i and Y_j, and lambda_r is the Lebesgue measure normalized by
#' the domain length.
#'
#' MBD is more robust than standard band depth because it doesn't require
#' complete containment. A curve that crosses the band boundaries still receives
#' partial depth based on the proportion of time it spends inside.
#'
#' MBD is the recommended depth for functional boxplots and outlier detection.
#'
#' @seealso \code{\link{depth.BD}} for standard band depth,
#'   \code{\link{boxplot.fdata}} for functional boxplots using MBD
#'
#' @export
#' @examples
#' # Create functional data with an outlier
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:19) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' X[20, ] <- sin(2*pi*t) + 2  # magnitude outlier
#' fd <- fdata(X, argvals = t)
#'
#' # Compute modified band depth
#' depths <- depth.MBD(fd)
#' print(depths)
#'
#' # The outlier should have lower depth
#' which.min(depths)  # Should be 20
depth.MBD <- function(fdataobj, fdataori = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d) || isTRUE(fdataori$fdata2d)) {
    stop("depth.MBD not yet implemented for 2D functional data")
  }

  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  if (nrow(fdataori$data) < 2) {
    stop("fdataori must have at least 2 curves")
  }

  .Call("wrap__depth_mbd_1d", fdataobj$data, fdataori$data)
}

#' Modal Depth
#'
#' Computes the modal depth for functional data based on kernel density
#' estimation in function space.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param h Bandwidth parameter. If NULL, computed automatically.
#' @param metric Distance metric function (currently ignored, uses L2).
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.mode(fd)
depth.mode <- function(fdataobj, fdataori = NULL, h = NULL, metric = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  # Auto-compute bandwidth if not provided
  if (is.null(h)) {
    # Use Silverman's rule of thumb adapted for functional data
    n <- nrow(fdataori$data)
    h <- 1.06 * sd(fdataori$data) * n^(-1/5)
    if (h < 1e-10) h <- 0.1  # fallback
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_mode_2d", fdataobj$data, fdataori$data, m1, m2, as.numeric(h)))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_mode_1d", fdataobj$data, fdataori$data, as.numeric(h))
}

#' Random Projection Depth
#'
#' Computes depth using random projections. Projects curves onto random
#' directions and averages the univariate depths.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.RP(fd, nproj = 100)
depth.RP <- function(fdataobj, fdataori = NULL, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_rp_2d", fdataobj$data, fdataori$data, m1, m2, as.integer(nproj)))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_rp_1d", fdataobj$data, fdataori$data, as.integer(nproj))
}

#' Random Tukey Depth
#'
#' Computes depth using random projections, taking the minimum depth
#' across all projections (Tukey halfspace depth approximation).
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.RT(fd, nproj = 100)
depth.RT <- function(fdataobj, fdataori = NULL, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_rt_2d", fdataobj$data, fdataori$data, m1, m2, as.integer(nproj)))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_rt_1d", fdataobj$data, fdataori$data, as.integer(nproj))
}

#' Functional Spatial Depth
#'
#' Computes the functional spatial depth based on the average of unit
#' vectors pointing from the curve to other curves.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.FSD(fd)
depth.FSD <- function(fdataobj, fdataori = NULL, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_fsd_2d", fdataobj$data, fdataori$data, m1, m2))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_fsd_1d", fdataobj$data, fdataori$data)
}

#' Kernel Functional Spatial Depth (KFSD)
#'
#' Computes the kernel functional spatial depth, which is a smoothed version
#' of the functional spatial depth (FSD) using Gaussian kernel weighting.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param trim Trimming proportion (not used, for compatibility).
#' @param h Bandwidth parameter for Gaussian kernel. If NULL, computed
#'   automatically using Silverman's rule.
#' @param scale Logical. Not used (for compatibility).
#' @param draw Logical. Not used (for compatibility).
#' @param ... Additional arguments (ignored).
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' depths <- depth.KFSD(fd)
depth.KFSD <- function(fdataobj, fdataori = NULL, trim = 0.25, h = NULL,
                       scale = FALSE, draw = FALSE, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  # Check for 2D functional data
  is_2d_obj <- isTRUE(fdataobj$fdata2d)
  is_2d_ori <- isTRUE(fdataori$fdata2d)

  if (is_2d_obj != is_2d_ori) {
    stop("Cannot compute depth between 1D and 2D functional data")
  }

  # Auto-compute bandwidth if not provided
  if (is.null(h)) {
    # Use Silverman's rule of thumb adapted for functional data
    n <- nrow(fdataori$data)
    h <- 1.06 * sd(fdataori$data) * n^(-1/5)
    if (h < 1e-10) h <- 0.15  # fallback
  }

  if (is_2d_obj) {
    # 2D functional data (surfaces)
    if (!identical(fdataobj$dims, fdataori$dims)) {
      stop("fdataobj and fdataori must have the same grid dimensions")
    }

    m1 <- as.integer(fdataobj$dims[1])
    m2 <- as.integer(fdataobj$dims[2])

    return(.Call("wrap__depth_kfsd_2d", fdataobj$data, fdataori$data, m1, m2, as.numeric(h)))
  }

  # 1D functional data (curves)
  # Validate dimensions
  if (ncol(fdataobj$data) != ncol(fdataori$data)) {
    stop("fdataobj and fdataori must have the same number of evaluation points")
  }

  .Call("wrap__depth_kfsd_1d", fdataobj$data, fdataori$data, as.numeric(fdataori$argvals), as.numeric(h))
}

#' Random Projection Depth with Derivatives (RPD)
#'
#' Computes depth using random projections that include derivative information.
#' This combines the original curves with their derivatives for a more robust
#' depth measure that is sensitive to shape changes.
#'
#' @param fdataobj An object of class 'fdata' to compute depth for.
#' @param fdataori An object of class 'fdata' as reference sample.
#'   If NULL, uses fdataobj as reference.
#' @param nproj Number of random projections. Default is 20.
#' @param deriv Vector of derivative orders to include. Default is c(0, 1)
#'   meaning original curves and first derivatives.
#' @param trim Trimming proportion (not used, for compatibility).
#' @param draw Logical. Not used (for compatibility).
#' @param ... Additional arguments passed to fdata.deriv.
#'
#' @return A numeric vector of depth values, one per curve in fdataobj.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' depths <- depth.RPD(fd)
depth.RPD <- function(fdataobj, fdataori = NULL, nproj = 20, deriv = c(0, 1),
                      trim = 0.25, draw = FALSE, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (is.null(fdataori)) {
    fdataori <- fdataobj
  }

  if (!inherits(fdataori, "fdata")) {
    stop("fdataori must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d) || isTRUE(fdataori$fdata2d)) {
    stop("depth.RPD not yet implemented for 2D functional data")
  }

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)
  n_ori <- nrow(fdataori$data)
  argvals <- fdataobj$argvals

  # Ensure deriv orders are valid
  max_deriv <- max(deriv)
  if (max_deriv >= m) {
    stop("Derivative order must be less than number of evaluation points")
  }

  # Compute derivatives for each order
  derivs_obj <- list()
  derivs_ori <- list()

  for (d in deriv) {
    if (d == 0) {
      derivs_obj[[as.character(d)]] <- fdataobj$data
      derivs_ori[[as.character(d)]] <- fdataori$data
    } else {
      fd_deriv_obj <- fdata.deriv(fdataobj, nderiv = d, ...)
      fd_deriv_ori <- fdata.deriv(fdataori, nderiv = d, ...)
      derivs_obj[[as.character(d)]] <- fd_deriv_obj$data
      derivs_ori[[as.character(d)]] <- fd_deriv_ori$data
    }
  }

  # For each derivative order, the number of points may differ
  # Use the minimum common dimension
  m_common <- min(sapply(derivs_obj, ncol))

  # Generate random projection vectors (smooth Gaussian processes)
  set.seed(NULL)  # Use random seed for different projections each call
  projections <- matrix(rnorm(nproj * m_common), nproj, m_common)

  # Normalize projections
  for (i in seq_len(nproj)) {
    projections[i, ] <- projections[i, ] / sqrt(sum(projections[i, ]^2))
  }

  # Integration weights for inner product
  argvals_common <- argvals[seq_len(m_common)]
  h <- diff(argvals_common)
  weights <- c(h[1]/2, (h[-length(h)] + h[-1])/2, h[length(h)]/2)

  # Compute inner product function
  inner_prod <- function(curve, proj, w) {
    sum(curve * proj * w)
  }

  # Accumulate depth from all projections
  depths <- rep(0, n)

  for (j in seq_len(nproj)) {
    proj <- projections[j, ]

    # For each derivative order, project curves
    n_derivs <- length(deriv)
    proj_scores_obj <- matrix(0, n, n_derivs)
    proj_scores_ori <- matrix(0, n_ori, n_derivs)

    for (k in seq_along(deriv)) {
      d_char <- as.character(deriv[k])
      data_obj <- derivs_obj[[d_char]][, seq_len(m_common), drop = FALSE]
      data_ori <- derivs_ori[[d_char]][, seq_len(m_common), drop = FALSE]

      for (i in seq_len(n)) {
        proj_scores_obj[i, k] <- inner_prod(data_obj[i, ], proj, weights)
      }
      for (i in seq_len(n_ori)) {
        proj_scores_ori[i, k] <- inner_prod(data_ori[i, ], proj, weights)
      }
    }

    # Compute multivariate halfspace depth for the projected scores
    # Using Tukey (location) depth: minimum fraction in any halfspace
    for (i in seq_len(n)) {
      x <- proj_scores_obj[i, ]

      # Simple univariate depth if n_derivs == 1
      if (n_derivs == 1) {
        vals <- proj_scores_ori[, 1]
        prop_below <- mean(vals <= x[1])
        prop_above <- mean(vals >= x[1])
        d_i <- min(prop_below, prop_above)
      } else {
        # Multivariate halfspace depth approximation
        # Use random directions in the projected space
        d_i <- 1
        n_dirs <- 50
        for (dir_idx in seq_len(n_dirs)) {
          direction <- rnorm(n_derivs)
          direction <- direction / sqrt(sum(direction^2))

          proj_x <- sum(x * direction)
          proj_ref <- proj_scores_ori %*% direction

          prop_below <- mean(proj_ref <= proj_x)
          prop_above <- mean(proj_ref >= proj_x)
          d_i <- min(d_i, min(prop_below, prop_above))
        }
      }

      depths[i] <- depths[i] + d_i
    }
  }

  # Average over projections
  depths / nproj
}

#' Compute functional median based on depth
#'
#' Returns the curve with maximum depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param depth.func Depth function to use. Default is depth.FM.
#' @param ... Additional arguments passed to depth function.
#'
#' @return The curve (as fdata object) with maximum depth.
#'
#' @export median.FM
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- median.FM(fd)
median.FM <- function(fdataobj, depth.func = depth.FM, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.func(fdataobj, fdataobj, ...)
  max_idx <- which.max(depths)

  fdataobj[max_idx, ]
}

#' Compute functional trimmed mean
#'
#' Computes the trimmed mean by excluding curves with lowest depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param depth.func Depth function to use. Default is depth.FM.
#' @param ... Additional arguments passed to depth function.
#'
#' @return A numeric vector containing the trimmed mean function.
#'
#' @export trimmed.FM
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- trimmed.FM(fd, trim = 0.2)
trimmed.FM <- function(fdataobj, trim = 0.1, depth.func = depth.FM, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.func(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  # Compute mean of kept curves and return as fdata object
  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Median using Modal Depth
#'
#' Returns the curve with maximum modal depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ... Additional arguments passed to depth.mode.
#'
#' @return The curve (as fdata object) with maximum modal depth.
#'
#' @export median.mode
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- median.mode(fd)
median.mode <- function(fdataobj, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.mode(fdataobj, fdataobj, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "Modal Median"
  result
}

#' Functional Median using Random Projection Depth
#'
#' Returns the curve with maximum RP depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RP.
#'
#' @return The curve (as fdata object) with maximum RP depth.
#'
#' @export median.RP
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- median.RP(fd)
median.RP <- function(fdataobj, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.RP(fdataobj, fdataobj, nproj = nproj, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "RP Median"
  result
}

#' Functional Median using Random Projection Depth with Derivatives
#'
#' Returns the curve with maximum RPD depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nproj Number of random projections. Default is 20.
#' @param deriv Vector of derivative orders to include. Default is c(0, 1).
#' @param ... Additional arguments passed to depth.RPD.
#'
#' @return The curve (as fdata object) with maximum RPD depth.
#'
#' @export median.RPD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' med <- median.RPD(fd)
median.RPD <- function(fdataobj, nproj = 20, deriv = c(0, 1), ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.RPD(fdataobj, fdataobj, nproj = nproj, deriv = deriv, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "RPD Median"
  result
}

#' Functional Median using Random Tukey Depth
#'
#' Returns the curve with maximum RT (Random Tukey) depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RT.
#'
#' @return The curve (as fdata object) with maximum RT depth.
#'
#' @export median.RT
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- median.RT(fd)
median.RT <- function(fdataobj, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.RT(fdataobj, fdataobj, nproj = nproj, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "RT Median"
  result
}

#' Functional Trimmed Mean using Modal Depth
#'
#' Computes the trimmed mean by excluding curves with lowest modal depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param ... Additional arguments passed to depth.mode.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.mode
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- trimmed.mode(fd, trim = 0.2)
trimmed.mode <- function(fdataobj, trim = 0.1, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.mode(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (mode)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Trimmed Mean using Random Projection Depth
#'
#' Computes the trimmed mean by excluding curves with lowest RP depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RP.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.RP
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- trimmed.RP(fd, trim = 0.2)
trimmed.RP <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RP(fdataobj, fdataobj, nproj = nproj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (RP)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Trimmed Mean using Random Projection Depth with Derivatives
#'
#' Computes the trimmed mean by excluding curves with lowest RPD depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 20.
#' @param deriv Vector of derivative orders to include. Default is c(0, 1).
#' @param ... Additional arguments passed to depth.RPD.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.RPD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tm <- trimmed.RPD(fd, trim = 0.2)
trimmed.RPD <- function(fdataobj, trim = 0.1, nproj = 20, deriv = c(0, 1), ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RPD(fdataobj, fdataobj, nproj = nproj, deriv = deriv, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (RPD)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Trimmed Mean using Random Tukey Depth
#'
#' Computes the trimmed mean by excluding curves with lowest RT depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RT.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.RT
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- trimmed.RT(fd, trim = 0.2)
trimmed.RT <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RT(fdataobj, fdataobj, nproj = nproj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (RT)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Variance
#'
#' Computes the pointwise variance function of functional data.
#' This is an S3 method for the generic \code{var} function.
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments (currently ignored).
#'
#' @return An fdata object containing the variance function (1D or 2D).
#'
#' @export var.fdata
#' @examples
#' # 1D functional data
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' v <- var(fd)
#'
#' # 2D functional data
#' X <- array(rnorm(500), dim = c(5, 10, 10))
#' fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
#' v2d <- var(fd2d)
var.fdata <- function(x, ...) {
  if (!inherits(x, "fdata")) {
    stop("x must be of class 'fdata'")
  }

  var_vals <- apply(x$data, 2, var)

  if (isTRUE(x$fdata2d)) {
    # Return as fdata2d
    result <- list(
      data = matrix(var_vals, nrow = 1),
      argvals = x$argvals,
      rangeval = x$rangeval,
      names = list(main = "Variance", xlab = x$names$xlab,
                   ylab = x$names$ylab, zlab = "Var"),
      fdata2d = TRUE,
      dims = x$dims
    )
    class(result) <- "fdata"
    return(result)
  }

  # 1D case
  fdata(matrix(var_vals, nrow = 1), argvals = x$argvals,
        names = list(main = "Variance", xlab = x$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Standard Deviation
#'
#' Computes the pointwise standard deviation function of functional data.
#' This is an S3 method for the generic \code{sd} function.
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments (currently ignored).
#'
#' @return An fdata object containing the standard deviation function (1D or 2D).
#'
#' @export sd.fdata
#' @examples
#' # 1D functional data
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' s <- sd(fd)
#'
#' # 2D functional data
#' X <- array(rnorm(500), dim = c(5, 10, 10))
#' fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
#' s2d <- sd(fd2d)
sd.fdata <- function(x, ...) {
  if (!inherits(x, "fdata")) {
    stop("x must be of class 'fdata'")
  }

  sd_vals <- apply(x$data, 2, sd)

  if (isTRUE(x$fdata2d)) {
    # Return as fdata2d
    result <- list(
      data = matrix(sd_vals, nrow = 1),
      argvals = x$argvals,
      rangeval = x$rangeval,
      names = list(main = "Standard Deviation", xlab = x$names$xlab,
                   ylab = x$names$ylab, zlab = "SD"),
      fdata2d = TRUE,
      dims = x$dims
    )
    class(result) <- "fdata"
    return(result)
  }

  # 1D case
  fdata(matrix(sd_vals, nrow = 1), argvals = x$argvals,
        names = list(main = "Standard Deviation", xlab = x$names$xlab,
                     ylab = "SD(X(t))"))
}

#' Geometric Median of Functional Data
#'
#' Computes the geometric median (L1 median) of functional data using
#' Weiszfeld's iterative algorithm. The geometric median minimizes the
#' sum of L2 distances to all curves/surfaces, making it robust to outliers.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param max.iter Maximum number of iterations (default 100).
#' @param tol Convergence tolerance (default 1e-6).
#'
#' @return An fdata object containing the geometric median function (1D or 2D).
#'
#' @details
#' The geometric median y minimizes:
#' \deqn{\sum_{i=1}^n ||X_i - y||_{L2}}
#'
#' Unlike the mean (L2 center), the geometric median is robust to outliers
#' because extreme values have bounded influence (influence function is bounded).
#'
#' The Weiszfeld algorithm is an iteratively reweighted least squares method
#' that converges to the geometric median.
#'
#' @seealso \code{\link{mean.fdata}} for the (non-robust) mean function,
#'   \code{\link{median.FM}} for depth-based median
#'
#' @export
#' @examples
#' # Create functional data with an outlier
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:19) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#' X[20, ] <- sin(2*pi*t) + 5  # Large outlier
#' fd <- fdata(X, argvals = t)
#'
#' # Compare mean vs geometric median
#' mean_curve <- mean(fd)
#' gmed_curve <- gmed(fd)
#'
#' # The geometric median is less affected by the outlier
gmed <- function(fdataobj, max.iter = 100, tol = 1e-6) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    # 2D case
    gmed_vals <- .Call("wrap__geometric_median_2d", fdataobj$data,
                       as.numeric(fdataobj$argvals[[1]]),
                       as.numeric(fdataobj$argvals[[2]]),
                       as.integer(max.iter), as.numeric(tol))

    result <- list(
      data = matrix(gmed_vals, nrow = 1),
      argvals = fdataobj$argvals,
      rangeval = fdataobj$rangeval,
      names = list(main = "Geometric Median", xlab = fdataobj$names$xlab,
                   ylab = fdataobj$names$ylab, zlab = fdataobj$names$zlab),
      fdata2d = TRUE,
      dims = fdataobj$dims
    )
    class(result) <- "fdata"
    return(result)
  }

  # 1D case
  gmed_vals <- .Call("wrap__geometric_median_1d", fdataobj$data,
                     as.numeric(fdataobj$argvals),
                     as.integer(max.iter), as.numeric(tol))

  fdata(matrix(gmed_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Geometric Median", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Covariance Function
#'
#' Computes the covariance function (surface) for functional data.
#' For 1D: \code{Cov(s, t) = E[(X(s) - mu(s))(X(t) - mu(t))]}
#' For 2D: Covariance across the flattened domain.
#' This is an S3 method for the generic \code{cov} function.
#'
#' @param x An object of class 'fdata'.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A list with components:
#' \describe{
#'   \item{cov}{The covariance matrix (m x m for 1D, (m1*m2) x (m1*m2) for 2D)}
#'   \item{argvals}{The evaluation points (same as input)}
#'   \item{mean}{The mean function}
#' }
#'
#' @export cov.fdata
#' @examples
#' # 1D functional data
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(0, 20, 50)
#' for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
#' fd <- fdata(X, argvals = t)
#' cov_result <- cov(fd)
#' image(cov_result$cov, main = "Covariance Surface")
cov.fdata <- function(x, ...) {
  if (!inherits(x, "fdata")) {
    stop("x must be of class 'fdata'")
  }

  # Compute mean and centered data
  n <- nrow(x$data)
  m <- ncol(x$data)

  mean_func <- colMeans(x$data)
  centered <- sweep(x$data, 2, mean_func)

  # Compute covariance matrix: (1/(n-1)) * t(X_centered) %*% X_centered
  cov_mat <- crossprod(centered) / (n - 1)

  list(
    cov = cov_mat,
    argvals = x$argvals,
    mean = mean_func,
    fdata2d = isTRUE(x$fdata2d),
    dims = x$dims
  )
}

#' Functional Trimmed Variance using FM Depth
#'
#' Computes the trimmed variance by excluding curves with lowest FM depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param ... Additional arguments passed to depth.FM.
#'
#' @return An fdata object containing the trimmed variance function.
#'
#' @export trimvar.FM
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- trimvar.FM(fd, trim = 0.2)
trimvar.FM <- function(fdataobj, trim = 0.1, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.FM(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  var_vals <- apply(fdataobj$data[keep_idx, , drop = FALSE], 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Variance (FM)", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Trimmed Variance using Modal Depth
#'
#' Computes the trimmed variance by excluding curves with lowest modal depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param ... Additional arguments passed to depth.mode.
#'
#' @return An fdata object containing the trimmed variance function.
#'
#' @export trimvar.mode
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- trimvar.mode(fd, trim = 0.2)
trimvar.mode <- function(fdataobj, trim = 0.1, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.mode(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  var_vals <- apply(fdataobj$data[keep_idx, , drop = FALSE], 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Variance (mode)", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Trimmed Variance using Random Projection Depth
#'
#' Computes the trimmed variance by excluding curves with lowest RP depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RP.
#'
#' @return An fdata object containing the trimmed variance function.
#'
#' @export trimvar.RP
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- trimvar.RP(fd, trim = 0.2)
trimvar.RP <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RP(fdataobj, fdataobj, nproj = nproj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  var_vals <- apply(fdataobj$data[keep_idx, , drop = FALSE], 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Variance (RP)", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Trimmed Variance using Random Projection Depth with Derivatives
#'
#' Computes the trimmed variance by excluding curves with lowest RPD depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 20.
#' @param deriv Vector of derivative orders to include. Default is c(0, 1).
#' @param ... Additional arguments passed to depth.RPD.
#'
#' @return An fdata object containing the trimmed variance function.
#'
#' @export trimvar.RPD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tv <- trimvar.RPD(fd, trim = 0.2)
trimvar.RPD <- function(fdataobj, trim = 0.1, nproj = 20, deriv = c(0, 1), ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RPD(fdataobj, fdataobj, nproj = nproj, deriv = deriv, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  var_vals <- apply(fdataobj$data[keep_idx, , drop = FALSE], 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Variance (RPD)", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Trimmed Variance using Random Tukey Depth
#'
#' Computes the trimmed variance by excluding curves with lowest RT depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param nproj Number of random projections. Default is 50.
#' @param ... Additional arguments passed to depth.RT.
#'
#' @return An fdata object containing the trimmed variance function.
#'
#' @export trimvar.RT
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- trimvar.RT(fd, trim = 0.2)
trimvar.RT <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.RT(fdataobj, fdataobj, nproj = nproj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  var_vals <- apply(fdataobj$data[keep_idx, , drop = FALSE], 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Variance (RT)", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
}

#' Functional Median using Band Depth
#'
#' Returns the curve with maximum band depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ... Additional arguments passed to depth.BD.
#'
#' @return The curve (as fdata object) with maximum band depth.
#'
#' @export median.BD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' med <- median.BD(fd)
median.BD <- function(fdataobj, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.BD(fdataobj, fdataobj, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "BD Median"
  result
}

#' Functional Median using Modified Band Depth
#'
#' Returns the curve with maximum modified band depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ... Additional arguments passed to depth.MBD.
#'
#' @return The curve (as fdata object) with maximum modified band depth.
#'
#' @export median.MBD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' med <- median.MBD(fd)
median.MBD <- function(fdataobj, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  depths <- depth.MBD(fdataobj, fdataobj, ...)
  max_idx <- which.max(depths)

  result <- fdataobj[max_idx, ]
  result$names$main <- "MBD Median"
  result
}

#' Functional Trimmed Mean using Band Depth
#'
#' Computes the trimmed mean by excluding curves with lowest band depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param ... Additional arguments passed to depth.BD.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.BD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tm <- trimmed.BD(fd, trim = 0.2)
trimmed.BD <- function(fdataobj, trim = 0.1, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.BD(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (BD)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}

#' Functional Trimmed Mean using Modified Band Depth
#'
#' Computes the trimmed mean by excluding curves with lowest modified band depth.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param trim Proportion of curves to trim (default 0.1).
#' @param ... Additional arguments passed to depth.MBD.
#'
#' @return An fdata object containing the trimmed mean function.
#'
#' @export trimmed.MBD
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tm <- trimmed.MBD(fd, trim = 0.2)
trimmed.MBD <- function(fdataobj, trim = 0.1, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)
  n_keep <- ceiling(n * (1 - trim))

  depths <- depth.MBD(fdataobj, fdataobj, ...)
  depth_order <- order(depths, decreasing = TRUE)
  keep_idx <- depth_order[seq_len(n_keep)]

  mean_vals <- colMeans(fdataobj$data[keep_idx, , drop = FALSE])
  fdata(matrix(mean_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Trimmed Mean (MBD)", xlab = fdataobj$names$xlab,
                     ylab = fdataobj$names$ylab))
}
