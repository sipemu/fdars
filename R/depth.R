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
#' @param scale Logical. If TRUE (default), scales depth to [0, 1] range
#'   using the FM1 formula from fda.usc. If FALSE, returns unscaled values
#'   in [0, 0.5] range.
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- func.med.FM(fd)
func.med.FM <- function(fdataobj, depth.func = depth.FM, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- func.trim.FM(fd, trim = 0.2)
func.trim.FM <- function(fdataobj, trim = 0.1, depth.func = depth.FM, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- func.med.mode(fd)
func.med.mode <- function(fdataobj, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- func.med.RP(fd)
func.med.RP <- function(fdataobj, nproj = 50, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' med <- func.med.RPD(fd)
func.med.RPD <- function(fdataobj, nproj = 20, deriv = c(0, 1), ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' med <- func.med.RT(fd)
func.med.RT <- function(fdataobj, nproj = 50, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- func.trim.mode(fd, trim = 0.2)
func.trim.mode <- function(fdataobj, trim = 0.1, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- func.trim.RP(fd, trim = 0.2)
func.trim.RP <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tm <- func.trim.RPD(fd, trim = 0.2)
func.trim.RPD <- function(fdataobj, trim = 0.1, nproj = 20, deriv = c(0, 1), ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tm <- func.trim.RT(fd, trim = 0.2)
func.trim.RT <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
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
#'
#' @param fdataobj An object of class 'fdata'.
#'
#' @return An fdata object containing the variance function.
#'
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' v <- func.var(fd)
func.var <- function(fdataobj) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  var_vals <- apply(fdataobj$data, 2, var)

  fdata(matrix(var_vals, nrow = 1), argvals = fdataobj$argvals,
        names = list(main = "Variance", xlab = fdataobj$names$xlab,
                     ylab = "Var(X(t))"))
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- func.trimvar.FM(fd, trim = 0.2)
func.trimvar.FM <- function(fdataobj, trim = 0.1, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- func.trimvar.mode(fd, trim = 0.2)
func.trimvar.mode <- function(fdataobj, trim = 0.1, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- func.trimvar.RP(fd, trim = 0.2)
func.trimvar.RP <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(200), 20, 10))
#' tv <- func.trimvar.RPD(fd, trim = 0.2)
func.trimvar.RPD <- function(fdataobj, trim = 0.1, nproj = 20, deriv = c(0, 1), ...) {
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
#' @export
#' @examples
#' fd <- fdata(matrix(rnorm(100), 10, 10))
#' tv <- func.trimvar.RT(fd, trim = 0.2)
func.trimvar.RT <- function(fdataobj, trim = 0.1, nproj = 50, ...) {
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
