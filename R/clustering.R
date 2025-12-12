#' Clustering Functions for Functional Data
#'
#' Functions for clustering functional data, including k-means and related
#' algorithms.

#' Functional K-Means Clustering
#'
#' Performs k-means clustering on functional data using the specified metric.
#' Uses k-means++ initialization for better initial centers.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ncl Number of clusters.
#' @param metric Either a string ("L2", "L1", "Linf") for fast Rust-based
#'   distance computation, or a metric/semimetric function (e.g., \code{metric.lp},
#'   \code{metric.hausdorff}, \code{semimetric.pca}). Using a function provides
#'   flexibility but may be slower for semimetrics computed in R.
#' @param max.iter Maximum number of iterations (default 100).
#' @param nstart Number of random starts (default 10). The best result
#'   (lowest within-cluster sum of squares) is returned.
#' @param seed Optional random seed for reproducibility.
#' @param draw Logical. If TRUE, plot the clustered curves (not yet implemented).
#' @param ... Additional arguments passed to the metric function.
#'
#' @return A list of class 'kmeans.fd' with components:
#' \describe{
#'   \item{cluster}{Integer vector of cluster assignments (1 to ncl).}
#'   \item{centers}{An fdata object containing the cluster centers.}
#'   \item{withinss}{Within-cluster sum of squares for each cluster.}
#'   \item{tot.withinss}{Total within-cluster sum of squares.}
#'   \item{size}{Number of observations in each cluster.}
#'   \item{fdataobj}{The input functional data object.}
#' }
#'
#' @details
#' When \code{metric} is a string ("L2", "L1", "Linf"), the entire k-means
#' algorithm runs in Rust with parallel processing, providing 50-200x speedup.
#'
#' When \code{metric} is a function, distances are computed using that function.
#' Functions like \code{metric.lp}, \code{metric.hausdorff}, and \code{metric.DTW}
#' have Rust backends and remain fast. Semimetric functions (\code{semimetric.*})
#' are computed in R and will be slower for large datasets.
#'
#' @export
#' @examples
#' # Create functional data with two groups
#' t <- seq(0, 1, length.out = 50)
#' n <- 30
#' X <- matrix(0, n, 50)
#' true_cluster <- rep(1:2, each = 15)
#' for (i in 1:n) {
#'   if (true_cluster[i] == 1) {
#'     X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
#'   } else {
#'     X[i, ] <- cos(2*pi*t) + rnorm(50, sd = 0.1)
#'   }
#' }
#' fd <- fdata(X, argvals = t)
#'
#' # Cluster with string metric (fast Rust path)
#' result <- kmeans.fd(fd, ncl = 2, metric = "L2")
#' table(result$cluster, true_cluster)
#'
#' # Cluster with metric function (also fast - Rust backend)
#' result2 <- kmeans.fd(fd, ncl = 2, metric = metric.lp)
#'
#' # Cluster with semimetric (flexible but slower)
#' result3 <- kmeans.fd(fd, ncl = 2, metric = semimetric.pca, ncomp = 3)
kmeans.fd <- function(fdataobj, ncl, metric = "L2",
                      max.iter = 100, nstart = 10, seed = NULL,
                      draw = FALSE, ...) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  if (isTRUE(fdataobj$fdata2d)) {
    stop("kmeans.fd for 2D functional data not yet implemented")
  }

  n <- nrow(fdataobj$data)

  if (ncl < 1 || ncl > n) {
    stop("ncl must be between 1 and the number of curves")
  }

  # Check if metric is a string or function
  if (is.character(metric)) {
    metric <- match.arg(metric, c("L2", "L1", "Linf"))
    result <- .kmeans_fd_rust(fdataobj, ncl, metric, max.iter, nstart, seed)
  } else if (is.function(metric)) {
    result <- .kmeans_fd_metric(fdataobj, ncl, metric, max.iter, nstart, seed, ...)
  } else {
    stop("metric must be a string ('L2', 'L1', 'Linf') or a metric/semimetric function")
  }

  # Create fdata object for centers
  centers <- fdata(result$centers, argvals = fdataobj$argvals,
                   names = list(main = "Cluster Centers",
                                xlab = fdataobj$names$xlab,
                                ylab = fdataobj$names$ylab))

  structure(
    list(
      cluster = result$cluster,
      centers = centers,
      withinss = result$withinss,
      tot.withinss = result$tot_withinss,
      size = result$size,
      fdataobj = fdataobj
    ),
    class = "kmeans.fd"
  )
}

#' Internal: K-means using Rust backend (string metrics)
#' @noRd
.kmeans_fd_rust <- function(fdataobj, ncl, metric, max.iter, nstart, seed) {
  seed_val <- if (!is.null(seed)) as.integer(seed) else NULL

  .Call("wrap__kmeans_fd", fdataobj$data,
        as.numeric(fdataobj$argvals),
        as.integer(ncl), as.integer(max.iter),
        as.integer(nstart), metric, seed_val)
}

#' Internal: K-means using metric/semimetric function
#' @noRd
.kmeans_fd_metric <- function(fdataobj, ncl, metric_func, max.iter, nstart, seed, ...) {
  if (!is.null(seed)) set.seed(seed)

  n <- nrow(fdataobj$data)
  m <- ncol(fdataobj$data)

  best_result <- NULL
  best_tot_withinss <- Inf

  for (start in seq_len(nstart)) {
    # K-means++ initialization
    centers_idx <- .kmeans_pp_init(fdataobj, ncl, metric_func, ...)

    centers <- fdataobj$data[centers_idx, , drop = FALSE]
    cluster <- integer(n)

    for (iter in seq_len(max.iter)) {
      old_cluster <- cluster

      # Assignment step: assign each curve to nearest center
      centers_fd <- fdata(centers, argvals = fdataobj$argvals)
      dist_to_centers <- metric_func(fdataobj, centers_fd, ...)

      cluster <- apply(dist_to_centers, 1, which.min)

      # Check convergence
      if (identical(cluster, old_cluster)) break

      # Update step: recompute centers
      for (k in seq_len(ncl)) {
        members <- which(cluster == k)
        if (length(members) > 0) {
          centers[k, ] <- colMeans(fdataobj$data[members, , drop = FALSE])
        }
      }
    }

    # Compute within-cluster sum of squares
    withinss <- numeric(ncl)
    for (k in seq_len(ncl)) {
      members <- which(cluster == k)
      if (length(members) > 0) {
        center_fd <- fdata(matrix(centers[k, ], nrow = 1), argvals = fdataobj$argvals)
        members_fd <- fdataobj[members, ]
        dists <- metric_func(members_fd, center_fd, ...)
        withinss[k] <- sum(dists^2)
      }
    }
    tot_withinss <- sum(withinss)

    # Keep best result
    if (tot_withinss < best_tot_withinss) {
      best_tot_withinss <- tot_withinss
      best_result <- list(
        cluster = cluster,
        centers = centers,
        withinss = withinss,
        tot_withinss = tot_withinss,
        size = tabulate(cluster, ncl)
      )
    }
  }

  best_result
}

#' Internal: K-means++ initialization with metric function
#' @noRd
.kmeans_pp_init <- function(fdataobj, ncl, metric_func, ...) {
  n <- nrow(fdataobj$data)
  centers_idx <- integer(ncl)

  # First center: random
  centers_idx[1] <- sample(n, 1)

  for (k in 2:ncl) {
    # Compute distances to nearest existing center
    current_centers <- fdataobj[centers_idx[1:(k-1)], ]
    dists <- metric_func(fdataobj, current_centers, ...)

    # Min distance to any center
    min_dists <- apply(dists, 1, min)

    # Probability proportional to D^2
    probs <- min_dists^2
    probs[centers_idx[1:(k-1)]] <- 0
    probs <- probs / sum(probs)

    centers_idx[k] <- sample(n, 1, prob = probs)
  }

  centers_idx
}

#' K-Means++ Center Initialization
#'
#' Initialize cluster centers using the k-means++ algorithm, which selects
#' centers with probability proportional to squared distance from existing
#' centers.
#'
#' @param fdataobj An object of class 'fdata'.
#' @param ncl Number of clusters.
#' @param metric Metric to use. One of "L2", "L1", or "Linf".
#' @param seed Optional random seed.
#'
#' @return An fdata object containing the initial cluster centers.
#'
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 50)
#' X <- matrix(rnorm(30 * 50), 30, 50)
#' fd <- fdata(X, argvals = t)
#' init_centers <- kmeans.center.ini(fd, ncl = 3)
kmeans.center.ini <- function(fdataobj, ncl, metric = "L2", seed = NULL) {
  if (!inherits(fdataobj, "fdata")) {
    stop("fdataobj must be of class 'fdata'")
  }

  n <- nrow(fdataobj$data)

  if (ncl < 1 || ncl > n) {
    stop("ncl must be between 1 and the number of curves")
  }

  # Set seed if provided
  if (!is.null(seed)) set.seed(seed)

  # Compute distance matrix
  if (metric == "L2") {
    dist_mat <- as.matrix(metric.lp(fdataobj))
  } else if (metric == "L1") {
    dist_mat <- as.matrix(metric.lp(fdataobj, p = 1))
  } else {
    dist_mat <- as.matrix(metric.lp(fdataobj, p = Inf))
  }

  # K-means++ initialization
  centers_idx <- integer(ncl)

  # First center: random
  centers_idx[1] <- sample(n, 1)

  # Remaining centers: probability proportional to D^2
  for (k in 2:ncl) {
    # Min distance to current centers
    min_dists <- apply(dist_mat[, centers_idx[1:(k-1)], drop = FALSE], 1, min)

    # Square and normalize
    probs <- min_dists^2
    probs[centers_idx[1:(k-1)]] <- 0  # Don't reselect existing centers
    probs <- probs / sum(probs)

    # Sample next center
    centers_idx[k] <- sample(n, 1, prob = probs)
  }

  # Return centers as fdata
  fdataobj[centers_idx, ]
}

#' Print Method for kmeans.fd Objects
#'
#' @param x An object of class 'kmeans.fd'.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.kmeans.fd <- function(x, ...) {
  cat("Functional K-Means Clustering\n")
  cat("=============================\n")
  cat("Number of clusters:", length(x$size), "\n")
  cat("Number of observations:", sum(x$size), "\n\n")

  cat("Cluster sizes:\n")
  print(x$size)

  cat("\nWithin-cluster sum of squares:\n")
  print(round(x$withinss, 4))

  cat("\nTotal within-cluster SS:", round(x$tot.withinss, 4), "\n")

  invisible(x)
}

#' Plot Method for kmeans.fd Objects
#'
#' @param x An object of class 'kmeans.fd'.
#' @param ... Additional arguments passed to matplot.
#'
#' @export
plot.kmeans.fd <- function(x, ...) {
  args <- list(...)
  ncl <- length(x$size)

  # Default colors for clusters
  if (is.null(args$col)) {
    args$col <- x$cluster
  }

  # Plot curves colored by cluster
  fd <- x$fdataobj
  if (is.null(args$type)) args$type <- "l"
  if (is.null(args$xlab)) args$xlab <- fd$names$xlab
  if (is.null(args$ylab)) args$ylab <- fd$names$ylab
  if (is.null(args$main)) args$main <- "Functional K-Means Clustering"

  do.call(matplot, c(list(x = fd$argvals, y = t(fd$data)), args))

  # Add centers
  lines_args <- list(lwd = 3, lty = 2)
  for (k in 1:ncl) {
    lines(fd$argvals, x$centers$data[k, ], col = k, lwd = 3, lty = 2)
  }

  invisible(x)
}
