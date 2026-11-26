# Golden values for curve_shape_confidence_interval(method = ModelBased) (fdars #71):
# fdasrvf 2.6.0 shape_CI(beta, a = 0.95, no = 5, Nsamp = 100, mode = "O",
# rotated = FALSE, scale = TRUE) on the 6 open planar curves of #54's depth golden,
# beta_i(t) = (t, a_i sin(pi t) + c_i sin(2 pi t^p_i)).
# RNG-independent quantities: the Karcher median (out.med$mu, betamean) and the
# eigenvalues of the tangent covariance K = multivariate_karcher_cov(out.med).
.libPaths(c("~/.cache/R-lib", .libPaths()))
suppressMessages(library(fdasrvf))
cat("fdasrvf", as.character(packageVersion("fdasrvf")), "\n")
a <- c(0.20, 0.25, 0.30, 0.35, 0.40, 0.60)
cc <- c(0.05, -0.03, 0.02, 0.04, -0.05, 0.30)
p <- c(1.0, 1.2, 0.9, 1.1, 1.0, 1.5)
N <- length(a)
for (M in c(60, 240)) {
  tt <- seq(0, 1, length.out = M)
  beta <- array(0, c(2, M, N))
  for (i in 1:N) beta[, , i] <- rbind(tt, a[i] * sin(pi * tt) + cc[i] * sin(2 * pi * tt^p[i]))
  set.seed(1)
  out <- shape_CI(beta, a = 0.95, no = 5, Nsamp = 100, mode = "O", rotated = FALSE,
                  scale = TRUE, parallel = FALSE)
  med <- out$out.med
  K <- multivariate_karcher_cov(med)
  ev <- svd(K)$d
  cat("T", M, "dims Q1a", dim(out$Q1a), "samples", dim(out$samples$betas), "\n")
  # eigenvalues in fdasrvf's Euclidean coordinates; L2 (Riemann sum / T) eigenvalues = ev / T
  cat("T", M, "eig_L2", sprintf("%.10e", ev[1:5] / M), "\n")
  cat("T", M, "mu_norm_L2", sprintf("%.10f", sqrt(sum(med$mu^2) / M)), "\n")
  # distances of the aligned data to the median (acos <mu, qn_i>, Riemann sum)
  mu <- med$mu / sqrt(sum(med$mu^2) / M)
  d <- sapply(1:N, function(i) acos(min(1, sum(mu * med$qn[, , i]) / M)))
  cat("T", M, "dist", sprintf("%.10f", d), "\n")
  # median curve (betamean) centred, at 7 sample positions
  bm <- med$betamean
  bm <- bm - rowMeans(bm)
  idx <- round(seq(1, M, length.out = 7))
  cat("T", M, "median_y", sprintf("%.8f", bm[2, idx]), "\n")
  cat("T", M, "median_x", sprintf("%.8f", bm[1, idx]), "\n")
  if (M == 60) {
    # full median curve at T = 60 (centred) for the shape-distance comparison
    cat("T", M, "median_full_x", sprintf("%.9e", bm[1, ]), "\n")
    cat("T", M, "median_full_y", sprintf("%.9e", bm[2, ]), "\n")
  }
  # model-sample summaries (RNG-dependent; reported for orientation only)
  cat("T", M, "box_dist_mean", sprintf("%.6f", mean(out$box$dist)), "\n")
}
