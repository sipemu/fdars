# Golden values for curve_shape_boxplot_from_aligned (fdars #54 ALN-NDDOWN).
# fdasrvf 2.5.0 fdasrvf:::curvebox_data on analytic "aligned" SRVF fields (no DP),
# with innerprod_q2 replaced by the trapezoid rule on t_j = j/(M-1) so the arithmetic
# matches fdars (fdasrvf uses the Riemann sum sum(q1*q2)/T).
suppressMessages(library(fdasrvf))
cat("fdasrvf", as.character(packageVersion("fdasrvf")), "\n")
M <- 21; tt <- seq(0, 1, length.out = M)
w <- rep(1/(M-1), M); w[1] <- w[M] <- 0.5/(M-1)
ip <- function(q1, q2) sum(sweep(q1 * q2, 2, w, "*"))
assignInNamespace("innerprod_q2", function(q1, q2) ip(q1, q2), "fdasrvf")
unitf <- function(q) q / sqrt(ip(q, q))
a <- c(0.10, -0.15, 0.20, -0.05, 0.30, -0.25, 0.05, 0.12, -0.18, 1.6, -1.1)
b <- c(0.05, 0.10, -0.12, 0.20, -0.08, 0.02, -0.22, 0.15, 0.07, 0.9, -1.3)
N <- length(a)
qn <- array(0, c(2, M, N))
for (i in 1:N) {
  q <- rbind(1 + a[i] * sin(pi * tt), b[i] * cos(pi * tt) + 0.1 * a[i] * tt)
  qn[, , i] <- unitf(q)
}
mu <- unitf(rbind(rep(1, M), 0.05 * sin(2 * pi * tt)))
for (scale in c(TRUE, FALSE)) {
  x <- list(betan = qn, betamean = mu, qn = qn, mu = mu, scale = scale)
  out <- fdasrvf:::curvebox_data(x, alpha = 0.05, ka = 1)
  idx <- function(f) which(sapply(1:N, function(i) isTRUE(all.equal(qn[, , i], f))))
  cat("scale", scale, "\n")
  cat("dist", sprintf("%.15e", out$dist), "\n")
  cat("Q1 Q3 Q1a Q3a (1-based)", out$Q1_index, out$Q3_index, out$Q1a_index, out$Q3a_index, "\n")
  cat("outliers", out$outlier_index, "\n")
  cat("min max", idx(out$minn), idx(out$maxx), "\n")
}
