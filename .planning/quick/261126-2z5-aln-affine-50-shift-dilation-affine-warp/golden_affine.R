# Golden values for fdars ALN-AFFINE (GH #50) from R fdacluster 0.4.2.
# Install: install.packages("fdacluster", lib = "~/.cache/R-lib")
.libPaths(c("~/.cache/R-lib", .libPaths())); suppressMessages(library(fdacluster))
cat("fdacluster", as.character(packageVersion("fdacluster")), "\n")
m <- 101; t <- seq(0, 1, length.out = m)
mu0 <- function(s) exp(-((s - 0.45) / 0.12)^2) + 0.6 * exp(-((s - 0.7) / 0.08)^2)
mu1 <- function(s) 0.8 * exp(-((s - 0.35) / 0.07)^2) - 0.5 * exp(-((s - 0.62) / 0.1)^2) + 0.3
a <- c(1.05, 0.95, 1.0, 1.08, 0.93, 1.02)
b <- c(-0.03, 0.04, 0.0, -0.06, 0.05, -0.01)
amp <- c(1.0, 1.2, 0.9, 1.1, 0.8, 1.05)
# k = 1 data: amplitude-scaled, slightly perturbed affine-warped copies of mu0
y1 <- t(sapply(1:6, function(i) amp[i] * mu0(a[i] * t + b[i]) + 0.02 * sin(2 * pi * t + i)))
# shift data: unscaled shifted copies (+ perturbation)
ys <- t(sapply(1:6, function(i) mu0(t + b[i]) + 0.02 * sin(2 * pi * t + i)))
ab_of <- function(res) t(apply(res$aligned_grids, 1, function(g) coef(lm(g ~ t))[c(2, 1)]))
rel <- function(ab) cbind(ab[, 1] / ab[1, 1], (ab[, 2] - ab[1, 2]) / ab[1, 1])  # h_1^{-1} o h_i (frame invariant)
show <- function(name, res) {
  ab <- ab_of(res); r <- rel(ab)
  cat("##", name, "n_iter", res$n_iterations, "\n")
  cat("rel_a:", sprintf("%.10f", r[, 1]), "\n"); cat("abs_a:", sprintf("%.10f", ab[, 1]), "\n"); cat("abs_b:", sprintf("%.10f", ab[, 2]), "\n")
  cat("rel_b:", sprintf("%.10f", r[, 2]), "\n")
  cat("memberships:", res$memberships, "\n")
  cat("dist:", sprintf("%.10e", res$distances_to_center), "\n")
}
show("k1_affine_pearson", fdakmeans(x = t, y = y1, n_clusters = 1, seeds = 1,
  warping_class = "affine", metric = "pearson", add_silhouettes = FALSE))
show("k1_dilation_pearson", fdakmeans(x = t, y = t(sapply(1:6, function(i)
  amp[i] * mu0(a[i] * t) + 0.02 * sin(2 * pi * t + i))), n_clusters = 1, seeds = 1,
  warping_class = "dilation", metric = "pearson", add_silhouettes = FALSE))
show("k1_shift_l2", fdakmeans(x = t, y = ys, n_clusters = 1, seeds = 1,
  warping_class = "shift", metric = "l2", add_silhouettes = FALSE))
# k = 2: 4 shifted copies of mu0 and 4 of mu1; seeds = curves 1 and 5
bb <- c(-0.04, 0.02, 0.05, -0.01)
y2 <- rbind(t(sapply(1:4, function(i) mu0(t + bb[i]) + 0.02 * sin(2 * pi * t + i))),
            t(sapply(1:4, function(i) mu1(t + bb[i]) + 0.02 * cos(2 * pi * t + i))))
show("k2_shift_l2", fdakmeans(x = t, y = y2, n_clusters = 2, seeds = c(1, 5),
  warping_class = "shift", metric = "l2", add_silhouettes = FALSE))
show("k2_affine_pearson", fdakmeans(x = t, y = y2, n_clusters = 2, seeds = c(1, 5),
  warping_class = "affine", metric = "pearson", add_silhouettes = FALSE))
