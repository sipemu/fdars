# Distributional (RNG-dependent) moments of fdasrvf 2.6.0 shape_CI model samples on the
# golden curves at T = 60 (rotated = FALSE, scale = TRUE, no = 5, Nsamp = 2000, set.seed(1)):
# mean/sd of the geodesic distance of the unposed samples (qns) and of the posed samples
# (qs, the box-plot distances) to the median. Riemann inner product sum/T as fdasrvf.
.libPaths(c("~/.cache/R-lib", .libPaths()))
suppressMessages(library(fdasrvf))
cat("fdasrvf", as.character(packageVersion("fdasrvf")), "\n")
a <- c(0.20, 0.25, 0.30, 0.35, 0.40, 0.60)
cc <- c(0.05, -0.03, 0.02, 0.04, -0.05, 0.30)
p <- c(1.0, 1.2, 0.9, 1.1, 1.0, 1.5)
N <- length(a); M <- 60
tt <- seq(0, 1, length.out = M)
beta <- array(0, c(2, M, N))
for (i in 1:N) beta[, , i] <- rbind(tt, a[i] * sin(pi * tt) + cc[i] * sin(2 * pi * tt^p[i]))
set.seed(1)
out <- shape_CI(beta, a = 0.95, no = 5, Nsamp = 2000, mode = "O", rotated = FALSE,
                scale = TRUE, parallel = FALSE)
mu <- out$out.med$mu; mu <- mu / sqrt(sum(mu^2) / M)
qns <- out$samples$qns
du <- sapply(1:dim(qns)[3], function(i) acos(max(-1, min(1, sum(mu * qns[, , i]) / M))))
db <- out$box$dist
cat("unposed mean", sprintf("%.6f", mean(du)), "sd", sprintf("%.6f", sd(du)), "\n")
cat("posed mean", sprintf("%.6f", mean(db)), "sd", sprintf("%.6f", sd(db)), "\n")
cat("sum eig_L2", sprintf("%.6f", sum(svd(multivariate_karcher_cov(out$out.med))$d[1:5]) / M), "\n")
