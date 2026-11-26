.libPaths(c("~/.cache/R-lib", .libPaths()))
suppressMessages(library(fdatest))
J <- 30; n1 <- 8; n2 <- 8
tt <- seq(0, 1, length.out = J)
set.seed(20261126)
d1 <- t(sapply(1:n1, function(i) sin(2*pi*tt) + 0.5*exp(-((tt-0.7)/0.1)^2) + rnorm(J, sd=0.6)))
d2 <- t(sapply(1:n2, function(i) sin(2*pi*tt) + rnorm(J, sd=0.6)))
B <- 40
set.seed(7)
invisible(capture.output(res <- ITP2bspline(d1, d2, mu = 0, order = 4, nknots = 5, B = B, paired = FALSE)))
# replicate permutation loop to export T_coeff
coeff <- res$coeff; n <- n1 + n2; p <- ncol(coeff)
set.seed(7)
T <- matrix(ncol = p, nrow = B)
for (perm in 1:B) { pm <- sample(n); cp <- coeff[pm, ]; T[perm, ] <- abs(colMeans(cp[1:n1, ]) - colMeans(cp[(n1+1):n, ])) }
pv <- sapply(1:p, function(i) sum(T[, i] >= abs(colMeans(coeff[1:n1,]) - colMeans(coeff[(n1+1):n,]))[i]) / B)
stopifnot(isTRUE(all.equal(pv, res$pval)))
cat("packageVersion fdatest", as.character(packageVersion("fdatest")), "\n")
f <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
cat("P =", p, "\n")
cat("T_COEFF (row-major B x p):\n", f(t(T)), "\n")
cat("PVAL:\n", f(res$pval), "\n")
cat("PVAL_MATRIX (row-major):\n", f(t(res$pval.matrix)), "\n")
cat("CORRECTED:\n", f(res$corrected.pval), "\n")
