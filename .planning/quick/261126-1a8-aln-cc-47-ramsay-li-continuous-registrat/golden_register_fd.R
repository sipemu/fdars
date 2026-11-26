# Golden values for fdars-core alignment::continuous_registration (ALN-CC, GH #47).
# Reference: R fda 6.3.0, register.fd(y0fd, yfd, WfdParobj, conv, iterlim, dbglev=0, crit).
suppressMessages(library(fda))
options(digits = 15)
t <- seq(0, 1, length.out = 101)
g <- function(s) exp(-(s - 0.3)^2 / 0.02) + 0.8 * exp(-(s - 0.7)^2 / 0.02)
a <- c(-0.3, -0.1, 0.15, 0.3)
Y <- sapply(a, function(ai) g(t + ai * t * (1 - t)))          # 101 x 4
ybasis <- create.bspline.basis(c(0, 1), nbasis = 43, norder = 6)
yfd <- smooth.basis(t, Y, ybasis)$fd
wbasis <- create.bspline.basis(c(0, 1), nbasis = 6, norder = 4)
cat("## basis values at t=c(0,0.1,0.37,0.5,0.9,1)\n")
print(eval.basis(c(0, 0.1, 0.37, 0.5, 0.9, 1), wbasis))
cat("## penalty Lfd=2\n"); print(getbasispenalty(wbasis, int2Lfd(2)))
nfine <- max(201, 10 * 43 + 1)
xfine <- seq(0, 1, length.out = nfine)
run <- function(crit, lambda) {
  WfdPar <- fdPar(fd(matrix(0, 6, 4), wbasis), int2Lfd(2), lambda)
  r <- register.fd(mean.fd(yfd), yfd, WfdPar, conv = 1e-6, iterlim = 50, dbglev = 0, crit = crit)
  cat(sprintf("\n## crit=%d lambda=%g\n", crit, lambda))
  cat("Wcoef (6x4 column-major):\n"); cat(format(c(r$Wfd$coefs), digits = 15), sep = ", "); cat("\n")
  idx <- c(11, 26, 51, 76, 91)  # t = 0.1, 0.25, 0.5, 0.75, 0.9
  cat("warp at t[idx] (5x4 column-major):\n"); cat(format(c(eval.fd(t[idx], r$warpfd)), digits = 15), sep = ", "); cat("\n")
  cat("registered at t[idx]:\n"); cat(format(c(eval.fd(t[idx], r$regfd)), digits = 15), sep = ", "); cat("\n")
  y0 <- eval.fd(xfine, mean.fd(yfd)); yr <- eval.fd(xfine, r$regfd)
  for (i in 1:4) {
    aa <- mean(y0^2); bb <- mean(y0 * yr[, i]); cc <- mean(yr[, i]^2)
    f <- if (crit == 1) mean((y0 - yr[, i])^2) else (aa + cc) - sqrt((aa - cc)^2 + 4 * bb^2)
    cat(sprintf("crit curve %d (data term): %.15g\n", i, f))
  }
}
run(1, 0); run(2, 0); run(2, 1e-3)
