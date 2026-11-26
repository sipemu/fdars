# Golden values for fdars-core alignment::continuous_registration periodic option
# and R-default W basis preset (ALN-CAV, GH #72).
# Reference: R fda 6.3.0 register.fd(..., periodic = TRUE) and WfdParobj = NULL.
suppressMessages(library(fda))
options(digits = 15)
t <- seq(0, 1, length.out = 101)
g <- function(s) sin(2 * pi * s) + 0.5 * cos(4 * pi * s) + 0.3 * sin(6 * pi * s)
d <- c(-0.06, -0.02, 0.03, 0.07)
b <- c(0.25, -0.2, 0.1, -0.15)
# periodic curve: shift + mild periodic-compatible time distortion
Y <- sapply(1:4, function(i) g(t + d[i] + b[i] * sin(2 * pi * t) / (2 * pi) * 0.3))
ybasis <- create.fourier.basis(c(0, 1), nbasis = 21)
yfd <- smooth.basis(t, Y, ybasis)$fd
nfine <- max(201, 10 * 21 + 1)
xfine <- seq(0, 1, length.out = nfine)
idx <- c(11, 26, 51, 76, 91)
report <- function(r, crit, label) {
  cat(sprintf("\n## %s\n", label))
  cat("Wcoef (column-major):\n"); cat(format(c(r$Wfd$coefs), digits = 15), sep = ", "); cat("\n")
  cat("shift:\n"); cat(format(r$shift, digits = 15), sep = ", "); cat("\n")
  cat("warp at t[idx]:\n"); cat(format(c(eval.fd(t[idx], r$warpfd)), digits = 15), sep = ", "); cat("\n")
  cat("registered at t[idx]:\n"); cat(format(c(eval.fd(t[idx], r$regfd)), digits = 15), sep = ", "); cat("\n")
  y0 <- eval.fd(xfine, mean.fd(yfd)); yr <- eval.fd(xfine, r$regfd)
  for (i in 1:4) {
    aa <- mean(y0^2); bb <- mean(y0 * yr[, i]); cc <- mean(yr[, i]^2)
    f <- if (crit == 1) mean((y0 - yr[, i])^2) else (aa + cc) - sqrt((aa - cc)^2 + 4 * bb^2)
    cat(sprintf("crit curve %d (data term): %.15g\n", i, f))
  }
}
for (crit in c(1, 2)) {
  r <- register.fd(mean.fd(yfd), yfd, NULL, conv = 1e-8, iterlim = 50, dbglev = 0,
                   periodic = TRUE, crit = crit)
  report(r, crit, sprintf("periodic crit=%d default W", crit))
}
# R-default W basis (2 linear B-splines), non-periodic, on the #47 golden data.
g2 <- function(s) exp(-(s - 0.3)^2 / 0.02) + 0.8 * exp(-(s - 0.7)^2 / 0.02)
a <- c(-0.3, -0.1, 0.15, 0.3)
Y2 <- sapply(a, function(ai) g2(t + ai * t * (1 - t)))
yfd2 <- smooth.basis(t, Y2, create.bspline.basis(c(0, 1), nbasis = 43, norder = 6))$fd
r <- register.fd(mean.fd(yfd2), yfd2, NULL, conv = 1e-8, iterlim = 50, dbglev = 0, crit = 2)
yfd <- yfd2
nfine <- max(201, 10 * 43 + 1); xfine <- seq(0, 1, length.out = nfine)
report(r, 2, "non-periodic crit=2 default W (R defaults)")
