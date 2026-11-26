# Golden values for curve_shape_depth (fdars #54 ALN-NDDOWN): fdasrvf 2.5.0
# curve_depth(beta, mode = "O", rotated = FALSE, scale = TRUE) on 6 open planar curves
# beta_i(t) = (t, a_i sin(pi t) + c_i sin(2 pi t^p_i)), at T = 60 ... 480 to show
# fdasrvf's own resolution drift (the Rust test uses T = 480 as the reference).
suppressMessages(library(fdasrvf))
cat("fdasrvf", as.character(packageVersion("fdasrvf")), "\n")
a <- c(0.20, 0.25, 0.30, 0.35, 0.40, 0.60)
cc <- c(0.05, -0.03, 0.02, 0.04, -0.05, 0.30)
p <- c(1.0, 1.2, 0.9, 1.1, 1.0, 1.5)
N <- length(a)
for (M in c(60, 120, 240, 480)) {
  tt <- seq(0, 1, length.out = M)
  beta <- array(0, c(2, M, N))
  for (i in 1:N) beta[, , i] <- rbind(tt, a[i] * sin(pi * tt) + cc[i] * sin(2 * pi * tt^p[i]))
  out <- curve_depth(beta, mode = "O", rotated = FALSE, scale = TRUE)
  cat("T", M, "amp", sprintf("%.10f", out$amp), "\n")
  cat("T", M, "phase", sprintf("%.10f", out$phase), "\n")
}
