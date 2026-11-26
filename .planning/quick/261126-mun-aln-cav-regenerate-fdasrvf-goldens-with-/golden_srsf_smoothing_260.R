# Regenerated golden values for SrsfSmoothing (fdars #56) against fdasrvf 2.6.0
# (CRAN, installed into ~/.cache/R-lib), fdars #74.
.libPaths(c("~/.cache/R-lib", .libPaths()))
suppressMessages(library(fdasrvf))
options(digits = 17)
cat("fdasrvf", as.character(packageVersion("fdasrvf")), "\n")
# 1) smooth.data golden
t <- seq(0, 1, length.out = 9)
f <- cbind(sin(3*t) + t^2, cos(5*t))
s <- smooth.data(f, 3)
cat("box1 <- ", sprintf("%.17g", s[,1]), "\n")
cat("box2 <- ", sprintf("%.17g", s[,2]), "\n")
# 2) the real fdasrvf 2.6.0 f_to_srvf(f, time, smooth = TRUE)
m <- 101; t <- seq(0, 1, length.out = m); j <- 0:(m-1)
f <- sin(2*pi*t) + 0.05*sin(1000*(j+1)*0.7071)
eps <- .Machine$double.eps
q_s <- f_to_srvf(f, t, smooth = TRUE)
q_0 <- f_to_srvf(f, t, smooth = FALSE)
# identical to the hand-written smooth.spline route used for 2.5.0 (#56)?
sp <- smooth.spline(t, f); g <- predict(sp, t, deriv = 1)$y
cat("max |f_to_srvf - manual smooth.spline route| ",
    sprintf("%.3g", max(abs(q_s - g / sqrt(abs(g) + eps)))), "\n")
gt <- 2*pi*cos(2*pi*t); q_true <- gt/sqrt(abs(gt)+eps)
l2 <- function(a){ d <- a^2; sqrt(sum((d[-1]+d[-m])/2*diff(t))) }
cat("R smooth err ", sprintf("%.6g", l2(q_s - q_true)), "\n")
cat("R raw err ", sprintf("%.6g", l2(q_0 - q_true)), "\n")
cat("df ", sp$df, " lambda ", sp$lambda, "\n")
cat("q_smooth <- ", sprintf("%.17g", q_s), "\n")
