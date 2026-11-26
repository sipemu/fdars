library(fdasrvf)
options(digits=17)
# 1) smooth.data golden (fdasrvf 2.5.0)
t <- seq(0, 1, length.out = 9)
f <- cbind(sin(3*t) + t^2, cos(5*t))
s <- smooth.data(f, 3)
cat("box1 <- ", sprintf("%.17g", s[,1]), "\n")
cat("box2 <- ", sprintf("%.17g", s[,2]), "\n")
# 2) f_to_srvf(smooth=TRUE) semantics of fdasrvf 2.5.1 (spline_derivative):
#    smooth.spline(time, y) then predict(deriv=1); q = g/sqrt(|g|+eps)
m <- 101; t <- seq(0, 1, length.out = m); j <- 0:(m-1)
f <- sin(2*pi*t) + 0.05*sin(1000*(j+1)*0.7071)
eps <- .Machine$double.eps
sp <- smooth.spline(t, f)
g <- predict(sp, t, deriv = 1)$y
q_s <- g / sqrt(abs(g) + eps)
gt <- 2*pi*cos(2*pi*t); q_true <- gt/sqrt(abs(gt)+eps)
g0 <- splinefun(t, f, method="fmm")(t, deriv=1); q_0 <- g0/sqrt(abs(g0)+eps)
l2 <- function(a){ d <- a^2; sqrt(sum((d[-1]+d[-m])/2*diff(t))) }
cat("R smooth err ", sprintf("%.6g", l2(q_s - q_true)), "\n")
cat("R raw err ", sprintf("%.6g", l2(q_0 - q_true)), "\n")
cat("df ", sp$df, " lambda ", sp$lambda, "\n")
