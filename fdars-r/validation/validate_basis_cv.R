# Validation script for basis function GCV/AIC/BIC
# Compares fdars implementation against mgcv

library(fdars)
library(mgcv)

cat("============================================\n")
cat("Basis Function GCV/AIC/BIC Validation\n")
cat("============================================\n\n")

set.seed(42)

# Test 1: Single curve GCV comparison
cat("Test 1: Single curve GCV comparison with mgcv\n")
cat("----------------------------------------------\n")
m <- 100
t_grid <- seq(0, 1, length.out = m)
y <- sin(2*pi*t_grid) + rnorm(m, sd = 0.2)

# fdars pspline
fd <- fdata(matrix(y, nrow = 1), argvals = t_grid)
ps_fdars <- pspline(fd, nbasis = 15, lambda = 0.1)

# mgcv GAM
gam_fit <- gam(y ~ s(t_grid, bs = "ps", k = 15, m = c(2, 2)), sp = 0.1)

cat("fdars GCV:", ps_fdars$gcv, "\n")
cat("mgcv GCV:", sum((y - fitted(gam_fit))^2) / m / (1 - sum(gam_fit$edf)/m)^2, "\n")
cat("fdars EDF:", ps_fdars$edf, "\n")
cat("mgcv EDF:", sum(gam_fit$edf), "\n")
cat("Fitted correlation:", cor(ps_fdars$fdata$data[1,], fitted(gam_fit)), "\n\n")

# Test 2: basis.gcv returns finite values for B-splines
cat("Test 2: B-spline basis.gcv returns finite values\n")
cat("------------------------------------------------\n")
y5 <- matrix(0, 5, m)
for (i in 1:5) y5[i,] <- sin(2*pi*t_grid) + rnorm(m, sd = 0.2)
fd5 <- fdata(y5, argvals = t_grid)

all_finite <- TRUE
for (nb in c(6, 8, 10, 12, 15)) {
  gcv_bsp <- basis.gcv(fd5, nbasis = nb, type = "bspline")
  gcv_four <- basis.gcv(fd5, nbasis = nb, type = "fourier")
  is_ok <- is.finite(gcv_bsp) && is.finite(gcv_four)
  all_finite <- all_finite && is_ok
  cat(sprintf("nbasis=%d: B-spline GCV=%.4f, Fourier GCV=%.4f [%s]\n",
              nb, gcv_bsp, gcv_four, if(is_ok) "OK" else "FAIL"))
}
cat(if(all_finite) "PASS: All GCV values are finite\n\n" else "FAIL: Some GCV values are Inf\n\n")

# Test 3: Coefficient dimensions match requested nbasis
cat("Test 3: Coefficient dimensions match requested nbasis\n")
cat("-----------------------------------------------------\n")
dims_match <- TRUE
for (nb in c(6, 8, 10, 12, 15)) {
  coefs <- fdata2basis(fd5, nbasis = nb, type = "bspline")
  match <- ncol(coefs) == nb
  dims_match <- dims_match && match
  cat(sprintf("Requested nbasis=%d, got ncol=%d [%s]\n",
              nb, ncol(coefs), if(match) "OK" else "FAIL"))
}
cat(if(dims_match) "PASS: All dimensions match\n\n" else "FAIL: Some dimensions mismatch\n\n")

# Test 4: fdata2basis_cv works with B-splines
cat("Test 4: fdata2basis_cv with B-splines\n")
cat("-------------------------------------\n")
cv_result <- fdata2basis_cv(fd5, nbasis.range = 6:15, type = "bspline", criterion = "GCV")
all_finite_cv <- all(is.finite(cv_result$scores))
cat("Optimal nbasis:", cv_result$optimal.nbasis, "\n")
cat("GCV scores:", round(cv_result$scores, 4), "\n")
cat(if(all_finite_cv) "PASS: All CV scores are finite\n\n" else "FAIL: Some CV scores are Inf\n\n")

# Test 5: Reconstruction accuracy
cat("Test 5: Basis reconstruction accuracy\n")
cat("-------------------------------------\n")
# Fourier (should be nearly perfect for sine wave)
coefs_f <- fdata2basis(fd, nbasis = 11, type = "fourier")
fd_recon_f <- basis2fdata(coefs_f, argvals = t_grid, type = "fourier")
err_f <- max(abs(fd$data - fd_recon_f$data))

# B-spline
coefs_b <- fdata2basis(fd, nbasis = 15, type = "bspline")
fd_recon_b <- basis2fdata(coefs_b, argvals = t_grid, type = "bspline")
err_b <- max(abs(fd$data - fd_recon_b$data))

cat(sprintf("Fourier (nbasis=11): max error = %.4f\n", err_f))
cat(sprintf("B-spline (nbasis=15): max error = %.4f\n", err_b))
cat(if(err_f < 0.5 && err_b < 1.0) "PASS: Reconstruction errors are acceptable\n\n"
    else "FAIL: Reconstruction errors too large\n\n")

# Summary
cat("============================================\n")
cat("Validation Summary\n")
cat("============================================\n")
tests_passed <- all_finite && dims_match && all_finite_cv
cat(if(tests_passed) "ALL TESTS PASSED\n" else "SOME TESTS FAILED\n")
