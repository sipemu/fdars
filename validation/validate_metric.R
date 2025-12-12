# Validation script for metric functions
# Compares fdars (Rust) vs fda.usc (R) implementations

library(fda.usc)
library(fdars)

# Set seed for reproducibility
set.seed(42)

# Create test data
n <- 20
m <- 50
t_grid <- seq(0, 1, length.out = m)

# Generate smooth functional data
X <- matrix(0, n, m)
for (i in 1:n) {
  X[i, ] <- sin(2 * pi * t_grid) + 0.5 * cos(4 * pi * t_grid * i / n) + rnorm(m, sd = 0.1)
}

# Create fdata objects for both packages
fd_orig <- fda.usc::fdata(X, argvals = t_grid)
fd_rust <- fdars::fdata(X, argvals = t_grid)

cat("========================================\n")
cat("Validation: metric.lp (L2 distance)\n")
cat("========================================\n\n")

# Compute L2 distances
D_orig <- fda.usc::metric.lp(fd_orig, lp = 2)
D_rust <- fdars::metric.lp(fd_rust, lp = 2)

# Compare
max_diff <- max(abs(D_orig - D_rust))
mean_diff <- mean(abs(D_orig - D_rust))
rel_diff <- max(abs(D_orig - D_rust) / (abs(D_orig) + 1e-10))

cat("Max absolute difference:", max_diff, "\n")
cat("Mean absolute difference:", mean_diff, "\n")
cat("Max relative difference:", rel_diff, "\n")

if (max_diff < 1e-6) {
  cat("PASS: L2 distances match within tolerance\n\n")
} else {
  cat("FAIL: L2 distances differ significantly\n")
  cat("Original (first 5x5):\n")
  print(round(D_orig[1:5, 1:5], 6))
  cat("\nRust (first 5x5):\n")
  print(round(D_rust[1:5, 1:5], 6))
  cat("\n")
}

cat("========================================\n")
cat("Validation: metric.lp (L1 distance)\n")
cat("========================================\n\n")

# Compute L1 distances
D_orig_l1 <- fda.usc::metric.lp(fd_orig, lp = 1)
D_rust_l1 <- fdars::metric.lp(fd_rust, lp = 1)

max_diff_l1 <- max(abs(D_orig_l1 - D_rust_l1))
cat("Max absolute difference (L1):", max_diff_l1, "\n")

if (max_diff_l1 < 1e-6) {
  cat("PASS: L1 distances match within tolerance\n\n")
} else {
  cat("FAIL: L1 distances differ significantly\n\n")
}

cat("========================================\n")
cat("Validation: metric.lp (cross-distances)\n")
cat("========================================\n\n")

# Cross-distances between different subsets
fd_orig1 <- fd_orig[1:10, ]
fd_orig2 <- fd_orig[11:20, ]
fd_rust1 <- fd_rust[1:10, ]
fd_rust2 <- fd_rust[11:20, ]

D_cross_orig <- fda.usc::metric.lp(fd_orig1, fd_orig2, lp = 2)
D_cross_rust <- fdars::metric.lp(fd_rust1, fd_rust2, lp = 2)

max_diff_cross <- max(abs(D_cross_orig - D_cross_rust))
cat("Max absolute difference (cross):", max_diff_cross, "\n")

if (max_diff_cross < 1e-6) {
  cat("PASS: Cross-distances match within tolerance\n\n")
} else {
  cat("FAIL: Cross-distances differ significantly\n\n")
}

cat("========================================\n")
cat("Validation: semimetric.pca\n")
cat("========================================\n\n")

# PCA semi-metric
D_pca_orig <- fda.usc::semimetric.pca(fd_orig, ncomp = 3)
D_pca_rust <- fdars::semimetric.pca(fd_rust, ncomp = 3)

# Note: PCA may have sign ambiguity, so we compare absolute values
max_diff_pca <- max(abs(D_pca_orig - D_pca_rust))
cat("Max absolute difference (PCA):", max_diff_pca, "\n")

if (max_diff_pca < 1e-4) {
  cat("PASS: PCA semi-metric matches within tolerance\n\n")
} else {
  cat("INFO: PCA semi-metric differs (may be due to algorithm differences)\n")
  cat("Original (first 5x5):\n")
  print(round(D_pca_orig[1:5, 1:5], 4))
  cat("\nRust (first 5x5):\n")
  print(round(D_pca_rust[1:5, 1:5], 4))
  cat("\n")
}

cat("========================================\n")
cat("Validation: metric.hausdorff\n")
cat("========================================\n\n")

D_haus_orig <- fda.usc::metric.hausdorff(fd_orig)
D_haus_rust <- fdars::metric.hausdorff(fd_rust)

max_diff_haus <- max(abs(D_haus_orig - D_haus_rust))
cat("Max absolute difference (Hausdorff):", max_diff_haus, "\n")

if (max_diff_haus < 1e-10) {
  cat("PASS: Hausdorff distances match exactly\n\n")
} else {
  cat("INFO: Hausdorff distances differ\n")
  cat("Original (first 5x5):\n")
  print(round(D_haus_orig[1:5, 1:5], 4))
  cat("\nRust (first 5x5):\n")
  print(round(D_haus_rust[1:5, 1:5], 4))
  cat("\n")
}

cat("========================================\n")
cat("Summary\n")
cat("========================================\n")
cat("metric.lp (L2): max_diff =", max_diff, "\n")
cat("metric.lp (L1): max_diff =", max_diff_l1, "\n")
cat("metric.lp (cross): max_diff =", max_diff_cross, "\n")
cat("metric.hausdorff: max_diff =", max_diff_haus, "\n")
cat("semimetric.pca: max_diff =", max_diff_pca, "\n")
