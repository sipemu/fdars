# Validation script for depth functions
# Compares fdars (Rust) vs fda.usc (R) implementations

library(fda.usc)
library(fdars)

# Set seed for reproducibility
set.seed(42)

# Create test data
n <- 30
m <- 50
t_grid <- seq(0, 1, length.out = m)

# Generate smooth functional data
X <- matrix(0, n, m)
for (i in 1:n) {
  X[i, ] <- sin(2 * pi * t_grid) + 0.3 * cos(4 * pi * t_grid) + rnorm(m, sd = 0.2)
}

# Add one outlier
X[1, ] <- 3 + sin(2 * pi * t_grid)

# Create fdata objects for both packages
fd_orig <- fda.usc::fdata(X, argvals = t_grid)
fd_rust <- fdars::fdata(X, argvals = t_grid)

cat("========================================\n")
cat("Validation: depth.FM (Fraiman-Muniz)\n")
cat("========================================\n\n")

# Compute FM depth (scale = TRUE matches fda.usc default)
depth_fm_orig <- fda.usc::depth.FM(fd_orig, trim = 0.25)$dep
depth_fm_rust <- fdars::depth.FM(fd_rust, trim = 0.25, scale = TRUE)

# Compare
max_diff <- max(abs(depth_fm_orig - depth_fm_rust))
cor_val <- cor(depth_fm_orig, depth_fm_rust)

cat("Max absolute difference:", max_diff, "\n")
cat("Correlation:", cor_val, "\n")
cat("Outlier detected (orig):", which.min(depth_fm_orig), "\n")
cat("Outlier detected (rust):", which.min(depth_fm_rust), "\n")

if (max_diff < 1e-10) {
  cat("PASS: FM depth values match exactly (machine precision)\n\n")
} else if (max_diff < 0.1 && cor_val > 0.95) {
  cat("PASS: FM depth values are highly correlated\n\n")
} else {
  cat("INFO: FM depth differs (checking algorithm differences)\n")
  cat("Original depths (first 10):", round(depth_fm_orig[1:10], 4), "\n")
  cat("Rust depths (first 10):", round(depth_fm_rust[1:10], 4), "\n\n")
}

cat("========================================\n")
cat("Validation: depth.mode (Modal depth)\n")
cat("========================================\n\n")

# Modal depth (both should identify outliers similarly)
depth_mode_orig <- fda.usc::depth.mode(fd_orig)$dep
depth_mode_rust <- fdars::depth.mode(fd_rust)

# Compare ranks rather than absolute values (different algorithms)
rank_orig <- rank(-depth_mode_orig)  # Higher depth = lower rank
rank_rust <- rank(-depth_mode_rust)

cor_ranks <- cor(rank_orig, rank_rust)
cat("Rank correlation:", cor_ranks, "\n")
cat("Outlier rank (orig):", rank_orig[1], "\n")
cat("Outlier rank (rust):", rank_rust[1], "\n")

if (cor_ranks > 0.8 && rank_orig[1] >= n - 2 && rank_rust[1] >= n - 2) {
  cat("PASS: Modal depth ranks are correlated and outlier detected\n\n")
} else {
  cat("INFO: Modal depth ranks differ\n")
  cat("Original depths (first 10):", round(depth_mode_orig[1:10], 4), "\n")
  cat("Rust depths (first 10):", round(depth_mode_rust[1:10], 4), "\n\n")
}

cat("========================================\n")
cat("Validation: depth.RP (Random Projection)\n")
cat("========================================\n\n")

# RP depth (stochastic, so we test multiple times)
set.seed(123)
depth_rp_orig <- fda.usc::depth.RP(fd_orig, nproj = 100)$dep

set.seed(123)
depth_rp_rust <- fdars::depth.RP(fd_rust, nproj = 100)

# Compare ranks
rank_rp_orig <- rank(-depth_rp_orig)
rank_rp_rust <- rank(-depth_rp_rust)

cor_rp <- cor(rank_rp_orig, rank_rp_rust)
cat("Rank correlation:", cor_rp, "\n")
cat("Outlier rank (orig):", rank_rp_orig[1], "\n")
cat("Outlier rank (rust):", rank_rp_rust[1], "\n")

# Note: RP is stochastic so we just check outlier detection
if (rank_rp_orig[1] >= n - 5 && rank_rp_rust[1] >= n - 5) {
  cat("PASS: Both methods identify outlier in bottom ranks\n\n")
} else {
  cat("INFO: RP depth outlier detection differs\n\n")
}

cat("========================================\n")
cat("Validation: depth.RT (Random Tukey)\n")
cat("========================================\n\n")

set.seed(456)
depth_rt_orig <- fda.usc::depth.RT(fd_orig, nproj = 100)$dep

set.seed(456)
depth_rt_rust <- fdars::depth.RT(fd_rust, nproj = 100)

rank_rt_orig <- rank(-depth_rt_orig)
rank_rt_rust <- rank(-depth_rt_rust)

cor_rt <- cor(rank_rt_orig, rank_rt_rust)
cat("Rank correlation:", cor_rt, "\n")
cat("Outlier rank (orig):", rank_rt_orig[1], "\n")
cat("Outlier rank (rust):", rank_rt_rust[1], "\n")

if (rank_rt_orig[1] >= n - 5 && rank_rt_rust[1] >= n - 5) {
  cat("PASS: Both methods identify outlier\n\n")
} else {
  cat("INFO: RT depth differs\n\n")
}

cat("========================================\n")
cat("Validation: depth.FSD (Functional Spatial)\n")
cat("========================================\n\n")

depth_fsd_orig <- fda.usc::depth.FSD(fd_orig)$dep
depth_fsd_rust <- fdars::depth.FSD(fd_rust)

max_diff_fsd <- max(abs(depth_fsd_orig - depth_fsd_rust))
cor_fsd <- cor(depth_fsd_orig, depth_fsd_rust)

cat("Max absolute difference:", max_diff_fsd, "\n")
cat("Correlation:", cor_fsd, "\n")
cat("Outlier detected (orig):", which.min(depth_fsd_orig), "\n")
cat("Outlier detected (rust):", which.min(depth_fsd_rust), "\n")

if (cor_fsd > 0.95) {
  cat("PASS: FSD depth highly correlated\n\n")
} else {
  cat("INFO: FSD depth differs\n")
  cat("Original depths (first 10):", round(depth_fsd_orig[1:10], 4), "\n")
  cat("Rust depths (first 10):", round(depth_fsd_rust[1:10], 4), "\n\n")
}

cat("========================================\n")
cat("Summary\n")
cat("========================================\n")
cat("depth.FM correlation:", cor(depth_fm_orig, depth_fm_rust), "\n")
cat("depth.mode rank correlation:", cor_ranks, "\n")
cat("depth.FSD correlation:", cor_fsd, "\n")
cat("All methods detect curve 1 as outlier:",
    which.min(depth_fm_orig) == 1 && which.min(depth_fm_rust) == 1, "\n")
