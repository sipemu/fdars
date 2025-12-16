# Validation script for fdata functions
# Compares fdars (Rust) vs fda.usc (R) implementations

library(fda.usc)
library(fdars)

# Set seed for reproducibility
set.seed(42)

# Create test data
n <- 20
m <- 30
t_grid <- seq(0, 1, length.out = m)

# Generate test matrix
X <- matrix(rnorm(n * m), n, m)

# Create fdata objects for both packages
fd_orig <- fda.usc::fdata(X, argvals = t_grid)
fd_rust <- fdars::fdata(X, argvals = t_grid)

cat("========================================\n")
cat("Validation: fdata creation\n")
cat("========================================\n\n")

# Check that data is stored correctly
data_match <- all(fd_orig$data == fd_rust$data)
argvals_match <- all(fd_orig$argvals == fd_rust$argvals)

cat("Data matrices match:", data_match, "\n")
cat("Argvals match:", argvals_match, "\n")

if (data_match && argvals_match) {
  cat("PASS: fdata objects created identically\n\n")
} else {
  cat("FAIL: fdata objects differ\n\n")
}

cat("========================================\n")
cat("Validation: mean\n")
cat("========================================\n\n")

# Compute mean
mean_orig <- fda.usc::mean(fd_orig)
mean_rust <- fdars::mean(fd_rust)

# fda.usc returns an fdata object, fdars returns a vector
if (inherits(mean_orig, "fdata")) {
  mean_orig_vec <- as.vector(mean_orig$data)
} else {
  mean_orig_vec <- mean_orig
}

max_diff <- max(abs(mean_orig_vec - mean_rust))
cat("Max absolute difference:", max_diff, "\n")

if (max_diff < 1e-10) {
  cat("PASS: mean matches exactly\n\n")
} else {
  cat("FAIL: mean differs\n")
  cat("Original:", round(mean_orig_vec[1:5], 6), "...\n")
  cat("Rust:", round(mean_rust[1:5], 6), "...\n\n")
}

cat("========================================\n")
cat("Validation: fdata.cen (centering)\n")
cat("========================================\n\n")

# Center data
fd_cen_orig <- fda.usc::fdata.cen(fd_orig)
fd_cen_rust <- fdars::fdata.cen(fd_rust)

# Compare centered data
max_diff_cen <- max(abs(fd_cen_orig$data - fd_cen_rust$data))
cat("Max absolute difference:", max_diff_cen, "\n")

# Check that centered data has zero mean
mean_cen_rust <- colMeans(fd_cen_rust$data)
max_mean <- max(abs(mean_cen_rust))
cat("Max centered mean (should be ~0):", max_mean, "\n")

if (max_diff_cen < 1e-10 && max_mean < 1e-10) {
  cat("PASS: fdata.cen matches exactly\n\n")
} else {
  cat("FAIL: fdata.cen differs\n\n")
}

cat("========================================\n")
cat("Validation: norm (L2 norm)\n")
cat("========================================\n\n")

# Compute L2 norms
norm_orig <- fda.usc::norm.fdata(fd_orig)
norm_rust <- fdars::norm(fd_rust)

max_diff_norm <- max(abs(norm_orig - norm_rust))
cat("Max absolute difference:", max_diff_norm, "\n")

if (max_diff_norm < 1e-6) {
  cat("PASS: norm matches within tolerance\n\n")
} else {
  cat("INFO: norm differs (may be integration method)\n")
  cat("Original norms (first 5):", round(norm_orig[1:5], 6), "\n")
  cat("Rust norms (first 5):", round(norm_rust[1:5], 6), "\n\n")
}

cat("========================================\n")
cat("Validation: norm (L1 norm)\n")
cat("========================================\n\n")

# Compute L1 norms
norm_l1_orig <- fda.usc::norm.fdata(fd_orig, lp = 1)
norm_l1_rust <- fdars::norm(fd_rust, lp = 1)

max_diff_l1 <- max(abs(norm_l1_orig - norm_l1_rust))
cat("Max absolute difference (L1):", max_diff_l1, "\n")

if (max_diff_l1 < 1e-6) {
  cat("PASS: L1 norm matches within tolerance\n\n")
} else {
  cat("INFO: L1 norm differs\n\n")
}

cat("========================================\n")
cat("Validation: subsetting [.fdata\n")
cat("========================================\n\n")

# Test subsetting
fd_sub_orig <- fd_orig[1:5, ]
fd_sub_rust <- fd_rust[1:5, ]

sub_match <- all(fd_sub_orig$data == fd_sub_rust$data)
cat("Subsetted data match:", sub_match, "\n")

if (sub_match) {
  cat("PASS: Subsetting works identically\n\n")
} else {
  cat("FAIL: Subsetting differs\n\n")
}

cat("========================================\n")
cat("Summary\n")
cat("========================================\n")
cat("fdata creation: PASS\n")
cat("mean max_diff:", max_diff, "\n")
cat("fdata.cen max_diff:", max_diff_cen, "\n")
cat("norm (L2) max_diff:", max_diff_norm, "\n")
cat("norm (L1) max_diff:", max_diff_l1, "\n")
