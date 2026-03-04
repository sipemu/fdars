#!/usr/bin/env Rscript
# Validate functional equivalence test (TOST) against R reference implementation.
# No R package exists for functional TOST, so this is a custom implementation.
#
# Usage: Rscript validation/R/validate_equivalence.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Equivalence test validation ===")

d <- load_data("equivalence_groups")
mat1 <- to_matrix(d$data1, d$n, d$m)
mat2 <- to_matrix(d$data2, d$n, d$m)
argvals <- d$argvals

result <- list()

n1 <- d$n
n2 <- d$n
m <- d$m
delta <- 0.5
alpha <- 0.05

# ---------------------------------------------------------------------------
# 1. Deterministic quantities
# ---------------------------------------------------------------------------
message("  Computing deterministic quantities...")
mean1 <- colMeans(mat1)
mean2 <- colMeans(mat2)
d_hat <- mean1 - mean2
test_statistic <- max(abs(d_hat))

result$d_hat <- as.vector(d_hat)
result$test_statistic <- test_statistic

# Pointwise standard deviations (population sd, not sample sd, to match Rust's /n)
sig1 <- sqrt(colMeans((mat1 - matrix(mean1, nrow = n1, ncol = m, byrow = TRUE))^2))
sig2 <- sqrt(colMeans((mat2 - matrix(mean2, nrow = n2, ncol = m, byrow = TRUE))^2))
pooled_se <- sqrt(sig1^2 / n1 + sig2^2 / n2)
pooled_se <- pmax(pooled_se, 1e-15)
result$pooled_se <- as.vector(pooled_se)

# ---------------------------------------------------------------------------
# 2. Gaussian multiplier bootstrap
# ---------------------------------------------------------------------------
message("  Computing Gaussian multiplier bootstrap...")
nb <- 1000

# Center the data
centered1 <- mat1 - matrix(mean1, nrow = n1, ncol = m, byrow = TRUE)
centered2 <- mat2 - matrix(mean2, nrow = n2, ncol = m, byrow = TRUE)

set.seed(42)
sup_stats <- numeric(nb)
for (b in 1:nb) {
  g1 <- rnorm(n1)
  g2 <- rnorm(n2)
  boot_stat <- numeric(m)
  for (j in 1:m) {
    s1 <- sum(g1 * centered1[, j]) / n1
    s2 <- sum(g2 * centered2[, j]) / n2
    boot_stat[j] <- abs((s1 - s2) / pooled_se[j])
  }
  sup_stats[b] <- max(boot_stat)
}

# Critical value at 1 - 2*alpha quantile
c_alpha <- as.numeric(quantile(sup_stats, probs = 1 - 2 * alpha, type = 1))
result$critical_value <- c_alpha

# SCB bounds
scb_lower <- d_hat - c_alpha * pooled_se
scb_upper <- d_hat + c_alpha * pooled_se
result$scb_lower <- as.vector(scb_lower)
result$scb_upper <- as.vector(scb_upper)

# Equivalence decision
equivalent <- all(scb_upper < delta) && all(scb_lower > -delta)
result$equivalent <- equivalent

# P-value (TOST style)
c_threshold <- min((delta - abs(d_hat)) / pooled_se)
if (c_threshold <= 0) {
  p_value <- 1.0
} else {
  p_value <- min(1.0, sum(sup_stats >= c_threshold) / (2 * nb))
}
result$p_value <- p_value

save_expected(result, "equivalence_expected")
message("=== Equivalence test validation complete ===")
