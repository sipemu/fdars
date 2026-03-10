#!/usr/bin/env Rscript
# Validate elastic regression against R's fdasrvf.
#
# Reference package: fdasrvf
# Usage: Rscript validation/R/validate_elastic_regression.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fdasrvf)

message("=== Validating elastic regression ===\n")

results <- list()

# Generate test data
set.seed(42)
n <- 15
m <- 51
argvals <- seq(0, 1, length.out = m)
data_mat <- matrix(0, nrow = m, ncol = n)  # fdasrvf: m x n
y <- numeric(n)

for (i in 1:n) {
  amp <- 1.0 + 0.5 * (i / n)
  shift <- 0.1 * (i - n / 2)
  data_mat[, i] <- amp * sin(2 * pi * (argvals + shift))
  y[i] <- amp
}

results$data <- as.numeric(data_mat)
results$y <- y
results$n <- n
results$m <- m
results$argvals <- argvals

# Helper to extract PCR results, computing fitted values from scores
extract_pcr_result <- function(epcr, y) {
  # Compute fitted values: yhat = alpha + scores %*% b
  scores <- epcr$pca$coef  # n x ncomp matrix
  b <- epcr$b              # ncomp vector
  yhat <- as.numeric(epcr$alpha + scores %*% b)
  sse <- epcr$SSE
  r_squared <- 1.0 - sse / sum((y - mean(y))^2)
  list(
    alpha = epcr$alpha,
    coefficients = as.numeric(b),
    fitted_values = yhat,
    sse = sse,
    r_squared = r_squared
  )
}

# ---- (a) Elastic PCR (vertical) ---
message("  Computing elastic PCR (vertical)...")
results$elastic_pcr_vert <- tryCatch({
  epcr <- elastic.pcr.regression(data_mat, y, time = argvals,
                                  pca.method = "vert", no = 3)
  extract_pcr_result(epcr, y)
}, error = function(e) {
  warning(sprintf("Elastic PCR (vert) failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Elastic PCR (horizontal) ---
message("  Computing elastic PCR (horizontal)...")
results$elastic_pcr_horiz <- tryCatch({
  epcr <- elastic.pcr.regression(data_mat, y, time = argvals,
                                  pca.method = "horiz", no = 3)
  extract_pcr_result(epcr, y)
}, error = function(e) {
  warning(sprintf("Elastic PCR (horiz) failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Elastic PCR (combined/joint) ---
message("  Computing elastic PCR (combined)...")
results$elastic_pcr_combined <- tryCatch({
  epcr <- elastic.pcr.regression(data_mat, y, time = argvals,
                                  pca.method = "combined", no = 3)
  extract_pcr_result(epcr, y)
}, error = function(e) {
  warning(sprintf("Elastic PCR (combined) failed: %s", conditionMessage(e)))
  NULL
})

save_expected(results, "elastic_regression")
message("Done.\n")
