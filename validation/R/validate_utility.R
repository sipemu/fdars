#!/usr/bin/env Rscript
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

message("=== Validating utility functions ===\n")

results <- list()

# ---- (a) Simpson's weights on 101 points ------------------------------------
message("  Computing Simpson's weights on seq(0, 1, length.out = 101)...")
results$simpsons_weights <- tryCatch({
  argvals_101 <- seq(0, 1, length.out = 101)
  n <- length(argvals_101)
  h <- argvals_101[2] - argvals_101[1]
  # Composite Simpson's rule: h/3 * [1, 4, 2, 4, 2, ..., 4, 1]
  # n must be odd (101 is odd)
  w <- rep(0, n)
  w[1] <- 1
  w[n] <- 1
  for (i in 2:(n - 1)) {
    if (i %% 2 == 0) {
      w[i] <- 4
    } else {
      w[i] <- 2
    }
  }
  w <- w * h / 3
  as.numeric(w)
}, error = function(e) {
  warning(sprintf("Simpson's weights (101) failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Simpson's weights on 11 points -------------------------------------
message("  Computing Simpson's weights on seq(0, 1, length.out = 11)...")
results$simpsons_weights_11 <- tryCatch({
  argvals_11 <- seq(0, 1, length.out = 11)
  n <- length(argvals_11)
  h <- argvals_11[2] - argvals_11[1]
  # Composite Simpson's rule: h/3 * [1, 4, 2, 4, 2, ..., 4, 1]
  w <- rep(0, n)
  w[1] <- 1
  w[n] <- 1
  for (i in 2:(n - 1)) {
    if (i %% 2 == 0) {
      w[i] <- 4
    } else {
      w[i] <- 2
    }
  }
  w <- w * h / 3
  as.numeric(w)
}, error = function(e) {
  warning(sprintf("Simpson's weights (11) failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Inner product of first two curves using Simpson's rule ---------------
message("  Computing inner product of curves 1 and 2 via Simpson's rule...")
results$inner_product_12 <- tryCatch({
  dat <- load_data("standard_50x101")
  n_curves <- dat$n
  m_pts <- dat$m
  argvals <- dat$argvals
  mat <- to_matrix(dat$data, n_curves, m_pts)

  # Simpson's weights for argvals (101 points)
  h <- argvals[2] - argvals[1]
  n_pts <- length(argvals)
  w <- rep(0, n_pts)
  w[1] <- 1
  w[n_pts] <- 1
  for (i in 2:(n_pts - 1)) {
    if (i %% 2 == 0) {
      w[i] <- 4
    } else {
      w[i] <- 2
    }
  }
  w <- w * h / 3

  curve1 <- mat[1, ]
  curve2 <- mat[2, ]
  sum(w * curve1 * curve2)
}, error = function(e) {
  warning(sprintf("Inner product (1,2) failed: %s", conditionMessage(e)))
  NULL
})

# ---- (d) Inner product matrix of first 5 curves ------------------------------
message("  Computing 5x5 inner product matrix via Simpson's rule...")
results$inner_product_matrix <- tryCatch({
  dat <- load_data("standard_50x101")
  n_curves <- dat$n
  m_pts <- dat$m
  argvals <- dat$argvals
  mat <- to_matrix(dat$data, n_curves, m_pts)

  # Simpson's weights for argvals (101 points)
  h <- argvals[2] - argvals[1]
  n_pts <- length(argvals)
  w <- rep(0, n_pts)
  w[1] <- 1
  w[n_pts] <- 1
  for (i in 2:(n_pts - 1)) {
    if (i %% 2 == 0) {
      w[i] <- 4
    } else {
      w[i] <- 2
    }
  }
  w <- w * h / 3

  sub5 <- mat[1:5, ]
  ip_mat <- matrix(0, 5, 5)
  for (i in 1:5) {
    for (j in 1:5) {
      ip_mat[i, j] <- sum(w * sub5[i, ] * sub5[j, ])
    }
  }
  list(
    n = 5,
    data = as.numeric(ip_mat)
  )
}, error = function(e) {
  warning(sprintf("Inner product matrix failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results -------------------------------------------------------------
save_expected(results, "utility_expected")
message("\nDone.\n")
