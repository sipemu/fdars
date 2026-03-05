#!/usr/bin/env Rscript
# Validate new modules (Issues #4-#8) against R reference implementations.
#
# Generates expected outputs for:
# - Scalar-on-function regression (fregre.lm from fda.usc)
# - Function-on-scalar regression (fRegress from fda)
# - GMM clustering (Mclust from mclust)
# - Functional classification (classif.* from fda.usc)
# - Functional mixed models (comparison with lme4 on FPC scores)
#
# Reference packages: fda, fda.usc, mclust, lme4, MASS
# Usage: Rscript validation/R/validate_new_modules.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)
library(MASS)

message("=== Validating new module computations ===\n")

set.seed(42)
results <- list()

# ── Shared data: regression dataset ─────────────────────────────────────────
dat <- load_data("regression_30x51")
n <- dat$n
m <- dat$m
argvals <- dat$argvals
mat <- to_matrix(dat$data, n, m)
y <- dat$y
fd_obj <- fdata(mat, argvals = argvals)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Scalar-on-function regression (fregre.lm from fda.usc)
# ═══════════════════════════════════════════════════════════════════════════════
message("  [1/5] Scalar-on-function regression (fregre.lm)...")
results$scalar_on_function <- tryCatch({
  # Use FPC basis with 3 components
  fit <- fregre.pc(fd_obj, y, l = 1:3)

  fitted_vals <- as.numeric(fit$fitted.values)
  residuals_raw <- as.numeric(fit$residuals)
  r_squared <- 1 - sum(residuals_raw^2) / sum((y - mean(y))^2)

  message(sprintf("    R² = %.6f", r_squared))
  message(sprintf("    Residual SS = %.6f", sum(residuals_raw^2)))

  list(
    fitted = fitted_vals,
    residuals = residuals_raw,
    r_squared = r_squared,
    residual_ss = sum(residuals_raw^2)
  )
}, error = function(e) {
  warning(sprintf("Scalar-on-function failed: %s", conditionMessage(e)))
  NULL
})

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Function-on-scalar regression (manually via lm on FPC scores)
# ═══════════════════════════════════════════════════════════════════════════════
message("  [2/5] Function-on-scalar regression...")
results$function_on_scalar <- tryCatch({
  # Create a scalar covariate (from the regression y as predictor of curve shape)
  x_cov <- y  # use scalar response as covariate

  # Compute FPC scores
  pca_obj <- fdata2pc(fd_obj, ncomp = 3)
  scores <- pca_obj$x[, 1:3]

  # Regress each FPC score on the covariate
  beta_scores <- numeric(3)
  for (k in 1:3) {
    fit_k <- lm(scores[, k] ~ x_cov)
    beta_scores[k] <- coef(fit_k)["x_cov"]
  }

  # Recover beta function: sum_k beta_k * phi_k(t)
  loadings <- pca_obj$rotation$data[1:3, ]  # 3 x m
  beta_function <- as.numeric(beta_scores %*% loadings)

  # Fitted curves: each curve gets x_i * beta(t)
  mean_curve <- as.numeric(colMeans(mat))
  fitted_mat <- outer(x_cov - mean(x_cov), beta_function) + rep(mean_curve, each = n)
  residual_mat <- mat - fitted_mat

  # Integrated squared residual per curve
  residual_l2 <- apply(residual_mat^2, 1, function(row) {
    sum(row) * (argvals[m] - argvals[1]) / (m - 1)
  })

  message(sprintf("    Beta function range: [%.4f, %.4f]", min(beta_function), max(beta_function)))
  message(sprintf("    Mean residual L2: %.6f", mean(residual_l2)))

  list(
    beta_function = beta_function,
    mean_residual_l2 = mean(residual_l2),
    beta_scores = beta_scores
  )
}, error = function(e) {
  warning(sprintf("Function-on-scalar failed: %s", conditionMessage(e)))
  NULL
})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GMM clustering (Mclust from mclust)
# ═══════════════════════════════════════════════════════════════════════════════
message("  [3/5] GMM clustering...")
results$gmm <- tryCatch({
  library(mclust)

  # Use cluster dataset
  clust_dat <- load_data("clusters_60x51")
  n_c <- clust_dat$n
  m_c <- clust_dat$m
  clust_mat <- to_matrix(clust_dat$data, n_c, m_c)
  true_labels <- clust_dat$true_labels

  # FPCA for dimensionality reduction
  clust_fd <- fdata(clust_mat, argvals = clust_dat$argvals)
  pca_clust <- fdata2pc(clust_fd, ncomp = 3)
  scores_clust <- pca_clust$x[, 1:3]

  # Fit GMM with K=3
  fit_gmm <- Mclust(scores_clust, G = 3, modelNames = "VVV")

  # Classification accuracy (after optimal label permutation)
  pred_labels <- fit_gmm$classification
  # Align labels via Hungarian matching (simple for K=3)
  best_acc <- 0
  for (perm in list(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))) {
    remapped <- perm[pred_labels]
    acc <- mean(remapped == true_labels)
    if (acc > best_acc) best_acc <- acc
  }

  message(sprintf("    GMM accuracy (K=3): %.4f", best_acc))
  message(sprintf("    BIC: %.2f", fit_gmm$bic))
  message(sprintf("    Log-likelihood: %.4f", fit_gmm$loglik))

  # BIC model selection
  fit_auto <- Mclust(scores_clust, G = 1:6)
  selected_k <- fit_auto$G

  message(sprintf("    BIC-selected K: %d", selected_k))

  list(
    accuracy = best_acc,
    bic = fit_gmm$bic,
    loglik = fit_gmm$loglik,
    selected_k = selected_k,
    weights = as.numeric(fit_gmm$parameters$pro)
  )
}, error = function(e) {
  warning(sprintf("GMM clustering failed: %s", conditionMessage(e)))
  NULL
})

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Functional classification (fda.usc classifiers)
# ═══════════════════════════════════════════════════════════════════════════════
message("  [4/5] Functional classification...")
results$classification <- tryCatch({
  # Use cluster dataset as classification problem
  clust_dat <- load_data("clusters_60x51")
  n_c <- clust_dat$n
  m_c <- clust_dat$m
  clust_mat <- to_matrix(clust_dat$data, n_c, m_c)
  true_labels <- as.factor(clust_dat$true_labels)

  clust_fd <- fdata(clust_mat, argvals = clust_dat$argvals)

  # Depth-based classification (DD-classifier)
  group <- as.integer(true_labels)
  fd_class1 <- fdata(clust_mat[group == 1, ], argvals = clust_dat$argvals)
  fd_class2 <- fdata(clust_mat[group == 2, ], argvals = clust_dat$argvals)
  fd_class3 <- fdata(clust_mat[group == 3, ], argvals = clust_dat$argvals)

  # Compute FM depth of all observations w.r.t. each class
  depth_c1 <- depth.FM(clust_fd, fdataori = fd_class1)$dep
  depth_c2 <- depth.FM(clust_fd, fdataori = fd_class2)$dep
  depth_c3 <- depth.FM(clust_fd, fdataori = fd_class3)$dep

  depth_matrix <- cbind(depth_c1, depth_c2, depth_c3)
  dd_predicted <- apply(depth_matrix, 1, which.max)
  dd_accuracy <- mean(dd_predicted == group)

  message(sprintf("    DD-classifier accuracy: %.4f", dd_accuracy))

  # k-NN on FPC scores
  pca_clust <- fdata2pc(clust_fd, ncomp = 3)
  scores_df <- as.data.frame(pca_clust$x[, 1:3])
  scores_df$label <- as.factor(group)

  # LOO k-NN with k=3
  knn_correct <- 0
  for (i in 1:n_c) {
    train_scores <- scores_df[-i, 1:3]
    train_labels <- scores_df$label[-i]
    test_score <- scores_df[i, 1:3]
    pred <- class::knn(train_scores, test_score, train_labels, k = 3)
    if (pred == scores_df$label[i]) knn_correct <- knn_correct + 1
  }
  knn_accuracy <- knn_correct / n_c

  message(sprintf("    k-NN accuracy (k=3, LOO): %.4f", knn_accuracy))

  list(
    dd_accuracy = dd_accuracy,
    dd_predicted = as.integer(dd_predicted),
    dd_depths = list(
      class1 = as.numeric(depth_c1),
      class2 = as.numeric(depth_c2),
      class3 = as.numeric(depth_c3)
    ),
    knn_accuracy = knn_accuracy
  )
}, error = function(e) {
  warning(sprintf("Classification failed: %s", conditionMessage(e)))
  NULL
})

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Functional mixed model (lme4 on FPC scores)
# ═══════════════════════════════════════════════════════════════════════════════
message("  [5/5] Functional mixed effects model...")
results$famm <- tryCatch({
  library(lme4)

  # Generate longitudinal data: 10 subjects, 3 visits each
  n_subj <- 10
  n_visits <- 3
  n_total <- n_subj * n_visits
  m_fmm <- 31
  t_fmm <- seq(0, 1, length.out = m_fmm)

  set.seed(123)
  curves_fmm <- matrix(0, nrow = n_total, ncol = m_fmm)
  subject_ids <- integer(n_total)
  x_cov <- numeric(n_total)

  for (s in 1:n_subj) {
    random_effect <- rnorm(1, 0, 0.5)  # subject-level random intercept in curve space
    for (v in 1:n_visits) {
      idx <- (s - 1) * n_visits + v
      subject_ids[idx] <- s
      x_cov[idx] <- (s - 1) / (n_subj - 1)
      # Curve = mean + covariate effect + random effect + noise
      curves_fmm[idx, ] <- sin(2 * pi * t_fmm) +
        x_cov[idx] * 0.5 * cos(2 * pi * t_fmm) +
        random_effect * rep(1, m_fmm) +
        rnorm(m_fmm, 0, 0.05)
    }
  }

  # FPCA
  fd_fmm <- fdata(curves_fmm, argvals = t_fmm)
  pca_fmm <- fdata2pc(fd_fmm, ncomp = 3)
  scores_fmm <- pca_fmm$x[, 1:3]

  # Fit scalar mixed models per FPC component
  gamma_estimates <- matrix(0, nrow = 1, ncol = 3)  # 1 covariate x 3 components
  sigma2_u <- numeric(3)
  sigma2_eps <- numeric(3)

  for (k in 1:3) {
    df_k <- data.frame(
      score = scores_fmm[, k],
      x = x_cov,
      subject = factor(subject_ids)
    )
    fit_k <- lmer(score ~ x + (1 | subject), data = df_k)
    gamma_estimates[1, k] <- fixef(fit_k)["x"]
    vc <- as.data.frame(VarCorr(fit_k))
    sigma2_u[k] <- vc$vcov[vc$grp == "subject"]
    sigma2_eps[k] <- vc$vcov[vc$grp == "Residual"]
  }

  message(sprintf("    Fixed effect estimates (3 components): %.4f, %.4f, %.4f",
                  gamma_estimates[1,1], gamma_estimates[1,2], gamma_estimates[1,3]))
  message(sprintf("    Random effect variances: %.4f, %.4f, %.4f",
                  sigma2_u[1], sigma2_u[2], sigma2_u[3]))

  list(
    gamma_estimates = as.numeric(gamma_estimates),
    sigma2_u = sigma2_u,
    sigma2_eps = sigma2_eps,
    n_subjects = n_subj,
    n_visits = n_visits,
    m = m_fmm,
    subject_ids = subject_ids,
    x_covariate = x_cov,
    data = as.vector(curves_fmm),  # column-major
    argvals = t_fmm
  )
}, error = function(e) {
  warning(sprintf("FAMM failed: %s", conditionMessage(e)))
  NULL
})

# ── Save results ────────────────────────────────────────────────────────────
message("\n  Saving expected new module validation values...")
save_expected(results, "new_modules_expected")

computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== New module validation complete: %d/%d computations succeeded ===", computed, total))
