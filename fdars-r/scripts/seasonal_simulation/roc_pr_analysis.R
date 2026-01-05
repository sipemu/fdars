#!/usr/bin/env Rscript
# ROC and Precision-Recall Curve Analysis
#
# This script generates publication-quality ROC and PR curves for the
# seasonality detection method comparison, including:
# - ROC curves with AUC values
# - PR curves with AUC-PR values
# - Operating point markers at default thresholds
# - Combined comparison plots

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# --- Load existing results ---
results_file <- "seasonality_detection_results.rds"

if (!file.exists(results_file)) {
  stop(paste0(
    "Results file '", results_file, "' not found.\n",
    "Please run 'seasonality_detection_comparison.R' first to generate the results."
  ))
}

results <- readRDS(results_file)
cat("=== ROC and PR Curve Analysis ===\n\n")
cat(sprintf("Loaded %d observations from '%s'\n", nrow(results), results_file))

# Ensure ground truth exists
if (!"ground_truth" %in% names(results)) {
  results$ground_truth <- results$strength >= 0.2
}

# --- Detection thresholds (for operating points) ---
detection_thresholds <- list(
  aic_score = 0,
  fft_score = 6.0,
  acf_score = 0.25,
  var_score = 0.2,
  spec_score = 0.3,
  wav_score = 0.26,
  sazed_score = 2,
  autoperiod_score = 0.3,
  cfd_score = 0.25
)

# --- Method configuration ---
method_config <- list(
  list(col = "var_score", name = "Variance", color = "#E41A1C"),
  list(col = "wav_score", name = "Wavelet", color = "#377EB8"),
  list(col = "sazed_score", name = "SAZED", color = "#4DAF4A"),
  list(col = "spec_score", name = "Spectral", color = "#984EA3"),
  list(col = "autoperiod_score", name = "Autoperiod", color = "#FF7F00"),
  list(col = "fft_score", name = "FFT", color = "#FFFF33"),
  list(col = "cfd_score", name = "CFD", color = "#A65628"),
  list(col = "aic_score", name = "AIC", color = "#F781BF"),
  list(col = "acf_score", name = "ACF", color = "#999999")
)

# Top 5 methods for focused analysis
top_methods <- c("Wavelet", "Variance", "SAZED", "Spectral", "Autoperiod")

# Color palette
method_colors <- setNames(
  sapply(method_config, function(x) x$color),
  sapply(method_config, function(x) x$name)
)

# --- Curve computation functions ---

#' Compute ROC curve for a method
#'
#' @param scores Continuous scores from the method
#' @param ground_truth Boolean vector of true labels
#' @param method_name Name of the method for labeling
#' @param n_points Number of threshold points
#' @return Data frame with TPR, FPR, threshold, Method
compute_roc_curve <- function(scores, ground_truth, method_name, n_points = 200) {
  valid <- !is.na(scores) & !is.na(ground_truth)
  scores <- scores[valid]
  ground_truth <- ground_truth[valid]

  if (length(unique(scores)) < 2) return(NULL)

  # Use quantiles for more uniform coverage
  thresholds <- quantile(scores, probs = seq(0, 1, length.out = n_points), na.rm = TRUE)
  thresholds <- unique(thresholds)

  roc_data <- lapply(thresholds, function(thresh) {
    detected <- scores > thresh

    tp <- sum(detected & ground_truth)
    fp <- sum(detected & !ground_truth)
    tn <- sum(!detected & !ground_truth)
    fn <- sum(!detected & ground_truth)

    tpr <- ifelse(tp + fn > 0, tp / (tp + fn), 0)  # Sensitivity/Recall
    fpr <- ifelse(fp + tn > 0, fp / (fp + tn), 0)  # 1 - Specificity

    data.frame(threshold = thresh, TPR = tpr, FPR = fpr)
  })

  roc_df <- do.call(rbind, roc_data)
  roc_df <- roc_df[order(roc_df$FPR, roc_df$TPR), ]
  roc_df$Method <- method_name
  return(roc_df)
}

#' Compute PR curve for a method
compute_pr_curve <- function(scores, ground_truth, method_name, n_points = 200) {
  valid <- !is.na(scores) & !is.na(ground_truth)
  scores <- scores[valid]
  ground_truth <- ground_truth[valid]

  if (length(unique(scores)) < 2) return(NULL)

  thresholds <- quantile(scores, probs = seq(0, 1, length.out = n_points), na.rm = TRUE)
  thresholds <- unique(thresholds)

  pr_data <- lapply(thresholds, function(thresh) {
    detected <- scores > thresh

    tp <- sum(detected & ground_truth)
    fp <- sum(detected & !ground_truth)
    fn <- sum(!detected & ground_truth)

    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 1)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)

    data.frame(threshold = thresh, Precision = precision, Recall = recall)
  })

  pr_df <- do.call(rbind, pr_data)
  pr_df$Method <- method_name
  return(pr_df)
}

#' Compute AUC using trapezoidal rule
compute_auc <- function(x, y) {
  # Sort by x
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]

  # Remove duplicates
  dup <- duplicated(x)
  x <- x[!dup]
  y <- y[!dup]

  if (length(x) < 2) return(NA)

  # Trapezoidal integration
  sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

#' Compute operating point (TPR, FPR, Precision, Recall at default threshold)
compute_operating_point <- function(scores, ground_truth, threshold) {
  valid <- !is.na(scores) & !is.na(ground_truth)
  scores <- scores[valid]
  ground_truth <- ground_truth[valid]

  detected <- scores > threshold

  tp <- sum(detected & ground_truth)
  fp <- sum(detected & !ground_truth)
  tn <- sum(!detected & !ground_truth)
  fn <- sum(!detected & ground_truth)

  tpr <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  fpr <- ifelse(fp + tn > 0, fp / (fp + tn), 0)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 1)
  recall <- tpr

  data.frame(TPR = tpr, FPR = fpr, Precision = precision, Recall = recall)
}

# --- Compute curves for all methods ---
cat("\nComputing ROC and PR curves...\n")

roc_curves <- list()
pr_curves <- list()
auc_roc <- list()
auc_pr <- list()
operating_points <- list()

for (cfg in method_config) {
  cat(sprintf("  Processing %s...\n", cfg$name))

  # ROC curve
  roc_df <- compute_roc_curve(results[[cfg$col]], results$ground_truth, cfg$name)
  if (!is.null(roc_df)) {
    roc_curves[[cfg$name]] <- roc_df
    auc_roc[[cfg$name]] <- compute_auc(roc_df$FPR, roc_df$TPR)
  }

  # PR curve
  pr_df <- compute_pr_curve(results[[cfg$col]], results$ground_truth, cfg$name)
  if (!is.null(pr_df)) {
    pr_curves[[cfg$name]] <- pr_df
    auc_pr[[cfg$name]] <- compute_auc(pr_df$Recall, pr_df$Precision)
  }

  # Operating point
  op <- compute_operating_point(
    results[[cfg$col]],
    results$ground_truth,
    detection_thresholds[[cfg$col]]
  )
  op$Method <- cfg$name
  operating_points[[cfg$name]] <- op
}

# Combine into data frames
roc_all <- do.call(rbind, roc_curves)
pr_all <- do.call(rbind, pr_curves)
op_all <- do.call(rbind, operating_points)

# Create AUC summary
auc_summary <- data.frame(
  Method = names(auc_roc),
  AUC_ROC = unlist(auc_roc),
  AUC_PR = unlist(auc_pr[names(auc_roc)])
)
auc_summary <- auc_summary[order(-auc_summary$AUC_ROC), ]
rownames(auc_summary) <- NULL

cat("\n=== AUC Summary ===\n\n")
print(auc_summary, row.names = FALSE)

# --- Set factor levels (ordered by AUC-ROC) ---
method_order <- auc_summary$Method
roc_all$Method <- factor(roc_all$Method, levels = method_order)
pr_all$Method <- factor(pr_all$Method, levels = method_order)
op_all$Method <- factor(op_all$Method, levels = method_order)

# --- Plot 1: Full ROC Curves (all methods) ---
p_roc_all <- ggplot(roc_all, aes(x = FPR, y = TPR, color = Method)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(data = op_all, aes(x = FPR, y = TPR, color = Method),
             size = 4, shape = 18) +
  scale_color_manual(values = method_colors) +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "ROC Curves: All Methods",
    subtitle = "Diamonds indicate default threshold operating points",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

# --- Plot 2: ROC Curves (top 5 methods only) ---
roc_top <- roc_all %>% filter(Method %in% top_methods)
op_top <- op_all %>% filter(Method %in% top_methods)
auc_top <- auc_summary %>% filter(Method %in% top_methods)

p_roc_top <- ggplot(roc_top, aes(x = FPR, y = TPR, color = Method)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(data = op_top, aes(x = FPR, y = TPR, color = Method),
             size = 5, shape = 18) +
  scale_color_manual(values = method_colors) +
  coord_fixed(xlim = c(0, 0.3), ylim = c(0.7, 1)) +
  labs(
    title = "ROC Curves: Top 5 Methods",
    subtitle = "Zoomed to region of interest (FPR < 0.3, TPR > 0.7)",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  # Add AUC annotations
  annotate("text", x = 0.02, y = 0.73, label = "AUC-ROC:", fontface = "bold", hjust = 0, size = 3.5) +
  geom_text(data = auc_top,
            aes(x = 0.02, y = seq(0.70, 0.70 - 0.03 * (nrow(auc_top) - 1), by = -0.03),
                label = sprintf("%s: %.3f", Method, AUC_ROC),
                color = Method),
            hjust = 0, size = 3, show.legend = FALSE) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

# --- Plot 3: Full PR Curves (all methods) ---
# Calculate baseline (random classifier precision)
baseline_precision <- sum(results$ground_truth) / nrow(results)

p_pr_all <- ggplot(pr_all, aes(x = Recall, y = Precision, color = Method)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = baseline_precision, linetype = "dashed", color = "gray50") +
  geom_point(data = op_all, aes(x = Recall, y = Precision, color = Method),
             size = 4, shape = 18) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Precision-Recall Curves: All Methods",
    subtitle = sprintf("Dashed line = baseline precision (%.2f)", baseline_precision),
    x = "Recall (Sensitivity)",
    y = "Precision"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

# --- Plot 4: PR Curves (top 5 methods, zoomed) ---
pr_top <- pr_all %>% filter(Method %in% top_methods)
auc_pr_top <- auc_summary %>% filter(Method %in% top_methods)

p_pr_top <- ggplot(pr_top, aes(x = Recall, y = Precision, color = Method)) +
  geom_line(linewidth = 1.2) +
  geom_point(data = op_top, aes(x = Recall, y = Precision, color = Method),
             size = 5, shape = 18) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(0.7, 1), ylim = c(0.85, 1)) +
  labs(
    title = "Precision-Recall Curves: Top 5 Methods",
    subtitle = "Zoomed to region of interest (Recall > 0.7, Precision > 0.85)",
    x = "Recall",
    y = "Precision"
  ) +
  annotate("text", x = 0.72, y = 0.88, label = "AUC-PR:", fontface = "bold", hjust = 0, size = 3.5) +
  geom_text(data = auc_pr_top,
            aes(x = 0.72, y = seq(0.865, 0.865 - 0.015 * (nrow(auc_pr_top) - 1), by = -0.015),
                label = sprintf("%s: %.3f", Method, AUC_PR),
                color = Method),
            hjust = 0, size = 3, show.legend = FALSE) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

# --- Plot 5: Operating Points Scatter ---
p_operating <- ggplot(op_all, aes(x = FPR, y = TPR, color = Method)) +
  geom_point(size = 6) +
  geom_text(aes(label = Method), vjust = -1.5, hjust = 0.5, size = 3) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(xlim = c(-0.02, 0.25), ylim = c(0.7, 1.02)) +
  labs(
    title = "Operating Points at Default Thresholds",
    subtitle = "Trade-off between True Positive Rate and False Positive Rate",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 9)
  )

# --- Save individual plots ---
cat("\n=== Saving Plots ===\n")

ggsave("plots/roc_curves_all.pdf", p_roc_all, width = 10, height = 8)
cat("Saved: plots/roc_curves_all.pdf\n")

ggsave("plots/roc_curves_top5.pdf", p_roc_top, width = 10, height = 8)
cat("Saved: plots/roc_curves_top5.pdf\n")

ggsave("plots/pr_curves_all.pdf", p_pr_all, width = 10, height = 8)
cat("Saved: plots/pr_curves_all.pdf\n")

ggsave("plots/pr_curves_top5.pdf", p_pr_top, width = 10, height = 8)
cat("Saved: plots/pr_curves_top5.pdf\n")

ggsave("plots/operating_points.pdf", p_operating, width = 8, height = 6)
cat("Saved: plots/operating_points.pdf\n")

# --- Combined comparison figure ---
p_combined <- grid.arrange(
  p_roc_top + theme(legend.position = "none"),
  p_pr_top + theme(legend.position = "none"),
  p_operating,
  ncol = 2, nrow = 2,
  layout_matrix = rbind(c(1, 2), c(3, 3)),
  heights = c(1, 0.8)
)

ggsave("plots/roc_pr_combined.pdf", p_combined, width = 14, height = 12)
cat("Saved: plots/roc_pr_combined.pdf\n")

# --- Save AUC summary ---
write.csv(auc_summary, "roc_pr_auc_summary.csv", row.names = FALSE)
cat("Saved: roc_pr_auc_summary.csv\n")

# --- Summary statistics ---
cat("\n=== Summary ===\n\n")

cat("Top 5 Methods by AUC-ROC:\n")
for (i in 1:5) {
  r <- auc_summary[i, ]
  cat(sprintf("  %d. %s: AUC-ROC = %.4f, AUC-PR = %.4f\n",
              i, r$Method, r$AUC_ROC, r$AUC_PR))
}

cat("\nOperating Points (default thresholds):\n")
cat(sprintf("%-12s %8s %8s %10s %8s\n", "Method", "TPR", "FPR", "Precision", "Recall"))
cat(paste(rep("-", 50), collapse = ""), "\n")
for (method in method_order) {
  r <- op_all[op_all$Method == method, ]
  cat(sprintf("%-12s %7.1f%% %7.1f%% %9.1f%% %7.1f%%\n",
              method, r$TPR * 100, r$FPR * 100, r$Precision * 100, r$Recall * 100))
}

# --- Key Findings ---
cat("\n=== Key Findings ===\n\n")

# Best by AUC-ROC
best_roc <- auc_summary[1, ]
cat(sprintf("Best AUC-ROC: %s (%.4f)\n", best_roc$Method, best_roc$AUC_ROC))

# Best by AUC-PR
best_pr_idx <- which.max(auc_summary$AUC_PR)
best_pr <- auc_summary[best_pr_idx, ]
cat(sprintf("Best AUC-PR: %s (%.4f)\n", best_pr$Method, best_pr$AUC_PR))

# Find method with lowest FPR at TPR >= 0.95
high_tpr_ops <- op_all[op_all$TPR >= 0.95, ]
if (nrow(high_tpr_ops) > 0) {
  best_low_fpr <- high_tpr_ops[which.min(high_tpr_ops$FPR), ]
  cat(sprintf("Lowest FPR at TPR >= 95%%: %s (FPR = %.1f%%, TPR = %.1f%%)\n",
              best_low_fpr$Method, best_low_fpr$FPR * 100, best_low_fpr$TPR * 100))
}

# Find method with highest precision at recall >= 0.90
high_recall_ops <- op_all[op_all$Recall >= 0.90, ]
if (nrow(high_recall_ops) > 0) {
  best_prec <- high_recall_ops[which.max(high_recall_ops$Precision), ]
  cat(sprintf("Highest Precision at Recall >= 90%%: %s (Precision = %.1f%%, Recall = %.1f%%)\n",
              best_prec$Method, best_prec$Precision * 100, best_prec$Recall * 100))
}

cat("\n=== ROC and PR Analysis Complete ===\n")
