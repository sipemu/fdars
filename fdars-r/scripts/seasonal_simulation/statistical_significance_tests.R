#!/usr/bin/env Rscript
# Statistical Significance Tests for Seasonality Detection Methods
#
# This script performs pairwise statistical comparisons between detection methods
# using McNemar's test to determine if performance differences are significant.
#
# McNemar's test compares paired binary classifications:
#                    Method B
#                  Correct  Wrong
# Method A Correct    a       b
#         Wrong       c       d
#
# H0: Methods have equal performance (b = c)
# Test statistic: chi^2 = (b-c)^2 / (b+c)

library(fdars)
library(ggplot2)
library(dplyr)
library(tidyr)

# --- Load existing results ---
results_file <- "seasonality_detection_results.rds"

if (!file.exists(results_file)) {
  stop(paste0(
    "Results file '", results_file, "' not found.\n",
    "Please run 'seasonality_detection_comparison.R' first to generate the results."
  ))
}

results <- readRDS(results_file)
cat("=== Statistical Significance Testing ===\n\n")
cat(sprintf("Loaded %d observations from '%s'\n", nrow(results), results_file))

# Ensure ground truth exists
if (!"ground_truth" %in% names(results)) {
  results$ground_truth <- results$strength >= 0.2
}

# --- Define method columns ---
method_cols <- list(
  "AIC" = "aic_detected",
  "FFT" = "fft_detected",
  "ACF" = "acf_detected",
  "Variance" = "var_detected",
  "Spectral" = "spec_detected",
  "Wavelet" = "wav_detected",
  "SAZED" = "sazed_detected",
  "Autoperiod" = "autoperiod_detected",
  "CFD" = "cfd_detected"
)

# --- Helper Functions ---

#' Build 2x2 contingency table for McNemar's test
#'
#' @param method_a Logical vector of Method A's predictions being correct
#' @param method_b Logical vector of Method B's predictions being correct
#' @return A 2x2 matrix suitable for mcnemar.test()
build_contingency_table <- function(method_a_correct, method_b_correct) {
  # Remove NAs
  valid <- !is.na(method_a_correct) & !is.na(method_b_correct)
  a_correct <- method_a_correct[valid]
  b_correct <- method_b_correct[valid]

  # Build contingency table
  #                    Method B
  #                  Correct  Wrong
  # Method A Correct    a       b    (discordant: A correct, B wrong)
  #         Wrong       c       d    (discordant: A wrong, B correct)

  a <- sum(a_correct & b_correct)      # Both correct
  b <- sum(a_correct & !b_correct)     # A correct, B wrong
  c <- sum(!a_correct & b_correct)     # A wrong, B correct
  d <- sum(!a_correct & !b_correct)    # Both wrong

  matrix(c(a, c, b, d), nrow = 2,
         dimnames = list("Method A" = c("Correct", "Wrong"),
                        "Method B" = c("Correct", "Wrong")))
}

#' Compute correctness vector (detection matches ground truth)
compute_correctness <- function(detected, ground_truth) {
  detected == ground_truth
}

#' Run McNemar's test for a pair of methods
#'
#' @param results Data frame with detection results
#' @param method_a_col Column name for Method A's detected values
#' @param method_b_col Column name for Method B's detected values
#' @param ground_truth Ground truth vector
#' @return List with test results
mcnemar_test_pair <- function(results, method_a_col, method_b_col, ground_truth) {
  # Get correctness vectors
  a_correct <- compute_correctness(results[[method_a_col]], ground_truth)
  b_correct <- compute_correctness(results[[method_b_col]], ground_truth)

  # Build contingency table
  cont_table <- build_contingency_table(a_correct, b_correct)

  # Get discordant counts
  b <- cont_table[1, 2]  # A correct, B wrong
  c <- cont_table[2, 1]  # A wrong, B correct

  # Run McNemar's test
  # Use exact test for small counts, chi-squared otherwise
  if (b + c < 25) {
    test_result <- tryCatch({
      mcnemar.test(cont_table, correct = FALSE)
    }, error = function(e) {
      # If standard test fails, try exact binomial test
      if (b + c > 0) {
        binom.test(b, b + c, p = 0.5)
      } else {
        list(statistic = NA, p.value = 1, method = "No discordant pairs")
      }
    })
  } else {
    test_result <- mcnemar.test(cont_table, correct = TRUE)
  }

  list(
    contingency_table = cont_table,
    n_discordant = b + c,
    a_better_count = b,  # Cases where A correct but B wrong
    b_better_count = c,  # Cases where B correct but A wrong
    statistic = if (is.null(test_result$statistic)) NA else test_result$statistic,
    p_value = test_result$p.value,
    method = test_result$method
  )
}

#' Run McNemar's test for all method pairs
#'
#' @param results Data frame with detection results
#' @param method_cols Named list mapping method names to column names
#' @return Data frame with all pairwise comparisons
mcnemar_all_pairs <- function(results, method_cols) {
  ground_truth <- results$ground_truth
  method_names <- names(method_cols)
  n_methods <- length(method_names)

  # Initialize results storage
  comparisons <- data.frame(
    Method_A = character(0),
    Method_B = character(0),
    A_Better = integer(0),
    B_Better = integer(0),
    N_Discordant = integer(0),
    Statistic = numeric(0),
    P_Value = numeric(0),
    stringsAsFactors = FALSE
  )

  # Compare all pairs
  for (i in 1:(n_methods - 1)) {
    for (j in (i + 1):n_methods) {
      method_a <- method_names[i]
      method_b <- method_names[j]

      test_result <- mcnemar_test_pair(
        results,
        method_cols[[method_a]],
        method_cols[[method_b]],
        ground_truth
      )

      comparisons <- rbind(comparisons, data.frame(
        Method_A = method_a,
        Method_B = method_b,
        A_Better = test_result$a_better_count,
        B_Better = test_result$b_better_count,
        N_Discordant = test_result$n_discordant,
        Statistic = test_result$statistic,
        P_Value = test_result$p_value,
        stringsAsFactors = FALSE
      ))
    }
  }

  comparisons
}

#' Adjust p-values for multiple comparisons
#'
#' @param pvalues Vector of p-values
#' @param method Correction method: "bonferroni", "holm", "BH" (Benjamini-Hochberg), "none"
#' @return Adjusted p-values
adjust_pvalues <- function(pvalues, method = "bonferroni") {
  p.adjust(pvalues, method = method)
}

#' Format p-value with significance stars
format_pvalue <- function(p, adjusted_p = NULL) {
  if (is.na(p)) return("NA")

  p_to_use <- if (!is.null(adjusted_p)) adjusted_p else p

  stars <- if (p_to_use < 0.001) "***"
           else if (p_to_use < 0.01) "**"
           else if (p_to_use < 0.05) "*"
           else ""

  if (p < 0.001) {
    sprintf("%.2e%s", p, stars)
  } else {
    sprintf("%.4f%s", p, stars)
  }
}

# --- Run Pairwise Comparisons ---
cat("\n=== McNemar's Test: Pairwise Method Comparisons ===\n\n")

comparisons <- mcnemar_all_pairs(results, method_cols)

# Add adjusted p-values
comparisons$P_Adjusted_Bonf <- adjust_pvalues(comparisons$P_Value, "bonferroni")
comparisons$P_Adjusted_BH <- adjust_pvalues(comparisons$P_Value, "BH")

# Add significance indicators
comparisons$Significant_Raw <- comparisons$P_Value < 0.05
comparisons$Significant_Bonf <- comparisons$P_Adjusted_Bonf < 0.05
comparisons$Significant_BH <- comparisons$P_Adjusted_BH < 0.05

# Print results table
cat(sprintf("%-12s %-12s %8s %8s %8s %12s %12s %12s\n",
            "Method A", "Method B", "A>B", "B>A", "Discord", "Raw p", "Bonf. p", "BH p"))
cat(paste(rep("-", 96), collapse = ""), "\n")

for (i in 1:nrow(comparisons)) {
  r <- comparisons[i, ]
  cat(sprintf("%-12s %-12s %8d %8d %8d %12s %12s %12s\n",
              r$Method_A, r$Method_B,
              r$A_Better, r$B_Better, r$N_Discordant,
              format_pvalue(r$P_Value),
              format_pvalue(r$P_Adjusted_Bonf),
              format_pvalue(r$P_Adjusted_BH)))
}

cat(paste(rep("-", 96), collapse = ""), "\n")
cat("Significance: * p<0.05, ** p<0.01, *** p<0.001 (based on adjusted p-value)\n\n")

# --- Summary Statistics ---
cat("=== Summary ===\n\n")
n_comparisons <- nrow(comparisons)
n_sig_raw <- sum(comparisons$Significant_Raw, na.rm = TRUE)
n_sig_bonf <- sum(comparisons$Significant_Bonf, na.rm = TRUE)
n_sig_bh <- sum(comparisons$Significant_BH, na.rm = TRUE)

cat(sprintf("Total pairwise comparisons: %d\n", n_comparisons))
cat(sprintf("Significant at alpha=0.05 (raw):        %d (%.1f%%)\n",
            n_sig_raw, 100 * n_sig_raw / n_comparisons))
cat(sprintf("Significant at alpha=0.05 (Bonferroni): %d (%.1f%%)\n",
            n_sig_bonf, 100 * n_sig_bonf / n_comparisons))
cat(sprintf("Significant at alpha=0.05 (BH/FDR):     %d (%.1f%%)\n",
            n_sig_bh, 100 * n_sig_bh / n_comparisons))

# --- Create P-Value Heatmap ---
cat("\n=== Generating P-Value Heatmap ===\n")

# Create symmetric matrix for heatmap
method_names <- names(method_cols)
n_methods <- length(method_names)

p_matrix <- matrix(1, nrow = n_methods, ncol = n_methods,
                   dimnames = list(method_names, method_names))
diff_matrix <- matrix(0, nrow = n_methods, ncol = n_methods,
                      dimnames = list(method_names, method_names))

for (i in 1:nrow(comparisons)) {
  r <- comparisons[i, ]
  idx_a <- which(method_names == r$Method_A)
  idx_b <- which(method_names == r$Method_B)

  # Use Bonferroni-adjusted p-value
  p_matrix[idx_a, idx_b] <- r$P_Adjusted_Bonf
  p_matrix[idx_b, idx_a] <- r$P_Adjusted_Bonf

  # Store difference direction (positive = row method better)
  diff_matrix[idx_a, idx_b] <- r$A_Better - r$B_Better
  diff_matrix[idx_b, idx_a] <- r$B_Better - r$A_Better
}

# Convert to long format for ggplot
p_df <- as.data.frame(as.table(p_matrix))
names(p_df) <- c("Method_A", "Method_B", "P_Value")

diff_df <- as.data.frame(as.table(diff_matrix))
names(diff_df) <- c("Method_A", "Method_B", "Difference")

heatmap_df <- merge(p_df, diff_df)
heatmap_df$Significant <- heatmap_df$P_Value < 0.05
heatmap_df$Log_P <- -log10(heatmap_df$P_Value + 1e-10)

# Order methods by F1 score (best to worst)
method_order <- c("Wavelet", "Variance", "SAZED", "Spectral", "Autoperiod",
                  "FFT", "CFD", "AIC", "ACF")
heatmap_df$Method_A <- factor(heatmap_df$Method_A, levels = method_order)
heatmap_df$Method_B <- factor(heatmap_df$Method_B, levels = method_order)

# Create heatmap
p_heatmap <- ggplot(heatmap_df, aes(x = Method_B, y = Method_A)) +
  geom_tile(aes(fill = Log_P), color = "white") +
  geom_text(aes(label = ifelse(Significant & Method_A != Method_B, "*", "")),
            color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient2(
    low = "white", mid = "steelblue", high = "darkblue",
    midpoint = 1.3,  # -log10(0.05)
    name = "-log10(p)",
    limits = c(0, 5)
  ) +
  labs(
    title = "McNemar's Test: Pairwise Method Comparisons",
    subtitle = "Bonferroni-adjusted p-values (* indicates p < 0.05)",
    x = "", y = ""
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  coord_fixed()

# Save heatmap
ggsave("plots/mcnemar_pvalue_heatmap.pdf", p_heatmap, width = 8, height = 7)
cat("Saved: plots/mcnemar_pvalue_heatmap.pdf\n")

# --- Create Difference Direction Heatmap ---
# Shows which method is better in each pair

# Only show upper triangle (avoid redundancy)
heatmap_df_upper <- heatmap_df %>%
  filter(as.numeric(Method_A) < as.numeric(Method_B))

p_diff <- ggplot(heatmap_df, aes(x = Method_B, y = Method_A)) +
  geom_tile(aes(fill = Difference), color = "white") +
  geom_text(aes(label = ifelse(Method_A != Method_B,
                               ifelse(Significant, sprintf("%+d*", Difference),
                                      sprintf("%+d", Difference)), "")),
            size = 3) +
  scale_fill_gradient2(
    low = "firebrick", mid = "white", high = "forestgreen",
    midpoint = 0,
    name = "Difference\n(Row - Col)",
    limits = c(-50, 50)
  ) +
  labs(
    title = "Method Performance Differences",
    subtitle = "Positive = row method better; * indicates significant (Bonferroni p < 0.05)",
    x = "", y = ""
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  coord_fixed()

ggsave("plots/mcnemar_difference_heatmap.pdf", p_diff, width = 8, height = 7)
cat("Saved: plots/mcnemar_difference_heatmap.pdf\n")

# --- Key Findings ---
cat("\n=== Key Findings ===\n\n")

# Find significant differences involving top methods
top_methods <- c("Wavelet", "Variance", "SAZED")
top_comparisons <- comparisons %>%
  filter(Method_A %in% top_methods | Method_B %in% top_methods) %>%
  arrange(P_Adjusted_Bonf)

cat("Comparisons involving top-3 methods (Wavelet, Variance, SAZED):\n\n")

for (i in 1:min(10, nrow(top_comparisons))) {
  r <- top_comparisons[i, ]
  winner <- if (r$A_Better > r$B_Better) r$Method_A else r$Method_B
  loser <- if (r$A_Better > r$B_Better) r$Method_B else r$Method_A
  diff <- abs(r$A_Better - r$B_Better)
  sig <- if (r$Significant_Bonf) " (SIGNIFICANT)" else ""

  cat(sprintf("%d. %s vs %s: %s better by %d cases, p=%.4f%s\n",
              i, r$Method_A, r$Method_B, winner, diff, r$P_Adjusted_Bonf, sig))
}

# --- Wavelet vs Variance (the key comparison from the report) ---
cat("\n=== Focus: Wavelet (97.8% F1) vs Variance (97.3% F1) ===\n\n")

wv_comparison <- comparisons %>%
  filter((Method_A == "Wavelet" & Method_B == "Variance") |
         (Method_A == "Variance" & Method_B == "Wavelet"))

if (nrow(wv_comparison) > 0) {
  r <- wv_comparison[1, ]
  cat(sprintf("Wavelet better in: %d cases\n",
              if (r$Method_A == "Wavelet") r$A_Better else r$B_Better))
  cat(sprintf("Variance better in: %d cases\n",
              if (r$Method_A == "Variance") r$A_Better else r$B_Better))
  cat(sprintf("Both agree: %d cases\n", nrow(results) - r$N_Discordant))
  cat(sprintf("Raw p-value: %.4f\n", r$P_Value))
  cat(sprintf("Bonferroni p-value: %.4f\n", r$P_Adjusted_Bonf))

  if (r$Significant_Bonf) {
    cat("\nConclusion: The difference IS statistically significant.\n")
  } else {
    cat("\nConclusion: The difference is NOT statistically significant.\n")
    cat("The 0.5% F1 difference (97.8% vs 97.3%) may be due to random variation.\n")
  }
}

# --- Save Results ---
saveRDS(comparisons, "statistical_significance_results.rds")
cat("\nSaved: statistical_significance_results.rds\n")

# Create summary table for report
summary_table <- comparisons %>%
  select(Method_A, Method_B, A_Better, B_Better, P_Value, P_Adjusted_Bonf, Significant_Bonf) %>%
  mutate(
    Winner = ifelse(A_Better > B_Better, Method_A,
                   ifelse(B_Better > A_Better, Method_B, "Tie")),
    Margin = abs(A_Better - B_Better)
  ) %>%
  arrange(P_Adjusted_Bonf)

write.csv(summary_table, "statistical_significance_summary.csv", row.names = FALSE)
cat("Saved: statistical_significance_summary.csv\n")

cat("\n=== Statistical Significance Testing Complete ===\n")
