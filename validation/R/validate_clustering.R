#!/usr/bin/env Rscript
# Validate clustering measures against R reference implementations.
# Computes silhouette scores, Calinski-Harabasz index, and within-cluster
# sum of squares on the clusters_60x51 dataset, then saves expected outputs
# for comparison with Rust implementations.
#
# Reference packages: fda.usc, cluster, fpc
# Usage: Rscript validation/R/validate_clustering.R

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "helpers.R"))

library(fda.usc)
library(cluster)
library(fpc)

message("=== Validating clustering measures ===\n")

# Load 3-cluster test data (60 curves, 51 grid points, 3 clusters of 20)
dat <- load_data("clusters_60x51")
n <- dat$n
m <- dat$m
argvals <- dat$argvals
true_labels <- dat$true_labels

# Build fda.usc::fdata object
fdataobj <- to_fdata(dat$data, n, m, argvals)

# Build raw matrix (n x m)
mat <- to_matrix(dat$data, n, m)

results <- list()

# ---- (a) Silhouette scores --------------------------------------------------
message("  Computing L2 distance matrix...")
results$silhouette <- tryCatch({
  dist_mat <- fda.usc::metric.lp(fdataobj, lp = 2)
  dist_obj <- as.dist(dist_mat)

  message("  Computing silhouette widths using true labels...")
  sil <- cluster::silhouette(true_labels, dist_obj)
  sil_widths <- as.numeric(sil[, "sil_width"])
  avg_sil <- mean(sil_widths)

  message(sprintf("    Average silhouette width: %.6f", avg_sil))
  list(
    widths = sil_widths,
    average = avg_sil
  )
}, error = function(e) {
  warning(sprintf("Silhouette computation failed: %s", conditionMessage(e)))
  NULL
})

# ---- (b) Calinski-Harabasz index --------------------------------------------
message("  Computing Calinski-Harabasz index...")
results$calinski_harabasz <- tryCatch({
  ch <- fpc::calinhara(x = mat, clustering = true_labels)
  message(sprintf("    CH index: %.6f", ch))
  ch
}, error = function(e) {
  warning(sprintf("Calinski-Harabasz computation failed: %s", conditionMessage(e)))
  NULL
})

# ---- (c) Within-cluster sum of squares (true labels) -------------------------
message("  Computing within-cluster sum of squares for true labels...")
results$withinss <- tryCatch({
  k <- max(true_labels)
  wcss <- numeric(k)
  for (cl in 1:k) {
    cluster_mat <- mat[true_labels == cl, , drop = FALSE]
    cluster_center <- colMeans(cluster_mat)
    # Sum of squared distances from each point to its cluster center
    diffs <- sweep(cluster_mat, 2, cluster_center)
    wcss[cl] <- sum(diffs^2)
  }
  total_wcss <- sum(wcss)
  message(sprintf("    Total within-cluster SS: %.6f", total_wcss))
  list(
    per_cluster = wcss,
    total = total_wcss
  )
}, error = function(e) {
  warning(sprintf("Within-cluster SS computation failed: %s", conditionMessage(e)))
  NULL
})

# ---- Save results ------------------------------------------------------------
message("\n  Saving expected clustering values...")
save_expected(results, "clustering_expected")

# Summary
computed <- sum(!vapply(results, is.null, logical(1)))
total <- length(results)
message(sprintf("\n=== Clustering validation complete: %d/%d computations succeeded ===", computed, total))
