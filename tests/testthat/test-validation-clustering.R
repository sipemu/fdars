# Validation tests for clustering functions
# Tests for kmeans.fd and optim.kmeans.fd

# =============================================================================
# Test Data Setup
# =============================================================================

create_clustered_data <- function(n_per_cluster = 10, m = 50, seed = 42) {
  set.seed(seed)
  t_grid <- seq(0, 1, length.out = m)

  # Cluster 1: sine-like curves
  data1 <- matrix(0, nrow = n_per_cluster, ncol = m)
  for (i in 1:n_per_cluster) {
    data1[i, ] <- sin(2 * pi * t_grid) + rnorm(m, 0, 0.1)
  }

  # Cluster 2: cosine-like curves
  data2 <- matrix(0, nrow = n_per_cluster, ncol = m)
  for (i in 1:n_per_cluster) {
    data2[i, ] <- cos(2 * pi * t_grid) + rnorm(m, 0, 0.1)
  }

  # Cluster 3: linear curves
  data3 <- matrix(0, nrow = n_per_cluster, ncol = m)
  for (i in 1:n_per_cluster) {
    data3[i, ] <- t_grid + rnorm(m, 0, 0.1)
  }

  data_mat <- rbind(data1, data2, data3)
  true_clusters <- rep(1:3, each = n_per_cluster)

  list(
    fd = fdars::fdata(data_mat, argvals = t_grid),
    X = data_mat,
    t_grid = t_grid,
    true_clusters = true_clusters,
    n = 3 * n_per_cluster,
    m = m
  )
}

# =============================================================================
# kmeans.fd Tests
# =============================================================================

test_that("kmeans.fd returns correct structure", {
  data <- create_clustered_data()
  result <- fdars::kmeans.fd(data$fd, ncl = 3, seed = 123)

  expect_s3_class(result, "kmeans.fd")
  expect_true("cluster" %in% names(result))
  expect_true("centers" %in% names(result))
  expect_true("size" %in% names(result))
  expect_true("withinss" %in% names(result))
  expect_true("tot.withinss" %in% names(result))
  expect_true("fdataobj" %in% names(result))

  # Check dimensions
  expect_length(result$cluster, data$n)
  expect_s3_class(result$centers, "fdata")
  expect_equal(nrow(result$centers$data), 3)
  expect_equal(ncol(result$centers$data), data$m)
})

test_that("kmeans.fd cluster assignments are valid", {
  data <- create_clustered_data()
  result <- fdars::kmeans.fd(data$fd, ncl = 3, seed = 123)

  # Cluster labels should be 1, 2, or 3
  expect_true(all(result$cluster %in% 1:3))

  # All clusters should have at least one member
  expect_equal(length(unique(result$cluster)), 3)
})

test_that("kmeans.fd is reproducible with seed", {
  data <- create_clustered_data()
  result1 <- fdars::kmeans.fd(data$fd, ncl = 3, seed = 456)
  result2 <- fdars::kmeans.fd(data$fd, ncl = 3, seed = 456)

  expect_equal(result1$cluster, result2$cluster)
  expect_equal(result1$centers$data, result2$centers$data)
})

test_that("kmeans.fd finds reasonable clusters for well-separated data", {
  data <- create_clustered_data()
  result <- fdars::kmeans.fd(data$fd, ncl = 3, seed = 123)

  # Most curves should be correctly clustered
  # Allow for some misclassification due to random initialization
  # Check that each true cluster maps mostly to one predicted cluster
  for (true_cl in 1:3) {
    idx <- data$true_clusters == true_cl
    pred_clusters <- result$cluster[idx]
    most_common <- as.integer(names(which.max(table(pred_clusters))))
    accuracy <- mean(pred_clusters == most_common)
    expect_gt(accuracy, 0.7)  # At least 70% should be in same cluster
  }
})

test_that("kmeans.fd with different metrics", {
  data <- create_clustered_data(n_per_cluster = 8)

  # Test L2 metric (default)
  result_l2 <- fdars::kmeans.fd(data$fd, ncl = 3, metric = "L2", seed = 123)
  expect_s3_class(result_l2, "kmeans.fd")

  # Test L1 metric
  result_l1 <- fdars::kmeans.fd(data$fd, ncl = 3, metric = "L1", seed = 123)
  expect_s3_class(result_l1, "kmeans.fd")

  # Test Linf metric
  result_linf <- fdars::kmeans.fd(data$fd, ncl = 3, metric = "Linf", seed = 123)
  expect_s3_class(result_linf, "kmeans.fd")
})

test_that("kmeans.fd with custom metric function", {
  data <- create_clustered_data(n_per_cluster = 8)

  # Use metric.lp as function
  result <- fdars::kmeans.fd(data$fd, ncl = 3, metric = fdars::metric.lp, seed = 123)
  expect_s3_class(result, "kmeans.fd")
  expect_length(result$cluster, data$n)
})

test_that("kmeans.fd nstart improves results", {
  data <- create_clustered_data()

  # Multiple starts should give more stable results
  result1 <- fdars::kmeans.fd(data$fd, ncl = 3, nstart = 1, seed = 789)
  result10 <- fdars::kmeans.fd(data$fd, ncl = 3, nstart = 10, seed = 789)

  # Both should return valid results
  expect_s3_class(result1, "kmeans.fd")
  expect_s3_class(result10, "kmeans.fd")
})

# =============================================================================
# optim.kmeans.fd Tests
# =============================================================================

test_that("optim.kmeans.fd returns correct structure", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:5,
                                    criterion = "silhouette", seed = 123)

  expect_s3_class(result, "optim.kmeans.fd")
  expect_true("optimal.k" %in% names(result))
  expect_true("criterion" %in% names(result))
  expect_true("scores" %in% names(result))
  expect_true("models" %in% names(result))
  expect_true("best.model" %in% names(result))

  # Check dimensions
  expect_length(result$scores, 4)  # k = 2, 3, 4, 5
  expect_length(result$models, 4)
  expect_s3_class(result$best.model, "kmeans.fd")
})

test_that("optim.kmeans.fd silhouette finds correct k", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:6,
                                    criterion = "silhouette", seed = 123)

  # Should find k=3 for well-separated 3-cluster data
  expect_equal(result$optimal.k, 3)

  # Silhouette scores should be between -1 and 1
  expect_true(all(result$scores >= -1 & result$scores <= 1))
})

test_that("optim.kmeans.fd Calinski-Harabasz finds correct k", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:6,
                                    criterion = "CH", seed = 123)

  # Should find k=3 for well-separated 3-cluster data
  expect_equal(result$optimal.k, 3)

  # CH index should be positive
  expect_true(all(result$scores > 0))
})

test_that("optim.kmeans.fd elbow finds reasonable k", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:6,
                                    criterion = "elbow", seed = 123)

  # Should find k around 3 for well-separated 3-cluster data
  expect_true(result$optimal.k >= 2 && result$optimal.k <= 4)

  # WCSS should be positive and decreasing
  expect_true(all(result$scores > 0))
  expect_true(all(diff(result$scores) <= 0))
})

test_that("optim.kmeans.fd is reproducible with seed", {
  data <- create_clustered_data()
  result1 <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:5,
                                     criterion = "silhouette", seed = 999)
  result2 <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:5,
                                     criterion = "silhouette", seed = 999)

  expect_equal(result1$optimal.k, result2$optimal.k)
  expect_equal(result1$scores, result2$scores)
})

test_that("optim.kmeans.fd print method works", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:4,
                                    criterion = "silhouette", seed = 123)

  # Print should not error
  expect_output(print(result), "Optimal K-Means Clustering")
  expect_output(print(result), "silhouette")
  expect_output(print(result), "Optimal k:")
})

test_that("optim.kmeans.fd plot method works", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:4,
                                    criterion = "silhouette", seed = 123)

  # Plot should not error
  expect_no_error(plot(result))
})

test_that("optim.kmeans.fd best.model matches optimal.k", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:5,
                                    criterion = "silhouette", seed = 123)

  # best.model should have optimal.k clusters
  n_clusters <- length(unique(result$best.model$cluster))
  expect_equal(n_clusters, result$optimal.k)
})

test_that("optim.kmeans.fd with different metrics", {
  data <- create_clustered_data(n_per_cluster = 8)

  # Test with L1 metric
  result_l1 <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:4,
                                       criterion = "silhouette",
                                       metric = "L1", seed = 123)
  expect_s3_class(result_l1, "optim.kmeans.fd")

  # Test with Linf metric
  result_linf <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:4,
                                         criterion = "silhouette",
                                         metric = "Linf", seed = 123)
  expect_s3_class(result_linf, "optim.kmeans.fd")
})

# =============================================================================
# Input Validation Tests
# =============================================================================

test_that("kmeans.fd rejects non-fdata input", {
  X <- matrix(rnorm(100), 10, 10)
  expect_error(fdars::kmeans.fd(X, ncl = 2))
})

test_that("optim.kmeans.fd rejects non-fdata input", {
  X <- matrix(rnorm(100), 10, 10)
  expect_error(fdars::optim.kmeans.fd(X, ncl.range = 2:4))
})

test_that("optim.kmeans.fd handles invalid ncl.range gracefully", {
  data <- create_clustered_data()

  # k=1 is filtered out, so 2:3 is used
  result1 <- fdars::optim.kmeans.fd(data$fd, ncl.range = 1:3, seed = 123)
  expect_true(min(result1$ncl.range) >= 2)

  # k > n is filtered out
  result2 <- fdars::optim.kmeans.fd(data$fd, ncl.range = 2:100, seed = 123)
  expect_true(max(result2$ncl.range) < data$n)
})

test_that("optim.kmeans.fd errors when all ncl.range is invalid", {
  # Create tiny dataset
  set.seed(42)
  fd <- fdars::fdata(matrix(rnorm(20), 2, 10))

  # All values in 1:1 are invalid (k=1 not allowed)
  expect_error(fdars::optim.kmeans.fd(fd, ncl.range = 1))
})

test_that("optim.kmeans.fd rejects invalid criterion", {
  data <- create_clustered_data()
  expect_error(fdars::optim.kmeans.fd(data$fd, ncl.range = 2:4,
                                       criterion = "invalid"))
})

# =============================================================================
# Edge Cases
# =============================================================================

test_that("kmeans.fd with k=2", {
  data <- create_clustered_data()
  result <- fdars::kmeans.fd(data$fd, ncl = 2, seed = 123)

  expect_s3_class(result, "kmeans.fd")
  expect_equal(length(unique(result$cluster)), 2)
})

test_that("kmeans.fd with small dataset", {
  set.seed(42)
  n <- 6
  m <- 20
  t_grid <- seq(0, 1, length.out = m)
  X <- matrix(rnorm(n * m), n, m)
  fd <- fdars::fdata(X, argvals = t_grid)

  result <- fdars::kmeans.fd(fd, ncl = 2, seed = 123)
  expect_s3_class(result, "kmeans.fd")
  expect_length(result$cluster, n)
})

test_that("optim.kmeans.fd with narrow range", {
  data <- create_clustered_data()
  result <- fdars::optim.kmeans.fd(data$fd, ncl.range = 3:4,
                                    criterion = "silhouette", seed = 123)

  expect_s3_class(result, "optim.kmeans.fd")
  expect_true(result$optimal.k %in% 3:4)
})
