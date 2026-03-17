//! Elastic clustering: k-means and hierarchical clustering using
//! Fisher-Rao elastic distances.
//!
//! Standard L2 clustering ignores phase variation. Elastic clustering
//! uses the Fisher-Rao distance (which factors out reparameterization)
//! and computes cluster centers as Karcher means.

use super::karcher::karcher_mean;
use super::pairwise::{elastic_distance, elastic_self_distance_matrix};
use super::KarcherMeanResult;
use crate::cv::subset_rows;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for elastic k-means clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticClusterConfig {
    /// Number of clusters.
    pub k: usize,
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
    /// Maximum number of k-means iterations.
    pub max_iter: usize,
    /// Convergence tolerance (reserved for future distance-based criteria).
    pub tol: f64,
    /// Maximum iterations for each Karcher mean computation.
    pub karcher_max_iter: usize,
    /// Convergence tolerance for each Karcher mean computation.
    pub karcher_tol: f64,
    /// Random seed for initialization.
    pub seed: u64,
}

impl Default for ElasticClusterConfig {
    fn default() -> Self {
        Self {
            k: 2,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
            karcher_max_iter: 15,
            karcher_tol: 1e-3,
            seed: 42,
        }
    }
}

/// Linkage method for elastic clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ElasticClusterMethod {
    /// K-means clustering with Karcher mean centers.
    #[default]
    KMeans,
    /// Hierarchical clustering with single (minimum) linkage.
    HierarchicalSingle,
    /// Hierarchical clustering with complete (maximum) linkage.
    HierarchicalComplete,
    /// Hierarchical clustering with average (UPGMA) linkage.
    HierarchicalAverage,
}

/// Result of elastic k-means clustering.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticClusterResult {
    /// Cluster label for each curve (0-indexed, length n).
    pub labels: Vec<usize>,
    /// Karcher mean for each cluster.
    pub centers: Vec<KarcherMeanResult>,
    /// Within-cluster sum of elastic distances for each cluster.
    pub within_distances: Vec<f64>,
    /// Total within-cluster distance (sum of `within_distances`).
    pub total_within_distance: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged (labels stabilized).
    pub converged: bool,
}

/// Result of hierarchical elastic clustering (dendrogram).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticDendrogram {
    /// Merge history: each entry `(i, j, distance)` records merging cluster
    /// indices i and j at the given distance.
    pub merges: Vec<(usize, usize, f64)>,
    /// Full elastic distance matrix used to build the dendrogram.
    pub distance_matrix: FdMatrix,
}

// ─── K-Means++ Initialization ───────────────────────────────────────────────

/// Select k initial center indices using k-means++ on a precomputed distance matrix.
fn kmeans_pp_init(dist_mat: &FdMatrix, k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = dist_mat.nrows();
    let mut centers = Vec::with_capacity(k);

    // Pick the first center uniformly at random.
    centers.push(rng.gen_range(0..n));

    // min_dist_sq[i] = min distance² from curve i to any chosen center.
    let mut min_dist_sq: Vec<f64> = (0..n)
        .map(|i| {
            let d = dist_mat[(i, centers[0])];
            d * d
        })
        .collect();

    for _ in 1..k {
        let total: f64 = min_dist_sq.iter().sum();
        if total <= 0.0 {
            // All remaining points are at distance 0; pick any unselected.
            for i in 0..n {
                if !centers.contains(&i) {
                    centers.push(i);
                    break;
                }
            }
        } else {
            let threshold = rng.gen::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = n - 1;
            for i in 0..n {
                cum += min_dist_sq[i];
                if cum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centers.push(chosen);
        }

        // Update minimum distances with the new center.
        let new_center = *centers.last().unwrap();
        for i in 0..n {
            let d = dist_mat[(i, new_center)];
            let d2 = d * d;
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
        }
    }

    centers
}

// ─── Empty Cluster Handling ─────────────────────────────────────────────────

/// Find the curve farthest from its assigned center (by avg peer distance)
/// in the largest cluster. Used to reassign when a cluster becomes empty.
fn reassign_empty_cluster(labels: &[usize], dist_mat: &FdMatrix) -> usize {
    let n = labels.len();

    // Find the largest cluster.
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; max_label + 1];
    for &l in labels {
        counts[l] += 1;
    }
    let largest_cluster = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &cnt)| cnt)
        .map(|(c, _)| c)
        .unwrap_or(0);

    // Find the member farthest from its peers in that cluster.
    let members: Vec<usize> = (0..n).filter(|&i| labels[i] == largest_cluster).collect();
    let mut max_avg_dist = -1.0_f64;
    let mut farthest = members[0];
    for &i in &members {
        let avg_d: f64 =
            members.iter().map(|&j| dist_mat[(i, j)]).sum::<f64>() / members.len() as f64;
        if avg_d > max_avg_dist {
            max_avg_dist = avg_d;
            farthest = i;
        }
    }
    farthest
}

// ─── K-Means ────────────────────────────────────────────────────────────────

/// Elastic k-means clustering using Fisher-Rao distances and Karcher means.
///
/// Partitions functional data into `k` clusters in the elastic metric. Cluster
/// centers are Karcher (Frechet) means, and assignment uses the Fisher-Rao
/// distance.
///
/// # Algorithm
/// 1. Compute the full elastic distance matrix.
/// 2. Initialize centers using k-means++.
/// 3. Iterate: assign curves to nearest center, recompute Karcher means,
///    recompute distances, and check for convergence (label stability).
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `config`  — Clustering configuration.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `k < 1` or `k > n`.
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_kmeans(
    data: &FdMatrix,
    argvals: &[f64],
    config: &ElasticClusterConfig,
) -> Result<ElasticClusterResult, FdarError> {
    let (n, m) = data.shape();

    if config.k < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "k must be >= 1".to_string(),
        });
    }
    if config.k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k ({}) must be <= n ({})", config.k, n),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    let k = config.k;

    // Step 1: Compute full elastic distance matrix.
    let dist_mat = elastic_self_distance_matrix(data, argvals, config.lambda);

    // Step 2: K-means++ initialization.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let center_indices = kmeans_pp_init(&dist_mat, k, &mut rng);

    // Initial assignment: each curve goes to its nearest initial center.
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mut best_d = f64::INFINITY;
        for (c, &ci) in center_indices.iter().enumerate() {
            let d = dist_mat[(i, ci)];
            if d < best_d {
                best_d = d;
                labels[i] = c;
            }
        }
    }

    // Step 3: Iterate.
    let mut converged = false;
    let mut n_iter = 0;
    let mut centers: Vec<KarcherMeanResult> = Vec::with_capacity(k);

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Compute Karcher mean for each cluster.
        centers = compute_cluster_centers(data, argvals, &labels, k, &dist_mat, config);

        // Reassign: compute distance from each curve to each center's mean.
        let new_labels: Vec<usize> = (0..n)
            .map(|i| {
                let fi = data.row(i);
                let mut best_d = f64::INFINITY;
                let mut best_c = 0;
                for (c, center) in centers.iter().enumerate() {
                    let d = elastic_distance(&fi, &center.mean, argvals, config.lambda);
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect();

        // Check convergence: labels unchanged.
        if new_labels == labels {
            converged = true;
            labels = new_labels;
            break;
        }

        labels = new_labels;
    }

    // If we exited without converging, recompute final centers.
    if !converged {
        centers = compute_cluster_centers(data, argvals, &labels, k, &dist_mat, config);
    }

    // Compute within-cluster distances.
    let mut within_distances = vec![0.0; k];
    for i in 0..n {
        let fi = data.row(i);
        let c = labels[i];
        let d = elastic_distance(&fi, &centers[c].mean, argvals, config.lambda);
        within_distances[c] += d;
    }
    let total_within_distance: f64 = within_distances.iter().sum();

    Ok(ElasticClusterResult {
        labels,
        centers,
        within_distances,
        total_within_distance,
        n_iter,
        converged,
    })
}

/// Compute Karcher mean centers for each cluster.
fn compute_cluster_centers(
    data: &FdMatrix,
    argvals: &[f64],
    labels: &[usize],
    k: usize,
    dist_mat: &FdMatrix,
    config: &ElasticClusterConfig,
) -> Vec<KarcherMeanResult> {
    let n = data.nrows();
    let mut centers = Vec::with_capacity(k);
    for c in 0..k {
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
        if members.is_empty() {
            // Empty cluster: steal the farthest point from the largest cluster.
            let singleton_idx = reassign_empty_cluster(labels, dist_mat);
            let sub = subset_rows(data, &[singleton_idx]);
            centers.push(karcher_mean(
                &sub,
                argvals,
                1,
                config.karcher_tol,
                config.lambda,
            ));
        } else {
            let sub = subset_rows(data, &members);
            centers.push(karcher_mean(
                &sub,
                argvals,
                config.karcher_max_iter,
                config.karcher_tol,
                config.lambda,
            ));
        }
    }
    centers
}

// ─── Hierarchical Clustering ────────────────────────────────────────────────

/// Hierarchical elastic clustering using Fisher-Rao distances.
///
/// Builds a dendrogram by agglomerative clustering. Supported linkage methods
/// are single, complete, and average. Passing [`ElasticClusterMethod::KMeans`]
/// is treated as single linkage.
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `method`  — Linkage method.
/// * `lambda`  — Roughness penalty for elastic alignment.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_hierarchical(
    data: &FdMatrix,
    argvals: &[f64],
    method: ElasticClusterMethod,
    lambda: f64,
) -> Result<ElasticDendrogram, FdarError> {
    let (n, m) = data.shape();

    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }

    // Step 1: Compute full distance matrix.
    let dist_mat = elastic_self_distance_matrix(data, argvals, lambda);

    // Step 2: Initialize — working cluster distance matrix and metadata.
    let mut active = vec![true; n];
    let mut cluster_sizes = vec![1usize; n];
    let mut cluster_dist = FdMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cluster_dist[(i, j)] = dist_mat[(i, j)];
        }
    }

    let mut merges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);

    // Step 3: n-1 merge steps.
    for _ in 0..(n - 1) {
        // Find the minimum-distance active pair.
        let mut min_d = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if cluster_dist[(i, j)] < min_d {
                    min_d = cluster_dist[(i, j)];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        merges.push((min_i, min_j, min_d));

        // Merge j into i: update distances to all other active clusters.
        let size_i = cluster_sizes[min_i];
        let size_j = cluster_sizes[min_j];
        for k in 0..n {
            if !active[k] || k == min_i || k == min_j {
                continue;
            }
            let d_ik = cluster_dist[(min_i.min(k), min_i.max(k))];
            let d_jk = cluster_dist[(min_j.min(k), min_j.max(k))];
            let new_d = match method {
                ElasticClusterMethod::HierarchicalSingle | ElasticClusterMethod::KMeans => {
                    d_ik.min(d_jk)
                }
                ElasticClusterMethod::HierarchicalComplete => d_ik.max(d_jk),
                ElasticClusterMethod::HierarchicalAverage => {
                    (d_ik * size_i as f64 + d_jk * size_j as f64) / (size_i + size_j) as f64
                }
            };
            let (lo, hi) = (min_i.min(k), min_i.max(k));
            cluster_dist[(lo, hi)] = new_d;
            cluster_dist[(hi, lo)] = new_d;
        }

        cluster_sizes[min_i] = size_i + size_j;
        active[min_j] = false;
    }

    Ok(ElasticDendrogram {
        merges,
        distance_matrix: dist_mat,
    })
}

// ─── Cut Dendrogram ─────────────────────────────────────────────────────────

/// Cut a dendrogram to produce k clusters.
///
/// Replays the merge history, stopping after `n - k` merges, and returns
/// cluster labels for each original observation.
///
/// # Arguments
/// * `dendrogram` — Result of [`elastic_hierarchical`].
/// * `k`          — Number of clusters desired.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `k < 1` or `k > n`.
pub fn cut_dendrogram(dendrogram: &ElasticDendrogram, k: usize) -> Result<Vec<usize>, FdarError> {
    let n = dendrogram.distance_matrix.nrows();

    if k < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "k must be >= 1".to_string(),
        });
    }
    if k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k ({k}) must be <= n ({n})"),
        });
    }

    // Start with n singleton clusters, each labeled by its own index.
    let mut cluster_of: Vec<usize> = (0..n).collect();
    let merges_to_apply = n - k;

    for &(ci, cj, _) in dendrogram.merges.iter().take(merges_to_apply) {
        // Relabel all points in cj's current cluster to ci's current cluster.
        let target = cluster_of[ci];
        let source = cluster_of[cj];
        for label in cluster_of.iter_mut() {
            if *label == source {
                *label = target;
            }
        }
    }

    // Compress labels to 0..k-1.
    let mut unique: Vec<usize> = cluster_of.clone();
    unique.sort_unstable();
    unique.dedup();

    let labels: Vec<usize> = cluster_of
        .iter()
        .map(|&c| unique.iter().position(|&u| u == c).unwrap())
        .collect();

    Ok(labels)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
        (data, t)
    }

    #[test]
    fn kmeans_smoke() {
        let (data, t) = make_data(8, 20);
        let config = ElasticClusterConfig {
            k: 2,
            max_iter: 3,
            karcher_max_iter: 3,
            ..Default::default()
        };
        let result = elastic_kmeans(&data, &t, &config).unwrap();
        assert_eq!(result.labels.len(), 8);
        assert_eq!(result.centers.len(), 2);
        assert_eq!(result.within_distances.len(), 2);
        assert!(result.total_within_distance >= 0.0);
        assert!(result.n_iter >= 1);
    }

    #[test]
    fn kmeans_single_cluster() {
        let (data, t) = make_data(5, 20);
        let config = ElasticClusterConfig {
            k: 1,
            max_iter: 3,
            karcher_max_iter: 3,
            ..Default::default()
        };
        let result = elastic_kmeans(&data, &t, &config).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
        assert_eq!(result.centers.len(), 1);
    }

    #[test]
    fn kmeans_k_too_large() {
        let (data, t) = make_data(3, 20);
        let config = ElasticClusterConfig {
            k: 5,
            ..Default::default()
        };
        assert!(elastic_kmeans(&data, &t, &config).is_err());
    }

    #[test]
    fn kmeans_k_zero() {
        let (data, t) = make_data(5, 20);
        let config = ElasticClusterConfig {
            k: 0,
            ..Default::default()
        };
        assert!(elastic_kmeans(&data, &t, &config).is_err());
    }

    #[test]
    fn hierarchical_single_smoke() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).unwrap();
        assert_eq!(dendro.merges.len(), 4);
        // Single linkage merge distances should be non-decreasing.
        for w in dendro.merges.windows(2) {
            assert!(
                w[1].2 >= w[0].2 - 1e-10,
                "single linkage should be non-decreasing"
            );
        }
    }

    #[test]
    fn hierarchical_complete_smoke() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalComplete, 0.0)
                .unwrap();
        assert_eq!(dendro.merges.len(), 4);
    }

    #[test]
    fn hierarchical_average_smoke() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalAverage, 0.0)
                .unwrap();
        assert_eq!(dendro.merges.len(), 4);
    }

    #[test]
    fn hierarchical_too_few_curves() {
        let t = uniform_grid(20);
        let curve: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let data = FdMatrix::from_slice(&curve, 1, 20).unwrap();
        assert!(
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).is_err()
        );
    }

    #[test]
    fn cut_dendrogram_all_singletons() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).unwrap();
        let labels = cut_dendrogram(&dendro, 5).unwrap();
        // Each point in its own cluster.
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn cut_dendrogram_one_cluster() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).unwrap();
        let labels = cut_dendrogram(&dendro, 1).unwrap();
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn cut_dendrogram_k_too_large() {
        let (data, t) = make_data(5, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).unwrap();
        assert!(cut_dendrogram(&dendro, 10).is_err());
    }

    #[test]
    fn cut_dendrogram_two_clusters() {
        let (data, t) = make_data(6, 20);
        let dendro =
            elastic_hierarchical(&data, &t, ElasticClusterMethod::HierarchicalSingle, 0.0).unwrap();
        let labels = cut_dendrogram(&dendro, 2).unwrap();
        assert_eq!(labels.len(), 6);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn default_config_values() {
        let cfg = ElasticClusterConfig::default();
        assert_eq!(cfg.k, 2);
        assert!((cfg.lambda - 0.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_iter, 20);
        assert!((cfg.tol - 1e-4).abs() < f64::EPSILON);
        assert_eq!(cfg.karcher_max_iter, 15);
        assert!((cfg.karcher_tol - 1e-3).abs() < f64::EPSILON);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn default_method() {
        assert_eq!(
            ElasticClusterMethod::default(),
            ElasticClusterMethod::KMeans
        );
    }
}
