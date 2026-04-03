//! Distance-based clustering: k-means (k-medoids) and hierarchical.
//!
//! These algorithms work with **any** precomputed distance matrix — elastic
//! (Fisher-Rao), DTW, Lp, amplitude-only, phase-only, or user-defined.
//!
//! # Examples
//!
//! ```
//! use fdars_core::alignment::{
//!     elastic_self_distance_matrix, hierarchical_from_distances,
//!     kmedoids_from_distances, cut_dendrogram, Linkage, KMedoidsConfig,
//! };
//! use fdars_core::matrix::FdMatrix;
//!
//! // Compute any distance matrix
//! let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
//! let data = FdMatrix::zeros(5, 20);
//! let dist = elastic_self_distance_matrix(&data, &t, 0.0);
//!
//! // Hierarchical clustering — works with any distance matrix
//! let dendro = hierarchical_from_distances(&dist, Linkage::Complete).unwrap();
//! let labels = cut_dendrogram(&dendro, 2).unwrap();
//!
//! // K-medoids — works with any distance matrix
//! let config = KMedoidsConfig { k: 2, ..Default::default() };
//! let result = kmedoids_from_distances(&dist, &config).unwrap();
//! ```

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for k-medoids clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct KMedoidsConfig {
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Random seed for k-means++ initialization.
    pub seed: u64,
}

impl Default for KMedoidsConfig {
    fn default() -> Self {
        Self {
            k: 2,
            max_iter: 100,
            seed: 42,
        }
    }
}

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Linkage {
    /// Minimum distance between clusters.
    #[default]
    Single,
    /// Maximum distance between clusters.
    Complete,
    /// Weighted average distance (UPGMA).
    Average,
}

/// Result of k-medoids clustering.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KMedoidsResult {
    /// Cluster label for each observation (0-indexed, length n).
    pub labels: Vec<usize>,
    /// Medoid index for each cluster (length k).
    pub medoid_indices: Vec<usize>,
    /// Within-cluster sum of distances for each cluster.
    pub within_distances: Vec<f64>,
    /// Total within-cluster distance.
    pub total_within_distance: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged (labels stabilized).
    pub converged: bool,
}

/// Result of hierarchical clustering (dendrogram).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Dendrogram {
    /// Merge history: each entry `(i, j, distance)` records merging cluster
    /// indices i and j at the given distance.
    pub merges: Vec<(usize, usize, f64)>,
    /// Number of observations.
    pub n: usize,
}

// ─── K-Means++ Initialization ───────────────────────────────────────────────

/// Select k initial center indices using k-means++ on a precomputed distance matrix.
fn kmeans_pp_init(dist_mat: &FdMatrix, k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = dist_mat.nrows();
    let mut centers = Vec::with_capacity(k);

    centers.push(rng.gen_range(0..n));

    let mut min_dist_sq: Vec<f64> = (0..n)
        .map(|i| {
            let d = dist_mat[(i, centers[0])];
            d * d
        })
        .collect();

    for _ in 1..k {
        let total: f64 = min_dist_sq.iter().sum();
        if total <= 0.0 {
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

// ─── K-Medoids ─────────────────────────────────────────────────────────────

/// K-medoids (PAM-style) clustering from a precomputed distance matrix.
///
/// Uses k-means++ initialization, then alternates between assigning each
/// observation to its nearest medoid and selecting the medoid that minimizes
/// within-cluster distances.
///
/// Works with **any** distance matrix — elastic, DTW, Lp, or user-defined.
///
/// # Arguments
/// * `dist_mat` — Symmetric n x n distance matrix.
/// * `config`   — Clustering configuration.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `k < 1` or `k > n`.
/// Returns [`FdarError::InvalidDimension`] if `dist_mat` is not square.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kmedoids_from_distances(
    dist_mat: &FdMatrix,
    config: &KMedoidsConfig,
) -> Result<KMedoidsResult, FdarError> {
    let n = dist_mat.nrows();
    if dist_mat.ncols() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "dist_mat",
            expected: format!("{n} x {n} (square)"),
            actual: format!("{} x {}", n, dist_mat.ncols()),
        });
    }
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

    let k = config.k;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut medoids = kmeans_pp_init(dist_mat, k, &mut rng);

    // Assign each point to nearest medoid.
    let mut labels = assign_to_medoids(dist_mat, &medoids, n);

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Update medoids: for each cluster, pick the member minimizing total distance.
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let mut best_cost = f64::INFINITY;
            let mut best_m = medoids[c];
            for &candidate in &members {
                let cost: f64 = members.iter().map(|&j| dist_mat[(candidate, j)]).sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_m = candidate;
                }
            }
            medoids[c] = best_m;
        }

        // Reassign.
        let new_labels = assign_to_medoids(dist_mat, &medoids, n);
        if new_labels == labels {
            converged = true;
            labels = new_labels;
            break;
        }
        labels = new_labels;
    }

    // Compute within-cluster distances.
    let mut within_distances = vec![0.0; k];
    for i in 0..n {
        within_distances[labels[i]] += dist_mat[(i, medoids[labels[i]])];
    }
    let total_within_distance: f64 = within_distances.iter().sum();

    Ok(KMedoidsResult {
        labels,
        medoid_indices: medoids,
        within_distances,
        total_within_distance,
        n_iter,
        converged,
    })
}

fn assign_to_medoids(dist_mat: &FdMatrix, medoids: &[usize], n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut best_d = f64::INFINITY;
            let mut best_c = 0;
            for (c, &med) in medoids.iter().enumerate() {
                let d = dist_mat[(i, med)];
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            best_c
        })
        .collect()
}

// ─── Hierarchical Clustering ───────────────────────────────────────────────

/// Hierarchical agglomerative clustering from a precomputed distance matrix.
///
/// Builds a [`Dendrogram`] by iteratively merging the closest pair of clusters.
/// Works with **any** distance matrix — elastic, DTW, Lp, or user-defined.
///
/// # Arguments
/// * `dist_mat` — Symmetric n x n distance matrix.
/// * `linkage`  — Linkage criterion.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `dist_mat` is not square or `n < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn hierarchical_from_distances(
    dist_mat: &FdMatrix,
    linkage: Linkage,
) -> Result<Dendrogram, FdarError> {
    let n = dist_mat.nrows();
    if dist_mat.ncols() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "dist_mat",
            expected: format!("{n} x {n} (square)"),
            actual: format!("{} x {}", n, dist_mat.ncols()),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "dist_mat",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }

    let mut active = vec![true; n];
    let mut cluster_sizes = vec![1usize; n];
    let mut cluster_dist = FdMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cluster_dist[(i, j)] = dist_mat[(i, j)];
        }
    }

    let mut merges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
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

        let size_i = cluster_sizes[min_i];
        let size_j = cluster_sizes[min_j];
        for k in 0..n {
            if !active[k] || k == min_i || k == min_j {
                continue;
            }
            let d_ik = cluster_dist[(min_i.min(k), min_i.max(k))];
            let d_jk = cluster_dist[(min_j.min(k), min_j.max(k))];
            let new_d = match linkage {
                Linkage::Single => d_ik.min(d_jk),
                Linkage::Complete => d_ik.max(d_jk),
                Linkage::Average => {
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

    Ok(Dendrogram { merges, n })
}

// ─── Cut Dendrogram ─────────────────────────────────────────────────────────

/// Cut a dendrogram to produce k clusters.
///
/// Replays the merge history, stopping after `n - k` merges, and returns
/// cluster labels for each original observation.
///
/// # Arguments
/// * `dendrogram` — Result of [`hierarchical_from_distances`].
/// * `k`          — Number of clusters desired.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `k < 1` or `k > n`.
pub fn cut_dendrogram(dendrogram: &Dendrogram, k: usize) -> Result<Vec<usize>, FdarError> {
    let n = dendrogram.n;

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

    let mut cluster_of: Vec<usize> = (0..n).collect();
    let merges_to_apply = n - k;

    for &(ci, cj, _) in dendrogram.merges.iter().take(merges_to_apply) {
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
    let labels = cluster_of
        .iter()
        .map(|&l| unique.iter().position(|&u| u == l).unwrap())
        .collect();

    Ok(labels)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::elastic_self_distance_matrix;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_dist_mat(n: usize, m: usize) -> FdMatrix {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
        elastic_self_distance_matrix(&data, &t, 0.0)
    }

    #[test]
    fn kmedoids_smoke() {
        let dist = make_dist_mat(8, 20);
        let config = KMedoidsConfig {
            k: 2,
            max_iter: 10,
            ..Default::default()
        };
        let result = kmedoids_from_distances(&dist, &config).unwrap();
        assert_eq!(result.labels.len(), 8);
        assert_eq!(result.medoid_indices.len(), 2);
        assert_eq!(result.within_distances.len(), 2);
        assert!(result.total_within_distance >= 0.0);
        assert!(result.n_iter >= 1);
    }

    #[test]
    fn kmedoids_single_cluster() {
        let dist = make_dist_mat(5, 20);
        let config = KMedoidsConfig {
            k: 1,
            max_iter: 10,
            ..Default::default()
        };
        let result = kmedoids_from_distances(&dist, &config).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
        assert_eq!(result.medoid_indices.len(), 1);
    }

    #[test]
    fn kmedoids_k_too_large() {
        let dist = make_dist_mat(3, 20);
        let config = KMedoidsConfig {
            k: 5,
            ..Default::default()
        };
        assert!(kmedoids_from_distances(&dist, &config).is_err());
    }

    #[test]
    fn kmedoids_k_zero() {
        let dist = make_dist_mat(5, 20);
        let config = KMedoidsConfig {
            k: 0,
            ..Default::default()
        };
        assert!(kmedoids_from_distances(&dist, &config).is_err());
    }

    #[test]
    fn hierarchical_single_smoke() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Single).unwrap();
        assert_eq!(dendro.merges.len(), 4);
        for w in dendro.merges.windows(2) {
            assert!(
                w[1].2 >= w[0].2 - 1e-10,
                "single linkage should be non-decreasing"
            );
        }
    }

    #[test]
    fn hierarchical_complete_smoke() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Complete).unwrap();
        assert_eq!(dendro.merges.len(), 4);
    }

    #[test]
    fn hierarchical_average_smoke() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Average).unwrap();
        assert_eq!(dendro.merges.len(), 4);
    }

    #[test]
    fn hierarchical_too_few() {
        let dist = FdMatrix::zeros(1, 1);
        assert!(hierarchical_from_distances(&dist, Linkage::Single).is_err());
    }

    #[test]
    fn cut_dendrogram_all_singletons() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Single).unwrap();
        let labels = cut_dendrogram(&dendro, 5).unwrap();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn cut_dendrogram_one_cluster() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Single).unwrap();
        let labels = cut_dendrogram(&dendro, 1).unwrap();
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn cut_dendrogram_k_too_large() {
        let dist = make_dist_mat(5, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Single).unwrap();
        assert!(cut_dendrogram(&dendro, 10).is_err());
    }

    #[test]
    fn cut_dendrogram_two_clusters() {
        let dist = make_dist_mat(6, 20);
        let dendro = hierarchical_from_distances(&dist, Linkage::Single).unwrap();
        let labels = cut_dendrogram(&dendro, 2).unwrap();
        assert_eq!(labels.len(), 6);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn default_config_values() {
        let cfg = KMedoidsConfig::default();
        assert_eq!(cfg.k, 2);
        assert_eq!(cfg.max_iter, 100);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn default_linkage() {
        assert_eq!(Linkage::default(), Linkage::Single);
    }

    #[test]
    fn non_square_dist_mat_error() {
        let dist = FdMatrix::zeros(3, 4);
        assert!(hierarchical_from_distances(&dist, Linkage::Single).is_err());
        let config = KMedoidsConfig::default();
        assert!(kmedoids_from_distances(&dist, &config).is_err());
    }
}
