//! Advanced functional clustering algorithms.
//!
//! This module provides four advanced functional clustering paradigms beyond the
//! basic k-means and fuzzy c-means found in [`clustering`](crate::clustering):
//!
//! - **DBSCAN** ([`dbscan_fd`]): Density-based clustering over precomputed functional
//!   L2 distances. Discovers clusters of arbitrary shape and flags noise curves as
//!   [`None`] in the assignment vector. No need to specify k.
//!
//! - **kCFC** ([`kcfc_cluster`]): K-means-style assignment loop where each cluster's
//!   centroid is replaced by a per-cluster FPCA model. A curve is assigned to the
//!   cluster whose FPCA basis produces the smallest reconstruction error.
//!
//! - **funFEM** and **Align-and-Cluster**: Discriminative-subspace GMM clustering and
//!   elastic joint clustering — implemented in plan 33-03.
//!
//! All algorithms are strictly additive relative to the existing `clustering` module.
//! No existing public signature is modified.

use crate::distance::l2_distance_matrix;
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use rand::prelude::*;

// ────────────────────────────────────────────────────────────────────────────
// DBSCAN over functional L2 distances
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for DBSCAN density clustering over functional data.
///
/// DBSCAN discovers clusters of arbitrary shape by expanding dense
/// regions of curves in functional L2 distance space. Curves that
/// do not belong to any dense region are flagged as noise ([`None`]
/// in [`DbscanResult::cluster`]).
///
/// ## Distance units
///
/// `eps` is in the same units as `l2_distance_matrix` — that is,
/// functional L2 distance with Simpson's-rule integration weights.
/// For a constant-1 curve on `argvals` spanning \[0, 1\], the L2 norm
/// is ≈ 1.0. As a practical starting point, set `eps` to a fraction
/// (e.g. 0.1–0.5) of the dataset's median pairwise L2 distance:
///
/// ```
/// # use fdars_core::distance::l2_distance_matrix;
/// # use fdars_core::matrix::FdMatrix;
/// # let data = FdMatrix::zeros(10, 20);
/// # let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let dist = l2_distance_matrix(&data, &argvals);
/// let n = data.nrows();
/// let mut upper: Vec<f64> = (0..n)
///     .flat_map(|i| ((i + 1)..n).map(move |j| dist[(i, j)]))
///     .collect();
/// upper.sort_by(|a, b| a.partial_cmp(b).unwrap());
/// let median_dist = upper[upper.len() / 2];
/// let eps = 0.3 * median_dist; // start here, tune as needed
/// let _ = eps;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbscanConfig {
    /// Neighbourhood radius in functional L2 distance units (default: 0.5).
    ///
    /// Must be strictly positive (`eps > 0`). A tiny positive value makes
    /// every curve a noise point; a very large value merges everything into
    /// one cluster.
    pub eps: f64,
    /// Minimum number of curves (including the point itself) required in the
    /// `eps`-neighbourhood for a point to be considered a core point (default: 3).
    ///
    /// Must be ≥ 1.
    pub min_points: usize,
}

impl Default for DbscanConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_points: 3,
        }
    }
}

/// Result of DBSCAN density clustering over functional data.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DbscanResult {
    /// Cluster assignment for each curve.
    ///
    /// `None` = noise (not part of any dense cluster).
    /// `Some(c)` = cluster index `c` (0-based, contiguous).
    pub cluster: Vec<Option<usize>>,
    /// Number of discovered clusters (excludes noise points).
    pub n_clusters: usize,
    /// Number of noise points (curves assigned `None`).
    pub n_noise: usize,
    /// Precomputed n × n pairwise L2 distance matrix used internally.
    pub distances: FdMatrix,
}

/// Run DBSCAN over functional L2 distances.
///
/// Discovers arbitrarily-shaped clusters in functional data by expanding
/// dense regions. Curves not reachable from any core point are labelled
/// noise (`None`).
///
/// # Arguments
///
/// * `data` — Functional data matrix (n × m, column-major).
/// * `argvals` — Evaluation grid (length m).
/// * `config` — Algorithm parameters; see [`DbscanConfig`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `n == 0`, `m == 0`, or
/// `argvals.len() != m`.  Returns [`FdarError::InvalidParameter`] if
/// `config.eps <= 0` or `config.min_points == 0`.
///
/// # Examples
///
/// ```
/// use fdars_core::clustering_advanced::{dbscan_fd, DbscanConfig};
/// use fdars_core::matrix::FdMatrix;
/// use std::f64::consts::PI;
///
/// // Two tight clusters of 5 sin curves each, well separated
/// let m = 30;
/// let n = 10;
/// let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let mut col_major = vec![0.0_f64; n * m];
/// for i in 0..5 {
///     for (j, &tj) in t.iter().enumerate() {
///         col_major[i + j * n] = (2.0 * PI * tj).sin();
///     }
/// }
/// for i in 5..10 {
///     for (j, &tj) in t.iter().enumerate() {
///         col_major[i + j * n] = (2.0 * PI * tj).sin() + 5.0;
///     }
/// }
/// let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
///
/// let mut cfg = DbscanConfig::default();
/// cfg.eps = 1.0;
/// cfg.min_points = 2;
/// let result = dbscan_fd(&data, &t, &cfg).unwrap();
/// assert_eq!(result.n_clusters, 2);
/// assert_eq!(result.n_noise, 0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dbscan_fd(
    data: &FdMatrix,
    argvals: &[f64],
    config: &DbscanConfig,
) -> Result<DbscanResult, FdarError> {
    let (n, m) = data.shape();

    // Validation
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 row and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if config.eps <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "eps",
            message: format!("eps must be > 0, got {}", config.eps),
        });
    }
    if config.min_points == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "min_points",
            message: "min_points must be >= 1".to_string(),
        });
    }

    let dist = l2_distance_matrix(data, argvals);

    // Standard DBSCAN
    // labels: None = unvisited/noise, Some(c) = cluster c
    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut visited: Vec<bool> = vec![false; n];
    let mut cluster_id: usize = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        // Compute eps-neighbourhood of i (excluding i itself)
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| j != i && dist[(i, j)] <= config.eps)
            .collect();

        // Core-point rule: i + its neighbors must reach min_points
        if neighbors.len() + 1 < config.min_points {
            // Leave as noise for now (may be absorbed as border point later)
            continue;
        }

        // i is a core point — start a new cluster
        labels[i] = Some(cluster_id);

        // BFS expansion
        let mut queue = neighbors.clone();
        let mut qi = 0;
        while qi < queue.len() {
            let j = queue[qi];
            qi += 1;

            if !visited[j] {
                visited[j] = true;
                let j_neighbors: Vec<usize> = (0..n)
                    .filter(|&k| k != j && dist[(j, k)] <= config.eps)
                    .collect();
                if j_neighbors.len() + 1 >= config.min_points {
                    // j is also a core point — add its unqueued neighbours
                    for nb in j_neighbors {
                        if !queue.contains(&nb) {
                            queue.push(nb);
                        }
                    }
                }
            }

            // Absorb j into cluster if not yet assigned
            if labels[j].is_none() {
                labels[j] = Some(cluster_id);
            }
        }

        cluster_id += 1;
    }

    let n_clusters = cluster_id;
    let n_noise = labels.iter().filter(|l| l.is_none()).count();

    Ok(DbscanResult {
        cluster: labels,
        n_clusters,
        n_noise,
        distances: dist,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// kCFC: per-cluster FPCA reassignment loop
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for kCFC (k-means-like clustering via Functional Components).
///
/// Each cluster is represented by a per-cluster FPCA model rather than a
/// centroid. A curve is assigned to the cluster whose FPCA basis gives the
/// smallest L2 reconstruction error.
///
/// **Reference:** Chiou & Li (2007), "Functional clustering and identifying
/// substructures of longitudinal data." R baseline: `fdapace::kCFC`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KcfcConfig {
    /// Number of clusters (default: 2). Must be ≥ 1 and ≤ n.
    pub k: usize,
    /// Number of per-cluster FPC components to use for reconstruction (default: 3).
    ///
    /// Clamped internally to `min(n_k, m)` where `n_k` is the cluster size.
    pub ncomp: usize,
    /// Maximum number of reassignment iterations (default: 50).
    pub max_iter: usize,
    /// Random seed for k-means++ initialization (default: 42).
    pub seed: u64,
}

impl Default for KcfcConfig {
    fn default() -> Self {
        Self {
            k: 2,
            ncomp: 3,
            max_iter: 50,
            seed: 42,
        }
    }
}

/// Result of kCFC per-cluster FPCA clustering.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KcfcResult {
    /// Cluster assignment for each curve (0-based, contiguous, length n).
    pub cluster: Vec<usize>,
    /// Per-cluster FPCA models.
    ///
    /// `fpca_models[k]` is `None` if cluster k was empty at convergence.
    pub fpca_models: Vec<Option<FpcaResult>>,
    /// Reconstruction error matrix (n × k).
    ///
    /// `reconstruction_errors[(i, k)]` is the L2 squared reconstruction error
    /// of curve i against cluster k's FPCA model.
    pub reconstruction_errors: FdMatrix,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged (no label changes in the last iteration).
    pub converged: bool,
}

/// Cluster functional data using per-cluster FPCA reconstruction errors (kCFC).
///
/// Initialises with k-means++ hard labels, then iterates: fit a per-cluster
/// FPCA model for each cluster, compute the reconstruction error of every
/// curve against every cluster's model, and reassign each curve to the cluster
/// with the smallest error. Repeats until convergence or `config.max_iter`.
///
/// # Arguments
///
/// * `data` — Functional data matrix (n × m, column-major).
/// * `argvals` — Evaluation grid (length m).
/// * `config` — Algorithm parameters; see [`KcfcConfig`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `n == 0`, `m == 0`, or
/// `argvals.len() != m`.  Returns [`FdarError::InvalidParameter`] if
/// `config.k == 0` or `config.k > n`.
///
/// # Examples
///
/// ```
/// use fdars_core::clustering_advanced::{kcfc_cluster, KcfcConfig};
/// use fdars_core::matrix::FdMatrix;
/// use std::f64::consts::PI;
///
/// let m = 30;
/// let n = 20;  // 10 per cluster
/// let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let mut col_major = vec![0.0_f64; n * m];
/// for i in 0..10 {
///     for (j, &tj) in t.iter().enumerate() {
///         col_major[i + j * n] = (2.0 * PI * tj).sin();
///     }
/// }
/// for i in 10..20 {
///     for (j, &tj) in t.iter().enumerate() {
///         col_major[i + j * n] = (2.0 * PI * tj).cos() + 5.0;
///     }
/// }
/// let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
///
/// let mut cfg = KcfcConfig::default();
/// cfg.k = 2;
/// cfg.ncomp = 2;
/// let result = kcfc_cluster(&data, &t, &cfg).unwrap();
/// assert_eq!(result.cluster.len(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kcfc_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &KcfcConfig,
) -> Result<KcfcResult, FdarError> {
    let (n, m) = data.shape();

    // Validation
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 row and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if config.k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "k must be >= 1".to_string(),
        });
    }
    if config.k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k={} exceeds number of curves n={}", config.k, n),
        });
    }

    let k = config.k;
    let weights = simpsons_weights(argvals);

    // ── k-means++ initialisation over raw row curves ──────────────────────
    let row_major = data.to_row_major(); // n * m, row-major buffer
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Select first center uniformly
    let mut center_indices: Vec<usize> = Vec::with_capacity(k);
    center_indices.push(rng.gen_range(0..n));

    // Maintain min-distance-squared to nearest chosen center
    let mut min_dist_sq: Vec<f64> = (0..n)
        .map(|i| {
            let c0 = center_indices[0];
            let d = l2_dist_rowmajor(&row_major, i, c0, m, &weights);
            d * d
        })
        .collect();

    while center_indices.len() < k {
        // Sample proportional to D^2
        let total: f64 = min_dist_sq.iter().sum();
        let chosen = if total < 1e-15 {
            rng.gen_range(0..n)
        } else {
            let r = rng.gen::<f64>() * total;
            let mut cumsum = 0.0;
            let mut sel = n - 1;
            for (i, &d) in min_dist_sq.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    sel = i;
                    break;
                }
            }
            sel
        };
        center_indices.push(chosen);

        // Update min_dist_sq with distance to new center
        for i in 0..n {
            let d = l2_dist_rowmajor(&row_major, i, chosen, m, &weights);
            let d2 = d * d;
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
        }
    }

    // Initial assignment: assign each curve to nearest center
    let mut cluster: Vec<usize> = (0..n)
        .map(|i| {
            center_indices
                .iter()
                .enumerate()
                .min_by(|(_, &c1), (_, &c2)| {
                    let d1 = l2_dist_rowmajor(&row_major, i, c1, m, &weights);
                    let d2 = l2_dist_rowmajor(&row_major, i, c2, m, &weights);
                    d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ki, _)| ki)
                .unwrap_or(0)
        })
        .collect();

    // ── Reassignment loop ─────────────────────────────────────────────────
    let mut fpca_models: Vec<Option<FpcaResult>> = vec![None; k];
    let mut reconstruction_errors = FdMatrix::zeros(n, k);
    let mut converged = false;
    let mut iterations = 0;

    for _iter in 0..config.max_iter {
        iterations += 1;

        // ── Fit per-cluster FPCA models ───────────────────────────────────
        for ki in 0..k {
            let member_indices: Vec<usize> = (0..n).filter(|&i| cluster[i] == ki).collect();

            if member_indices.is_empty() {
                // Keep previous model (or None on first iteration)
                continue;
            }

            // Gather member rows into a (n_k x m) FdMatrix
            let n_k = member_indices.len();
            let mut col_major_k = vec![0.0_f64; n_k * m];
            for (row_in_k, &orig_i) in member_indices.iter().enumerate() {
                for j in 0..m {
                    col_major_k[row_in_k + j * n_k] = data[(orig_i, j)];
                }
            }
            let data_k = FdMatrix::from_column_major(col_major_k, n_k, m)?;

            // Fit FPCA (ncomp clamped internally to min(n_k, m))
            match fdata_to_pc_1d(&data_k, config.ncomp, argvals) {
                Ok(fpca) => {
                    fpca_models[ki] = Some(fpca);
                }
                Err(_) => {
                    // Degenerate cluster; keep prior model
                }
            }
        }

        // ── Compute reconstruction errors for all curves vs all clusters ──
        for i in 0..n {
            let curve_row = data.row(i);
            let curve_mat = FdMatrix::from_slice(&curve_row, 1, m)?;

            for ki in 0..k {
                let err = match &fpca_models[ki] {
                    None => f64::INFINITY,
                    Some(fpca) => {
                        let ncomp_eff = fpca.rotation.ncols();
                        match fpca.project(&curve_mat) {
                            Ok(scores) => {
                                match fpca.reconstruct(&scores, ncomp_eff) {
                                    Ok(recon) => {
                                        // L2^2 reconstruction error with Simpson weights
                                        let mut err_sq = 0.0;
                                        for j in 0..m {
                                            let diff = curve_row[j] - recon[(0, j)];
                                            err_sq += diff * diff * weights[j];
                                        }
                                        err_sq
                                    }
                                    Err(_) => f64::INFINITY,
                                }
                            }
                            Err(_) => f64::INFINITY,
                        }
                    }
                };
                reconstruction_errors[(i, ki)] = err;
            }
        }

        // ── Reassign each curve to the cluster with minimum error ─────────
        let mut changed = false;
        for i in 0..n {
            let best_k = (0..k)
                .min_by(|&a, &b| {
                    reconstruction_errors[(i, a)]
                        .partial_cmp(&reconstruction_errors[(i, b)])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if cluster[i] != best_k {
                cluster[i] = best_k;
                changed = true;
            }
        }

        if !changed {
            converged = true;
            break;
        }
    }

    Ok(KcfcResult {
        cluster,
        fpca_models,
        reconstruction_errors,
        iterations,
        converged,
    })
}

/// Compute L2 distance between two rows in a flat row-major buffer.
///
/// `buf[i * m .. (i+1) * m]` is row `i`.
fn l2_dist_rowmajor(buf: &[f64], i: usize, j: usize, m: usize, weights: &[f64]) -> f64 {
    let mut sq = 0.0;
    for t in 0..m {
        let d = buf[i * m + t] - buf[j * m + t];
        sq += d * d * weights[t];
    }
    sq.sqrt()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{adjusted_rand_index, uniform_grid};
    use std::f64::consts::PI;

    // ── Synthetic data generators ────────────────────────────────────────

    /// Two tight clusters, n_per curves each. Cluster 0: sin wave.
    /// Cluster 1: sin wave shifted up by 5.
    fn two_tight_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
        let t = uniform_grid(m);
        let n = 2 * n_per;
        let mut col_major = vec![0.0_f64; n * m];
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin();
            }
        }
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[(i + n_per) + j * n] = (2.0 * PI * tj).sin() + 5.0;
            }
        }
        let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();
        (
            FdMatrix::from_column_major(col_major, n, m).unwrap(),
            t,
            labels,
        )
    }

    /// 2 tight clusters (n_per each) + 2 far constant-offset outlier curves.
    fn clusters_with_noise(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let n = 2 * n_per + 2;
        let mut col_major = vec![0.0_f64; n * m];
        // Cluster 0: sin wave
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin();
            }
        }
        // Cluster 1: sin wave + 5
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[(i + n_per) + j * n] = (2.0 * PI * tj).sin() + 5.0;
            }
        }
        // Outlier 0: constant 100
        let o0 = 2 * n_per;
        for j in 0..m {
            col_major[o0 + j * n] = 100.0;
        }
        // Outlier 1: constant -100
        let o1 = 2 * n_per + 1;
        for j in 0..m {
            col_major[o1 + j * n] = -100.0;
        }
        (FdMatrix::from_column_major(col_major, n, m).unwrap(), t)
    }

    /// Two well-separated clusters for kCFC testing.
    fn two_separated_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
        let t = uniform_grid(m);
        let n = 2 * n_per;
        let mut col_major = vec![0.0_f64; n * m];
        // Cluster 0: sin waves
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin() + 0.05 * (i as f64 / n_per as f64);
            }
        }
        // Cluster 1: cos waves shifted up by 8 (very different shape)
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[(i + n_per) + j * n] =
                    (2.0 * PI * tj).cos() + 8.0 + 0.05 * (i as f64 / n_per as f64);
            }
        }
        let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();
        (
            FdMatrix::from_column_major(col_major, n, m).unwrap(),
            t,
            labels,
        )
    }

    // ── DBSCAN tests ─────────────────────────────────────────────────────

    #[test]
    fn test_dbscan_core_points() {
        let m = 30;
        let n_per = 5;
        let (data, t, _labels) = two_tight_clusters(n_per, m);
        let result = dbscan_fd(
            &data,
            &t,
            &DbscanConfig {
                eps: 1.0,
                min_points: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.n_clusters, 2, "expected 2 clusters");
        assert_eq!(result.n_noise, 0, "expected 0 noise points");
        assert_eq!(result.cluster.len(), 2 * n_per);
    }

    #[test]
    fn test_dbscan_noise_flagging() {
        let m = 30;
        let n_per = 5;
        let (data, t) = clusters_with_noise(n_per, m);
        let result = dbscan_fd(
            &data,
            &t,
            &DbscanConfig {
                eps: 1.5,
                min_points: 2,
                ..Default::default()
            },
        )
        .unwrap();
        // The 2 outlier curves should be noise
        assert_eq!(
            result.n_noise, 2,
            "expected exactly 2 noise points, got {}",
            result.n_noise
        );
        assert_eq!(result.n_clusters, 2, "expected 2 clusters");
        // Verify outlier indices (last 2) are None
        let n = data.nrows();
        assert!(result.cluster[n - 2].is_none(), "outlier 0 should be noise");
        assert!(result.cluster[n - 1].is_none(), "outlier 1 should be noise");
    }

    #[test]
    fn test_dbscan_zero_eps_returns_err() {
        let m = 20;
        let (data, t, _) = two_tight_clusters(5, m);
        assert!(
            dbscan_fd(
                &data,
                &t,
                &DbscanConfig {
                    eps: 0.0,
                    ..Default::default()
                }
            )
            .is_err(),
            "eps=0 should return Err"
        );
    }

    #[test]
    fn test_dbscan_negative_eps_returns_err() {
        let m = 20;
        let (data, t, _) = two_tight_clusters(5, m);
        assert!(
            dbscan_fd(
                &data,
                &t,
                &DbscanConfig {
                    eps: -1.0,
                    ..Default::default()
                }
            )
            .is_err(),
            "eps=-1 should return Err"
        );
    }

    #[test]
    fn test_dbscan_invalid_min_points_zero() {
        let m = 20;
        let (data, t, _) = two_tight_clusters(5, m);
        assert!(
            dbscan_fd(
                &data,
                &t,
                &DbscanConfig {
                    min_points: 0,
                    ..Default::default()
                }
            )
            .is_err(),
            "min_points=0 should return Err"
        );
    }

    #[test]
    fn test_dbscan_empty_data() {
        let data = FdMatrix::zeros(0, 0);
        let t: Vec<f64> = vec![];
        assert!(
            dbscan_fd(&data, &t, &DbscanConfig::default()).is_err(),
            "empty data should return Err"
        );
    }

    #[test]
    fn test_dbscan_mismatched_argvals() {
        let m = 20;
        let (data, _t, _) = two_tight_clusters(5, m);
        let wrong_t = uniform_grid(m + 1);
        assert!(
            dbscan_fd(&data, &wrong_t, &DbscanConfig::default()).is_err(),
            "mismatched argvals should return Err"
        );
    }

    #[test]
    fn test_dbscan_distances_shape() {
        let m = 20;
        let n_per = 4;
        let (data, t, _) = two_tight_clusters(n_per, m);
        let result = dbscan_fd(
            &data,
            &t,
            &DbscanConfig {
                eps: 1.0,
                min_points: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let n = 2 * n_per;
        assert_eq!(
            result.distances.shape(),
            (n, n),
            "distance matrix must be n x n"
        );
    }

    // ── kCFC tests ───────────────────────────────────────────────────────

    #[test]
    fn test_kcfc_recovery() {
        let m = 40;
        let n_per = 10;
        let (data, t, ground_truth) = two_separated_clusters(n_per, m);
        let result = kcfc_cluster(
            &data,
            &t,
            &KcfcConfig {
                k: 2,
                ncomp: 3,
                max_iter: 50,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        let ari = adjusted_rand_index(&result.cluster, &ground_truth);
        assert!(
            ari >= 0.90,
            "kCFC ARI={ari:.3} should be >= 0.90 on well-separated data"
        );
    }

    #[test]
    fn test_kcfc_errors_ordering() {
        // Curves in cluster 0 should have smaller error against cluster 0's FPCA
        // than against cluster 1's FPCA, and vice versa.
        let m = 40;
        let n_per = 10;
        let (data, t, ground_truth) = two_separated_clusters(n_per, m);
        let result = kcfc_cluster(
            &data,
            &t,
            &KcfcConfig {
                k: 2,
                ncomp: 3,
                max_iter: 50,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();

        // Determine which result cluster corresponds to ground truth 0
        let gt0_cluster = result.cluster[0]; // first curve is in ground truth 0
        let gt1_cluster = 1 - gt0_cluster;

        let mut correct_ordering = 0;
        let mut total = 0;
        for i in 0..data.nrows() {
            let expected_cluster = if ground_truth[i] == 0 {
                gt0_cluster
            } else {
                gt1_cluster
            };
            let err_own = result.reconstruction_errors[(i, expected_cluster)];
            let err_other = result.reconstruction_errors[(i, 1 - expected_cluster)];
            if err_own.is_finite() && err_other.is_finite() {
                if err_own < err_other {
                    correct_ordering += 1;
                }
                total += 1;
            }
        }
        // At least 80% should have correct error ordering
        assert!(
            correct_ordering * 10 >= total * 8,
            "only {correct_ordering}/{total} curves had smaller error for their true cluster"
        );
    }

    #[test]
    fn test_kcfc_deterministic() {
        let m = 30;
        let n_per = 8;
        let (data, t, _) = two_separated_clusters(n_per, m);
        let cfg = KcfcConfig {
            k: 2,
            ncomp: 2,
            max_iter: 30,
            seed: 7,
            ..Default::default()
        };
        let r1 = kcfc_cluster(&data, &t, &cfg).unwrap();
        let r2 = kcfc_cluster(&data, &t, &cfg).unwrap();
        assert_eq!(
            r1.cluster, r2.cluster,
            "identical seed must produce identical assignments"
        );
    }

    #[test]
    fn test_kcfc_invalid_k_zero() {
        let m = 20;
        let (data, t, _) = two_tight_clusters(5, m);
        assert!(
            kcfc_cluster(
                &data,
                &t,
                &KcfcConfig {
                    k: 0,
                    ..Default::default()
                }
            )
            .is_err(),
            "k=0 should return Err"
        );
    }

    #[test]
    fn test_kcfc_invalid_k_gt_n() {
        let m = 20;
        let n = 4;
        let (data, t, _) = two_tight_clusters(n / 2, m);
        assert!(
            kcfc_cluster(
                &data,
                &t,
                &KcfcConfig {
                    k: n + 1,
                    ..Default::default()
                }
            )
            .is_err(),
            "k>n should return Err"
        );
    }

    #[test]
    fn test_kcfc_empty_data() {
        let data = FdMatrix::zeros(0, 0);
        let t: Vec<f64> = vec![];
        assert!(
            kcfc_cluster(&data, &t, &KcfcConfig::default()).is_err(),
            "empty data should return Err"
        );
    }

    #[test]
    fn test_kcfc_mismatched_argvals() {
        let m = 20;
        let (data, _t, _) = two_tight_clusters(5, m);
        let wrong_t = uniform_grid(m + 3);
        assert!(
            kcfc_cluster(&data, &wrong_t, &KcfcConfig::default()).is_err(),
            "mismatched argvals should return Err"
        );
    }

    #[test]
    fn test_kcfc_result_shapes() {
        let m = 20;
        let n_per = 5;
        let (data, t, _) = two_separated_clusters(n_per, m);
        let n = 2 * n_per;
        let result = kcfc_cluster(
            &data,
            &t,
            &KcfcConfig {
                k: 2,
                ncomp: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.cluster.len(), n);
        assert_eq!(result.fpca_models.len(), 2);
        assert_eq!(result.reconstruction_errors.shape(), (n, 2));
    }
}
