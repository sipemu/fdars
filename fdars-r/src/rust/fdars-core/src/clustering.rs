//! Clustering algorithms for functional data.
//!
//! This module provides k-means and fuzzy c-means clustering algorithms
//! for functional data.

use crate::helpers::{extract_curves, l2_distance, simpsons_weights, NUMERICAL_EPS};
use crate::{iter_maybe_parallel, slice_maybe_parallel};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of k-means clustering.
pub struct KmeansResult {
    /// Cluster assignments for each observation
    pub cluster: Vec<usize>,
    /// Cluster centers (k x m matrix, column-major)
    pub centers: Vec<f64>,
    /// Within-cluster sum of squares for each cluster
    pub withinss: Vec<f64>,
    /// Total within-cluster sum of squares
    pub tot_withinss: f64,
    /// Number of iterations
    pub iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// K-means++ initialization: select initial centers with probability proportional to D^2.
///
/// # Arguments
/// * `curves` - Vector of curve vectors
/// * `k` - Number of clusters
/// * `weights` - Integration weights for L2 distance
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of k initial cluster centers
fn kmeans_plusplus_init(
    curves: &[Vec<f64>],
    k: usize,
    weights: &[f64],
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let n = curves.len();
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First center: random
    let first_idx = rng.gen_range(0..n);
    centers.push(curves[first_idx].clone());

    // Remaining centers: probability proportional to D^2
    for _ in 1..k {
        let distances: Vec<f64> = curves
            .iter()
            .map(|curve| {
                centers
                    .iter()
                    .map(|c| l2_distance(curve, c, weights))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let dist_sq: Vec<f64> = distances.iter().map(|d| d * d).collect();
        let total: f64 = dist_sq.iter().sum();

        if total < NUMERICAL_EPS {
            let idx = rng.gen_range(0..n);
            centers.push(curves[idx].clone());
        } else {
            let r = rng.gen::<f64>() * total;
            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (i, &d) in dist_sq.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    chosen = i;
                    break;
                }
            }
            centers.push(curves[chosen].clone());
        }
    }

    centers
}

/// Compute fuzzy membership values for a single observation.
///
/// # Arguments
/// * `distances` - Distances from the observation to each cluster center
/// * `exponent` - Exponent for fuzzy membership (2 / (fuzziness - 1))
///
/// # Returns
/// Vector of membership values (one per cluster)
fn compute_fuzzy_membership(distances: &[f64], exponent: f64) -> Vec<f64> {
    let k = distances.len();
    let mut membership = vec![0.0; k];

    // Check if observation is very close to any center
    for (c, &dist) in distances.iter().enumerate() {
        if dist < NUMERICAL_EPS {
            // Assign full membership to this cluster
            membership[c] = 1.0;
            return membership;
        }
    }

    // Normal fuzzy membership computation
    for c in 0..k {
        let mut sum = 0.0;
        for c2 in 0..k {
            if distances[c2] > NUMERICAL_EPS {
                sum += (distances[c] / distances[c2]).powf(exponent);
            }
        }
        membership[c] = if sum > NUMERICAL_EPS { 1.0 / sum } else { 1.0 };
    }

    membership
}

/// K-means clustering for functional data.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `k` - Number of clusters
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed
pub fn kmeans_fd(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> KmeansResult {
    if n == 0 || m == 0 || k == 0 || k > n || argvals.len() != m {
        return KmeansResult {
            cluster: Vec::new(),
            centers: Vec::new(),
            withinss: Vec::new(),
            tot_withinss: 0.0,
            iter: 0,
            converged: false,
        };
    }

    let weights = simpsons_weights(argvals);
    let mut rng = StdRng::seed_from_u64(seed);

    // Extract curves using helper
    let curves = extract_curves(data, n, m);

    // K-means++ initialization using helper
    let mut centers = kmeans_plusplus_init(&curves, k, &weights, &mut rng);

    let mut cluster = vec![0usize; n];
    let mut converged = false;
    let mut iter = 0;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Assignment step
        let new_cluster: Vec<usize> = slice_maybe_parallel!(curves)
            .map(|curve| {
                let mut best_cluster = 0;
                let mut best_dist = f64::INFINITY;
                for (c, center) in centers.iter().enumerate() {
                    let dist = l2_distance(curve, center, &weights);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                best_cluster
            })
            .collect();

        // Check convergence
        if new_cluster == cluster {
            converged = true;
            break;
        }
        cluster = new_cluster;

        // Update step
        let new_centers: Vec<Vec<f64>> = (0..k)
            .map(|c| {
                let members: Vec<usize> = cluster
                    .iter()
                    .enumerate()
                    .filter(|(_, &cl)| cl == c)
                    .map(|(i, _)| i)
                    .collect();

                if members.is_empty() {
                    centers[c].clone()
                } else {
                    let mut center = vec![0.0; m];
                    for &i in &members {
                        for j in 0..m {
                            center[j] += curves[i][j];
                        }
                    }
                    let n_members = members.len() as f64;
                    for j in 0..m {
                        center[j] /= n_members;
                    }
                    center
                }
            })
            .collect();

        // Check convergence by center movement
        let max_movement: f64 = centers
            .iter()
            .zip(new_centers.iter())
            .map(|(old, new)| l2_distance(old, new, &weights))
            .fold(0.0, f64::max);

        centers = new_centers;

        if max_movement < tol {
            converged = true;
            break;
        }
    }

    // Compute within-cluster sum of squares
    let mut withinss = vec![0.0; k];
    for (i, curve) in curves.iter().enumerate() {
        let c = cluster[i];
        let dist = l2_distance(curve, &centers[c], &weights);
        withinss[c] += dist * dist;
    }
    let tot_withinss: f64 = withinss.iter().sum();

    // Flatten centers (column-major: k x m)
    let mut centers_flat = vec![0.0; k * m];
    for c in 0..k {
        for j in 0..m {
            centers_flat[c + j * k] = centers[c][j];
        }
    }

    KmeansResult {
        cluster,
        centers: centers_flat,
        withinss,
        tot_withinss,
        iter,
        converged,
    }
}

/// Result of fuzzy c-means clustering.
pub struct FuzzyCmeansResult {
    /// Membership matrix (n x k, column-major)
    pub membership: Vec<f64>,
    /// Cluster centers (k x m, column-major)
    pub centers: Vec<f64>,
    /// Number of iterations
    pub iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Fuzzy c-means clustering for functional data.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `k` - Number of clusters
/// * `fuzziness` - Fuzziness parameter (> 1)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed
pub fn fuzzy_cmeans_fd(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    k: usize,
    fuzziness: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> FuzzyCmeansResult {
    if n == 0 || m == 0 || k == 0 || k > n || argvals.len() != m || fuzziness <= 1.0 {
        return FuzzyCmeansResult {
            membership: Vec::new(),
            centers: Vec::new(),
            iter: 0,
            converged: false,
        };
    }

    let weights = simpsons_weights(argvals);
    let mut rng = StdRng::seed_from_u64(seed);

    // Extract curves using helper
    let curves = extract_curves(data, n, m);

    // Initialize membership matrix randomly
    let mut membership = vec![0.0; n * k];
    for i in 0..n {
        let mut row_sum = 0.0;
        for c in 0..k {
            let val = rng.gen::<f64>();
            membership[i + c * n] = val;
            row_sum += val;
        }
        for c in 0..k {
            membership[i + c * n] /= row_sum;
        }
    }

    let mut centers = vec![vec![0.0; m]; k];
    let mut converged = false;
    let mut iter = 0;
    let exponent = 2.0 / (fuzziness - 1.0);

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Update centers
        for c in 0..k {
            let mut numerator = vec![0.0; m];
            let mut denominator = 0.0;

            for (i, curve) in curves.iter().enumerate() {
                let weight = membership[i + c * n].powf(fuzziness);
                for j in 0..m {
                    numerator[j] += weight * curve[j];
                }
                denominator += weight;
            }

            if denominator > NUMERICAL_EPS {
                for j in 0..m {
                    centers[c][j] = numerator[j] / denominator;
                }
            }
        }

        // Update membership using helper
        let mut new_membership = vec![0.0; n * k];
        let mut max_change = 0.0;

        for (i, curve) in curves.iter().enumerate() {
            let distances: Vec<f64> = centers
                .iter()
                .map(|c| l2_distance(curve, c, &weights))
                .collect();

            // Use the helper function for membership computation
            let memberships = compute_fuzzy_membership(&distances, exponent);

            for c in 0..k {
                new_membership[i + c * n] = memberships[c];
                let change = (memberships[c] - membership[i + c * n]).abs();
                if change > max_change {
                    max_change = change;
                }
            }
        }

        membership = new_membership;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Flatten centers (column-major: k x m)
    let mut centers_flat = vec![0.0; k * m];
    for c in 0..k {
        for j in 0..m {
            centers_flat[c + j * k] = centers[c][j];
        }
    }

    FuzzyCmeansResult {
        membership,
        centers: centers_flat,
        iter,
        converged,
    }
}

/// Compute silhouette score for clustering result.
pub fn silhouette_score(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    cluster: &[usize],
) -> Vec<f64> {
    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    let curves = extract_curves(data, n, m);

    let k = cluster.iter().cloned().max().unwrap_or(0) + 1;

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let my_cluster = cluster[i];

            // a(i) = average distance to points in same cluster
            let same_cluster: Vec<usize> = cluster
                .iter()
                .enumerate()
                .filter(|(j, &c)| c == my_cluster && *j != i)
                .map(|(j, _)| j)
                .collect();

            let a_i = if same_cluster.is_empty() {
                0.0
            } else {
                let sum: f64 = same_cluster
                    .iter()
                    .map(|&j| l2_distance(&curves[i], &curves[j], &weights))
                    .sum();
                sum / same_cluster.len() as f64
            };

            // b(i) = min average distance to points in other clusters
            let mut b_i = f64::INFINITY;
            for c in 0..k {
                if c == my_cluster {
                    continue;
                }

                let other_cluster: Vec<usize> = cluster
                    .iter()
                    .enumerate()
                    .filter(|(_, &cl)| cl == c)
                    .map(|(j, _)| j)
                    .collect();

                if other_cluster.is_empty() {
                    continue;
                }

                let avg_dist: f64 = other_cluster
                    .iter()
                    .map(|&j| l2_distance(&curves[i], &curves[j], &weights))
                    .sum::<f64>()
                    / other_cluster.len() as f64;

                b_i = b_i.min(avg_dist);
            }

            if b_i.is_infinite() {
                0.0
            } else {
                let max_ab = a_i.max(b_i);
                if max_ab > NUMERICAL_EPS {
                    (b_i - a_i) / max_ab
                } else {
                    0.0
                }
            }
        })
        .collect()
}

/// Compute Calinski-Harabasz index for clustering result.
pub fn calinski_harabasz(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    cluster: &[usize],
) -> f64 {
    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let curves = extract_curves(data, n, m);

    let k = cluster.iter().cloned().max().unwrap_or(0) + 1;
    if k < 2 {
        return 0.0;
    }

    // Global mean
    let mut global_mean = vec![0.0; m];
    for curve in &curves {
        for j in 0..m {
            global_mean[j] += curve[j];
        }
    }
    for j in 0..m {
        global_mean[j] /= n as f64;
    }

    // Cluster centers
    let mut centers = vec![vec![0.0; m]; k];
    let mut counts = vec![0usize; k];
    for (i, curve) in curves.iter().enumerate() {
        let c = cluster[i];
        counts[c] += 1;
        for j in 0..m {
            centers[c][j] += curve[j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..m {
                centers[c][j] /= counts[c] as f64;
            }
        }
    }

    // Between-cluster sum of squares
    let mut bgss = 0.0;
    for c in 0..k {
        let dist = l2_distance(&centers[c], &global_mean, &weights);
        bgss += counts[c] as f64 * dist * dist;
    }

    // Within-cluster sum of squares
    let mut wgss = 0.0;
    for (i, curve) in curves.iter().enumerate() {
        let c = cluster[i];
        let dist = l2_distance(curve, &centers[c], &weights);
        wgss += dist * dist;
    }

    if wgss < NUMERICAL_EPS {
        return f64::INFINITY;
    }

    (bgss / (k - 1) as f64) / (wgss / (n - k) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a uniform grid of points
    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    /// Generate two clearly separated clusters of curves
    fn generate_two_clusters(n_per_cluster: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let t = uniform_grid(m);
        let mut data = Vec::with_capacity(2 * n_per_cluster * m);

        // Cluster 0: sine waves with low amplitude
        for i in 0..n_per_cluster {
            for &ti in &t {
                data.push((2.0 * PI * ti).sin() + 0.1 * (i as f64 / n_per_cluster as f64));
            }
        }

        // Cluster 1: sine waves shifted up by 5
        for i in 0..n_per_cluster {
            for &ti in &t {
                data.push((2.0 * PI * ti).sin() + 5.0 + 0.1 * (i as f64 / n_per_cluster as f64));
            }
        }

        // Column-major reordering
        let n = 2 * n_per_cluster;
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                col_major[i + j * n] = data[i * m + j];
            }
        }

        (col_major, t)
    }

    // ============== K-means tests ==============

    #[test]
    fn test_kmeans_fd_basic() {
        let m = 50;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = kmeans_fd(&data, n, m, &t, 2, 100, 1e-6, 42);

        assert_eq!(result.cluster.len(), n);
        assert!(result.converged);
        assert!(result.iter > 0 && result.iter <= 100);
    }

    #[test]
    fn test_kmeans_fd_finds_clusters() {
        let m = 50;
        let n_per = 10;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = kmeans_fd(&data, n, m, &t, 2, 100, 1e-6, 42);

        // First half should be one cluster, second half the other
        let cluster_0 = result.cluster[0];
        let cluster_1 = result.cluster[n_per];

        assert_ne!(cluster_0, cluster_1, "Clusters should be different");

        // Check that first half is in same cluster
        for i in 0..n_per {
            assert_eq!(result.cluster[i], cluster_0);
        }

        // Check that second half is in same cluster
        for i in n_per..n {
            assert_eq!(result.cluster[i], cluster_1);
        }
    }

    #[test]
    fn test_kmeans_fd_deterministic() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result1 = kmeans_fd(&data, n, m, &t, 2, 100, 1e-6, 42);
        let result2 = kmeans_fd(&data, n, m, &t, 2, 100, 1e-6, 42);

        // Same seed should give same results
        assert_eq!(result1.cluster, result2.cluster);
    }

    #[test]
    fn test_kmeans_fd_withinss() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = kmeans_fd(&data, n, m, &t, 2, 100, 1e-6, 42);

        // Within-cluster sum of squares should be non-negative
        for &wss in &result.withinss {
            assert!(wss >= 0.0);
        }

        // Total should equal sum
        let sum: f64 = result.withinss.iter().sum();
        assert!((sum - result.tot_withinss).abs() < 1e-10);
    }

    #[test]
    fn test_kmeans_fd_centers_shape() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;
        let k = 3;

        let result = kmeans_fd(&data, n, m, &t, k, 100, 1e-6, 42);

        // Centers should be k x m matrix (column-major)
        assert_eq!(result.centers.len(), k * m);
    }

    #[test]
    fn test_kmeans_fd_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let result = kmeans_fd(&[], 0, 30, &t, 2, 100, 1e-6, 42);
        assert!(result.cluster.is_empty());
        assert!(!result.converged);

        // k > n
        let data = vec![0.0; 5 * 30];
        let result = kmeans_fd(&data, 5, 30, &t, 10, 100, 1e-6, 42);
        assert!(result.cluster.is_empty());
    }

    #[test]
    fn test_kmeans_fd_single_cluster() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let data = vec![0.0; n * m];

        let result = kmeans_fd(&data, n, m, &t, 1, 100, 1e-6, 42);

        // All should be in cluster 0
        for &c in &result.cluster {
            assert_eq!(c, 0);
        }
    }

    // ============== Fuzzy C-means tests ==============

    #[test]
    fn test_fuzzy_cmeans_fd_basic() {
        let m = 50;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = fuzzy_cmeans_fd(&data, n, m, &t, 2, 2.0, 100, 1e-6, 42);

        assert_eq!(result.membership.len(), n * 2);
        assert!(result.iter > 0);
    }

    #[test]
    fn test_fuzzy_cmeans_fd_membership_sums_to_one() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;
        let k = 2;

        let result = fuzzy_cmeans_fd(&data, n, m, &t, k, 2.0, 100, 1e-6, 42);

        // Each observation's membership should sum to 1
        for i in 0..n {
            let sum: f64 = (0..k).map(|c| result.membership[i + c * n]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Membership should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_fuzzy_cmeans_fd_membership_in_range() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = fuzzy_cmeans_fd(&data, n, m, &t, 2, 2.0, 100, 1e-6, 42);

        // All memberships should be in [0, 1]
        for &mem in &result.membership {
            assert!((0.0..=1.0 + 1e-10).contains(&mem));
        }
    }

    #[test]
    fn test_fuzzy_cmeans_fd_fuzziness_effect() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result_low = fuzzy_cmeans_fd(&data, n, m, &t, 2, 1.5, 100, 1e-6, 42);
        let result_high = fuzzy_cmeans_fd(&data, n, m, &t, 2, 3.0, 100, 1e-6, 42);

        // Higher fuzziness should give more diffuse memberships
        // Measure by entropy-like metric
        let entropy_low: f64 = result_low
            .membership
            .iter()
            .map(|&m| if m > 1e-10 { -m * m.ln() } else { 0.0 })
            .sum();

        let entropy_high: f64 = result_high
            .membership
            .iter()
            .map(|&m| if m > 1e-10 { -m * m.ln() } else { 0.0 })
            .sum();

        assert!(
            entropy_high >= entropy_low - 0.1,
            "Higher fuzziness should give higher entropy"
        );
    }

    #[test]
    fn test_fuzzy_cmeans_fd_invalid_fuzziness() {
        let t = uniform_grid(30);
        let data = vec![0.0; 10 * 30];

        // Fuzziness <= 1 should fail
        let result = fuzzy_cmeans_fd(&data, 10, 30, &t, 2, 1.0, 100, 1e-6, 42);
        assert!(result.membership.is_empty());

        let result = fuzzy_cmeans_fd(&data, 10, 30, &t, 2, 0.5, 100, 1e-6, 42);
        assert!(result.membership.is_empty());
    }

    #[test]
    fn test_fuzzy_cmeans_fd_centers_shape() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let k = 3;
        let data = vec![0.0; n * m];

        let result = fuzzy_cmeans_fd(&data, n, m, &t, k, 2.0, 100, 1e-6, 42);

        assert_eq!(result.centers.len(), k * m);
    }

    // ============== Silhouette score tests ==============

    #[test]
    fn test_silhouette_score_well_separated() {
        let m = 30;
        let n_per = 10;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        // Perfect clustering: first half in 0, second in 1
        let cluster: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        let scores = silhouette_score(&data, n, m, &t, &cluster);

        assert_eq!(scores.len(), n);

        // Well-separated clusters should have high silhouette scores
        let mean_score: f64 = scores.iter().sum::<f64>() / n as f64;
        assert!(
            mean_score > 0.5,
            "Well-separated clusters should have high silhouette: {}",
            mean_score
        );
    }

    #[test]
    fn test_silhouette_score_range() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let cluster: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        let scores = silhouette_score(&data, n, m, &t, &cluster);

        // Silhouette scores should be in [-1, 1]
        for &s in &scores {
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&s));
        }
    }

    #[test]
    fn test_silhouette_score_single_cluster() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let data = vec![0.0; n * m];

        // All in one cluster
        let cluster = vec![0usize; n];

        let scores = silhouette_score(&data, n, m, &t, &cluster);

        // Single cluster should give zeros
        for &s in &scores {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_silhouette_score_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let scores = silhouette_score(&[], 0, 30, &t, &[]);
        assert!(scores.is_empty());

        // Mismatched cluster length
        let data = vec![0.0; 10 * 30];
        let cluster = vec![0; 5]; // Wrong length
        let scores = silhouette_score(&data, 10, 30, &t, &cluster);
        assert!(scores.is_empty());
    }

    // ============== Calinski-Harabasz tests ==============

    #[test]
    fn test_calinski_harabasz_well_separated() {
        let m = 30;
        let n_per = 10;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let cluster: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        let ch = calinski_harabasz(&data, n, m, &t, &cluster);

        // Well-separated clusters should have high CH index
        assert!(
            ch > 1.0,
            "Well-separated clusters should have high CH: {}",
            ch
        );
    }

    #[test]
    fn test_calinski_harabasz_positive() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let cluster: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        let ch = calinski_harabasz(&data, n, m, &t, &cluster);

        assert!(ch >= 0.0, "CH index should be non-negative");
    }

    #[test]
    fn test_calinski_harabasz_single_cluster() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let data = vec![0.0; n * m];

        // All in one cluster
        let cluster = vec![0usize; n];

        let ch = calinski_harabasz(&data, n, m, &t, &cluster);

        // Single cluster should give 0
        assert!(ch.abs() < 1e-10);
    }

    #[test]
    fn test_calinski_harabasz_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let ch = calinski_harabasz(&[], 0, 30, &t, &[]);
        assert!(ch.abs() < 1e-10);
    }
}
