//! Clustering algorithms for functional data.
//!
//! This module provides k-means and fuzzy c-means clustering algorithms
//! for functional data.

use crate::helpers::simpsons_weights;
use rand::prelude::*;
use rayon::prelude::*;

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

/// Compute L2 distance between two curves.
fn l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64 {
    let mut dist_sq = 0.0;
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * weights[i];
    }
    dist_sq.sqrt()
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

    // Extract curves
    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

    // k-means++ initialization
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
                    .map(|c| l2_distance(curve, c, &weights))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let dist_sq: Vec<f64> = distances.iter().map(|d| d * d).collect();
        let total: f64 = dist_sq.iter().sum();

        if total < 1e-10 {
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

    let mut cluster = vec![0usize; n];
    let mut converged = false;
    let mut iter = 0;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Assignment step
        let new_cluster: Vec<usize> = curves
            .par_iter()
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
                let members: Vec<usize> =
                    cluster.iter().enumerate().filter(|(_, &cl)| cl == c).map(|(i, _)| i).collect();

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

    // Extract curves
    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

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

            if denominator > 1e-10 {
                for j in 0..m {
                    centers[c][j] = numerator[j] / denominator;
                }
            }
        }

        // Update membership
        let mut new_membership = vec![0.0; n * k];
        let mut max_change = 0.0;

        for (i, curve) in curves.iter().enumerate() {
            let distances: Vec<f64> =
                centers.iter().map(|c| l2_distance(curve, c, &weights)).collect();

            for c in 0..k {
                if distances[c] < 1e-10 {
                    new_membership[i + c * n] = 1.0;
                    for c2 in 0..k {
                        if c2 != c {
                            new_membership[i + c2 * n] = 0.0;
                        }
                    }
                    break;
                }

                let mut sum = 0.0;
                for c2 in 0..k {
                    if distances[c2] > 1e-10 {
                        sum += (distances[c] / distances[c2]).powf(exponent);
                    }
                }
                new_membership[i + c * n] = if sum > 1e-10 { 1.0 / sum } else { 1.0 };
            }

            for c in 0..k {
                let change = (new_membership[i + c * n] - membership[i + c * n]).abs();
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
pub fn silhouette_score(data: &[f64], n: usize, m: usize, argvals: &[f64], cluster: &[usize]) -> Vec<f64> {
    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

    let k = cluster.iter().cloned().max().unwrap_or(0) + 1;

    (0..n)
        .into_par_iter()
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
                if max_ab > 1e-10 {
                    (b_i - a_i) / max_ab
                } else {
                    0.0
                }
            }
        })
        .collect()
}

/// Compute Calinski-Harabasz index for clustering result.
pub fn calinski_harabasz(data: &[f64], n: usize, m: usize, argvals: &[f64], cluster: &[usize]) -> f64 {
    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

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

    if wgss < 1e-10 {
        return f64::INFINITY;
    }

    (bgss / (k - 1) as f64) / (wgss / (n - k) as f64)
}
