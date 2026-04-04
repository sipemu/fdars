//! Clustering algorithms for functional data.
//!
//! This module provides k-means and fuzzy c-means clustering algorithms
//! for functional data.

use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use crate::{iter_maybe_parallel, maybe_par_chunks_mut_enumerate, slice_maybe_parallel};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of k-means clustering.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KmeansResult {
    /// Cluster assignments for each observation
    pub cluster: Vec<usize>,
    /// Cluster centers (k x m matrix)
    pub centers: FdMatrix,
    /// Within-cluster sum of squares for each cluster
    pub withinss: Vec<f64>,
    /// Total within-cluster sum of squares
    pub tot_withinss: f64,
    /// Number of iterations
    pub iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl KmeansResult {
    /// Assign new observations to the nearest cluster center.
    ///
    /// For each row in `data`, computes the weighted L2 distance to each
    /// cluster center using Simpson's integration weights derived from
    /// `argvals`, and assigns the observation to the cluster with the
    /// minimum distance.
    ///
    /// # Arguments
    /// * `data` - Matrix (n_new x m) of new observations
    /// * `argvals` - Evaluation points (length m)
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if the number of columns in
    /// `data` does not match the number of columns in the cluster centers, or
    /// if `argvals.len()` does not match the number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::clustering::kmeans_fd;
    ///
    /// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
    /// let data = FdMatrix::from_column_major(
    ///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     10, 20,
    /// ).unwrap();
    /// let result = kmeans_fd(&data, &argvals, 2, 100, 1e-6, 42).unwrap();
    ///
    /// // Predict cluster assignments for new data
    /// let new_data = FdMatrix::from_column_major(
    ///     (0..60).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     3, 20,
    /// ).unwrap();
    /// let assignments = result.predict(&new_data, &argvals).unwrap();
    /// assert_eq!(assignments.len(), 3);
    /// assert!(assignments.iter().all(|&c| c < 2));
    /// ```
    pub fn predict(&self, data: &FdMatrix, argvals: &[f64]) -> Result<Vec<usize>, FdarError> {
        let (n, m) = data.shape();
        let m_centers = self.centers.ncols();
        let k = self.centers.nrows();
        if m != m_centers {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{m_centers} columns"),
                actual: format!("{m} columns"),
            });
        }
        if argvals.len() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: format!("{m}"),
                actual: format!("{}", argvals.len()),
            });
        }

        let weights = simpsons_weights(argvals);
        let curves = data.to_row_major();
        let centers = self.centers.to_row_major();
        Ok(assign_clusters(&curves, n, m, &centers, k, &weights))
    }
}

/// K-means++ initialization: select initial centers with probability proportional to D^2.
///
/// Uses incremental distance tracking: maintains a `min_dist_sq` vector and only
/// computes distances to the newest center on each iteration, avoiding redundant
/// distance computations to all existing centers.
///
/// # Arguments
/// * `curves` - Flat row-major buffer of curves (n curves, each m values)
/// * `n` - Number of curves
/// * `m` - Number of evaluation points per curve
/// * `k` - Number of clusters
/// * `weights` - Integration weights for L2 distance
/// * `rng` - Random number generator
///
/// # Returns
/// Flat buffer of k initial cluster centers (k * m values)
/// Select an index with probability proportional to the given weights.
fn weighted_random_select(dist_sq: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = dist_sq.iter().sum();
    if total < NUMERICAL_EPS {
        return rng.gen_range(0..dist_sq.len());
    }
    let r = rng.gen::<f64>() * total;
    let mut cumsum = 0.0;
    for (i, &d) in dist_sq.iter().enumerate() {
        cumsum += d;
        if cumsum >= r {
            return i;
        }
    }
    dist_sq.len() - 1
}

fn kmeans_plusplus_init(
    curves: &[f64],
    n: usize,
    m: usize,
    k: usize,
    weights: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * m];

    // First center: random
    let first_idx = rng.gen_range(0..n);
    centers[..m].copy_from_slice(&curves[first_idx * m..(first_idx + 1) * m]);

    // Initialize min_dist_sq with squared distances to first center
    let center0 = &centers[..m];
    let mut min_dist_sq: Vec<f64> = (0..n)
        .map(|i| {
            let d = l2_distance(&curves[i * m..(i + 1) * m], center0, weights);
            d * d
        })
        .collect();

    // Remaining centers: probability proportional to D^2
    for c_idx in 1..k {
        let chosen = weighted_random_select(&min_dist_sq, rng);
        centers[c_idx * m..(c_idx + 1) * m].copy_from_slice(&curves[chosen * m..(chosen + 1) * m]);

        // Update min_dist_sq: only compute distance to the newest center
        let new_center = &centers[c_idx * m..(c_idx + 1) * m];
        maybe_par_chunks_mut_enumerate!(min_dist_sq, 1, |(i, chunk): (usize, &mut [f64])| {
            let d_sq = l2_distance(&curves[i * m..(i + 1) * m], new_center, weights).powi(2);
            if d_sq < chunk[0] {
                chunk[0] = d_sq;
            }
        });
    }

    centers
}

/// Compute fuzzy membership values for a single observation, writing into `out`.
///
/// # Arguments
/// * `distances` - Distances from the observation to each cluster center
/// * `k` - Number of clusters
/// * `exponent` - Exponent for fuzzy membership (2 / (fuzziness - 1))
/// * `out` - Output slice (length k) to write membership values into
fn compute_fuzzy_membership_into(distances: &[f64], k: usize, exponent: f64, out: &mut [f64]) {
    out[..k].fill(0.0);

    // Check if observation is very close to any center
    for (c, &dist) in distances[..k].iter().enumerate() {
        if dist < NUMERICAL_EPS {
            // Assign full membership to this cluster
            out[c] = 1.0;
            return;
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
        out[c] = if sum > NUMERICAL_EPS { 1.0 / sum } else { 1.0 };
    }
}

/// Build an FdMatrix (k x m) from flat row-major centers buffer.
fn centers_to_matrix(centers: &[f64], k: usize, m: usize) -> FdMatrix {
    let mut flat = vec![0.0; k * m];
    for c in 0..k {
        for j in 0..m {
            flat[c + j * k] = centers[c * m + j];
        }
    }
    FdMatrix::from_column_major(flat, k, m).expect("dimension invariant: data.len() == n * m")
}

/// Initialize a random membership matrix (n x k) with rows summing to 1.
fn init_random_membership(n: usize, k: usize, rng: &mut StdRng) -> FdMatrix {
    let mut membership = FdMatrix::zeros(n, k);
    for i in 0..n {
        let mut row_sum = 0.0;
        for c in 0..k {
            let val = rng.gen::<f64>();
            membership[(i, c)] = val;
            row_sum += val;
        }
        for c in 0..k {
            membership[(i, c)] /= row_sum;
        }
    }
    membership
}

/// Group sample indices by their cluster assignment.
fn cluster_member_indices(cluster: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut indices = vec![Vec::new(); k];
    for (i, &c) in cluster.iter().enumerate() {
        indices[c].push(i);
    }
    indices
}

/// Assign each curve to its nearest center, returning cluster indices.
fn assign_clusters(
    curves: &[f64],
    n: usize,
    m: usize,
    centers: &[f64],
    k: usize,
    weights: &[f64],
) -> Vec<usize> {
    // Build a slice of curve slices for parallel iteration
    let curve_indices: Vec<usize> = (0..n).collect();
    slice_maybe_parallel!(curve_indices)
        .map(|&i| {
            let curve = &curves[i * m..(i + 1) * m];
            let mut best_cluster = 0;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let center = &centers[c * m..(c + 1) * m];
                let dist = l2_distance(curve, center, weights);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            best_cluster
        })
        .collect()
}

/// Compute new cluster centers from curve assignments.
fn update_kmeans_centers(
    curves: &[f64],
    n: usize,
    m: usize,
    assignments: &[usize],
    old_centers: &[f64],
    k: usize,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * m];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        let curve = &curves[i * m..(i + 1) * m];
        let center = &mut centers[c * m..(c + 1) * m];
        for j in 0..m {
            center[j] += curve[j];
        }
    }

    for c in 0..k {
        if counts[c] > 0 {
            let center = &mut centers[c * m..(c + 1) * m];
            let n_members = counts[c] as f64;
            for j in 0..m {
                center[j] /= n_members;
            }
        } else {
            // Keep old center for empty clusters
            centers[c * m..(c + 1) * m].copy_from_slice(&old_centers[c * m..(c + 1) * m]);
        }
    }

    centers
}

/// Compute within-cluster sum of squares for each cluster.
fn compute_within_ss(
    curves: &[f64],
    n: usize,
    m: usize,
    centers: &[f64],
    assignments: &[usize],
    k: usize,
    weights: &[f64],
) -> Vec<f64> {
    let mut withinss = vec![0.0; k];
    for i in 0..n {
        let c = assignments[i];
        let dist = l2_distance(
            &curves[i * m..(i + 1) * m],
            &centers[c * m..(c + 1) * m],
            weights,
        );
        withinss[c] += dist * dist;
    }
    withinss
}

/// Update fuzzy c-means cluster centers from membership values.
fn update_fuzzy_centers(
    curves: &[f64],
    n: usize,
    m: usize,
    membership: &FdMatrix,
    k: usize,
    fuzziness: f64,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * m];
    for c in 0..k {
        let mut denominator = 0.0;
        let center = &mut centers[c * m..(c + 1) * m];

        for i in 0..n {
            let weight = membership[(i, c)].powf(fuzziness);
            let curve = &curves[i * m..(i + 1) * m];
            for j in 0..m {
                center[j] += weight * curve[j];
            }
            denominator += weight;
        }

        if denominator > NUMERICAL_EPS {
            for j in 0..m {
                center[j] /= denominator;
            }
        }
    }
    centers
}

/// Update fuzzy membership values and compute max change.
fn update_fuzzy_membership_step(
    curves: &[f64],
    n: usize,
    m: usize,
    centers: &[f64],
    k: usize,
    old_membership: &FdMatrix,
    exponent: f64,
    weights: &[f64],
) -> (FdMatrix, f64) {
    let mut new_membership = FdMatrix::zeros(n, k);
    let mut max_change = 0.0;
    let mut distances = vec![0.0; k];
    let mut memberships = vec![0.0; k];

    for i in 0..n {
        let curve = &curves[i * m..(i + 1) * m];
        for c in 0..k {
            distances[c] = l2_distance(curve, &centers[c * m..(c + 1) * m], weights);
        }

        compute_fuzzy_membership_into(&distances, k, exponent, &mut memberships);

        for c in 0..k {
            new_membership[(i, c)] = memberships[c];
            let change = (memberships[c] - old_membership[(i, c)]).abs();
            if change > max_change {
                max_change = change;
            }
        }
    }

    (new_membership, max_change)
}

/// Compute mean L2 distance from a curve to a set of curve indices.
fn mean_cluster_distance(
    curve: &[f64],
    curves: &[f64],
    m: usize,
    indices: &[usize],
    weights: &[f64],
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f64 = indices
        .iter()
        .map(|&j| l2_distance(curve, &curves[j * m..(j + 1) * m], weights))
        .sum();
    sum / indices.len() as f64
}

/// Compute cluster centers, global mean, and counts from curves and assignments.
fn compute_centers_and_global_mean(
    curves: &[f64],
    n: usize,
    m: usize,
    assignments: &[usize],
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut global_mean = vec![0.0; m];
    for i in 0..n {
        let curve = &curves[i * m..(i + 1) * m];
        for j in 0..m {
            global_mean[j] += curve[j];
        }
    }
    for j in 0..m {
        global_mean[j] /= n as f64;
    }

    let mut centers = vec![0.0; k * m];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        let curve = &curves[i * m..(i + 1) * m];
        let center = &mut centers[c * m..(c + 1) * m];
        for j in 0..m {
            center[j] += curve[j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            let center = &mut centers[c * m..(c + 1) * m];
            for j in 0..m {
                center[j] /= counts[c] as f64;
            }
        }
    }

    (centers, global_mean, counts)
}

/// Run one k-means iteration: assign clusters, update centers, compute movement.
fn kmeans_step(
    curves: &[f64],
    n: usize,
    m: usize,
    centers: &[f64],
    k: usize,
    weights: &[f64],
) -> (Vec<usize>, Vec<f64>, f64) {
    let new_cluster = assign_clusters(curves, n, m, centers, k, weights);
    let new_centers = update_kmeans_centers(curves, n, m, &new_cluster, centers, k);
    let max_movement = (0..k)
        .map(|c| {
            l2_distance(
                &centers[c * m..(c + 1) * m],
                &new_centers[c * m..(c + 1) * m],
                weights,
            )
        })
        .fold(0.0, f64::max);
    (new_cluster, new_centers, max_movement)
}

/// Run the k-means iteration loop until convergence or max iterations.
fn kmeans_iterate(
    curves: &[f64],
    n: usize,
    m: usize,
    mut centers: Vec<f64>,
    k: usize,
    weights: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<usize>, Vec<f64>, usize, bool) {
    let mut cluster = vec![0usize; n];
    let mut converged = false;
    let mut iter = 0;

    for iteration in 0..max_iter {
        iter = iteration + 1;
        let (new_cluster, new_centers, max_movement) =
            kmeans_step(curves, n, m, &centers, k, weights);

        if new_cluster == cluster {
            converged = true;
            break;
        }
        cluster = new_cluster;
        centers = new_centers;

        if max_movement < tol {
            converged = true;
            break;
        }
    }

    (cluster, centers, iter, converged)
}

/// K-means clustering for functional data.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points
/// * `k` - Number of clusters
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::clustering::kmeans_fd;
///
/// // 10 curves at 20 evaluation points
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),
///     10, 20,
/// ).unwrap();
/// let result = kmeans_fd(&data, &argvals, 2, 100, 1e-6, 42).unwrap();
/// assert_eq!(result.cluster.len(), 10);
/// assert_eq!(result.centers.nrows(), 2);
/// assert!(result.converged);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kmeans_fd(
    data: &FdMatrix,
    argvals: &[f64],
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<KmeansResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".into(),
            actual: format!("{n}x{m}"),
        });
    }
    if k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "number of clusters must be > 0".into(),
        });
    }
    if k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k={k} exceeds number of observations n={n}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    let weights = simpsons_weights(argvals);
    let mut rng = StdRng::seed_from_u64(seed);

    // Extract curves as flat row-major buffer
    let curves = data.to_row_major();

    // K-means++ initialization
    let centers = kmeans_plusplus_init(&curves, n, m, k, &weights, &mut rng);

    let (cluster, centers, iter, converged) =
        kmeans_iterate(&curves, n, m, centers, k, &weights, max_iter, tol);

    let withinss = compute_within_ss(&curves, n, m, &centers, &cluster, k, &weights);
    let tot_withinss: f64 = withinss.iter().sum();
    let centers_mat = centers_to_matrix(&centers, k, m);

    Ok(KmeansResult {
        cluster,
        centers: centers_mat,
        withinss,
        tot_withinss,
        iter,
        converged,
    })
}

/// Result of fuzzy c-means clustering.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FuzzyCmeansResult {
    /// Membership matrix (n x k)
    pub membership: FdMatrix,
    /// Cluster centers (k x m)
    pub centers: FdMatrix,
    /// Fuzziness parameter used during fitting
    pub fuzziness: f64,
    /// Number of iterations
    pub iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl FuzzyCmeansResult {
    /// Compute fuzzy membership values for new observations.
    ///
    /// For each new observation, computes the weighted L2 distance to each
    /// cluster center and derives fuzzy membership values using the same
    /// fuzziness parameter used during fitting.
    ///
    /// Each row of the returned matrix sums to 1.0, with values in \[0, 1\]
    /// indicating the degree of membership in each cluster.
    ///
    /// # Arguments
    /// * `data` - Matrix (n_new x m) of new observations
    /// * `argvals` - Evaluation points (length m)
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if the number of columns in
    /// `data` does not match the number of columns in the cluster centers, or
    /// if `argvals.len()` does not match the number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::clustering::fuzzy_cmeans_fd;
    ///
    /// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
    /// let data = FdMatrix::from_column_major(
    ///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     10, 20,
    /// ).unwrap();
    /// let result = fuzzy_cmeans_fd(&data, &argvals, 2, 2.0, 100, 1e-6, 42).unwrap();
    ///
    /// // Predict membership for new data
    /// let new_data = FdMatrix::from_column_major(
    ///     (0..60).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     3, 20,
    /// ).unwrap();
    /// let membership = result.predict(&new_data, &argvals).unwrap();
    /// assert_eq!(membership.shape(), (3, 2));
    /// // Each row should sum to 1
    /// for i in 0..3 {
    ///     let sum: f64 = (0..2).map(|c| membership[(i, c)]).sum();
    ///     assert!((sum - 1.0).abs() < 1e-6);
    /// }
    /// ```
    pub fn predict(&self, data: &FdMatrix, argvals: &[f64]) -> Result<FdMatrix, FdarError> {
        let (n, m) = data.shape();
        let m_centers = self.centers.ncols();
        let k = self.centers.nrows();
        if m != m_centers {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{m_centers} columns"),
                actual: format!("{m} columns"),
            });
        }
        if argvals.len() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: format!("{m}"),
                actual: format!("{}", argvals.len()),
            });
        }

        let weights = simpsons_weights(argvals);
        let curves = data.to_row_major();
        let centers = self.centers.to_row_major();
        let exponent = 2.0 / (self.fuzziness - 1.0);

        let mut membership = FdMatrix::zeros(n, k);
        let mut distances = vec![0.0; k];
        let mut memberships = vec![0.0; k];

        for i in 0..n {
            let curve = &curves[i * m..(i + 1) * m];
            for c in 0..k {
                distances[c] = l2_distance(curve, &centers[c * m..(c + 1) * m], &weights);
            }

            compute_fuzzy_membership_into(&distances, k, exponent, &mut memberships);

            for c in 0..k {
                membership[(i, c)] = memberships[c];
            }
        }

        Ok(membership)
    }
}

/// Fuzzy c-means clustering for functional data.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points
/// * `k` - Number of clusters
/// * `fuzziness` - Fuzziness parameter (> 1)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fuzzy_cmeans_fd(
    data: &FdMatrix,
    argvals: &[f64],
    k: usize,
    fuzziness: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<FuzzyCmeansResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".into(),
            actual: format!("{n}x{m}"),
        });
    }
    if k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "number of clusters must be > 0".into(),
        });
    }
    if k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k={k} exceeds number of observations n={n}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if fuzziness <= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "fuzziness",
            message: format!("fuzziness must be > 1.0, got {fuzziness}"),
        });
    }

    let weights = simpsons_weights(argvals);
    let mut rng = StdRng::seed_from_u64(seed);

    // Extract curves as flat row-major buffer
    let curves = data.to_row_major();

    let mut membership = init_random_membership(n, k, &mut rng);

    let mut centers = vec![0.0; k * m];
    let mut converged = false;
    let mut iter = 0;
    let exponent = 2.0 / (fuzziness - 1.0);

    for iteration in 0..max_iter {
        iter = iteration + 1;

        centers = update_fuzzy_centers(&curves, n, m, &membership, k, fuzziness);

        let (new_membership, max_change) = update_fuzzy_membership_step(
            &curves,
            n,
            m,
            &centers,
            k,
            &membership,
            exponent,
            &weights,
        );

        membership = new_membership;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    let centers_mat = centers_to_matrix(&centers, k, m);

    Ok(FuzzyCmeansResult {
        membership,
        centers: centers_mat,
        fuzziness,
        iter,
        converged,
    })
}

/// Compute silhouette score for clustering result.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::clustering::silhouette_score;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..60).map(|i| (i as f64 * 0.1).sin()).collect(),
///     6, 10,
/// ).unwrap();
/// let cluster = vec![0, 0, 0, 1, 1, 1];
/// let scores = silhouette_score(&data, &argvals, &cluster);
/// assert_eq!(scores.len(), 6);
/// // Silhouette scores are in [-1, 1]
/// assert!(scores.iter().all(|&s| s >= -1.0 - 1e-10 && s <= 1.0 + 1e-10));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn silhouette_score(data: &FdMatrix, argvals: &[f64], cluster: &[usize]) -> Vec<f64> {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    let curves = data.to_row_major();

    let k = cluster.iter().copied().max().unwrap_or(0) + 1;
    let members = cluster_member_indices(cluster, k);

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let my_cluster = cluster[i];
            let curve_i = &curves[i * m..(i + 1) * m];

            let same_indices: Vec<usize> = members[my_cluster]
                .iter()
                .copied()
                .filter(|&j| j != i)
                .collect();
            let a_i = mean_cluster_distance(curve_i, &curves, m, &same_indices, &weights);

            let mut b_i = f64::INFINITY;
            for c in 0..k {
                if c != my_cluster && !members[c].is_empty() {
                    b_i = b_i.min(mean_cluster_distance(
                        curve_i,
                        &curves,
                        m,
                        &members[c],
                        &weights,
                    ));
                }
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

/// Silhouette score from a precomputed distance matrix.
///
/// Works with any distance matrix (elastic, DTW, Lp, or custom).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn silhouette_score_from_distances(dist_mat: &FdMatrix, cluster: &[usize]) -> Vec<f64> {
    let n = dist_mat.nrows();
    if n == 0 || dist_mat.ncols() != n || cluster.len() != n {
        return Vec::new();
    }

    let k = cluster.iter().copied().max().unwrap_or(0) + 1;
    let members = cluster_member_indices(cluster, k);

    (0..n)
        .map(|i| {
            let my_cluster = cluster[i];

            // a(i) = mean distance to same-cluster members
            let same: Vec<usize> = members[my_cluster]
                .iter()
                .copied()
                .filter(|&j| j != i)
                .collect();
            let a_i = if same.is_empty() {
                0.0
            } else {
                same.iter().map(|&j| dist_mat[(i, j)]).sum::<f64>() / same.len() as f64
            };

            // b(i) = min over other clusters of mean distance
            let mut b_i = f64::INFINITY;
            for c in 0..k {
                if c != my_cluster && !members[c].is_empty() {
                    let mean_d = members[c].iter().map(|&j| dist_mat[(i, j)]).sum::<f64>()
                        / members[c].len() as f64;
                    b_i = b_i.min(mean_d);
                }
            }

            if b_i.is_infinite() {
                0.0
            } else {
                let max_ab = a_i.max(b_i);
                if max_ab > 1e-15 {
                    (b_i - a_i) / max_ab
                } else {
                    0.0
                }
            }
        })
        .collect()
}

/// Compute Calinski-Harabasz index for clustering result.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn calinski_harabasz(data: &FdMatrix, argvals: &[f64], cluster: &[usize]) -> f64 {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m == 0 || cluster.len() != n || argvals.len() != m {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let curves = data.to_row_major();

    let k = cluster.iter().copied().max().unwrap_or(0) + 1;
    if k < 2 {
        return 0.0;
    }

    let (centers, global_mean, counts) = compute_centers_and_global_mean(&curves, n, m, cluster, k);

    let mut bgss = 0.0;
    for c in 0..k {
        let dist = l2_distance(&centers[c * m..(c + 1) * m], &global_mean, &weights);
        bgss += counts[c] as f64 * dist * dist;
    }

    let wgss_vec = compute_within_ss(&curves, n, m, &centers, cluster, k, &weights);
    let wgss: f64 = wgss_vec.iter().sum();

    if wgss < NUMERICAL_EPS {
        return f64::INFINITY;
    }

    (bgss / (k - 1) as f64) / (wgss / (n - k) as f64)
}

/// Calinski-Harabasz index from a precomputed distance matrix.
///
/// Uses the distance-based formulation: CH = [B/(k-1)] / [W/(n-k)]
/// where B = total between-cluster distance, W = total within-cluster distance.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn calinski_harabasz_from_distances(dist_mat: &FdMatrix, cluster: &[usize]) -> f64 {
    let n = dist_mat.nrows();
    if n == 0 || dist_mat.ncols() != n || cluster.len() != n {
        return 0.0;
    }

    let k = cluster.iter().copied().max().unwrap_or(0) + 1;
    if k < 2 || n <= k {
        return 0.0;
    }

    // Total dispersion: sum of all pairwise squared distances
    let total_disp: f64 = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| dist_mat[(i, j)].powi(2)))
        .sum::<f64>();

    // Within-cluster dispersion
    let members = cluster_member_indices(cluster, k);
    let mut within = 0.0;
    for c in 0..k {
        let nc = members[c].len();
        if nc < 2 {
            continue;
        }
        for ii in 0..nc {
            for jj in (ii + 1)..nc {
                within += dist_mat[(members[c][ii], members[c][jj])].powi(2);
            }
        }
    }

    let between = total_disp - within;
    // Normalize: account for cluster sizes
    let w_norm = within / (n - k) as f64;
    let b_norm = between / (k - 1) as f64;

    if w_norm > 1e-15 {
        b_norm / w_norm
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

    /// Generate two clearly separated clusters of curves as an FdMatrix
    fn generate_two_clusters(n_per_cluster: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let n = 2 * n_per_cluster;
        let mut col_major = vec![0.0; n * m];

        // Cluster 0: sine waves with low amplitude
        for i in 0..n_per_cluster {
            for (j, &ti) in t.iter().enumerate() {
                col_major[i + j * n] =
                    (2.0 * PI * ti).sin() + 0.1 * (i as f64 / n_per_cluster as f64);
            }
        }

        // Cluster 1: sine waves shifted up by 5
        for i in 0..n_per_cluster {
            for (j, &ti) in t.iter().enumerate() {
                col_major[(i + n_per_cluster) + j * n] =
                    (2.0 * PI * ti).sin() + 5.0 + 0.1 * (i as f64 / n_per_cluster as f64);
            }
        }

        (FdMatrix::from_column_major(col_major, n, m).unwrap(), t)
    }

    // ============== K-means tests ==============

    #[test]
    fn test_kmeans_fd_basic() {
        let m = 50;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

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

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

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

        let result1 = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();
        let result2 = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

        // Same seed should give same results
        assert_eq!(result1.cluster, result2.cluster);
    }

    #[test]
    fn test_kmeans_fd_withinss() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

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
        let k = 3;

        let result = kmeans_fd(&data, &t, k, 100, 1e-6, 42).unwrap();

        // Centers should be k x m matrix
        assert_eq!(result.centers.nrows(), k);
        assert_eq!(result.centers.ncols(), m);
    }

    #[test]
    fn test_kmeans_fd_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let data = FdMatrix::zeros(0, 0);
        assert!(kmeans_fd(&data, &t, 2, 100, 1e-6, 42).is_err());

        // k > n
        let data = FdMatrix::zeros(5, 30);
        assert!(kmeans_fd(&data, &t, 10, 100, 1e-6, 42).is_err());
    }

    #[test]
    fn test_kmeans_fd_single_cluster() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let data = FdMatrix::zeros(n, m);

        let result = kmeans_fd(&data, &t, 1, 100, 1e-6, 42).unwrap();

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

        let result = fuzzy_cmeans_fd(&data, &t, 2, 2.0, 100, 1e-6, 42).unwrap();

        assert_eq!(result.membership.nrows(), n);
        assert_eq!(result.membership.ncols(), 2);
        assert!(result.iter > 0);
    }

    #[test]
    fn test_fuzzy_cmeans_fd_membership_sums_to_one() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let n = 2 * n_per;
        let k = 2;

        let result = fuzzy_cmeans_fd(&data, &t, k, 2.0, 100, 1e-6, 42).unwrap();

        // Each observation's membership should sum to 1
        for i in 0..n {
            let sum: f64 = (0..k).map(|c| result.membership[(i, c)]).sum();
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

        let result = fuzzy_cmeans_fd(&data, &t, 2, 2.0, 100, 1e-6, 42).unwrap();

        // All memberships should be in [0, 1]
        for &mem in result.membership.as_slice() {
            assert!((0.0..=1.0 + 1e-10).contains(&mem));
        }
    }

    #[test]
    fn test_fuzzy_cmeans_fd_fuzziness_effect() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);

        let result_low = fuzzy_cmeans_fd(&data, &t, 2, 1.5, 100, 1e-6, 42).unwrap();
        let result_high = fuzzy_cmeans_fd(&data, &t, 2, 3.0, 100, 1e-6, 42).unwrap();

        // Higher fuzziness should give more diffuse memberships
        // Measure by entropy-like metric
        let entropy_low: f64 = result_low
            .membership
            .as_slice()
            .iter()
            .map(|&m| if m > 1e-10 { -m * m.ln() } else { 0.0 })
            .sum();

        let entropy_high: f64 = result_high
            .membership
            .as_slice()
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
        let data = FdMatrix::zeros(10, 30);

        // Fuzziness <= 1 should fail
        assert!(fuzzy_cmeans_fd(&data, &t, 2, 1.0, 100, 1e-6, 42).is_err());
        assert!(fuzzy_cmeans_fd(&data, &t, 2, 0.5, 100, 1e-6, 42).is_err());
    }

    #[test]
    fn test_fuzzy_cmeans_fd_centers_shape() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let k = 3;
        let data = FdMatrix::zeros(n, m);

        let result = fuzzy_cmeans_fd(&data, &t, k, 2.0, 100, 1e-6, 42).unwrap();

        assert_eq!(result.centers.nrows(), k);
        assert_eq!(result.centers.ncols(), m);
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

        let scores = silhouette_score(&data, &t, &cluster);

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

        let scores = silhouette_score(&data, &t, &cluster);

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
        let data = FdMatrix::zeros(n, m);

        // All in one cluster
        let cluster = vec![0usize; n];

        let scores = silhouette_score(&data, &t, &cluster);

        // Single cluster should give zeros
        for &s in &scores {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_silhouette_score_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let data = FdMatrix::zeros(0, 0);
        let scores = silhouette_score(&data, &t, &[]);
        assert!(scores.is_empty());

        // Mismatched cluster length
        let data = FdMatrix::zeros(10, 30);
        let cluster = vec![0; 5]; // Wrong length
        let scores = silhouette_score(&data, &t, &cluster);
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

        let ch = calinski_harabasz(&data, &t, &cluster);

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

        let ch = calinski_harabasz(&data, &t, &cluster);

        assert!(ch >= 0.0, "CH index should be non-negative");
    }

    #[test]
    fn test_calinski_harabasz_single_cluster() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        let data = FdMatrix::zeros(n, m);

        // All in one cluster
        let cluster = vec![0usize; n];

        let ch = calinski_harabasz(&data, &t, &cluster);

        // Single cluster should give 0
        assert!(ch.abs() < 1e-10);
    }

    #[test]
    fn test_calinski_harabasz_invalid_input() {
        let t = uniform_grid(30);

        // Empty data
        let data = FdMatrix::zeros(0, 0);
        let ch = calinski_harabasz(&data, &t, &[]);
        assert!(ch.abs() < 1e-10);
    }

    #[test]
    fn test_identical_curves_kmeans() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 10;
        // All curves identical
        let data_vec: Vec<f64> = (0..n * m)
            .map(|idx| (2.0 * PI * (idx % m) as f64 / (m - 1) as f64).sin())
            .collect();
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();
        // Should not panic with identical data
        assert_eq!(result.cluster.len(), n);
    }

    #[test]
    fn test_k_equals_n() {
        let m = 30;
        let t = uniform_grid(m);
        let n = 5;
        let (data, _) = generate_two_clusters(n, m);
        let result = kmeans_fd(&data, &t, 2 * n, 100, 1e-6, 42).unwrap();
        // k == n: each curve is its own cluster
        assert_eq!(result.cluster.len(), 2 * n);
    }

    #[test]
    fn test_n2_kmeans() {
        let m = 30;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(2, m);
        for j in 0..m {
            data[(0, j)] = 0.0;
            data[(1, j)] = 10.0;
        }
        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();
        assert_eq!(result.cluster.len(), 2);
        assert_ne!(result.cluster[0], result.cluster[1]);
    }

    // ============== KmeansResult::predict tests ==============

    #[test]
    fn test_kmeans_predict_shape() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

        let new_data = FdMatrix::zeros(3, m);
        let assignments = result.predict(&new_data, &t).unwrap();
        assert_eq!(assignments.len(), 3);
    }

    #[test]
    fn test_kmeans_predict_reproduces_training() {
        let m = 30;
        let n_per = 10;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

        // Predicting on training data should reproduce original assignments
        let predicted = result.predict(&data, &t).unwrap();
        assert_eq!(predicted, result.cluster);
    }

    #[test]
    fn test_kmeans_predict_correct_cluster() {
        let m = 30;
        let n_per = 10;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

        // Create a new curve clearly in cluster 0 (low amplitude sine)
        let mut new_data = FdMatrix::zeros(1, m);
        for j in 0..m {
            new_data[(0, j)] = (2.0 * PI * t[j]).sin();
        }
        let pred = result.predict(&new_data, &t).unwrap();
        assert_eq!(pred[0], result.cluster[0]); // same cluster as first training group

        // Create a new curve clearly in cluster 1 (shifted up by 5)
        let mut new_data2 = FdMatrix::zeros(1, m);
        for j in 0..m {
            new_data2[(0, j)] = (2.0 * PI * t[j]).sin() + 5.0;
        }
        let pred2 = result.predict(&new_data2, &t).unwrap();
        assert_eq!(pred2[0], result.cluster[n_per]); // same cluster as second group
    }

    #[test]
    fn test_kmeans_predict_dimension_mismatch() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let result = kmeans_fd(&data, &t, 2, 100, 1e-6, 42).unwrap();

        // Wrong number of columns
        let wrong_data = FdMatrix::zeros(3, 20);
        assert!(result.predict(&wrong_data, &t).is_err());

        // Wrong argvals length
        let new_data = FdMatrix::zeros(3, m);
        let wrong_t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        assert!(result.predict(&new_data, &wrong_t).is_err());
    }

    // ============== FuzzyCmeansResult::predict tests ==============

    #[test]
    fn test_fuzzy_predict_shape() {
        let m = 30;
        let n_per = 5;
        let k = 2;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = fuzzy_cmeans_fd(&data, &t, k, 2.0, 100, 1e-6, 42).unwrap();

        let new_data = FdMatrix::zeros(3, m);
        let membership = result.predict(&new_data, &t).unwrap();
        assert_eq!(membership.shape(), (3, k));
    }

    #[test]
    fn test_fuzzy_predict_membership_sums_to_one() {
        let m = 30;
        let n_per = 5;
        let k = 2;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = fuzzy_cmeans_fd(&data, &t, k, 2.0, 100, 1e-6, 42).unwrap();

        let new_data = FdMatrix::zeros(4, m);
        let membership = result.predict(&new_data, &t).unwrap();

        for i in 0..4 {
            let sum: f64 = (0..k).map(|c| membership[(i, c)]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Row {} membership should sum to 1, got {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_fuzzy_predict_membership_in_range() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = fuzzy_cmeans_fd(&data, &t, 2, 2.0, 100, 1e-6, 42).unwrap();

        let new_data = FdMatrix::zeros(4, m);
        let membership = result.predict(&new_data, &t).unwrap();

        for &mem in membership.as_slice() {
            assert!((0.0..=1.0 + 1e-10).contains(&mem));
        }
    }

    #[test]
    fn test_fuzzy_predict_reproduces_training() {
        let m = 30;
        let n_per = 5;
        let n = 2 * n_per;
        let k = 2;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = fuzzy_cmeans_fd(&data, &t, k, 2.0, 100, 1e-6, 42).unwrap();

        // Predicting on training data should reproduce similar memberships
        let predicted = result.predict(&data, &t).unwrap();
        for i in 0..n {
            for c in 0..k {
                assert!(
                    (predicted[(i, c)] - result.membership[(i, c)]).abs() < 1e-4,
                    "Membership mismatch at ({}, {}): {} vs {}",
                    i,
                    c,
                    predicted[(i, c)],
                    result.membership[(i, c)]
                );
            }
        }
    }

    #[test]
    fn test_fuzzy_predict_dimension_mismatch() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);
        let result = fuzzy_cmeans_fd(&data, &t, 2, 2.0, 100, 1e-6, 42).unwrap();

        // Wrong number of columns
        let wrong_data = FdMatrix::zeros(3, 20);
        assert!(result.predict(&wrong_data, &t).is_err());

        // Wrong argvals length
        let new_data = FdMatrix::zeros(3, m);
        let wrong_t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        assert!(result.predict(&new_data, &wrong_t).is_err());
    }

    #[test]
    fn test_fuzzy_predict_fuzziness_stored() {
        let m = 30;
        let n_per = 5;
        let (data, t) = generate_two_clusters(n_per, m);

        let result = fuzzy_cmeans_fd(&data, &t, 2, 2.5, 100, 1e-6, 42).unwrap();
        assert!((result.fuzziness - 2.5).abs() < 1e-10);
    }
}
