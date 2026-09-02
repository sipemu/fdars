//! Kernel-k-means clustering of curve sets through the Global Alignment Kernel.
//!
//! Standard k-means minimizes Euclidean distance to an explicit centroid. Kernel
//! k-means lifts the data into the reproducing-kernel feature space induced by the
//! GAK kernel and minimizes the *feature-space* distance to each cluster mean —
//! **without ever materializing a centroid curve**. Every quantity the algorithm
//! needs is read from the n×n GAK Gram matrix `K` (built once via
//! [`gak_gram_train`]).
//!
//! ## The kernel trick (no centroid)
//!
//! For a cluster `C_k` the squared feature-space distance of point `i` is
//! ```text
//! d²(i, k) = K[i,i] − (2/|C_k|)·Σ_{j∈C_k} K[i,j] + (1/|C_k|²)·Σ_{j,l∈C_k} K[j,l]
//! ```
//! The last term `within_k = (1/|C_k|²)·Σ_{j,l∈C_k} K[j,l]` depends only on the
//! cluster, so it is precomputed once per cluster per iteration and reused across
//! all points — the assignment sweep is then O(n²) once the Gram is in memory.
//! With the normalized GAK `K[i,i] = 1`, so the diagonal term is a constant that
//! drops out of the argmin (kept in the returned inertia for interpretability).
//!
//! Because there is no centroid, [`KernelKmeansResult`] has **no `centers` field**
//! — this is a hard correctness property of kernel k-means, not an omission.
//!
//! ## Robustness
//!
//! - **Init:** `n_init` *random-partition* restarts (k-means++ is wrong here — it
//!   assumes L2 curve vectors, but we only have similarity-valued Gram entries).
//!   Each restart is seeded `seed_from_u64(seed + restart_idx)` for reproducibility.
//!   The lowest-total-inertia restart is returned.
//! - **Empty clusters:** if a cluster empties mid-iteration (or `k` exceeds the
//!   number of natural clusters), it is reseeded with the point currently farthest
//!   (max `d²`) from its assigned cluster — the algorithm never panics.
//! - **Determinism:** the Gram is built once and reused across all restarts; the
//!   same `seed` yields identical labels.
//!
//! ## Out-of-sample prediction
//!
//! [`KernelKmeansResult::predict`] assigns new curves via the cross-Gram from
//! [`gak_gram_predict`] (n_test × n_train, normalized so `k(test,test)=1`), reusing
//! the fitted σ, the training within-cluster sums, and the training cluster sizes —
//! no re-estimation.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::metric::gak::{gak_gram_predict, gak_gram_train, GakConfig, GakGramTrain};
use rand::prelude::*;

/// Configuration for kernel-k-means clustering ([`kernel_kmeans_fd`]).
///
/// Defaults: `n_init = 10` (robustness over tslearn's default of 1 — kernel
/// k-means routinely lands in poor local minima from a single random partition),
/// `max_iter = 300`, `tol = 1e-4`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KernelKmeansConfig {
    /// Number of clusters `k` (≥ 1, ≤ number of curves).
    pub n_clusters: usize,
    /// Number of random-partition restarts; the lowest-inertia run is returned (≥ 1).
    pub n_init: usize,
    /// Maximum Lloyd iterations per restart.
    pub max_iter: usize,
    /// Convergence tolerance on the relative inertia decrease.
    pub tol: f64,
    /// Base RNG seed; restart `r` is seeded `seed + r`.
    pub seed: u64,
    /// GAK kernel configuration (bandwidth σ, or the median heuristic).
    pub gak: GakConfig,
}

impl Default for KernelKmeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_init: 10,
            max_iter: 300,
            tol: 1e-4,
            seed: 0,
            gak: GakConfig::default(),
        }
    }
}

impl KernelKmeansConfig {
    /// Construct a config for `n_clusters` with an explicit GAK bandwidth σ,
    /// keeping the other defaults.
    #[must_use]
    pub fn new(n_clusters: usize, sigma: f64) -> Self {
        Self {
            n_clusters,
            gak: GakConfig::with_sigma(sigma),
            ..Self::default()
        }
    }
}

/// Result of [`kernel_kmeans_fd`].
///
/// Carries the cluster assignments and fit diagnostics plus the internal state
/// [`KernelKmeansResult::predict`] needs (the fitted [`GakGramTrain`], the per-cluster
/// within-cluster kernel sums, and the cluster sizes). There is **no centroid /
/// `centers` field**: kernel k-means has no explicit centroid curve.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KernelKmeansResult {
    /// Cluster assignment for each training curve (values in `0..n_clusters`).
    pub cluster: Vec<usize>,
    /// Total feature-space inertia `Σ_i d²(i, cluster[i])` of the returned run.
    pub inertia: f64,
    /// Number of Lloyd iterations the winning restart ran.
    pub iter: usize,
    /// Whether the winning restart converged before hitting `max_iter`.
    pub converged: bool,
    /// Index of the restart (0-based) that produced this result.
    pub n_init_best: usize,
    /// Fitted GAK training Gram + σ, retained so `predict` reuses the exact kernel.
    train: GakGramTrain,
    /// Per-cluster within-cluster kernel sum `within_k = (1/|C_k|²)·ΣΣ K[j,l]`.
    within: Vec<f64>,
    /// Per-cluster size `|C_k|` at convergence (used by `predict`).
    sizes: Vec<usize>,
}

impl KernelKmeansResult {
    /// The number of clusters (`k`).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.within.len()
    }

    /// Assign new (out-of-sample) curves to the fitted clusters.
    ///
    /// Builds the normalized cross-Gram `Kcross = gak_gram_predict(train, new_data)`
    /// (n_test × n_train) and assigns each new curve `t` to the cluster minimizing
    /// ```text
    /// d²(t, k) = 1 − (2/|C_k|)·Σ_{j∈C_k} Kcross[t,j] + within_k
    /// ```
    /// where `1 = k(test,test)` (normalized GAK), and `within_k`, `|C_k|` are the
    /// **fitted training** quantities. The same σ and normalization as the fit are
    /// reused (via [`gak_gram_predict`]) — nothing is re-estimated on the test set.
    ///
    /// # Errors
    /// Propagates [`gak_gram_predict`] errors: [`FdarError::InvalidDimension`] if
    /// `new_data` is empty or its evaluation-grid width differs from the training
    /// set's, or [`FdarError::InvalidParameter`] if the stored σ is invalid.
    pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> {
        let kcross = gak_gram_predict(&self.train, new_data)?;
        let n_test = kcross.nrows();
        let n_train = kcross.ncols();
        let k = self.within.len();

        let mut labels = vec![0usize; n_test];
        for t in 0..n_test {
            // Per-cluster cross-kernel sums Σ_{j∈C_k} Kcross[t,j].
            let mut cross_sum = vec![0.0f64; k];
            for j in 0..n_train {
                cross_sum[self.cluster[j]] += kcross[(t, j)];
            }
            let mut best = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (c, &sz) in self.sizes.iter().enumerate() {
                // k(test,test) = 1 (normalized); constant across clusters but kept
                // for a faithful d². Empty training clusters are unreachable.
                let d2 = if sz == 0 {
                    f64::INFINITY
                } else {
                    1.0 - (2.0 / sz as f64) * cross_sum[c] + self.within[c]
                };
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = c;
                }
            }
            labels[t] = best;
        }
        Ok(labels)
    }
}

/// Cluster a curve set with kernel-k-means through the GAK kernel.
///
/// Builds the GAK Gram **once** ([`gak_gram_train`]), then runs `config.n_init`
/// random-partition restarts (each seeded `config.seed + restart_idx`), keeping the
/// lowest-total-inertia run. Assignments are computed purely from Gram-matrix kernel
/// distances via the kernel trick — the result has **no centroid field**. Empty
/// clusters are recovered by reseeding the farthest point; the algorithm never panics.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (no curves or no points).
/// - [`FdarError::InvalidParameter`] if `n_clusters < 1`, `n_clusters > n`, or
///   `n_init < 1`.
/// - Propagates any error from [`gak_gram_train`] (e.g. an invalid σ).
///
/// # Examples
/// ```
/// use fdars_core::{kernel_kmeans_fd, KernelKmeansConfig, FdMatrix};
/// // Two well-separated groups of two curves each (column-major FdMatrix).
/// let rows = [
///     [0.0, 0.1, 0.2, 0.3],
///     [0.0, 0.1, 0.2, 0.25],
///     [9.0, 9.1, 9.2, 9.3],
///     [9.0, 9.1, 9.2, 9.25],
/// ];
/// let (n, m) = (4, 4);
/// let mut data = vec![0.0; n * m];
/// for (i, r) in rows.iter().enumerate() {
///     for (j, &v) in r.iter().enumerate() {
///         data[i + j * n] = v; // column-major
///     }
/// }
/// let data = FdMatrix::from_slice(&data, n, m).unwrap();
///
/// let cfg = KernelKmeansConfig::new(2, 1.0);
/// let res = kernel_kmeans_fd(&data, &cfg).unwrap();
/// assert_eq!(res.cluster.len(), 4);
/// // The two curves in each group land in the same cluster.
/// assert_eq!(res.cluster[0], res.cluster[1]);
/// assert_eq!(res.cluster[2], res.cluster[3]);
/// assert_ne!(res.cluster[0], res.cluster[2]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kernel_kmeans_fd(
    data: &FdMatrix,
    config: &KernelKmeansConfig,
) -> Result<KernelKmeansResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0, ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    let k = config.n_clusters;
    if k < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_clusters",
            message: "number of clusters must be >= 1".to_string(),
        });
    }
    if k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_clusters",
            message: format!("n_clusters={k} exceeds number of curves n={n}"),
        });
    }
    if config.n_init < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_init",
            message: "n_init must be >= 1".to_string(),
        });
    }

    // Build the GAK Gram ONCE; reused across every restart.
    let train = gak_gram_train(data, &config.gak)?;
    let gram = &train.gram;

    let mut best: Option<RestartOutcome> = None;
    for restart in 0..config.n_init {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64));
        let outcome = run_restart(gram, n, k, config.max_iter, config.tol, &mut rng, restart);
        let take = match &best {
            None => true,
            Some(b) => outcome.inertia < b.inertia,
        };
        if take {
            best = Some(outcome);
        }
    }

    // n_init >= 1 guarantees at least one restart ran.
    let RestartOutcome {
        cluster,
        inertia,
        iter,
        converged,
        within,
        sizes,
        restart_idx,
    } = best.expect("n_init >= 1 guarantees a restart outcome");

    Ok(KernelKmeansResult {
        cluster,
        inertia,
        iter,
        converged,
        n_init_best: restart_idx,
        train,
        within,
        sizes,
    })
}

/// Outcome of a single random-partition restart.
struct RestartOutcome {
    cluster: Vec<usize>,
    inertia: f64,
    iter: usize,
    converged: bool,
    within: Vec<f64>,
    sizes: Vec<usize>,
    restart_idx: usize,
}

/// Run one kernel-k-means restart (Lloyd iterations) on the fixed Gram.
fn run_restart(
    gram: &FdMatrix,
    n: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
    restart_idx: usize,
) -> RestartOutcome {
    // Random-partition init: assign each point to one of k clusters, then repair
    // any empty cluster so every cluster starts non-empty.
    let mut cluster: Vec<usize> = (0..n).map(|_| rng.gen_range(0..k)).collect();
    ensure_no_empty_random(&mut cluster, n, k, rng);

    let mut sizes = vec![0usize; k];
    let mut within = vec![0.0f64; k];
    let mut d2 = vec![0.0f64; n * k]; // row-major d²(i, c) scratch
    let mut prev_inertia = f64::INFINITY;
    let mut iter = 0usize;
    let mut converged = false;

    while iter < max_iter {
        iter += 1;

        // Per-cluster sizes and within-cluster sums for the current assignment.
        compute_cluster_stats(gram, &cluster, n, k, &mut sizes, &mut within);

        // d²(i, c) = K[i,i] − (2/|C_c|)·Σ_{j∈C_c} K[i,j] + within_c.
        for i in 0..n {
            // Cross sums Σ_{j∈C_c} K[i,j] per cluster.
            let mut cross = vec![0.0f64; k];
            for j in 0..n {
                cross[cluster[j]] += gram[(i, j)];
            }
            let kii = gram[(i, i)];
            for c in 0..k {
                d2[i * k + c] = if sizes[c] == 0 {
                    f64::INFINITY
                } else {
                    kii - (2.0 / sizes[c] as f64) * cross[c] + within[c]
                };
            }
        }

        // Reassign each point to its argmin cluster.
        let mut new_cluster = vec![0usize; n];
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let v = d2[i * k + c];
                if v < best_d {
                    best_d = v;
                    best_c = c;
                }
            }
            new_cluster[i] = best_c;
        }

        // Empty-cluster recovery: reseed each empty cluster with the point that is
        // currently farthest (max d²) from its assigned cluster. Never panics.
        recover_empty_clusters(&mut new_cluster, &d2, n, k);

        // Inertia of the new assignment.
        let inertia: f64 = (0..n).map(|i| d2[i * k + new_cluster[i]]).sum();

        let changed = new_cluster != cluster;
        cluster = new_cluster;

        // Convergence: labels stable, or relative inertia change below tol.
        let rel = if prev_inertia.is_finite() && prev_inertia.abs() > 0.0 {
            (prev_inertia - inertia).abs() / prev_inertia.abs()
        } else {
            f64::INFINITY
        };
        if !changed || rel < tol {
            converged = true;
            break;
        }
        prev_inertia = inertia;
    }

    // Final stats for the converged assignment (used by predict).
    compute_cluster_stats(gram, &cluster, n, k, &mut sizes, &mut within);
    let inertia = final_inertia(gram, &cluster, &sizes, &within, n, k);

    RestartOutcome {
        cluster,
        inertia,
        iter,
        converged,
        within,
        sizes,
        restart_idx,
    }
}

/// Compute per-cluster sizes and `within_c = (1/|C_c|²)·Σ_{j,l∈C_c} K[j,l]`.
fn compute_cluster_stats(
    gram: &FdMatrix,
    cluster: &[usize],
    n: usize,
    k: usize,
    sizes: &mut [usize],
    within: &mut [f64],
) {
    sizes.iter_mut().for_each(|s| *s = 0);
    within.iter_mut().for_each(|w| *w = 0.0);
    for &c in cluster.iter() {
        sizes[c] += 1;
    }
    // Σ_{j,l∈C_c} K[j,l] accumulated by scanning all pairs once.
    let mut sums = vec![0.0f64; k];
    for j in 0..n {
        let cj = cluster[j];
        for l in 0..n {
            if cluster[l] == cj {
                sums[cj] += gram[(j, l)];
            }
        }
    }
    for c in 0..k {
        if sizes[c] > 0 {
            let sz = sizes[c] as f64;
            within[c] = sums[c] / (sz * sz);
        } else {
            within[c] = 0.0;
        }
    }
}

/// Total inertia `Σ_i d²(i, cluster[i])` for a settled assignment.
fn final_inertia(
    gram: &FdMatrix,
    cluster: &[usize],
    sizes: &[usize],
    within: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let mut total = 0.0;
    for i in 0..n {
        let mut cross = vec![0.0f64; k];
        for j in 0..n {
            cross[cluster[j]] += gram[(i, j)];
        }
        let c = cluster[i];
        if sizes[c] > 0 {
            let d2 = gram[(i, i)] - (2.0 / sizes[c] as f64) * cross[c] + within[c];
            total += d2;
        }
    }
    total
}

/// Ensure a random-partition init leaves no cluster empty by moving distinct
/// points into empty clusters (deterministic given the RNG state).
fn ensure_no_empty_random(cluster: &mut [usize], n: usize, k: usize, rng: &mut StdRng) {
    loop {
        let mut sizes = vec![0usize; k];
        for &c in cluster.iter() {
            sizes[c] += 1;
        }
        let empty: Vec<usize> = (0..k).filter(|&c| sizes[c] == 0).collect();
        if empty.is_empty() {
            return;
        }
        for c in empty {
            // Steal a random point from a cluster that currently has ≥ 2 members.
            let donors: Vec<usize> = (0..n).filter(|&i| sizes[cluster[i]] > 1).collect();
            if donors.is_empty() {
                // Not enough distinct points to fill every cluster (k == n edge);
                // give up — the assignment loop tolerates this.
                return;
            }
            let pick = donors[rng.gen_range(0..donors.len())];
            sizes[cluster[pick]] -= 1;
            cluster[pick] = c;
            sizes[c] += 1;
        }
    }
}

/// Recover empty clusters after a reassignment by moving the point currently
/// farthest from its assigned cluster into each empty cluster. Never panics.
fn recover_empty_clusters(cluster: &mut [usize], d2: &[f64], n: usize, k: usize) {
    loop {
        let mut sizes = vec![0usize; k];
        for &c in cluster.iter() {
            sizes[c] += 1;
        }
        let Some(empty) = (0..k).find(|&c| sizes[c] == 0) else {
            return;
        };
        // Farthest point (max d² to its own cluster) that can be safely moved
        // (its current cluster has > 1 member, so moving it never creates a new
        // empty cluster).
        let mut best_i = None;
        let mut best_d = f64::NEG_INFINITY;
        for i in 0..n {
            if sizes[cluster[i]] <= 1 {
                continue;
            }
            let d = d2[i * k + cluster[i]];
            if d > best_d {
                best_d = d;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => {
                cluster[i] = empty;
            }
            None => return, // no movable point (k == n); leave as-is.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an FdMatrix from row-major curves (each inner Vec is one curve/row).
    fn matrix_from_rows(rows: &[Vec<f64>]) -> FdMatrix {
        let n = rows.len();
        let m = rows[0].len();
        let mut data = vec![0.0; n * m];
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                data[i + j * n] = v; // column-major
            }
        }
        FdMatrix::from_slice(&data, n, m).unwrap()
    }

    /// Two well-separated groups: a low-flat band and a high-flat band.
    fn two_groups() -> (FdMatrix, Vec<usize>) {
        let m = 20;
        let mut rows = Vec::new();
        let mut truth = Vec::new();
        // Group 0: near 0.
        for i in 0..5 {
            let off = i as f64 * 0.01;
            rows.push(
                (0..m)
                    .map(|k| (k as f64 * 0.05).sin() * 0.2 + off)
                    .collect(),
            );
            truth.push(0);
        }
        // Group 1: near 10.
        for i in 0..5 {
            let off = i as f64 * 0.01;
            rows.push(
                (0..m)
                    .map(|k| (k as f64 * 0.05).sin() * 0.2 + 10.0 + off)
                    .collect(),
            );
            truth.push(1);
        }
        (matrix_from_rows(&rows), truth)
    }

    /// Permutation-invariant purity of `labels` against `truth`.
    fn purity(labels: &[usize], truth: &[usize], k: usize) -> f64 {
        let n = labels.len();
        let n_truth = truth.iter().copied().max().unwrap_or(0) + 1;
        let mut correct = 0usize;
        for c in 0..k {
            // Majority true-label count within predicted cluster c.
            let mut counts = vec![0usize; n_truth];
            for i in 0..n {
                if labels[i] == c {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        correct as f64 / n as f64
    }

    #[test]
    fn test_kernel_kmeans_recovers_groups() {
        let (data, truth) = two_groups();
        let cfg = KernelKmeansConfig::new(2, 1.0);
        let res = kernel_kmeans_fd(&data, &cfg).unwrap();
        assert_eq!(res.cluster.len(), 10);
        let p = purity(&res.cluster, &truth, 2);
        assert!((p - 1.0).abs() < 1e-12, "purity {p} != 1.0");
        assert_eq!(res.n_clusters(), 2);
    }

    #[test]
    fn test_kernel_kmeans_deterministic() {
        let (data, _) = two_groups();
        let cfg = KernelKmeansConfig::new(2, 1.0);
        let a = kernel_kmeans_fd(&data, &cfg).unwrap();
        let b = kernel_kmeans_fd(&data, &cfg).unwrap();
        assert_eq!(a.cluster, b.cluster, "same seed must give identical labels");
        assert_eq!(a.inertia.to_bits(), b.inertia.to_bits());
        assert_eq!(a.n_init_best, b.n_init_best);
    }

    #[test]
    fn test_kernel_kmeans_empty_cluster_recovery() {
        // k = 4 but only two natural groups (n = 10). Must not panic and must
        // return valid labels with exactly k distinct non-empty clusters.
        let (data, _) = two_groups();
        let cfg = KernelKmeansConfig {
            n_clusters: 4,
            ..KernelKmeansConfig::new(4, 1.0)
        };
        let res = kernel_kmeans_fd(&data, &cfg).unwrap();
        assert_eq!(res.cluster.len(), 10);
        assert!(res.cluster.iter().all(|&c| c < 4));
        // Sizes are internally consistent (no empty cluster left behind).
        let mut sizes = vec![0usize; 4];
        for &c in &res.cluster {
            sizes[c] += 1;
        }
        assert!(
            sizes.iter().all(|&s| s >= 1),
            "an empty cluster survived: {sizes:?}"
        );
    }

    #[test]
    fn test_kernel_kmeans_empty_cluster_k_equals_n() {
        // Extreme case k == n: every point its own cluster; must not panic.
        let rows: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..12).map(|k| (k as f64 * 0.1 + i as f64).sin()).collect())
            .collect();
        let data = matrix_from_rows(&rows);
        let cfg = KernelKmeansConfig::new(4, 1.0);
        let res = kernel_kmeans_fd(&data, &cfg).unwrap();
        assert_eq!(res.cluster.len(), 4);
        assert!(res.cluster.iter().all(|&c| c < 4));
    }

    #[test]
    fn test_kernel_kmeans_n_init() {
        // n_init > 1 must return inertia no worse than a single-init baseline on
        // the same seed, and build the Gram once (implicit — single train call).
        let (data, _) = two_groups();
        let multi = KernelKmeansConfig {
            n_init: 10,
            ..KernelKmeansConfig::new(2, 1.0)
        };
        let single = KernelKmeansConfig {
            n_init: 1,
            ..KernelKmeansConfig::new(2, 1.0)
        };
        let rm = kernel_kmeans_fd(&data, &multi).unwrap();
        let rs = kernel_kmeans_fd(&data, &single).unwrap();
        assert!(
            rm.inertia <= rs.inertia + 1e-12,
            "multi-init inertia {} worse than single-init {}",
            rm.inertia,
            rs.inertia
        );
    }

    #[test]
    fn test_kernel_kmeans_predict() {
        let (data, _) = two_groups();
        let cfg = KernelKmeansConfig::new(2, 1.0);
        let res = kernel_kmeans_fd(&data, &cfg).unwrap();

        // Cluster label of the low band (training curve 0) and high band (curve 5).
        let low_label = res.cluster[0];
        let high_label = res.cluster[5];
        assert_ne!(low_label, high_label);

        let m = 20;
        // Test set: a novel low curve, a novel high curve, and an exact copy of
        // training curve 0.
        let low_curve: Vec<f64> = (0..m)
            .map(|k| (k as f64 * 0.05).sin() * 0.2 + 0.03)
            .collect();
        let high_curve: Vec<f64> = (0..m)
            .map(|k| (k as f64 * 0.05).sin() * 0.2 + 10.03)
            .collect();
        let copy0 = data.row(0);
        let test = matrix_from_rows(&[low_curve, high_curve, copy0]);

        let preds = res.predict(&test).unwrap();
        assert_eq!(preds.len(), 3);
        assert_eq!(
            preds[0], low_label,
            "novel low curve should route to low cluster"
        );
        assert_eq!(
            preds[1], high_label,
            "novel high curve should route to high cluster"
        );
        assert_eq!(
            preds[2], res.cluster[0],
            "exact copy should match its training label"
        );
    }

    #[test]
    fn test_kernel_kmeans_validation() {
        let (data, _) = two_groups();
        // n_clusters = 0.
        let cfg0 = KernelKmeansConfig {
            n_clusters: 0,
            ..KernelKmeansConfig::new(0, 1.0)
        };
        assert!(matches!(
            kernel_kmeans_fd(&data, &cfg0),
            Err(FdarError::InvalidParameter { .. })
        ));
        // n_clusters > n.
        let cfg_big = KernelKmeansConfig {
            n_clusters: 999,
            ..KernelKmeansConfig::new(999, 1.0)
        };
        assert!(matches!(
            kernel_kmeans_fd(&data, &cfg_big),
            Err(FdarError::InvalidParameter { .. })
        ));
        // n_init = 0.
        let cfg_ni = KernelKmeansConfig {
            n_init: 0,
            ..KernelKmeansConfig::new(2, 1.0)
        };
        assert!(matches!(
            kernel_kmeans_fd(&data, &cfg_ni),
            Err(FdarError::InvalidParameter { .. })
        ));
        // Empty data.
        let empty = FdMatrix::zeros(0, 0);
        assert!(matches!(
            kernel_kmeans_fd(&empty, &KernelKmeansConfig::new(2, 1.0)),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_kernel_kmeans_no_centroid() {
        // Structural: the result exposes only the documented public fields; there
        // is no centroid/centers field. This test destructures the public API —
        // if a `centers` field were added it would still compile, so we assert the
        // intent by using ONLY the documented fields and confirming no centroid is
        // needed for predict (predict works from stored Gram state alone).
        let (data, _) = two_groups();
        let res = kernel_kmeans_fd(&data, &KernelKmeansConfig::new(2, 1.0)).unwrap();
        let KernelKmeansResult {
            cluster,
            inertia,
            iter,
            converged,
            n_init_best,
            ..
        } = &res;
        assert_eq!(cluster.len(), 10);
        assert!(inertia.is_finite());
        assert!(*iter >= 1);
        let _ = converged;
        let _ = n_init_best;
        // predict needs no centroid — it works purely from the stored kernel state.
        let preds = res.predict(&data).unwrap();
        assert_eq!(preds, res.cluster);
    }
}
