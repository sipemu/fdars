//! k-Shape clustering of curve sets through the Shape-Based Distance (SBD).
//!
//! k-Shape (Paparrizos & Gravano, *k-Shape*, SIGMOD 2015) is a partitional
//! clustering algorithm for time series that is invariant to amplitude scaling
//! and phase (circular) shift. It alternates two steps, Lloyd-style, until
//! convergence:
//!
//! 1. **Assignment.** Each z-normalized series is assigned to the cluster whose
//!    centroid minimizes the [`sbd`] distance. The optimal SBD *shift* from that
//!    same call is stored — it is needed to shift-align the member before the
//!    centroid update (a common transcription bug is to discard it).
//! 2. **Refinement (shape extraction).** Each centroid is recomputed as the
//!    **top eigenvector** of a shift-aligned, mean-centered cross-product matrix
//!    — *not* the arithmetic mean (that would be k-means, not k-Shape). See
//!    [`shape_extraction`] for the exact matrix algebra.
//!
//! ## Shape extraction (the only genuinely-new numerical piece)
//!
//! For a cluster with members `X` (`n_k × m`, each row shift-aligned to the
//! current centroid and z-normalized):
//!
//! ```text
//! S = XᵀX                        (m × m)
//! Q = I_m − (1/m)·O_m            (centering over the TIME dim; m = series length)
//! M = Qᵀ S Q                     (symmetric m × m)
//! μ = argmax eigenvector of M    (LARGEST eigenvalue)
//! ```
//!
//! The eigenvector is defined up to sign: the sign that minimizes the total SBD
//! to the members is chosen, then `μ` is z-normalized. Two subtle points that
//! silently corrupt k-Shape if wrong: the centering divisor is `m` (the series
//! length), **not** `n_k`; and nalgebra returns eigenvalues **ascending**, so the
//! centroid is the eigenvector at the *largest* eigenvalue (argmax), not index 0.
//!
//! ## Robustness (mirrors [`crate::kernel_kmeans`])
//!
//! - **Init:** `n_init` *random-partition* restarts, each seeded
//!   `seed_from_u64(seed + restart_idx)`; the lowest-total-inertia restart wins.
//!   The default `n_init = 10` (an fdars convention exceeding tslearn's 1).
//! - **Empty clusters:** a cluster that empties mid-iteration is reseeded in
//!   place from the series currently farthest (max SBD) from its centroid — a
//!   documented divergence from tslearn's full restart. The algorithm never
//!   panics; `k > natural clusters` returns valid labels.
//! - **Determinism:** the same `seed` yields byte-identical labels and inertia;
//!   sequential and `parallel` builds agree (SBD is RNG-free).
//!
//! ## Out-of-sample prediction
//!
//! [`KShapeResult::predict`] z-normalizes each new series, computes [`sbd`] to
//! every stored (already-z-normalized) centroid, and takes the argmin — the
//! centroids are used as-is, so `predict(train_data)` reproduces the training
//! labels.

use crate::alignment::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::metric::sbd::{sbd, sbd_distance_matrix};
use crate::shapelet::z_normalize_window;
use nalgebra::{DMatrix, SymmetricEigen};
use rand::prelude::*;

/// Configuration for k-Shape clustering ([`kshape_fd`]).
///
/// Defaults: `n_clusters = 2`, `n_init = 10` (robustness over tslearn's default
/// of 1 — k-Shape is sensitive to initialization), `max_iter = 100`,
/// `tol = 1e-6`, `seed = 0`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KShapeConfig {
    /// Number of clusters `k` (≥ 1, ≤ number of series).
    pub n_clusters: usize,
    /// Number of random-partition restarts; the lowest-inertia run is returned (≥ 1).
    pub n_init: usize,
    /// Maximum Lloyd iterations per restart.
    pub max_iter: usize,
    /// Convergence tolerance on the absolute inertia decrease.
    pub tol: f64,
    /// Base RNG seed; restart `r` is seeded `seed + r`.
    pub seed: u64,
}

impl Default for KShapeConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_init: 10,
            max_iter: 100,
            tol: 1e-6,
            seed: 0,
        }
    }
}

impl KShapeConfig {
    /// Construct a config for `n_clusters`, keeping the other defaults.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Result of [`kshape_fd`].
///
/// Carries the fitted centroids (k × m, already z-normalized) plus cluster
/// assignments and fit diagnostics. [`KShapeResult::predict`] reuses the stored
/// centroids directly — nothing is re-estimated on new data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KShapeResult {
    /// Cluster centroids as a `k × m` [`FdMatrix`] (each row a z-normalized shape prototype).
    pub centroids: FdMatrix,
    /// Cluster assignment for each training series (values in `0..n_clusters`).
    pub cluster: Vec<usize>,
    /// Total SBD inertia `Σ_i SBD(series_i, centroid[cluster[i]])` of the returned run.
    pub inertia: f64,
    /// Number of Lloyd iterations the winning restart ran.
    pub iter: usize,
    /// Whether the winning restart converged before hitting `max_iter`.
    pub converged: bool,
    /// Index of the restart (0-based) that produced this result.
    pub n_init_best: usize,
}

impl KShapeResult {
    /// The fitted centroids (`k × m`, each row a z-normalized shape prototype).
    #[must_use]
    pub fn centroids(&self) -> &FdMatrix {
        &self.centroids
    }

    /// The per-series cluster assignments.
    #[must_use]
    pub fn cluster(&self) -> &[usize] {
        &self.cluster
    }

    /// Total SBD inertia of the returned run.
    #[must_use]
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// The number of clusters (`k`).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.centroids.nrows()
    }

    /// Assign new (out-of-sample) series to the fitted clusters.
    ///
    /// Each new series is z-normalized and compared by [`sbd`] against every
    /// stored (already-z-normalized) centroid; the argmin cluster is returned.
    /// The centroids are used as-is, so `predict(train_data)` reproduces the
    /// training labels.
    ///
    /// # Errors
    /// Returns [`FdarError::InvalidDimension`] if `new_data` is empty or its
    /// evaluation-grid width differs from the fitted centroids' length `m`.
    pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> {
        let p = new_data.nrows();
        let m = new_data.ncols();
        let k = self.centroids.nrows();
        let cm = self.centroids.ncols();
        if p == 0 || m == 0 {
            return Err(FdarError::InvalidDimension {
                parameter: "new_data",
                expected: "non-empty matrix (nrows > 0, ncols > 0)".to_string(),
                actual: format!("{p}x{m}"),
            });
        }
        if m != cm {
            return Err(FdarError::InvalidDimension {
                parameter: "new_data",
                expected: format!("series length m={cm} matching fitted centroids"),
                actual: format!("m={m}"),
            });
        }

        // Materialize the stored centroids as contiguous rows once.
        let mut centroid_rows: Vec<Vec<f64>> = Vec::with_capacity(k);
        for c in 0..k {
            centroid_rows.push(self.centroids.row(c));
        }

        let mut labels = vec![0usize; p];
        let mut row = vec![0.0f64; m];
        for t in 0..p {
            new_data.row_to_buf(t, &mut row);
            let z = z_normalize_window(&row);
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (c, cent) in centroid_rows.iter().enumerate() {
                let d = sbd(&z, cent).map(|r| r.distance).unwrap_or(1.0);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            labels[t] = best;
        }
        Ok(labels)
    }
}

/// Cluster a curve set with k-Shape.
///
/// Runs `config.n_init` random-partition restarts (each seeded
/// `config.seed + restart_idx`), keeping the lowest-total-inertia run. Every
/// series is z-normalized once up front; each restart iterates SBD assignment +
/// shape-extraction centroid refinement to convergence or `max_iter`. Empty
/// clusters are recovered in place by farthest-point reassignment; the algorithm
/// never panics.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (no series or no points).
/// - [`FdarError::InvalidParameter`] if `n_clusters < 1`, `n_clusters > n`, or
///   `n_init < 1`.
///
/// # Examples
/// ```
/// use fdars_core::kshape::{kshape_fd, KShapeConfig};
/// use fdars_core::FdMatrix;
/// // Two shape groups: rising ramps vs. falling ramps (column-major FdMatrix).
/// let rows = [
///     vec![0.0, 1.0, 2.0, 3.0, 4.0],
///     vec![0.1, 1.1, 2.0, 3.1, 3.9],
///     vec![4.0, 3.0, 2.0, 1.0, 0.0],
///     vec![3.9, 3.1, 2.0, 0.9, 0.1],
/// ];
/// let (n, m) = (4, 5);
/// let mut data = vec![0.0; n * m];
/// for (i, r) in rows.iter().enumerate() {
///     for (j, &v) in r.iter().enumerate() {
///         data[i + j * n] = v; // column-major
///     }
/// }
/// let data = FdMatrix::from_slice(&data, n, m).unwrap();
///
/// let cfg = KShapeConfig::new(2);
/// let res = kshape_fd(&data, &cfg).unwrap();
/// assert_eq!(res.cluster.len(), 4);
/// // The two rising ramps share a cluster; the two falling ramps share the other.
/// assert_eq!(res.cluster[0], res.cluster[1]);
/// assert_eq!(res.cluster[2], res.cluster[3]);
/// assert_ne!(res.cluster[0], res.cluster[2]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError> {
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
            message: format!("n_clusters={k} exceeds number of series n={n}"),
        });
    }
    if config.n_init < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_init",
            message: "n_init must be >= 1".to_string(),
        });
    }

    // Z-normalize every series ONCE up front; all assignment + shape extraction
    // operate on these z-normed rows.
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut row = vec![0.0f64; m];
    for i in 0..n {
        data.row_to_buf(i, &mut row);
        series.push(z_normalize_window(&row));
    }

    let mut best: Option<RestartOutcome> = None;
    for restart in 0..config.n_init {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64));
        let outcome = run_restart(
            &series,
            n,
            m,
            k,
            config.max_iter,
            config.tol,
            &mut rng,
            restart,
        );
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
        centroids,
        inertia,
        iter,
        converged,
        restart_idx,
    } = best.expect("n_init >= 1 guarantees a restart outcome");

    // Assemble the k × m centroid matrix (rows already z-normalized).
    let mut cmat = FdMatrix::zeros(k, m);
    for (c, cent) in centroids.iter().enumerate() {
        for (j, &v) in cent.iter().enumerate() {
            cmat[(c, j)] = v;
        }
    }

    Ok(KShapeResult {
        centroids: cmat,
        cluster,
        inertia,
        iter,
        converged,
        n_init_best: restart_idx,
    })
}

/// Cluster a curve set with **k-medoids over the Shape-Based Distance**.
///
/// A shape-based clustering path distinct from [`kshape_fd`]: instead of
/// estimating shape-extraction centroids, this builds the full n×n SBD distance
/// matrix ([`sbd_distance_matrix`]) and feeds it to the existing
/// [`kmedoids_from_distances`] solver. The returned medoids are *actual input
/// series* (indices into `data`), which makes the result directly
/// interpretable. Because the backend distance is SBD, the clustering is
/// invariant to amplitude scaling and circular phase shift — unlike an L2- or
/// DTW-backed k-medoids.
///
/// Reuses [`KMedoidsConfig`] / [`KMedoidsResult`] unchanged.
///
/// # Errors
/// Propagates any error from [`sbd_distance_matrix`] (e.g. empty `data`) or from
/// [`kmedoids_from_distances`] (`config.k < 1`, `config.k > n`).
///
/// # Examples
/// ```
/// use fdars_core::{sbd_kmedoids, sbd_distance_matrix, KMedoidsConfig, FdMatrix};
/// use fdars_core::alignment::kmedoids_from_distances;
///
/// // Two shape groups: rising ramps vs. falling ramps (column-major FdMatrix).
/// let rows = [
///     vec![0.0, 1.0, 2.0, 3.0, 4.0],
///     vec![0.1, 1.1, 2.0, 3.1, 3.9],
///     vec![4.0, 3.0, 2.0, 1.0, 0.0],
///     vec![3.9, 3.1, 2.0, 0.9, 0.1],
/// ];
/// let (n, m) = (4, 5);
/// let mut flat = vec![0.0; n * m];
/// for (i, r) in rows.iter().enumerate() {
///     for (j, &v) in r.iter().enumerate() {
///         flat[i + j * n] = v; // column-major
///     }
/// }
/// let data = FdMatrix::from_slice(&flat, n, m).unwrap();
///
/// let cfg = KMedoidsConfig { k: 2, ..Default::default() };
/// let res = sbd_kmedoids(&data, &cfg).unwrap();
/// assert_eq!(res.labels.len(), 4);
/// assert_eq!(res.medoid_indices.len(), 2);
///
/// // Equivalent to the explicit SBD-matrix → k-medoids flow:
/// let dist = sbd_distance_matrix(&data).unwrap();
/// let manual = kmedoids_from_distances(&dist, &cfg).unwrap();
/// assert_eq!(res.labels, manual.labels);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn sbd_kmedoids(data: &FdMatrix, config: &KMedoidsConfig) -> Result<KMedoidsResult, FdarError> {
    let dist = sbd_distance_matrix(data)?;
    kmedoids_from_distances(&dist, config)
}

/// Outcome of a single random-partition restart.
struct RestartOutcome {
    cluster: Vec<usize>,
    centroids: Vec<Vec<f64>>,
    inertia: f64,
    iter: usize,
    converged: bool,
    restart_idx: usize,
}

/// Run one k-Shape restart (Lloyd iterations) on the z-normalized series.
#[allow(clippy::too_many_arguments)]
fn run_restart(
    series: &[Vec<f64>],
    n: usize,
    m: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
    restart_idx: usize,
) -> RestartOutcome {
    // Random-partition init, repaired so every cluster starts non-empty.
    let mut cluster: Vec<usize> = (0..n).map(|_| rng.gen_range(0..k)).collect();
    ensure_no_empty_random(&mut cluster, n, k, rng);

    // Centroids: initialized to zero; the first refinement fills them from the
    // random partition (assignment before the first refinement is skipped).
    let mut centroids: Vec<Vec<f64>> = vec![vec![0.0f64; m]; k];
    refine_centroids(series, &cluster, k, m, &mut centroids);

    let mut prev_inertia = f64::INFINITY;
    let mut iter = 0usize;
    let mut converged = false;
    let mut inertia = f64::INFINITY;

    while iter < max_iter {
        iter += 1;

        // --- Assignment: min-SBD centroid, storing the SBD shift per series. ---
        let mut new_cluster = vec![0usize; n];
        let mut dist_to_own = vec![0.0f64; n];
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = f64::INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let d = sbd(&series[i], cent).map(|r| r.distance).unwrap_or(1.0);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            new_cluster[i] = best_c;
            dist_to_own[i] = best_d;
        }

        // Empty-cluster recovery: reseed each empty cluster from the series
        // currently farthest (max SBD) from its assigned centroid.
        recover_empty_clusters(&mut new_cluster, &dist_to_own, n, k);

        // --- Refinement: shape-extraction centroids. ---
        refine_centroids(series, &new_cluster, k, m, &mut centroids);

        // --- Inertia (after refinement, against the just-assigned labels). ---
        inertia = 0.0;
        for i in 0..n {
            let d = sbd(&series[i], &centroids[new_cluster[i]])
                .map(|r| r.distance)
                .unwrap_or(1.0);
            inertia += d;
        }

        let changed = new_cluster != cluster;
        cluster = new_cluster;

        if !changed || (prev_inertia - inertia).abs() < tol {
            converged = true;
            break;
        }
        prev_inertia = inertia;
    }

    RestartOutcome {
        cluster,
        centroids,
        inertia,
        iter,
        converged,
        restart_idx,
    }
}

/// Recompute every centroid via shape extraction from its current members.
///
/// An empty cluster keeps its previous centroid (recovery happens in the
/// assignment step); a non-empty cluster's centroid is overwritten in place.
fn refine_centroids(
    series: &[Vec<f64>],
    cluster: &[usize],
    k: usize,
    m: usize,
    centroids: &mut [Vec<f64>],
) {
    for c in 0..k {
        let members: Vec<usize> = (0..series.len()).filter(|&i| cluster[i] == c).collect();
        if members.is_empty() {
            continue;
        }
        centroids[c] = shape_extraction(series, &members, &centroids[c], m);
    }
}

/// Shape-extraction centroid for one cluster (decision 5 — the k-Shape crux).
///
/// Each member is aligned to `centroid` by its SBD optimal shift, stacked into
/// `X` (`n_k × m`); then `S = XᵀX`, `Q = I_m − O_m/m` (centering over the TIME
/// dimension, divisor `m` = series length), `M = QᵀSQ`. The centroid is the
/// eigenvector of `M` at the **largest** eigenvalue (nalgebra returns ascending,
/// so argmax), sign-fixed to minimize total SBD to the members, then
/// z-normalized.
fn shape_extraction(
    series: &[Vec<f64>],
    members: &[usize],
    centroid: &[f64],
    m: usize,
) -> Vec<f64> {
    let n_k = members.len();

    // Align each member to the current centroid by its SBD shift, then
    // re-z-normalize the shifted vector. If the centroid is all-zero (first
    // refinement, before any assignment produced a shape), SBD's constant-series
    // guard returns shift 0 — the members are used unshifted, which is correct
    // for a random-partition seed.
    let mut x_aligned: Vec<Vec<f64>> = Vec::with_capacity(n_k);
    for &i in members {
        let shift = sbd(centroid, &series[i]).map(|r| r.shift).unwrap_or(0);
        let shifted = circular_shift(&series[i], shift);
        x_aligned.push(z_normalize_window(&shifted));
    }

    // S = XᵀX  (m × m). S[a][b] = Σ_i X[i][a] · X[i][b].
    let mut s = DMatrix::<f64>::zeros(m, m);
    for row_vec in &x_aligned {
        for a in 0..m {
            let va = row_vec[a];
            if va == 0.0 {
                continue;
            }
            for b in 0..m {
                s[(a, b)] += va * row_vec[b];
            }
        }
    }

    // M = Qᵀ S Q with Q = I_m − O_m/m (centering over time; divisor m).
    // Q is symmetric, so M = Q S Q. Compute QS then (QS)Q.
    // (Q A)[a][b] = A[a][b] − mean_over_a(A[:,b]); (B Q)[a][b] = B[a][b] − mean_over_b(B[a][:]).
    let inv_m = 1.0 / m as f64;
    // QS: subtract, from each column, that column's mean over rows.
    let mut qs = s.clone();
    for b in 0..m {
        let mut col_mean = 0.0;
        for a in 0..m {
            col_mean += s[(a, b)];
        }
        col_mean *= inv_m;
        for a in 0..m {
            qs[(a, b)] -= col_mean;
        }
    }
    // (QS)Q: subtract, from each row, that row's mean over columns.
    let mut mmat = qs.clone();
    for a in 0..m {
        let mut row_mean = 0.0;
        for b in 0..m {
            row_mean += qs[(a, b)];
        }
        row_mean *= inv_m;
        for b in 0..m {
            mmat[(a, b)] -= row_mean;
        }
    }
    // Symmetrize defensively (M is symmetric in exact arithmetic).
    for a in 0..m {
        for b in (a + 1)..m {
            let avg = 0.5 * (mmat[(a, b)] + mmat[(b, a)]);
            mmat[(a, b)] = avg;
            mmat[(b, a)] = avg;
        }
    }

    // Top eigenvector: nalgebra returns eigenvalues ASCENDING → take argmax.
    let eig = SymmetricEigen::new(mmat);
    let mut arg = 0usize;
    let mut best_eval = f64::NEG_INFINITY;
    for (i, &ev) in eig.eigenvalues.iter().enumerate() {
        if ev > best_eval {
            best_eval = ev;
            arg = i;
        }
    }
    let mut v: Vec<f64> = eig.eigenvectors.column(arg).iter().copied().collect();

    // Sign fix: choose ±v minimizing Σ SBD(±v, member).
    let neg: Vec<f64> = v.iter().map(|x| -x).collect();
    let mut sum_pos = 0.0;
    let mut sum_neg = 0.0;
    for row_vec in &x_aligned {
        sum_pos += sbd(&v, row_vec).map(|r| r.distance).unwrap_or(1.0);
        sum_neg += sbd(&neg, row_vec).map(|r| r.distance).unwrap_or(1.0);
    }
    if sum_neg < sum_pos {
        v = neg;
    }

    // z-normalize the centroid (mean 0, std 1) for the next iteration's SBD.
    z_normalize_window(&v)
}

/// Circularly shift `x` by `shift` positions (positive = right / later).
fn circular_shift(x: &[f64], shift: isize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let n_i = n as isize;
    let s = ((shift % n_i) + n_i) % n_i; // normalize to 0..n
    let mut out = vec![0.0f64; n];
    for (i, &v) in x.iter().enumerate() {
        let j = ((i as isize + s) % n_i) as usize;
        out[j] = v;
    }
    out
}

/// Ensure a random-partition init leaves no cluster empty by moving distinct
/// series into empty clusters (deterministic given the RNG state).
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
            let donors: Vec<usize> = (0..n).filter(|&i| sizes[cluster[i]] > 1).collect();
            if donors.is_empty() {
                return; // k == n edge; assignment loop tolerates this.
            }
            let pick = donors[rng.gen_range(0..donors.len())];
            sizes[cluster[pick]] -= 1;
            cluster[pick] = c;
            sizes[c] += 1;
        }
    }
}

/// Recover empty clusters after a reassignment by moving the series currently
/// farthest (max SBD to its centroid) into each empty cluster. Never panics.
fn recover_empty_clusters(cluster: &mut [usize], dist_to_own: &[f64], n: usize, k: usize) {
    loop {
        let mut sizes = vec![0usize; k];
        for &c in cluster.iter() {
            sizes[c] += 1;
        }
        let Some(empty) = (0..k).find(|&c| sizes[c] == 0) else {
            return;
        };
        let mut best_i = None;
        let mut best_d = f64::NEG_INFINITY;
        for i in 0..n {
            if sizes[cluster[i]] <= 1 {
                continue;
            }
            let d = dist_to_own[i];
            if d > best_d {
                best_d = d;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(i) => cluster[i] = empty,
            None => return, // no movable series (k == n); leave as-is.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build an FdMatrix from row-major curves (each inner Vec is one series/row).
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

    /// Permutation-invariant purity of `labels` against `truth`.
    fn purity(labels: &[usize], truth: &[usize], k: usize) -> f64 {
        let n = labels.len();
        let n_truth = truth.iter().copied().max().unwrap_or(0) + 1;
        let mut correct = 0usize;
        for c in 0..k {
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

    /// Two shape groups with random per-series CIRCULAR shifts + light noise.
    /// Group 0 is a single-period sine; group 1 is a single-period cosine-like
    /// (double-frequency) bump — distinct base shapes.
    fn shifted_groups(seed: u64) -> (FdMatrix, Vec<usize>) {
        let m = 40usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::new();
        let mut truth = Vec::new();
        let base_a: Vec<f64> = (0..m)
            .map(|j| (2.0 * PI * j as f64 / m as f64).sin())
            .collect();
        let base_b: Vec<f64> = (0..m)
            .map(|j| (4.0 * PI * j as f64 / m as f64).sin())
            .collect();
        for (label, base) in [(0usize, &base_a), (1usize, &base_b)] {
            for _ in 0..8 {
                let shift = rng.gen_range(0..m) as isize;
                let shifted = circular_shift(base, shift);
                let noisy: Vec<f64> = shifted
                    .iter()
                    .map(|&v| v + (rng.gen::<f64>() - 0.5) * 0.05)
                    .collect();
                rows.push(noisy);
                truth.push(label);
            }
        }
        (matrix_from_rows(&rows), truth)
    }

    #[test]
    fn test_kshape_recovers_shifted_groups() {
        let (data, truth) = shifted_groups(7);
        let cfg = KShapeConfig {
            n_clusters: 2,
            n_init: 10,
            seed: 3,
            ..Default::default()
        };
        let res = kshape_fd(&data, &cfg).unwrap();
        assert_eq!(res.cluster.len(), 16);
        let p = purity(&res.cluster, &truth, 2);
        assert!((p - 1.0).abs() < 1e-12, "purity {p} != 1.0");
        assert_eq!(res.n_clusters(), 2);
        // Centroids are z-normalized (mean ~0).
        for c in 0..2 {
            let row = res.centroids.row(c);
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!(mean.abs() < 1e-8, "centroid {c} not zero-mean: {mean}");
        }
    }

    #[test]
    fn test_kshape_centroid_sign() {
        // A clean single-shape cluster: 6 identical (up to tiny noise) sine curves,
        // no shifts. The extracted centroid must correlate POSITIVELY with members.
        let m = 32usize;
        let base: Vec<f64> = (0..m)
            .map(|j| (2.0 * PI * j as f64 / m as f64).sin())
            .collect();
        let mut rng = StdRng::seed_from_u64(1);
        let rows: Vec<Vec<f64>> = (0..6)
            .map(|_| {
                base.iter()
                    .map(|&v| v + (rng.gen::<f64>() - 0.5) * 0.01)
                    .collect()
            })
            .collect();
        let data = matrix_from_rows(&rows);
        let cfg = KShapeConfig::new(1);
        let res = kshape_fd(&data, &cfg).unwrap();
        let cent = res.centroids.row(0);
        let base_z = z_normalize_window(&base);
        let cent_z = z_normalize_window(&cent);
        // Pearson correlation (both zero-mean, unit-std) = mean of products.
        let corr: f64 = cent_z
            .iter()
            .zip(base_z.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / m as f64;
        assert!(
            corr > 0.99,
            "centroid must correlate positively, corr={corr}"
        );
    }

    #[test]
    fn test_kshape_empty_cluster_recovery() {
        // k = 5 but only two natural groups. Must not panic; all clusters non-empty.
        let (data, _) = shifted_groups(11);
        let cfg = KShapeConfig {
            n_clusters: 5,
            n_init: 3,
            seed: 2,
            ..Default::default()
        };
        let res = kshape_fd(&data, &cfg).unwrap();
        assert_eq!(res.cluster.len(), 16);
        assert!(res.cluster.iter().all(|&c| c < 5));
        let mut sizes = vec![0usize; 5];
        for &c in &res.cluster {
            sizes[c] += 1;
        }
        assert!(
            sizes.iter().all(|&s| s >= 1),
            "an empty cluster survived: {sizes:?}"
        );
    }

    #[test]
    fn test_kshape_deterministic() {
        let (data, _) = shifted_groups(5);
        let cfg = KShapeConfig {
            n_clusters: 2,
            n_init: 5,
            seed: 42,
            ..Default::default()
        };
        let a = kshape_fd(&data, &cfg).unwrap();
        let b = kshape_fd(&data, &cfg).unwrap();
        assert_eq!(a.cluster, b.cluster, "same seed must give identical labels");
        assert_eq!(a.inertia.to_bits(), b.inertia.to_bits());
        assert_eq!(a.n_init_best, b.n_init_best);
        // Centroids byte-identical too (sequential==parallel: SBD is RNG-free).
        let n = a.centroids.nrows();
        let m = a.centroids.ncols();
        for i in 0..n {
            for j in 0..m {
                assert_eq!(a.centroids[(i, j)].to_bits(), b.centroids[(i, j)].to_bits());
            }
        }
    }

    #[test]
    fn test_kshape_best_of_n_init() {
        // n_init > 1 must return inertia no worse than a single-init baseline on
        // the same base seed.
        let (data, _) = shifted_groups(9);
        let multi = KShapeConfig {
            n_clusters: 2,
            n_init: 10,
            seed: 4,
            ..Default::default()
        };
        let single = KShapeConfig {
            n_init: 1,
            ..multi.clone()
        };
        let rm = kshape_fd(&data, &multi).unwrap();
        let rs = kshape_fd(&data, &single).unwrap();
        assert!(
            rm.inertia <= rs.inertia + 1e-12,
            "multi-init inertia {} worse than single-init {}",
            rm.inertia,
            rs.inertia
        );
    }

    #[test]
    fn test_kshape_predict() {
        let (data, _) = shifted_groups(13);
        let cfg = KShapeConfig {
            n_clusters: 2,
            n_init: 10,
            seed: 6,
            ..Default::default()
        };
        let res = kshape_fd(&data, &cfg).unwrap();

        // predict on the training data reproduces the training labels exactly.
        let preds = res.predict(&data).unwrap();
        assert_eq!(preds, res.cluster, "predict(train) must reproduce cluster");

        // A new series near group A (a shifted copy of series 0) routes to A.
        let m = data.ncols();
        let src = data.row(0);
        let novel = circular_shift(&src, 7);
        let test = matrix_from_rows(&[novel]);
        let p = res.predict(&test).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(
            p[0], res.cluster[0],
            "shifted copy of series 0 should route to its cluster"
        );
        let _ = m;
    }

    #[test]
    fn test_kshape_validation() {
        let (data, _) = shifted_groups(1);
        // n_clusters = 0.
        let cfg0 = KShapeConfig::new(0);
        assert!(matches!(
            kshape_fd(&data, &cfg0),
            Err(FdarError::InvalidParameter { .. })
        ));
        // n_clusters > n.
        let cfg_big = KShapeConfig::new(999);
        assert!(matches!(
            kshape_fd(&data, &cfg_big),
            Err(FdarError::InvalidParameter { .. })
        ));
        // n_init = 0.
        let cfg_ni = KShapeConfig {
            n_init: 0,
            ..KShapeConfig::new(2)
        };
        assert!(matches!(
            kshape_fd(&data, &cfg_ni),
            Err(FdarError::InvalidParameter { .. })
        ));
        // Empty data.
        let empty = FdMatrix::zeros(0, 0);
        assert!(matches!(
            kshape_fd(&empty, &KShapeConfig::new(2)),
            Err(FdarError::InvalidDimension { .. })
        ));
        // predict dimension mismatch.
        let res = kshape_fd(&data, &KShapeConfig::new(2)).unwrap();
        let wrong = matrix_from_rows(&[vec![1.0, 2.0, 3.0]]);
        assert!(matches!(
            res.predict(&wrong),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_sbd_kmedoids_recovers_groups() {
        // Two shifted-shape groups → k-medoids over SBD recovers them at high
        // purity, proving it uses the shape-invariant SBD matrix (an L2/DTW
        // backend would be fooled by the circular shifts).
        let (data, truth) = shifted_groups(7);
        let cfg = KMedoidsConfig {
            k: 2,
            max_iter: 100,
            seed: 3,
        };
        let res = sbd_kmedoids(&data, &cfg).unwrap();
        assert_eq!(res.labels.len(), 16);
        assert_eq!(res.medoid_indices.len(), 2);
        let p = purity(&res.labels, &truth, 2);
        assert!(p >= 0.9, "SBD k-medoids purity {p} too low (< 0.9)");
    }

    #[test]
    fn test_sbd_kmedoids_uses_sbd_matrix() {
        // sbd_kmedoids == manual composition sbd_distance_matrix + kmedoids_from_distances
        // (same seed → identical labels and medoids).
        let (data, _) = shifted_groups(5);
        let cfg = KMedoidsConfig {
            k: 2,
            max_iter: 100,
            seed: 42,
        };
        let res = sbd_kmedoids(&data, &cfg).unwrap();
        let dist = sbd_distance_matrix(&data).unwrap();
        let manual = kmedoids_from_distances(&dist, &cfg).unwrap();
        assert_eq!(
            res.labels, manual.labels,
            "labels must match manual composition"
        );
        assert_eq!(
            res.medoid_indices, manual.medoid_indices,
            "medoids must match manual composition"
        );
        assert_eq!(
            res.total_within_distance.to_bits(),
            manual.total_within_distance.to_bits()
        );
    }

    #[test]
    fn test_sbd_kmedoids_validation() {
        let (data, _) = shifted_groups(1);
        // k = 0 → error (propagated from kmedoids_from_distances).
        let cfg0 = KMedoidsConfig {
            k: 0,
            ..Default::default()
        };
        assert!(matches!(
            sbd_kmedoids(&data, &cfg0),
            Err(FdarError::InvalidParameter { .. })
        ));
        // k > n → error.
        let cfg_big = KMedoidsConfig {
            k: 999,
            ..Default::default()
        };
        assert!(matches!(
            sbd_kmedoids(&data, &cfg_big),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    /// Crate-root re-exports for the full v0.34.0 SBD + k-Shape surface resolve.
    ///
    /// Uses `crate::` paths (the same items published at the crate root); a full
    /// external `use fdars_core::{...}` resolution is additionally covered by the
    /// `sbd_kmedoids` doctest, which is compiled as an out-of-crate binary.
    #[test]
    fn test_kshape_reexports() {
        use crate::{
            kshape_fd, sbd, sbd_distance_matrix, sbd_kmedoids, KMedoidsConfig, KMedoidsResult,
            KShapeConfig, KShapeResult, SbdResult,
        };
        // Reference each item so an unresolved name fails to compile.
        let _f: fn(&FdMatrix, &KShapeConfig) -> Result<KShapeResult, FdarError> = kshape_fd;
        let _k: fn(&FdMatrix, &KMedoidsConfig) -> Result<KMedoidsResult, FdarError> = sbd_kmedoids;
        let _s: fn(&[f64], &[f64]) -> Result<SbdResult, FdarError> = sbd;
        let _m: fn(&FdMatrix) -> Result<FdMatrix, FdarError> = sbd_distance_matrix;
    }
}
