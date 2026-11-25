//! DTW k-means with DTW Barycenter Averaging (DBA) centroids.
//!
//! The Lloyd loop of tslearn's `TimeSeriesKMeans(metric="dtw")`:
//!
//! 1. **Assignment.** Each series goes to the centroid with the smallest DTW
//!    cost (squared local cost, optional [`DtwWindow`]; lowest index on ties).
//! 2. **Update.** Each centroid is recomputed by
//!    [DBA](crate::metric::dba) over its members, warm-started from the
//!    previous centroid (tslearn passes `init_barycenter=cluster_centers_[k]`).
//!
//! The loop stops when the inertia (mean DTW cost of each series to its
//! assigned centroid) changes by less than `tol`. After the loop the series are
//! re-assigned to the final centroids, as tslearn's `_post_fit` does.
//!
//! ## Differences from tslearn
//!
//! - **Inertia** is reported for the *final* centroids and labels; tslearn's
//!   `inertia_` is the value computed before the last centroid update.
//! - **Empty clusters** are re-seeded in place from the series farthest from
//!   its own centroid (the [`crate::kshape`] convention) instead of restarting
//!   the whole run; the algorithm never fails on an empty cluster.
//! - **Restarts:** `n_init` k-means++ restarts, restart `r` seeded `seed + r`;
//!   the lowest-inertia run wins. k-means++ uses DTW costs and tslearn's greedy
//!   `2 + ⌊ln k⌋` local trials. Given the same explicit initial centroids
//!   ([`DtwKMeansInit::Centroids`]) and no empty clusters, labels and centroids
//!   match tslearn to rounding.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::metric::dba::{dba_from_init, DBA_P};
use crate::metric::dtw::dtw_distance_windowed;
use crate::metric::DtwWindow;
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Initial centroids for [`dtw_kmeans`].
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DtwKMeansInit {
    /// k-means++ seeding on DTW costs (tslearn's `_k_init_metric`), one
    /// independent seeding per restart.
    #[default]
    KMeansPlusPlus,
    /// Explicit initial centroids (`n_clusters × m`, finite). Every restart
    /// starts from them, so `n_init > 1` only repeats the same run.
    Centroids(FdMatrix),
}

/// Configuration for [`dtw_kmeans`].
///
/// Defaults: `n_clusters = 2`, `n_init = 10` (the fdars clustering convention;
/// tslearn's default is 1), `max_iter = 50`, `tol = 1e-6`, `seed = 0`, no
/// window, k-means++ init, and DBA with `dba_max_iter = 30`, `dba_tol = 1e-5`
/// (the values tslearn uses inside `TimeSeriesKMeans`).
///
/// ```
/// use fdars_core::{DtwKMeansConfig, DtwWindow};
/// let mut cfg = DtwKMeansConfig::default();
/// cfg.n_clusters = 3;
/// cfg.window = DtwWindow::SakoeChiba { radius: 5 };
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwKMeansConfig {
    /// Number of clusters `k` (`1 ≤ k ≤ n`).
    pub n_clusters: usize,
    /// Number of restarts (`≥ 1`); the lowest-inertia run is returned.
    pub n_init: usize,
    /// Maximum Lloyd iterations per restart.
    pub max_iter: usize,
    /// Stop when the absolute inertia change is below `tol` (finite, `≥ 0`).
    pub tol: f64,
    /// Base RNG seed; restart `r` is seeded `seed + r`.
    pub seed: u64,
    /// Global warping window for assignment and DBA alignments.
    pub window: DtwWindow,
    /// Initial centroids.
    pub init: DtwKMeansInit,
    /// Maximum DBA iterations per centroid update.
    pub dba_max_iter: usize,
    /// DBA cost-change tolerance per centroid update.
    pub dba_tol: f64,
}

impl Default for DtwKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_init: 10,
            max_iter: 50,
            tol: 1e-6,
            seed: 0,
            window: DtwWindow::None,
            init: DtwKMeansInit::KMeansPlusPlus,
            dba_max_iter: 30,
            dba_tol: 1e-5,
        }
    }
}

/// Result of [`dtw_kmeans`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwKMeansResult {
    /// DBA centroids as a `k × m` matrix.
    pub centroids: FdMatrix,
    /// Cluster label of each training series (values in `0..k`).
    pub cluster: Vec<usize>,
    /// Mean DTW cost `(1/n) Σ_i DTW²(x_i, centroid[cluster[i]])` for the
    /// returned centroids and labels.
    pub inertia: f64,
    /// Lloyd iterations run by the winning restart.
    pub iter: usize,
    /// Whether the winning restart met `tol` before `max_iter`.
    pub converged: bool,
    /// Index (0-based) of the winning restart.
    pub n_init_best: usize,
    /// The window used for fitting, reused by [`DtwKMeansResult::predict`].
    pub window: DtwWindow,
}

impl DtwKMeansResult {
    /// Assign new series to the nearest fitted centroid (DTW cost, same window).
    ///
    /// `predict` on the training data reproduces [`DtwKMeansResult::cluster`].
    ///
    /// # Errors
    /// [`FdarError::InvalidDimension`] if `new_data` is empty;
    /// [`FdarError::InvalidParameter`] if it holds non-finite values or the
    /// window is unfeasible for its length;
    /// [`FdarError::ComputationFailed`] if no admissible path exists.
    pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> {
        let rows = validated_rows(new_data, "new_data")?;
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let cents = matrix_rows(&self.centroids);
        let (labels, _) = nearest(&refs, &cents, self.window)?;
        Ok(labels)
    }
}

/// DTW k-means clustering with DBA centroids (tslearn
/// `TimeSeriesKMeans(metric="dtw")`).
///
/// See the [module docs](self) for the algorithm and the documented
/// differences from tslearn.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty, or explicit initial
///   centroids do not have shape `n_clusters × m`.
/// - [`FdarError::InvalidParameter`] for `n_clusters` outside `1..=n`,
///   `n_init == 0`, a negative / non-finite `tol` or `dba_tol`, non-finite
///   data or centroids, or an invalid / unfeasible window.
/// - [`FdarError::ComputationFailed`] if no admissible warping path exists.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_kmeans, DtwKMeansConfig};
///
/// // Two groups: a shifted up-bump and a shifted down-bump (column-major).
/// let (n, m) = (6, 30);
/// let mut flat = vec![0.0; n * m];
/// for i in 0..n {
///     let (sign, c) = if i < 3 { (1.0, 0.3 + 0.05 * i as f64) } else { (-1.0, 0.5 + 0.05 * i as f64) };
///     for t in 0..m {
///         let u = t as f64 / (m - 1) as f64;
///         flat[i + t * n] = sign * (-((u - c) / 0.08f64).powi(2)).exp();
///     }
/// }
/// let data = FdMatrix::from_column_major(flat, n, m).unwrap();
/// let res = dtw_kmeans(&data, &DtwKMeansConfig::default()).unwrap();
/// assert_eq!(res.cluster[0], res.cluster[2]);
/// assert_ne!(res.cluster[0], res.cluster[3]);
/// assert_eq!(res.predict(&data).unwrap(), res.cluster);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_kmeans(data: &FdMatrix, config: &DtwKMeansConfig) -> Result<DtwKMeansResult, FdarError> {
    let rows = validated_rows(data, "data")?;
    validate_config(config, rows.len(), rows[0].len())?;
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut best: Option<DtwKMeansResult> = None;
    for r in 0..config.n_init {
        let mut run = run_restart(&refs, config, config.seed.wrapping_add(r as u64))?;
        run.n_init_best = r;
        if best.as_ref().map_or(true, |b| run.inertia < b.inertia) {
            best = Some(run);
        }
    }
    best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "dtw_kmeans",
        detail: "no restart produced a result".to_string(),
    })
}

/// Rows of a non-empty, finite matrix.
fn validated_rows(data: &FdMatrix, parameter: &'static str) -> Result<Vec<Vec<f64>>, FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "non-empty matrix (nrows > 0, ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "all values must be finite".to_string(),
        });
    }
    Ok(matrix_rows(data))
}

fn matrix_rows(data: &FdMatrix) -> Vec<Vec<f64>> {
    (0..data.nrows()).map(|i| data.row(i)).collect()
}

fn validate_config(config: &DtwKMeansConfig, n: usize, m: usize) -> Result<(), FdarError> {
    let k = config.n_clusters;
    if k == 0 || k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_clusters",
            message: format!("must be in 1..={n}, got {k}"),
        });
    }
    if config.n_init == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_init",
            message: "must be at least 1".to_string(),
        });
    }
    for (name, v) in [("tol", config.tol), ("dba_tol", config.dba_tol)] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: name,
                message: format!("must be finite and >= 0, got {v}"),
            });
        }
    }
    config.window.row_bounds(m, m)?;
    match &config.init {
        DtwKMeansInit::Centroids(c) => validate_init_centroids(c, k, m),
        DtwKMeansInit::KMeansPlusPlus => Ok(()),
    }
}

fn validate_init_centroids(c: &FdMatrix, k: usize, m: usize) -> Result<(), FdarError> {
    if c.shape() != (k, m) {
        return Err(FdarError::InvalidDimension {
            parameter: "init",
            expected: format!("{k}x{m} initial centroids"),
            actual: format!("{}x{}", c.nrows(), c.ncols()),
        });
    }
    if c.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "init",
            message: "initial centroids must be finite".to_string(),
        });
    }
    Ok(())
}

/// One full Lloyd run from its own initialisation.
fn run_restart(
    rows: &[&[f64]],
    config: &DtwKMeansConfig,
    seed: u64,
) -> Result<DtwKMeansResult, FdarError> {
    let window = config.window;
    let mut centroids = match &config.init {
        DtwKMeansInit::Centroids(c) => matrix_rows(c),
        DtwKMeansInit::KMeansPlusPlus => kmeans_plus_plus(rows, config.n_clusters, window, seed)?,
    };
    let mut old_inertia = f64::INFINITY;
    let mut iter = 0;
    let mut converged = false;
    for _ in 0..config.max_iter {
        let (labels, inertia) = assign_with_recovery(rows, &mut centroids, window)?;
        update_centroids(rows, &labels, &mut centroids, config)?;
        iter += 1;
        if (old_inertia - inertia).abs() < config.tol {
            converged = true;
            break;
        }
        old_inertia = inertia;
    }
    let (cluster, inertia) = assign_with_recovery(rows, &mut centroids, window)?;
    let m = centroids[0].len();
    let flat: Vec<f64> = (0..m)
        .flat_map(|j| centroids.iter().map(move |c| c[j]))
        .collect();
    Ok(DtwKMeansResult {
        centroids: FdMatrix::from_column_major(flat, centroids.len(), m)?,
        cluster,
        inertia,
        iter,
        converged,
        n_init_best: 0,
        window,
    })
}

/// DTW cost of every series to every centroid → (argmin labels, own costs).
fn nearest(
    rows: &[&[f64]],
    centroids: &[Vec<f64>],
    window: DtwWindow,
) -> Result<(Vec<usize>, Vec<f64>), FdarError> {
    let best = iter_maybe_parallel!(0..rows.len())
        .map(|i| {
            let mut arg = 0;
            let mut best = f64::INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let d = dtw_distance_windowed(rows[i], cent, DBA_P, window)?;
                if d < best {
                    best = d;
                    arg = c;
                }
            }
            Ok((arg, best))
        })
        .collect::<Result<Vec<(usize, f64)>, FdarError>>()?;
    Ok(best.into_iter().unzip())
}

/// Assign, re-seed empty clusters from the farthest series, return
/// `(labels, inertia)`.
fn assign_with_recovery(
    rows: &[&[f64]],
    centroids: &mut [Vec<f64>],
    window: DtwWindow,
) -> Result<(Vec<usize>, f64), FdarError> {
    let (mut labels, mut own) = nearest(rows, centroids, window)?;
    let k = centroids.len();
    loop {
        let mut sizes = vec![0usize; k];
        for &c in &labels {
            sizes[c] += 1;
        }
        let Some(empty) = (0..k).find(|&c| sizes[c] == 0) else {
            break;
        };
        // k ≤ n guarantees some cluster holds ≥ 2 series while one is empty.
        let far = (0..rows.len()).filter(|&i| sizes[labels[i]] > 1).fold(
            None,
            |acc: Option<usize>, i| match acc {
                Some(b) if own[b] >= own[i] => Some(b),
                _ => Some(i),
            },
        );
        let Some(i) = far else { break };
        centroids[empty] = rows[i].to_vec();
        labels[i] = empty;
        own[i] = 0.0;
    }
    let inertia = own.iter().sum::<f64>() / rows.len() as f64;
    Ok((labels, inertia))
}

/// DBA update of every centroid over its members, warm-started in place.
fn update_centroids(
    rows: &[&[f64]],
    labels: &[usize],
    centroids: &mut [Vec<f64>],
    config: &DtwKMeansConfig,
) -> Result<(), FdarError> {
    for (c, cent) in centroids.iter_mut().enumerate() {
        let members: Vec<&[f64]> = rows
            .iter()
            .zip(labels)
            .filter(|(_, l)| **l == c)
            .map(|(r, _)| *r)
            .collect();
        if members.is_empty() {
            continue;
        }
        let weights = vec![1.0; members.len()];
        let res = dba_from_init(
            &members,
            &weights,
            std::mem::take(cent),
            config.dba_max_iter,
            config.dba_tol,
            config.window,
        )?;
        *cent = res.barycenter;
    }
    Ok(())
}

/// k-means++ seeding on DTW costs with greedy local trials (tslearn
/// `_k_init_metric`).
fn kmeans_plus_plus(
    rows: &[&[f64]],
    k: usize,
    window: DtwWindow,
    seed: u64,
) -> Result<Vec<Vec<f64>>, FdarError> {
    let n = rows.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let trials = 2 + (k as f64).ln().floor() as usize;
    let first = rng.gen_range(0..n);
    let mut centers = vec![rows[first].to_vec()];
    let mut closest = costs_to(rows, rows[first], window)?;
    let mut pot: f64 = closest.iter().sum();
    for _ in 1..k {
        let mut cum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &d in &closest {
            acc += d;
            cum.push(acc);
        }
        let mut best: Option<(f64, usize, Vec<f64>)> = None;
        for _ in 0..trials {
            let target = rng.gen::<f64>() * pot;
            let cand = cum.partition_point(|&c| c < target).min(n - 1);
            let d = costs_to(rows, rows[cand], window)?;
            let merged: Vec<f64> = closest.iter().zip(&d).map(|(a, b)| a.min(*b)).collect();
            let p: f64 = merged.iter().sum();
            if best.as_ref().map_or(true, |(bp, _, _)| p < *bp) {
                best = Some((p, cand, merged));
            }
        }
        if let Some((p, cand, merged)) = best {
            centers.push(rows[cand].to_vec());
            closest = merged;
            pot = p;
        }
    }
    Ok(centers)
}

fn costs_to(rows: &[&[f64]], center: &[f64], window: DtwWindow) -> Result<Vec<f64>, FdarError> {
    iter_maybe_parallel!(0..rows.len())
        .map(|i| dtw_distance_windowed(center, rows[i], DBA_P, window))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values: tslearn 0.10.0
    // `TimeSeriesKMeans(n_clusters=2, metric="dtw", init=K[[0, 1]], n_init=1,
    // max_iter=50, tol=1e-6).fit(K)` on the `k_data()` set below (same formula
    // in Python); KM_INERTIA_FINAL = mean of cdist_dtw(K, centers)^2 at labels.
    const KM_C0: [f64; 24] = [
        0.01727064499150713,
        0.04210481115535555,
        0.025774808780113134,
        0.03454162581842836,
        0.08917980365328583,
        0.34239703232201213,
        0.7606374811911788,
        1.0108722198366777,
        0.7631293396856507,
        0.3320797890107903,
        0.10176846366086134,
        0.041024658620717656,
        0.026544898185433242,
        0.015302714016630081,
        0.01119071517081647,
        0.03991711895815464,
        0.031473489720816684,
        0.02530868845630737,
        0.015035115970690266,
        0.004777013183551332,
        0.044519326115977795,
        0.034261645193487636,
        0.028107036669971053,
        0.02012512388503469,
    ];
    const KM_C1: [f64; 24] = [
        0.032754062571095574,
        0.02427983845963168,
        0.021531347993521646,
        0.012323461614222415,
        -0.00598244833272101,
        0.0011165105026163288,
        -0.09286503260687717,
        -0.3352522305042176,
        -0.6869060153758835,
        -0.949722151572702,
        -0.8874216428029605,
        -0.5688333805286607,
        -0.24004190568793912,
        -0.06243208723701897,
        0.04901428279159727,
        0.24864843220330268,
        0.5008154854056139,
        0.3959537987231543,
        0.12247626687355476,
        0.037853713498510924,
        0.028751900811052215,
        0.02748157078324337,
        0.02562519470001636,
        0.021526320991529287,
    ];
    const KM_INERTIA_FINAL: f64 = 0.04695699675443099;
    const KM_LABELS: [usize; 10] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

    fn k_data() -> FdMatrix {
        let (n, m) = (10usize, 24usize);
        let mut flat = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let u = t as f64 / (m - 1) as f64;
                let noise = ((i * 7919 + t * 104729) % 1009) as f64 / 1009.0;
                let v = if i % 2 == 0 {
                    let c = 0.3 + 0.05 * (i % 5) as f64;
                    (-((u - c) / 0.08).powi(2)).exp()
                } else {
                    let c = 0.35 + 0.05 * (i % 5) as f64;
                    -(-((u - c) / 0.1).powi(2)).exp()
                        + 0.5 * (-((u - c - 0.3) / 0.06).powi(2)).exp()
                };
                flat[i + t * n] = v + 0.05 * noise;
            }
        }
        FdMatrix::from_column_major(flat, n, m).unwrap()
    }

    fn rows_to_matrix(rows: &[Vec<f64>]) -> FdMatrix {
        let (n, m) = (rows.len(), rows[0].len());
        let flat: Vec<f64> = (0..m)
            .flat_map(|j| rows.iter().map(move |r| r[j]))
            .collect();
        FdMatrix::from_column_major(flat, n, m).unwrap()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "{what}[{k}]: got {g}, want {w}");
        }
    }

    #[test]
    fn matches_tslearn_from_fixed_init() {
        let data = k_data();
        let init = rows_to_matrix(&[data.row(0), data.row(1)]);
        let cfg = DtwKMeansConfig {
            init: DtwKMeansInit::Centroids(init),
            n_init: 1,
            ..DtwKMeansConfig::default()
        };
        let res = dtw_kmeans(&data, &cfg).unwrap();
        assert_eq!(res.cluster, KM_LABELS.to_vec());
        assert_close(&res.centroids.row(0), &KM_C0, 1e-10, "c0");
        assert_close(&res.centroids.row(1), &KM_C1, 1e-10, "c1");
        assert!((res.inertia - KM_INERTIA_FINAL).abs() < 1e-10);
        assert_eq!(res.iter, 3); // tslearn n_iter_ == 3
        assert!(res.converged);
    }

    #[test]
    fn kmeans_plus_plus_recovers_groups_deterministically() {
        let data = k_data();
        let res = dtw_kmeans(&data, &DtwKMeansConfig::default()).unwrap();
        let even = res.cluster[0];
        for (i, &l) in res.cluster.iter().enumerate() {
            assert_eq!(l == even, i % 2 == 0, "series {i}");
        }
        let again = dtw_kmeans(&data, &DtwKMeansConfig::default()).unwrap();
        assert_eq!(res, again);
        assert_eq!(res.predict(&data).unwrap(), res.cluster);
    }

    #[test]
    fn labels_are_argmin_and_inertia_consistent() {
        let data = k_data();
        let cfg = DtwKMeansConfig {
            n_clusters: 3,
            n_init: 3,
            window: DtwWindow::SakoeChiba { radius: 4 },
            ..DtwKMeansConfig::default()
        };
        let res = dtw_kmeans(&data, &cfg).unwrap();
        let mut total = 0.0;
        for i in 0..data.nrows() {
            let costs: Vec<f64> = (0..3)
                .map(|c| {
                    dtw_distance_windowed(&data.row(i), &res.centroids.row(c), 2.0, cfg.window)
                        .unwrap()
                })
                .collect();
            let own = costs[res.cluster[i]];
            assert!(costs.iter().all(|&d| own <= d));
            total += own;
        }
        assert!((res.inertia - total / data.nrows() as f64).abs() < 1e-12);
        assert!(res.n_init_best < 3);
    }

    #[test]
    fn zero_radius_window_is_euclidean_kmeans() {
        let data = k_data();
        let cfg = DtwKMeansConfig {
            window: DtwWindow::SakoeChiba { radius: 0 },
            n_init: 2,
            ..DtwKMeansConfig::default()
        };
        let res = dtw_kmeans(&data, &cfg).unwrap();
        // A radius-0 band forces the diagonal path, so DBA is the arithmetic
        // mean of the members; once converged the final labels are those the
        // centroids were last updated from.
        assert!(res.converged);
        for c in 0..2 {
            let members: Vec<usize> = (0..10).filter(|&i| res.cluster[i] == c).collect();
            assert!(!members.is_empty());
            let mean: Vec<f64> = (0..24)
                .map(|j| members.iter().map(|&i| data[(i, j)]).sum::<f64>() / members.len() as f64)
                .collect();
            assert_close(&res.centroids.row(c), &mean, 1e-12, "euclidean centroid");
        }
    }

    #[test]
    fn empty_cluster_is_reseeded_from_farthest_series() {
        let data = k_data();
        // Centroid 1 is far from every series, so the first assignment leaves
        // it empty; it must be re-seeded instead of failing (tslearn restarts).
        let init = rows_to_matrix(&[data.row(0), vec![100.0; 24]]);
        let cfg = DtwKMeansConfig {
            init: DtwKMeansInit::Centroids(init),
            n_init: 1,
            ..DtwKMeansConfig::default()
        };
        let res = dtw_kmeans(&data, &cfg).unwrap();
        assert!((0..2).all(|c| res.cluster.contains(&c)));
        assert!(res.centroids.as_slice().iter().all(|v| v.abs() < 10.0));
        assert_eq!(res.predict(&data).unwrap(), res.cluster);
    }

    #[test]
    fn k_equals_n_gives_singletons() {
        let data = k_data();
        let cfg = DtwKMeansConfig {
            n_clusters: 10,
            n_init: 1,
            ..DtwKMeansConfig::default()
        };
        let res = dtw_kmeans(&data, &cfg).unwrap();
        let mut seen = res.cluster.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
        assert!(res.inertia.abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let data = k_data();
        let bad = |cfg: DtwKMeansConfig| dtw_kmeans(&data, &cfg).unwrap_err();
        for k in [0, 11] {
            let cfg = DtwKMeansConfig {
                n_clusters: k,
                ..DtwKMeansConfig::default()
            };
            assert!(matches!(bad(cfg), FdarError::InvalidParameter { .. }));
        }
        let cfg = DtwKMeansConfig {
            n_init: 0,
            ..DtwKMeansConfig::default()
        };
        assert!(matches!(bad(cfg), FdarError::InvalidParameter { .. }));
        let cfg = DtwKMeansConfig {
            tol: f64::NAN,
            ..DtwKMeansConfig::default()
        };
        assert!(matches!(bad(cfg), FdarError::InvalidParameter { .. }));
        let cfg = DtwKMeansConfig {
            init: DtwKMeansInit::Centroids(FdMatrix::zeros(3, 24)),
            ..DtwKMeansConfig::default()
        };
        assert!(matches!(bad(cfg), FdarError::InvalidDimension { .. }));
        let cfg = DtwKMeansConfig {
            window: DtwWindow::Itakura { max_slope: 0.5 },
            ..DtwKMeansConfig::default()
        };
        assert!(matches!(bad(cfg), FdarError::InvalidParameter { .. }));
        assert!(matches!(
            dtw_kmeans(&FdMatrix::zeros(0, 3), &DtwKMeansConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
        let res = dtw_kmeans(&data, &DtwKMeansConfig::default()).unwrap();
        assert!(matches!(
            res.predict(&FdMatrix::zeros(0, 24)),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
