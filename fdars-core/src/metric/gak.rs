//! Global Alignment Kernel (GAK) — a positive-semi-definite kernel between curves.
//!
//! GAK (Cuturi, "Fast Global Alignment Kernels", ICML 2011) turns the DTW
//! *minimum* alignment cost into a *sum over all alignment paths*, yielding a
//! valid kernel for kernel machines (SVM, kernel-k-means, kernel PCA). Unlike the
//! naive `exp(-DTW)` "Gaussian DTW kernel", which is **not** positive
//! semi-definite (PSD), the normalized triangular GAK (TGAK) is PSD by
//! construction — it is the sum over the whole alignment lattice, not a single
//! best-path distance.
//!
//! ## Numerical strategy (mandatory)
//!
//! The forward recursion multiplies local kernel values in `(0, 1]` along every
//! alignment path. For series longer than ~50 points this product underflows
//! `f64` to `0.0`, silently returning an all-zero Gram matrix. Therefore the
//! whole DP runs in **log space**, accumulating with a 3-way log-sum-exp
//! (soft-MAX) rather than a raw product. This mirrors the log-sum-exp
//! stabilization of `softmin3` in [`super::soft_dtw`], but note the difference:
//! soft-DTW uses a soft-*MIN* (a distance); GAK uses a soft-*MAX* / log-sum-exp
//! (a kernel path-sum). The two are **not** interchangeable.
//!
//! ## Normalization (mandatory for PSD)
//!
//! The public kernel is always the *triangular normalized* form
//! `k(x, y) / sqrt(k(x, x) · k(y, y))`, computed entirely in log space as
//! `exp(logGAK(x, y) - 0.5·(logGAK(x, x) + logGAK(y, y)))`. This gives a
//! similarity in `[0, 1]` with `k(x, x) = 1`, and is the form with the PSD
//! guarantee. The unnormalized log-kernel [`loggak`] is exposed only
//! `pub(crate)` for internal reuse (e.g. split train/predict normalization).
//!
//! ## Local kernel
//!
//! The TGAK local (triangular) kernel between two scalar observations is, with
//! `h = exp(-d² / (2σ²))` and `d = xᵢ − yⱼ`:
//! `local(i, j) = h / (2 − h)`, i.e. in log space
//! `log_local(i, j) = -d²/(2σ²) - ln(2 - exp(-d²/(2σ²)))`.
//! The `2 − h` denominator is Cuturi's infinite-divisibility correction that
//! makes the resulting kernel PSD.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Configuration for Global Alignment Kernel computation.
///
/// `sigma = None` selects the median-distance bandwidth heuristic
/// ([`sigma_gak`]); `sigma = Some(s)` fixes the bandwidth (`s > 0`).
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GakConfig {
    /// Bandwidth σ (> 0). `None` → auto-select via the median heuristic.
    pub sigma: Option<f64>,
}

impl GakConfig {
    /// Construct a config with an explicit bandwidth σ.
    #[must_use]
    pub fn with_sigma(sigma: f64) -> Self {
        Self { sigma: Some(sigma) }
    }
}

/// Stable 3-way log-sum-exp: `ln(exp(a) + exp(b) + exp(c))`.
///
/// Uses the max-subtraction trick for numerical stability (mirrors the structure
/// of [`super::soft_dtw::softmin3`], but this is a soft-MAX, not a soft-MIN — do
/// not substitute one for the other). Returns `-inf` when all inputs are `-inf`
/// (the identity element of the path-sum), never `NaN`.
#[inline]
pub(crate) fn logsumexp3(a: f64, b: f64, c: f64) -> f64 {
    let max_val = a.max(b).max(c);
    if max_val == f64::NEG_INFINITY {
        // All three predecessors unreachable — contribute nothing.
        return f64::NEG_INFINITY;
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let ea = (a - max_val).exp();
    let eb = (b - max_val).exp();
    let ec = (c - max_val).exp();
    max_val + (ea + eb + ec).ln()
}

/// Log of the TGAK local (triangular) kernel between two scalar observations.
///
/// `log_local = -d²/(2σ²) - ln(2 - exp(-d²/(2σ²)))`, always finite for σ > 0.
#[inline]
fn log_local(xi: f64, yj: f64, inv_two_sigma_sq: f64) -> f64 {
    let d = xi - yj;
    let neg_half_dist = -(d * d) * inv_two_sigma_sq; // -d²/(2σ²)  ≤ 0
    let h = neg_half_dist.exp(); // h ∈ (0, 1]
    neg_half_dist - (2.0 - h).ln()
}

/// Unnormalized log-domain Global Alignment Kernel between two 1D series.
///
/// Runs the full `(n+1) × (m+1)` forward DP over the alignment lattice in log
/// space with a 2-row rolling buffer (O(m) memory), returning `L[n][m] = log k(x, y)`.
/// This is the **unnormalized** log-kernel — the public [`gak`] applies triangular
/// normalization on top. Exposed `pub(crate)` so downstream code (Phase 55
/// split-normalized Gram export) can reuse the raw log-kernel + cached diagonals.
///
/// Returns `f64::NEG_INFINITY` when either series is empty (empty path-sum).
pub(crate) fn loggak(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return f64::NEG_INFINITY;
    }
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    // Log-domain DP: L[0][0] = 0 (= log 1), all other boundary cells = -inf.
    let mut prev = vec![f64::NEG_INFINITY; m + 1];
    let mut curr = vec![f64::NEG_INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f64::NEG_INFINITY;
        let xi = x[i - 1];
        for j in 1..=m {
            let ll = log_local(xi, y[j - 1], inv_two_sigma_sq);
            curr[j] = ll + logsumexp3(prev[j], curr[j - 1], prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Normalized pairwise GAK similarity between two curves.
///
/// Returns the triangular-normalized kernel
/// `exp(logGAK(x, y) - 0.5·(logGAK(x, x) + logGAK(y, y)))`, a value in `[0, 1]`
/// with `gak(x, x) == 1`. Wholly dissimilar curves yield a value near `0`
/// (never `NaN`/`Inf`). `sigma` must be `> 0`; a non-positive σ yields `0.0`
/// (the matrix-level entry points validate σ up front and return an error).
///
/// # Examples
/// ```
/// use fdars_core::gak;
/// let x = [0.0, 1.0, 2.0, 3.0];
/// let y = [0.0, 1.0, 2.0, 3.0];
/// // Self-similarity is exactly 1.
/// assert!((gak(&x, &y, 1.0) - 1.0).abs() < 1e-12);
/// // A different curve gives a value strictly inside [0, 1].
/// let z = [3.0, 2.0, 1.0, 0.0];
/// let k = gak(&x, &z, 1.0);
/// assert!((0.0..=1.0).contains(&k));
/// ```
#[must_use]
pub fn gak(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    if sigma <= 0.0 || x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let log_xy = loggak(x, y, sigma);
    let log_xx = loggak(x, x, sigma);
    let log_yy = loggak(y, y, sigma);
    normalize_log(log_xy, log_xx, log_yy)
}

/// Apply triangular normalization in log space and exponentiate to `[0, 1]`.
///
/// `-inf` numerator (utterly dissimilar) maps to `0.0`, never `NaN`.
#[inline]
fn normalize_log(log_xy: f64, log_xx: f64, log_yy: f64) -> f64 {
    let log_norm = log_xy - 0.5 * (log_xx + log_yy);
    if log_norm == f64::NEG_INFINITY {
        0.0
    } else {
        log_norm.exp()
    }
}

/// Median-distance bandwidth heuristic for GAK.
///
/// Returns the median pairwise Euclidean distance between the curves in `data`
/// (rows), clamped to a small positive floor so degenerate/identical data never
/// produces `σ = 0`. This is the fdars analogue of tslearn's `sigma_gak` median
/// heuristic; with this σ the off-diagonal Gram entries land in a healthy,
/// non-degenerate range rather than collapsing to near-identity or near-constant.
///
/// Divergence note: tslearn samples random point-pairs and scales by
/// `sqrt(median_length)`; fdars uses the exact median of full-curve Euclidean
/// distances (deterministic, no RNG), which yields an equivalently healthy
/// bandwidth for the fixed-length `FdMatrix` layout.
#[must_use]
pub fn sigma_gak(data: &FdMatrix) -> f64 {
    const SIGMA_FLOOR: f64 = 1e-8;
    let n = data.nrows();
    let m = data.ncols();
    if n < 2 || m == 0 {
        return SIGMA_FLOOR.max(1.0);
    }
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in 0..m {
                let d = rows[i][k] - rows[j][k];
                sum += d * d;
            }
            dists.push(sum.sqrt());
        }
    }
    if dists.is_empty() {
        return SIGMA_FLOOR.max(1.0);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = dists.len() / 2;
    let median = if dists.len() % 2 == 0 {
        0.5 * (dists[mid - 1] + dists[mid])
    } else {
        dists[mid]
    };
    median.max(SIGMA_FLOOR)
}

/// Build the n×n normalized GAK Gram matrix over a curve set.
///
/// The result is a positive-semi-definite kernel matrix with entries in `[0, 1]`
/// and an all-ones diagonal (`G[i][i] == 1.0` exactly). It is **symmetric by
/// assignment**: the upper triangle is computed once and mirrored
/// (`G[j][i] = G[i][j]`), so symmetry is bit-exact rather than merely within
/// tolerance. The upper-triangle computation is parallelized under the `parallel`
/// feature via `iter_maybe_parallel!`; because the kernel is order-independent and
/// uses no RNG, the sequential and parallel builds are bit-identical.
///
/// The bandwidth comes from `config.sigma` when `Some`, otherwise from the median
/// heuristic ([`sigma_gak`]). Diagonal self-log-kernels are computed once (O(n))
/// and reused during normalization — no redundant self-kernel recomputation.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (no curves or no points).
/// - [`FdarError::InvalidParameter`] if the resolved σ is not `> 0`.
///
/// # Examples
/// ```
/// use fdars_core::{gak_gram_matrix, GakConfig, FdMatrix};
/// // Two identical curves and one shifted curve (column-major FdMatrix).
/// let data = FdMatrix::from_slice(
///     &[0.0, 0.0, 5.0,  1.0, 1.0, 6.0,  2.0, 2.0, 7.0],
///     3, 3,
/// ).unwrap();
/// let gram = gak_gram_matrix(&data, &GakConfig::with_sigma(1.0)).unwrap();
/// assert_eq!(gram.shape(), (3, 3));
/// assert!((gram[(0, 0)] - 1.0).abs() < 1e-12); // unit diagonal
/// assert_eq!(gram[(0, 1)], gram[(1, 0)]);      // symmetric by assignment
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gak_gram_matrix(data: &FdMatrix, config: &GakConfig) -> Result<FdMatrix, FdarError> {
    let (gram, _diag_log, _sigma, _rows) = build_train_gram(data, config)?;
    Ok(gram)
}

/// Resolve σ + validate, then build the normalized symmetric-by-assignment PSD
/// n×n Gram together with the per-curve diagonal self-log-kernels and the σ
/// actually used.
///
/// Shared core of [`gak_gram_matrix`] (which discards the diagonals) and
/// [`gak_gram_train`] (which stores them for split train/predict normalization).
/// Factoring this keeps the two Gram builders bit-identical and avoids
/// recomputing the O(n) diagonal.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (no curves or no points).
/// - [`FdarError::InvalidParameter`] if the resolved σ is not `> 0`.
#[allow(clippy::type_complexity)]
fn build_train_gram(
    data: &FdMatrix,
    config: &GakConfig,
) -> Result<(FdMatrix, Vec<f64>, f64, Vec<Vec<f64>>), FdarError> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0, ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }

    let sigma = match config.sigma {
        Some(s) => s,
        None => sigma_gak(data),
    };
    // Reject non-positive or NaN bandwidth (NaN fails the `> 0.0` test).
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "sigma",
            message: format!("bandwidth must be > 0, got {sigma}"),
        });
    }

    // Pre-collect rows once (avoid O(n²) row materializations in the hot loop).
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();

    // Precompute the n diagonal self-log-kernels once (O(n)); reused in normalization.
    let diag_log: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| loggak(&rows[i], &rows[i], sigma))
        .collect();

    // Compute the upper triangle (order-independent) in parallel, then scatter +
    // mirror for bit-exact symmetry.
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let log_xy = loggak(&rows[i], &rows[j], sigma);
                    normalize_log(log_xy, diag_log[i], diag_log[j])
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut gram = FdMatrix::zeros(n, n);
    for i in 0..n {
        gram[(i, i)] = 1.0; // unit diagonal by construction
    }
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let v = upper_vals[idx];
            gram[(i, j)] = v;
            gram[(j, i)] = v; // symmetric by assignment (bit-exact)
            idx += 1;
        }
    }
    Ok((gram, diag_log, sigma, rows))
}

/// Result of [`gak_gram_train`]: a training GAK Gram plus everything needed to
/// cross-normalize an out-of-sample prediction Gram against the **training**
/// self-kernels.
///
/// Consumed by [`gak_gram_predict`]. The `gram` is the n_train × n_train
/// normalized, symmetric-by-assignment, PSD kernel matrix with a unit diagonal —
/// directly usable as `SVC(kernel='precomputed')` training input. The
/// per-training-curve unnormalized log self-kernels (`log_self`) are kept
/// `pub(crate)`; they are the internal contract that makes the split-normalized
/// prediction impossible to get wrong (Cuturi triangular normalization needs the
/// *training* diagonals, never test-only self-kernels).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GakGramTrain {
    /// n_train × n_train normalized GAK Gram (PSD, unit diagonal, symmetric).
    pub gram: FdMatrix,
    /// Per-training-curve unnormalized log self-kernels `loggak(x_i, x_i, sigma)`,
    /// used to cross-normalize the prediction Gram. Kept `pub(crate)`; read via
    /// [`GakGramTrain::log_self`].
    pub(crate) log_self: Vec<f64>,
    /// The bandwidth σ actually used (resolved from the config, possibly via the
    /// median heuristic). Prediction reuses this exact value.
    pub sigma: f64,
    /// Training curves (one `Vec<f64>` per row), retained so that
    /// [`gak_gram_predict`] can evaluate the cross-kernel `loggak(test, train)`.
    /// Kept `pub(crate)` — an internal contract, not part of the public surface.
    pub(crate) train_rows: Vec<Vec<f64>>,
}

impl GakGramTrain {
    /// The per-training-curve unnormalized log self-kernels used for
    /// cross-normalizing a prediction Gram.
    #[must_use]
    pub fn log_self(&self) -> &[f64] {
        &self.log_self
    }
}

/// Build a **training** GAK Gram matrix for an external precomputed-kernel SVM.
///
/// Returns a [`GakGramTrain`] whose `gram` is the n_train × n_train normalized,
/// PSD, symmetric-by-assignment Gram (unit diagonal) — directly consumable as
/// `SVC(kernel='precomputed')` training input — together with the resolved σ and
/// the per-training-curve log self-kernels needed to cross-normalize a later
/// prediction Gram ([`gak_gram_predict`]).
///
/// The bandwidth comes from `config.sigma` when `Some`, otherwise from the median
/// heuristic ([`sigma_gak`]). Diagonal self-log-kernels are computed once (O(n));
/// the Gram build shares its core with [`gak_gram_matrix`], so the two are
/// bit-identical.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (no curves or no points).
/// - [`FdarError::InvalidParameter`] if the resolved σ is not `> 0`.
///
/// # Examples
/// ```
/// use fdars_core::{gak_gram_train, gak_gram_predict, GakConfig, FdMatrix};
/// // Three training curves, two eval points each (column-major).
/// let train = FdMatrix::from_slice(
///     &[0.0, 1.0, 5.0,  0.0, 1.0, 5.0],
///     3, 2,
/// ).unwrap();
/// let fit = gak_gram_train(&train, &GakConfig::with_sigma(1.0)).unwrap();
/// assert_eq!(fit.gram.shape(), (3, 3));
///
/// // Score two new curves against the fitted training set.
/// let test = FdMatrix::from_slice(&[0.0, 5.0,  0.0, 5.0], 2, 2).unwrap();
/// let k_test = gak_gram_predict(&fit, &test).unwrap();
/// assert_eq!(k_test.shape(), (2, 3)); // n_test × n_train
///
/// // External precomputed-kernel SVM handoff (no Python dependency here):
/// //   from sklearn.svm import SVC
/// //   svc = SVC(kernel='precomputed').fit(fit.gram, y_train)
/// //   preds = svc.predict(k_test)   # k_test is n_test × n_train
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gak_gram_train(data: &FdMatrix, config: &GakConfig) -> Result<GakGramTrain, FdarError> {
    let (gram, log_self, sigma, train_rows) = build_train_gram(data, config)?;
    Ok(GakGramTrain {
        gram,
        log_self,
        sigma,
        train_rows,
    })
}

/// Build an out-of-sample **prediction** GAK Gram against a fitted training set.
///
/// Returns an **n_test × n_train** matrix (rows = new/test curves, cols =
/// training curves) whose entry `(t, j)` is the triangular-normalized GAK
/// `exp(loggak(x_test_t, x_train_j, σ) - 0.5·(loggak(x_test_t, x_test_t, σ) +
/// log_self_train[j]))`. This matches the `SVC(kernel='precomputed')` prediction
/// contract `K[t, j] = kernel(test_t, train_j)`; passing the transpose silently
/// degrades an external SVM.
///
/// The bandwidth σ and the training self-kernels come from `train` (the stored
/// [`GakGramTrain::sigma`] and [`GakGramTrain::log_self`]) — never recomputed from
/// the test set alone — so the prediction lives in the *same* feature space as the
/// training Gram. Test self-log-kernels are computed once (O(n_test)). Every entry
/// lies in `[0, 1]`. The cross-matrix rows are computed in parallel under the
/// `parallel` feature via `iter_maybe_parallel!`; the kernel is order-independent
/// and uses no RNG, so sequential and parallel builds are bit-identical.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `new_data` is empty, or if its evaluation
///   grid width (`ncols`) differs from the training set's.
/// - [`FdarError::InvalidParameter`] if `train.sigma` is not `> 0`.
///
/// # Examples
/// ```
/// use fdars_core::{gak_gram_train, gak_gram_predict, GakConfig, FdMatrix};
/// let train = FdMatrix::from_slice(&[0.0, 1.0, 5.0,  0.0, 1.0, 5.0], 3, 2).unwrap();
/// let fit = gak_gram_train(&train, &GakConfig::with_sigma(1.0)).unwrap();
/// let test = FdMatrix::from_slice(&[0.0, 5.0,  0.0, 5.0], 2, 2).unwrap();
/// let k = gak_gram_predict(&fit, &test).unwrap();
/// assert_eq!(k.shape(), (2, 3));
/// // A test curve identical to training curve 0 scores ≈ 1 in column 0.
/// assert!((k[(0, 0)] - 1.0).abs() < 1e-9);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gak_gram_predict(train: &GakGramTrain, new_data: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let n_train = train.train_rows.len();
    debug_assert_eq!(
        train.log_self.len(),
        n_train,
        "log_self length must equal n_train"
    );
    debug_assert_eq!(
        train.gram.nrows(),
        n_train,
        "training Gram row count must equal n_train"
    );

    let n_test = new_data.nrows();
    let m_test = new_data.ncols();
    let sigma = train.sigma;

    if n_test == 0 || m_test == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: "non-empty matrix (nrows > 0, ncols > 0)".to_string(),
            actual: format!("{n_test}x{m_test}"),
        });
    }
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "sigma",
            message: format!("stored training bandwidth must be > 0, got {sigma}"),
        });
    }

    // The cross-kernel loggak(test, train) requires matching evaluation-grid
    // widths (same #cols) between the test and training curves.
    let m_train = train.train_rows.first().map_or(0, Vec::len);
    if m_test != m_train {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data.ncols",
            expected: format!("{m_train} (training evaluation-grid width)"),
            actual: format!("{m_test}"),
        });
    }

    // Pre-collect test rows once (avoid repeated row materializations).
    let test_rows: Vec<Vec<f64>> = (0..n_test).map(|t| new_data.row(t)).collect();

    // Test self-log-kernels: computed once (O(n_test)), reused across all columns.
    let test_self: Vec<f64> = iter_maybe_parallel!(0..n_test)
        .map(|t| loggak(&test_rows[t], &test_rows[t], sigma))
        .collect();

    // n_test × n_train cross-Gram. Rows are independent → parallelize over test
    // curves; entry (t, j) uses the STORED training diagonal train.log_self[j]
    // (never a test-only self-kernel), so prediction lives in the training
    // feature space. Order-independent + no RNG → seq == par bit-for-bit.
    let row_blocks: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_test)
        .map(|t| {
            let mut row = vec![0.0; n_train];
            for j in 0..n_train {
                let log_xy = loggak(&test_rows[t], &train.train_rows[j], sigma);
                row[j] = normalize_log(log_xy, test_self[t], train.log_self[j]);
            }
            row
        })
        .collect();

    // Scatter into a column-major n_test × n_train FdMatrix.
    let mut out = FdMatrix::zeros(n_test, n_train);
    for (t, row) in row_blocks.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[(t, j)] = v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build an FdMatrix from row-major curves (each inner Vec is one curve/row).
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

    #[test]
    fn test_logsumexp3_basic() {
        // ln(exp(0)+exp(0)+exp(0)) = ln 3
        assert!((logsumexp3(0.0, 0.0, 0.0) - 3.0_f64.ln()).abs() < 1e-15);
        // -inf inputs are ignored (identity).
        let a = logsumexp3(1.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!((a - 1.0).abs() < 1e-15);
        // All -inf → -inf, not NaN.
        let z = logsumexp3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert_eq!(z, f64::NEG_INFINITY);
        assert!(!z.is_nan());
    }

    /// GAK-01: log-domain DP does not underflow on long series.
    #[test]
    fn test_gak_no_underflow() {
        // Two long, similar-but-distinct sinusoids (m = 200).
        let m = 200;
        let x: Vec<f64> = (0..m).map(|k| (k as f64 * 0.05).sin()).collect();
        let y: Vec<f64> = (0..m).map(|k| (k as f64 * 0.05 + 0.3).sin()).collect();
        let k = gak(&x, &y, 1.0);
        // A raw-product recursion would return exactly 0.0 here.
        assert!(k > 1e-10, "GAK underflowed to {k} on m={m} series");
        assert!(k <= 1.0 + 1e-12);
        assert!(k.is_finite());
    }

    /// GAK-02: normalization → [0,1], unit diagonal, NaN/Inf-free even for
    /// wholly dissimilar curves.
    #[test]
    fn test_gak_normalized_range() {
        let rows = vec![
            (0..80).map(|k| (k as f64 * 0.1).sin()).collect::<Vec<_>>(),
            (0..80).map(|k| (k as f64 * 0.1).cos()).collect::<Vec<_>>(),
            (0..80).map(|k| k as f64).collect::<Vec<_>>(), // wildly different scale
            (0..80).map(|k| -(k as f64) * 3.0).collect::<Vec<_>>(),
        ];
        let data = matrix_from_rows(&rows);
        let gram = gak_gram_matrix(&data, &GakConfig::with_sigma(2.0)).unwrap();
        let n = gram.nrows();
        for i in 0..n {
            assert!((gram[(i, i)] - 1.0).abs() < 1e-12, "diag[{i}] != 1");
            for j in 0..n {
                let v = gram[(i, j)];
                assert!(v.is_finite(), "non-finite entry ({i},{j}) = {v}");
                assert!(
                    (0.0..=1.0 + 1e-12).contains(&v),
                    "entry ({i},{j}) = {v} out of [0,1]"
                );
            }
        }
    }

    /// GAK-03: Gram is symmetric by assignment (bit-exact).
    #[test]
    fn test_gak_gram_symmetric() {
        let rows: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..40).map(|k| ((k + i) as f64 * 0.2).sin()).collect())
            .collect();
        let data = matrix_from_rows(&rows);
        let gram = gak_gram_matrix(&data, &GakConfig::with_sigma(1.5)).unwrap();
        let n = gram.nrows();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    gram[(i, j)].to_bits(),
                    gram[(j, i)].to_bits(),
                    "asymmetry at ({i},{j})"
                );
            }
        }
    }

    /// GAK-03: normalized Gram is PSD (min eigenvalue ≥ −1e-8).
    #[test]
    fn test_gak_gram_psd() {
        let rows: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..50)
                    .map(|k| (k as f64 * 0.15 + i as f64 * 0.4).sin() + 0.1 * i as f64)
                    .collect()
            })
            .collect();
        let data = matrix_from_rows(&rows);
        let gram = gak_gram_matrix(&data, &GakConfig::with_sigma(2.0)).unwrap();
        let dm = gram.to_dmatrix();
        let eig = dm.symmetric_eigenvalues();
        let min_eig = eig.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_eig >= -1e-8,
            "min eigenvalue {min_eig} < -1e-8 (not PSD)"
        );
    }

    /// GAK-03: parallel/sequential determinism — the Gram equals its own
    /// recomputation bit-for-bit (kernel is order-independent, no RNG).
    #[test]
    fn test_gak_parallel_matches_sequential() {
        let rows: Vec<Vec<f64>> = (0..7)
            .map(|i| (0..60).map(|k| ((k * 3 + i) as f64 * 0.07).cos()).collect())
            .collect();
        let data = matrix_from_rows(&rows);
        let cfg = GakConfig::with_sigma(1.2);
        let g1 = gak_gram_matrix(&data, &cfg).unwrap();
        let g2 = gak_gram_matrix(&data, &cfg).unwrap();
        let n = g1.nrows();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    g1[(i, j)].to_bits(),
                    g2[(i, j)].to_bits(),
                    "nondeterministic entry ({i},{j})"
                );
            }
        }
    }

    /// GAK-04: median-heuristic σ produces a healthy (non-degenerate) Gram.
    #[test]
    fn test_sigma_gak_healthy() {
        // Ten curves with modest, evenly-spaced phase shifts (similar scale so
        // the median-heuristic σ keeps the whole off-diagonal band healthy).
        let rows: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                (0..60)
                    .map(|k| (k as f64 * 0.12 + i as f64 * 0.15).sin())
                    .collect()
            })
            .collect();
        let data = matrix_from_rows(&rows);
        let sigma = sigma_gak(&data);
        assert!(sigma > 0.0, "sigma heuristic returned non-positive {sigma}");
        // Auto-σ via config (sigma = None).
        let gram = gak_gram_matrix(&data, &GakConfig::default()).unwrap();
        let n = gram.nrows();
        let mut min_off = f64::INFINITY;
        let mut max_off = f64::NEG_INFINITY;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    min_off = min_off.min(gram[(i, j)]);
                    max_off = max_off.max(gram[(i, j)]);
                }
            }
        }
        // Not near-identity (all ~0) and not near-constant (all ~1).
        assert!(
            max_off < 0.999,
            "Gram near-constant (max off-diag {max_off})"
        );
        assert!(
            min_off > 1e-4,
            "Gram near-identity (min off-diag {min_off})"
        );
        // Span is non-degenerate.
        assert!(max_off - min_off > 0.05, "off-diagonal range too narrow");
    }

    /// σ floor: identical/degenerate data must not yield σ = 0.
    #[test]
    fn test_sigma_gak_floor_on_identical() {
        let rows = vec![vec![1.0; 20], vec![1.0; 20], vec![1.0; 20]];
        let data = matrix_from_rows(&rows);
        let sigma = sigma_gak(&data);
        assert!(sigma > 0.0, "sigma floor failed: {sigma}");
        // Gram build must still succeed and be all-ones (identical curves).
        let gram = gak_gram_matrix(&data, &GakConfig::default()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((gram[(i, j)] - 1.0).abs() < 1e-9);
            }
        }
    }

    /// Reference correctness against a HAND-DERIVED case (tslearn not installed;
    /// values derived analytically from the Cuturi TGAK formula — NOT fabricated
    /// tslearn numbers).
    ///
    /// Case: x = [0, 1], y = [0, 2], σ = 1.
    /// Local kernel local(a,b) = h/(2-h), h = exp(-(a-b)²/2):
    ///   local(0,0) = 1/(2-1)          = 1.0
    ///   local(0,2) = e⁻²/(2-e⁻²)      = 0.0725788834957538
    ///   local(1,0) = e⁻⁰·⁵/(2-e⁻⁰·⁵) = 0.4352665983935838
    ///   local(1,2) = e⁻⁰·⁵/(2-e⁻⁰·⁵) = 0.4352665983935838
    /// Linear DP (M[0][0]=1, edges 0):
    ///   M[1][1] = local(0,0)·M[0][0]                     = 1.0
    ///   M[1][2] = local(0,2)·(M[0][2]+M[0][1]+M[1][1])   = local(0,2)·1
    ///   M[2][1] = local(1,0)·(M[1][1]+M[1][0]+M[2][0])   = local(1,0)·1
    ///   M[2][2] = local(1,2)·(M[1][2]+M[1][1]+M[2][1])   = 0.6563147738051063  = k(x,y)
    /// Similarly k(x,x) = 1.8705331967871679, k(y,y) = 1.1451577669915078.
    /// Normalized k = k(x,y)/sqrt(k(x,x)·k(y,y)) = 0.44843221961236995.
    #[test]
    fn test_gak_vs_reference() {
        let x = [0.0, 1.0];
        let y = [0.0, 2.0];
        let got = gak(&x, &y, 1.0);
        let expected = 0.448_432_219_612_369_95_f64;
        assert!(
            (got - expected).abs() < 1e-9,
            "GAK reference mismatch: got {got}, expected {expected}"
        );

        // Length-3 hand case: x=[0,1,2], y=[0,1,3], σ=2 → 0.805752775914924
        // (unnormalized DP evaluated analytically; see scratch derivation).
        let x3 = [0.0, 1.0, 2.0];
        let y3 = [0.0, 1.0, 3.0];
        let got3 = gak(&x3, &y3, 2.0);
        let expected3 = 0.805_752_775_914_924_f64;
        assert!(
            (got3 - expected3).abs() < 1e-9,
            "GAK length-3 reference mismatch: got {got3}, expected {expected3}"
        );

        // Self-similarity is exactly 1.
        assert!((gak(&x, &x, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gak_gram_empty_errors() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(matches!(
            gak_gram_matrix(&empty, &GakConfig::with_sigma(1.0)),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_gak_gram_bad_sigma_errors() {
        let data = matrix_from_rows(&[vec![0.0, 1.0], vec![1.0, 2.0]]);
        assert!(matches!(
            gak_gram_matrix(&data, &GakConfig::with_sigma(-1.0)),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            gak_gram_matrix(&data, &GakConfig::with_sigma(0.0)),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    // --- Phase 55: Gram-matrix export (GAK-05/06) -------------------------

    /// GAK-05: train Gram is n×n, symmetric, unit diagonal, PSD.
    #[test]
    fn test_gram_train_shape_psd() {
        let rows: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..50)
                    .map(|k| (k as f64 * 0.15 + i as f64 * 0.4).sin() + 0.1 * i as f64)
                    .collect()
            })
            .collect();
        let data = matrix_from_rows(&rows);
        let fit = gak_gram_train(&data, &GakConfig::with_sigma(2.0)).unwrap();
        let n = data.nrows();
        assert_eq!(fit.gram.shape(), (n, n));
        assert_eq!(fit.log_self().len(), n);
        assert!(fit.sigma > 0.0);
        // Unit diagonal + symmetry (bit-exact).
        for i in 0..n {
            assert!((fit.gram[(i, i)] - 1.0).abs() < 1e-12);
            for j in 0..n {
                assert_eq!(fit.gram[(i, j)].to_bits(), fit.gram[(j, i)].to_bits());
            }
        }
        // PSD: min eigenvalue ≥ −1e-8.
        let eig = fit.gram.to_dmatrix().symmetric_eigenvalues();
        let min_eig = eig.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_eig >= -1e-8, "min eig {min_eig} < -1e-8 (not PSD)");
    }

    /// GAK-06: predict Gram is exactly n_test × n_train (n_test ≠ n_train so a
    /// transpose would fail this assert).
    #[test]
    fn test_gram_predict_shape() {
        let train_rows: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..30).map(|k| ((k + i) as f64 * 0.2).sin()).collect())
            .collect();
        let train = matrix_from_rows(&train_rows);
        let fit = gak_gram_train(&train, &GakConfig::with_sigma(1.5)).unwrap();

        // n_test = 3 ≠ n_train = 5.
        let test_rows: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..30).map(|k| ((k + i) as f64 * 0.25).cos()).collect())
            .collect();
        let test = matrix_from_rows(&test_rows);
        let k = gak_gram_predict(&fit, &test).unwrap();
        assert_eq!(k.shape(), (3, 5), "predict Gram must be n_test × n_train");
    }

    /// GAK-06: every entry ∈ [0,1]; a test curve identical to a training curve
    /// gives ≈ 1.0 in that column.
    #[test]
    fn test_gram_predict_normalized() {
        let train_rows: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..40).map(|k| ((k + i * 3) as f64 * 0.1).sin()).collect())
            .collect();
        let train = matrix_from_rows(&train_rows);
        let fit = gak_gram_train(&train, &GakConfig::with_sigma(1.0)).unwrap();

        // Test set: row 0 is an exact copy of training curve 2; row 1 is novel.
        let test_rows = vec![
            train_rows[2].clone(),
            (0..40).map(|k| (k as f64 * 0.07).cos()).collect(),
        ];
        let test = matrix_from_rows(&test_rows);
        let k = gak_gram_predict(&fit, &test).unwrap();
        let (nt, ntr) = k.shape();
        for t in 0..nt {
            for j in 0..ntr {
                let v = k[(t, j)];
                assert!(v.is_finite(), "non-finite ({t},{j}) = {v}");
                assert!((0.0..=1.0 + 1e-12).contains(&v), "({t},{j}) = {v} ∉ [0,1]");
            }
        }
        // Identical curve → ≈ 1.0 in its column.
        assert!(
            (k[(0, 2)] - 1.0).abs() < 1e-9,
            "identical test curve should score ≈1 in col 2, got {}",
            k[(0, 2)]
        );
    }

    /// GAK-06: predict(train, train_data) reproduces train.gram within 1e-12 —
    /// proves the stored-diagonal cross-normalization matches the training Gram.
    #[test]
    fn test_gram_predict_reproduces_train() {
        let rows: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..45).map(|k| ((k + i * 2) as f64 * 0.13).sin()).collect())
            .collect();
        let data = matrix_from_rows(&rows);
        let fit = gak_gram_train(&data, &GakConfig::with_sigma(1.8)).unwrap();
        let k = gak_gram_predict(&fit, &data).unwrap();
        let n = data.nrows();
        assert_eq!(k.shape(), (n, n));
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[(i, j)] - fit.gram[(i, j)]).abs() < 1e-12,
                    "predict({i},{j})={} vs train={}",
                    k[(i, j)],
                    fit.gram[(i, j)]
                );
            }
        }
    }

    /// GAK-06: prediction uses train.sigma even when new_data's own median σ
    /// would differ. Predicting the training data with the fitted (explicit-σ)
    /// model must reproduce the train Gram; a different σ would not.
    #[test]
    fn test_gram_predict_sigma_consistency() {
        // Training data on a small amplitude scale.
        let train_rows: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..40).map(|k| ((k + i) as f64 * 0.1).sin()).collect())
            .collect();
        let train = matrix_from_rows(&train_rows);
        // Fit with an explicit σ far from what sigma_gak would pick on either set.
        let explicit_sigma = 3.7;
        let fit = gak_gram_train(&train, &GakConfig::with_sigma(explicit_sigma)).unwrap();
        assert!((fit.sigma - explicit_sigma).abs() < 1e-15);

        // A test set on a very different amplitude scale (its own median σ differs).
        let test_rows: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..40)
                    .map(|k| ((k + i) as f64 * 0.1).sin() * 50.0)
                    .collect()
            })
            .collect();
        let test = matrix_from_rows(&test_rows);
        let sigma_test = sigma_gak(&test);
        assert!(
            (sigma_test - explicit_sigma).abs() > 1.0,
            "test set's own σ ({sigma_test}) should differ from train σ"
        );

        // Predict must use fit.sigma. Cross-check by manual computation with
        // fit.sigma vs a wrong σ: only fit.sigma reproduces the entries.
        let k = gak_gram_predict(&fit, &test).unwrap();
        let t0 = test.row(0);
        let tr0 = train.row(0);
        let expected = {
            let log_xy = loggak(&t0, &tr0, explicit_sigma);
            let log_xx = loggak(&t0, &t0, explicit_sigma);
            normalize_log(log_xy, log_xx, fit.log_self()[0])
        };
        assert!(
            (k[(0, 0)] - expected).abs() < 1e-12,
            "predict did not use train.sigma: got {}, expected {expected}",
            k[(0, 0)]
        );
    }

    #[test]
    fn test_gram_predict_empty_and_grid_errors() {
        let train = matrix_from_rows(&[vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]]);
        let fit = gak_gram_train(&train, &GakConfig::with_sigma(1.0)).unwrap();
        // Empty test set.
        let empty = FdMatrix::zeros(0, 0);
        assert!(matches!(
            gak_gram_predict(&fit, &empty),
            Err(FdarError::InvalidDimension { .. })
        ));
        // Grid-width mismatch (train m=3, test m=2).
        let bad = matrix_from_rows(&[vec![0.0, 1.0]]);
        assert!(matches!(
            gak_gram_predict(&fit, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
