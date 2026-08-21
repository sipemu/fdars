//! Specialized functional-PCA variants.
//!
//! This module collects the specialized FPCA / cross-covariance tools that the R
//! ecosystems (`fdapace`, `refund`) expose and that fdars previously lacked:
//!
//! - [`fpca_der`] — FPCA of curve derivatives (differentiate curves, then FPCA).
//! - [`fsvd`] — functional SVD / cross-FPCA between two paired functional samples.
//! - [`cross_covariance`] — the cross-covariance surface between two samples.
//! - [`dynamical_correlation`] — a scalar dynamical/functional correlation.
//! - [`ssvd`] — a sandwich-smoother / sparse-SVD FPCA path.
//!
//! All entry points are **additive and non-breaking**: they reuse the dense FPCA
//! engine ([`crate::regression::fdata_to_pc_1d`]) and the covariance/derivative
//! helpers in [`crate::fdata`] / [`crate::covariance`] rather than introducing a
//! new subsystem, and they add **no new crate dependency**. Every public function
//! returns [`Result`] and validates its inputs up front (empty matrix, mismatched
//! argument grids, mismatched sample sizes, `ncomp` out of range) rather than
//! panicking. Outputs are numeric only — no plotting/rendering.

use crate::error::FdarError;
use crate::fdata;
use crate::helpers::{gaussian_kernel, simpsons_weights};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use nalgebra::DMatrix;

/// Result of a functional SVD ([`fsvd`]) between two paired functional samples.
///
/// `fsvd` decomposes the empirical cross-covariance surface between two samples
/// `X` (n×p) and `Y` (n×q) — observed on the same `n` subjects — into paired
/// left/right singular functions and singular values. The singular functions are
/// scaled to unit functional (L2) norm on their respective argument grids, and
/// the scores are the projections of each sample onto its singular functions.
///
/// This struct is defined alongside [`cross_covariance`] but is **populated by
/// [`fsvd`]**. It is `#[non_exhaustive]` so fields may be added without breaking
/// downstream code.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FsvdResult {
    /// Singular values of the cross-covariance decomposition (length `ncomp`,
    /// non-increasing).
    pub singular_values: Vec<f64>,
    /// Left singular functions, shape p×`ncomp` (column-major). Each column has
    /// unit functional L2 norm on `argvals_x`.
    pub left_functions: FdMatrix,
    /// Right singular functions, shape q×`ncomp` (column-major). Each column has
    /// unit functional L2 norm on `argvals_y`.
    pub right_functions: FdMatrix,
    /// Scores of sample `X` on the left singular functions, shape n×`ncomp`.
    pub left_scores: FdMatrix,
    /// Scores of sample `Y` on the right singular functions, shape n×`ncomp`.
    pub right_scores: FdMatrix,
}

/// Cross-covariance surface between two paired functional samples.
///
/// Given two samples `x` (n×p) and `y` (n×q) observed on the same `n` subjects,
/// returns the p×q sample-centered empirical cross-covariance surface
///
/// ```text
/// C[(s, t)] = (1 / (n - 1)) * Σ_i (x_i(s) - x̄(s)) * (y_i(t) - ȳ(t))
/// ```
///
/// with a Bessel (`1/(n-1)`) divisor. Each sample is centered separately by its
/// own column means (this is *not* the covariance of the concatenated data). When
/// `x` and `y` are the same sample this reduces to
/// [`crate::fdata::functional_covariance`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the two samples have different row
/// counts, if `n < 2` (Bessel correction needs ≥ 2 observations), or if either
/// sample has zero columns. Returns [`FdarError::InvalidParameter`] if `p * q`
/// would overflow `usize`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::cross_covariance;
///
/// let x = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
/// let y = FdMatrix::from_column_major(vec![2.0, 4.0, 6.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
/// let c = cross_covariance(&x, &y).unwrap();
/// assert_eq!(c.shape(), (2, 2));
/// ```
#[must_use = "cross_covariance returns the surface; ignoring it wastes the computation"]
pub fn cross_covariance(x: &FdMatrix, y: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let (nx, p) = x.shape();
    let (ny, q) = y.shape();

    if nx != ny {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{nx} rows (matching x)"),
            actual: format!("{ny} rows"),
        });
    }
    if nx < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "x",
            expected: ">= 2 rows".to_string(),
            actual: nx.to_string(),
        });
    }
    if p == 0 || q == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: if p == 0 { "x" } else { "y" },
            expected: ">= 1 column".to_string(),
            actual: format!("p = {p}, q = {q}"),
        });
    }
    // Guard against usize overflow in the p×q allocation (mirrors functional_covariance).
    p.checked_mul(q)
        .ok_or_else(|| FdarError::InvalidParameter {
            parameter: "x",
            message: format!(
                "p={p}, q={q} too large: p*q would overflow usize (max {})",
                usize::MAX
            ),
        })?;

    // Center each sample by its own column means.
    let xc = fdata::center_1d(x);
    let yc = fdata::center_1d(y);

    let denom = (nx - 1) as f64;
    let mut cov = FdMatrix::zeros(p, q);
    for s in 0..p {
        let cxs = xc.column(s);
        for t in 0..q {
            let cyt = yc.column(t);
            let val: f64 = cxs
                .iter()
                .zip(cyt.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                / denom;
            cov[(s, t)] = val;
        }
    }
    Ok(cov)
}

/// FPCA of the *derivatives* of a functional sample.
///
/// Differentiates each curve `nderiv` times (finite differences via
/// [`crate::fdata::deriv_1d`]) and then runs the dense FPCA engine
/// ([`fdata_to_pc_1d`]) on the differentiated sample. The returned [`FpcaResult`]
/// (loadings, scores, mean, singular values) therefore describes the
/// **differentiated process**. Passing `nderiv = 0` differentiates nothing and is
/// exactly equivalent to `fdata_to_pc_1d(data, ncomp, argvals)`. A `nderiv` of 1
/// is the usual convention.
///
/// # Divergence from `fdapace::FPCAder`
///
/// fdars differentiates the **curves first** and then decomposes the derivative
/// process (its eigenfunctions are eigenfunctions of the differentiated data). The
/// R `fdapace::FPCAder` instead differentiates the **eigenfunctions** of an
/// already-fitted FPCA of the original process. The two agree on the leading modes
/// for smooth data but are not identical in finite samples; this function follows
/// the differentiate-then-decompose convention.
///
/// # Errors
///
/// Returns [`FdarError`] for an empty matrix (`n == 0` or `m == 0`), an `argvals`
/// length that does not match the number of evaluation points, `ncomp < 1`, or
/// `nderiv > 0` with fewer than two columns (a numerical derivative needs ≥ 2
/// points). Inputs are validated **before** calling `deriv_1d`, which otherwise
/// silently returns a zero matrix on malformed input.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fpca_der;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let fpca = fpca_der(&data, 2, &argvals, 1).unwrap();
/// assert_eq!(fpca.rotation.shape().1, 2);
/// ```
#[must_use = "fpca_der returns the derivative FPCA result; ignoring it wastes the computation"]
pub fn fpca_der(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
    nderiv: usize,
) -> Result<FpcaResult, FdarError> {
    let (n, m) = data.shape();
    // Validate BEFORE calling deriv_1d (which silently returns zeros on bad input).
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0 columns".to_string(),
            actual: format!("m = {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }
    if nderiv > 0 && m < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: format!("need >= 2 columns for a numerical derivative, got m = {m}"),
        });
    }

    let deriv = fdata::deriv_1d(data, argvals, nderiv);
    fdata_to_pc_1d(&deriv, ncomp, argvals)
}

/// Dynamical (functional) correlation between two paired functional samples.
///
/// Implements the Dubin–Müller dynamical correlation (as in `fdapace::DynCorr`):
/// each curve is centered by its own integrated mean, then by the population mean
/// at each point, standardized to unit functional L2 norm, and the per-subject
/// integrated inner product (divided by the domain length) is averaged over the
/// sample. The result is a scalar in `[-1, 1]`: it is `1` when the two samples
/// co-vary perfectly (e.g. `x == y`), `-1` when they are exact negatives, and
/// near `0` for independent samples.
///
/// Both samples must be observed on the **same** argument grid `argvals`
/// (dynamical correlation is a same-domain pointwise construction).
///
/// # Errors
///
/// Returns [`FdarError`] if the two samples have different row counts or column
/// counts, if `argvals.len()` does not match the number of evaluation points, if
/// `n < 2`, or if the domain has zero length.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::dynamical_correlation;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.3).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let r = dynamical_correlation(&data, &data, &argvals).unwrap();
/// assert!((r - 1.0).abs() < 1e-9);
/// ```
#[must_use = "dynamical_correlation returns the scalar association; ignoring it wastes the computation"]
pub fn dynamical_correlation(
    x: &FdMatrix,
    y: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (nx, mx) = x.shape();
    let (ny, my) = y.shape();
    if nx != ny {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{nx} rows (matching x)"),
            actual: format!("{ny} rows"),
        });
    }
    if mx != my {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{mx} columns (matching x)"),
            actual: format!("{my} columns"),
        });
    }
    if argvals.len() != mx {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{mx} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if nx < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "x",
            expected: ">= 2 rows".to_string(),
            actual: nx.to_string(),
        });
    }
    let m = mx;
    let n = nx;
    let domain_length = argvals[m - 1] - argvals[0];
    if domain_length <= 0.0 || domain_length.is_nan() {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be increasing with positive domain length".to_string(),
        });
    }
    let w = simpsons_weights(argvals);

    // Step 1: per-curve integrated-mean centering.
    let mut xc1 = x.clone();
    let mut yc1 = y.clone();
    for (src, dst) in [(x, &mut xc1), (y, &mut yc1)] {
        for i in 0..n {
            let aver: f64 = (0..m).map(|j| src[(i, j)] * w[j]).sum::<f64>() / domain_length;
            for j in 0..m {
                dst[(i, j)] -= aver;
            }
        }
    }

    // Step 2: population (pointwise) centering.
    let center_pop = |mat: &mut FdMatrix| {
        for j in 0..m {
            let mean_j: f64 = (0..n).map(|i| mat[(i, j)]).sum::<f64>() / n as f64;
            for i in 0..n {
                mat[(i, j)] -= mean_j;
            }
        }
    };
    center_pop(&mut xc1);
    center_pop(&mut yc1);

    // Step 3: functional L2 standardization per curve.
    let standardize = |mat: &mut FdMatrix| {
        for i in 0..n {
            let norm_sq: f64 =
                (0..m).map(|j| mat[(i, j)].powi(2) * w[j]).sum::<f64>() / domain_length;
            let norm = norm_sq.sqrt();
            if norm < 1e-15 {
                for j in 0..m {
                    mat[(i, j)] = 0.0;
                }
            } else {
                for j in 0..m {
                    mat[(i, j)] /= norm;
                }
            }
        }
    };
    standardize(&mut xc1);
    standardize(&mut yc1);

    // Step 4: per-subject integrated inner product / domain_length, averaged.
    let total: f64 = (0..n)
        .map(|i| {
            let z: f64 = (0..m)
                .map(|j| xc1[(i, j)] * yc1[(i, j)] * w[j])
                .sum::<f64>()
                / domain_length;
            z
        })
        .sum();
    Ok(total / n as f64)
}

/// Functional SVD / cross-FPCA between two paired functional samples.
///
/// Decomposes the Simpson-weighted empirical cross-covariance surface between two
/// samples `x` (n×p) and `y` (n×q) — observed on the same `n` subjects — into
/// paired left/right singular functions and singular values. The singular
/// functions are rescaled to unit functional L2 norm on their respective grids,
/// and a deterministic sign convention is applied (the largest-magnitude element
/// of each left singular function is made positive, and the paired right function
/// is flipped together so the singular value stays non-negative). Per-sample
/// scores are the weighted projections of each sample onto its singular functions.
///
/// `ncomp` is clamped to `min(ncomp, p, q)`.
///
/// # Errors
///
/// Returns [`FdarError`] if the samples have different row counts, if `n < 2`, if
/// either `argvals` length does not match its sample's column count, or if
/// `ncomp < 1`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fsvd;
///
/// let ax: Vec<f64> = (0..6).map(|i| i as f64 / 5.0).collect();
/// let ay: Vec<f64> = (0..6).map(|i| i as f64 / 5.0).collect();
/// let x = FdMatrix::from_column_major((0..24).map(|i| (i as f64 * 0.2).sin()).collect(), 4, 6).unwrap();
/// let y = FdMatrix::from_column_major((0..24).map(|i| (i as f64 * 0.2).cos()).collect(), 4, 6).unwrap();
/// let res = fsvd(&x, &ax, &y, &ay, 2).unwrap();
/// assert_eq!(res.left_functions.shape(), (6, 2));
/// ```
#[must_use = "fsvd returns the cross-FPCA result; ignoring it wastes the computation"]
pub fn fsvd(
    x: &FdMatrix,
    argvals_x: &[f64],
    y: &FdMatrix,
    argvals_y: &[f64],
    ncomp: usize,
) -> Result<FsvdResult, FdarError> {
    let (nx, p) = x.shape();
    let (ny, q) = y.shape();
    if nx != ny {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{nx} rows (matching x)"),
            actual: format!("{ny} rows"),
        });
    }
    if nx < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "x",
            expected: ">= 2 rows".to_string(),
            actual: nx.to_string(),
        });
    }
    if argvals_x.len() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_x",
            expected: format!("{p} elements"),
            actual: format!("{} elements", argvals_x.len()),
        });
    }
    if argvals_y.len() != q {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_y",
            expected: format!("{q} elements"),
            actual: format!("{} elements", argvals_y.len()),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }
    let k = ncomp.min(p).min(q);

    // 1. Empirical cross-covariance (p×q).
    let c = cross_covariance(x, y)?;

    // 2. Integration weights and their square roots on each grid.
    let wx = simpsons_weights(argvals_x);
    let wy = simpsons_weights(argvals_y);
    let sqrt_wx: Vec<f64> = wx.iter().map(|v| v.sqrt()).collect();
    let sqrt_wy: Vec<f64> = wy.iter().map(|v| v.sqrt()).collect();

    // 3. Weighted cross-covariance Cw[(s,t)] = sqrt_wx[s]·C[(s,t)]·sqrt_wy[t].
    let mut cw = FdMatrix::zeros(p, q);
    for s in 0..p {
        for t in 0..q {
            cw[(s, t)] = sqrt_wx[s] * c[(s, t)] * sqrt_wy[t];
        }
    }

    // 4. SVD of Cw via the symmetric eigendecomposition of the smaller Gram
    //    matrix (robust for rank-deficient Cw; nalgebra's general SVD can fail to
    //    converge on near-rank-1 inputs). Decompose the smaller of Cw·Cwᵀ / Cwᵀ·Cw.
    let gram_on_right = q <= p; // eigendecompose the (min×min) Gram matrix
    let g_dim = q.min(p);
    let mut gram = vec![0.0_f64; g_dim * g_dim];
    if gram_on_right {
        // Cwᵀ·Cw is q×q: gram[(a,b)] = Σ_s cw[(s,a)]·cw[(s,b)].
        for a in 0..q {
            for b in 0..q {
                gram[a + b * q] = (0..p).map(|s| cw[(s, a)] * cw[(s, b)]).sum();
            }
        }
    } else {
        // Cw·Cwᵀ is p×p: gram[(a,b)] = Σ_t cw[(a,t)]·cw[(b,t)].
        for a in 0..p {
            for b in 0..p {
                gram[a + b * p] = (0..q).map(|t| cw[(a, t)] * cw[(b, t)]).sum();
            }
        }
    }
    let eigen = DMatrix::from_column_slice(g_dim, g_dim, &gram).symmetric_eigen();
    let mut pairs: Vec<(f64, usize)> = (0..eigen.eigenvalues.len())
        .map(|idx| (eigen.eigenvalues[idx], idx))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let pairs: Vec<(f64, usize)> = pairs.into_iter().take(k).collect();

    // 5. Recover singular values + both singular vectors, unscaling to unit L2.
    let mut singular_values = Vec::with_capacity(k);
    let mut left = FdMatrix::zeros(p, k);
    let mut right = FdMatrix::zeros(q, k);
    for (comp, &(lam, col_idx)) in pairs.iter().enumerate() {
        let sigma = lam.max(0.0).sqrt();
        singular_values.push(sigma);
        // Raw (Euclidean-orthonormal) singular vectors of Cw.
        let mut uk = vec![0.0_f64; p];
        let mut vk = vec![0.0_f64; q];
        if gram_on_right {
            // Eigenvector is the right vector v_k (length q); u_k = Cw·v_k / σ.
            for t in 0..q {
                vk[t] = eigen.eigenvectors[(t, col_idx)];
            }
            if sigma > 1e-12 {
                for s in 0..p {
                    uk[s] = (0..q).map(|t| cw[(s, t)] * vk[t]).sum::<f64>() / sigma;
                }
            }
        } else {
            // Eigenvector is the left vector u_k (length p); v_k = Cwᵀ·u_k / σ.
            for s in 0..p {
                uk[s] = eigen.eigenvectors[(s, col_idx)];
            }
            if sigma > 1e-12 {
                for t in 0..q {
                    vk[t] = (0..p).map(|s| cw[(s, t)] * uk[s]).sum::<f64>() / sigma;
                }
            }
        }
        // Unscale to unit functional L2 norm on each grid.
        for s in 0..p {
            left[(s, comp)] = if sqrt_wx[s] > 1e-15 {
                uk[s] / sqrt_wx[s]
            } else {
                uk[s]
            };
        }
        for t in 0..q {
            right[(t, comp)] = if sqrt_wy[t] > 1e-15 {
                vk[t] / sqrt_wy[t]
            } else {
                vk[t]
            };
        }
    }

    // 6. Deterministic sign convention: flip left AND right together.
    for comp in 0..k {
        let s_max = (0..p)
            .max_by(|&a, &b| {
                left[(a, comp)]
                    .abs()
                    .partial_cmp(&left[(b, comp)].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        if left[(s_max, comp)] < 0.0 {
            for s in 0..p {
                left[(s, comp)] = -left[(s, comp)];
            }
            for t in 0..q {
                right[(t, comp)] = -right[(t, comp)];
            }
        }
    }

    // 7. Per-sample scores from centered samples.
    let xc = fdata::center_1d(x);
    let yc = fdata::center_1d(y);
    let mut left_scores = FdMatrix::zeros(nx, k);
    let mut right_scores = FdMatrix::zeros(nx, k);
    for comp in 0..k {
        for i in 0..nx {
            left_scores[(i, comp)] = (0..p)
                .map(|s| xc[(i, s)] * left[(s, comp)] * wx[s])
                .sum::<f64>();
            right_scores[(i, comp)] = (0..q)
                .map(|t| yc[(i, t)] * right[(t, comp)] * wy[t])
                .sum::<f64>();
        }
    }

    Ok(FsvdResult {
        singular_values,
        left_functions: left,
        right_functions: right,
        left_scores,
        right_scores,
    })
}

/// Separable row-then-column Gaussian smoothing of an m×m covariance surface.
///
/// `pub(crate)` so the FACE sparse-covariance path (`irreg_fdata::face`) can reuse
/// the same sandwich smoother without duplicating it. Not part of the public API.
pub(crate) fn gaussian_smooth_cov(cov: &FdMatrix, argvals: &[f64], bandwidth: f64) -> FdMatrix {
    let m = argvals.len();
    // Precompute the normalized kernel weight matrix K[(a,b)].
    let mut kernel = vec![0.0_f64; m * m];
    for a in 0..m {
        let mut row_sum = 0.0;
        for b in 0..m {
            let kv = gaussian_kernel((argvals[a] - argvals[b]).abs(), bandwidth);
            kernel[a + b * m] = kv;
            row_sum += kv;
        }
        if row_sum > 1e-15 {
            for b in 0..m {
                kernel[a + b * m] /= row_sum;
            }
        }
    }
    // Row pass: tmp = K · cov.
    let mut tmp = FdMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for a in 0..m {
                acc += kernel[i + a * m] * cov[(a, j)];
            }
            tmp[(i, j)] = acc;
        }
    }
    // Column pass: out = tmp · K^T (symmetric result).
    let mut out = FdMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for b in 0..m {
                acc += tmp[(i, b)] * kernel[j + b * m];
            }
            out[(i, j)] = acc;
        }
    }
    out
}

/// Sandwich-smoother / sparse-SVD FPCA path.
///
/// An alternative to the raw thin-SVD FPCA ([`fdata_to_pc_1d`]) that estimates the
/// loadings/scores from a **smoothed** covariance surface. The empirical
/// covariance is smoothed with a separable Gaussian kernel of the given
/// `bandwidth`, then decomposed via the symmetric sandwich
/// `W^{1/2}·Cov·W^{1/2}` (the same pattern used by the PACE FPCA path). Returns an
/// [`FpcaResult`] with the same field conventions as [`fdata_to_pc_1d`], so the two
/// are directly comparable.
///
/// A `bandwidth <= 1e-10` is treated as **no smoothing** (identity smoother): the
/// empirical covariance is decomposed directly. In this dense limit the result
/// agrees with [`fdata_to_pc_1d`] within a small tolerance (~1e-4 on the singular
/// values). Note this special-case is required because the underlying
/// [`gaussian_kernel`] returns `0` at zero bandwidth rather than an identity.
///
/// # Errors
///
/// Returns [`FdarError`] for an empty matrix, an `argvals` length that does not
/// match the number of evaluation points, `ncomp < 1`, or a negative `bandwidth`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::ssvd;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.3).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let res = ssvd(&data, 2, &argvals, 0.1).unwrap();
/// assert_eq!(res.rotation.shape().0, 10);
/// ```
#[must_use = "ssvd returns the smoothed FPCA result; ignoring it wastes the computation"]
pub fn ssvd(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
    bandwidth: f64,
) -> Result<FpcaResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0 columns".to_string(),
            actual: format!("m = {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }
    if bandwidth < 0.0 || bandwidth.is_nan() {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("bandwidth must be >= 0, got {bandwidth}"),
        });
    }

    // Centered data + means for the FpcaResult.
    let (centered, means) = {
        let c = fdata::center_1d(data);
        let mut means = vec![0.0; m];
        for j in 0..m {
            means[j] = (0..n).map(|i| data[(i, j)]).sum::<f64>() / n as f64;
        }
        (c, means)
    };

    // Empirical covariance (m×m, 1/(n-1)); requires n >= 2.
    let emp = fdata::functional_covariance(data)?;

    // Smoothing: identity in the dense limit, separable Gaussian otherwise.
    let smooth_cov = if bandwidth <= 1e-10 {
        emp
    } else {
        gaussian_smooth_cov(&emp, argvals, bandwidth)
    };

    // Sandwich eigendecompose: W^{1/2}·Cov·W^{1/2} (inlined pace_fpca pattern).
    let w = simpsons_weights(argvals);
    let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let mut c_scaled = vec![0.0_f64; m * m];
    for col in 0..m {
        for row in 0..m {
            c_scaled[row + col * m] = sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col];
        }
    }
    let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();
    let mut pairs: Vec<(f64, usize)> = (0..eigen.eigenvalues.len())
        .map(|k| (eigen.eigenvalues[k], k))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let pairs: Vec<(f64, usize)> = pairs
        .into_iter()
        .filter(|&(lam, _)| lam > 0.0)
        .take(ncomp)
        .collect();
    let actual = pairs.len();

    let denom = (n - 1) as f64;
    let mut singular_values = Vec::with_capacity(actual);
    let mut rotation = FdMatrix::zeros(m, actual);
    for (comp, &(lam, col_idx)) in pairs.iter().enumerate() {
        // eigenvalue of W^{1/2}·Cov·W^{1/2} == S^2/(n-1) ⇒ S = sqrt(lam·(n-1)).
        singular_values.push((lam * denom).sqrt());
        for j in 0..m {
            let raw = eigen.eigenvectors[(j, col_idx)];
            rotation[(j, comp)] = if sqrt_w[j] > 1e-15 {
                raw / sqrt_w[j]
            } else {
                raw
            };
        }
    }
    // Sign convention: max-|abs| element positive.
    for comp in 0..actual {
        let j_max = (0..m)
            .max_by(|&a, &b| {
                rotation[(a, comp)]
                    .abs()
                    .partial_cmp(&rotation[(b, comp)].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        if rotation[(j_max, comp)] < 0.0 {
            for j in 0..m {
                rotation[(j, comp)] = -rotation[(j, comp)];
            }
        }
    }

    // Scores: weighted projection of centered data onto eigenfunctions.
    let mut scores = FdMatrix::zeros(n, actual);
    for comp in 0..actual {
        for i in 0..n {
            scores[(i, comp)] = (0..m)
                .map(|j| centered[(i, j)] * rotation[(j, comp)] * w[j])
                .sum::<f64>();
        }
    }

    Ok(FpcaResult {
        singular_values,
        rotation,
        scores,
        mean: means,
        centered,
        weights: w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Squared functional L2 norm of a column `c` under integration weights `w`.
    fn weighted_l2_sq(c: &[f64], w: &[f64]) -> f64 {
        c.iter().zip(w.iter()).map(|(&v, &wj)| v * v * wj).sum()
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- cross_covariance ------------------------------------------------

    #[test]
    fn test_cross_cov_shape() {
        // X: 4×2, Y: 4×3 -> C: 2×3
        let x = FdMatrix::from_column_major((0..8).map(|i| i as f64).collect(), 4, 2).unwrap();
        let y =
            FdMatrix::from_column_major((0..12).map(|i| (i as f64).sin()).collect(), 4, 3).unwrap();
        let c = cross_covariance(&x, &y).unwrap();
        assert_eq!(c.shape(), (2, 3));
    }

    #[test]
    fn test_cross_cov_self() {
        // cross_covariance(X, X) == functional_covariance(X) elementwise.
        let x = FdMatrix::from_column_major(vec![1.0, 2.0, 5.0, 3.0, 0.0, 4.0, 2.0, 7.0], 4, 2)
            .unwrap();
        let c = cross_covariance(&x, &x).unwrap();
        let fc = fdata::functional_covariance(&x).unwrap();
        assert_eq!(c.shape(), fc.shape());
        for s in 0..2 {
            for t in 0..2 {
                assert!(
                    approx(c[(s, t)], fc[(s, t)], 1e-12),
                    "c[{s},{t}]={} fc={}",
                    c[(s, t)],
                    fc[(s, t)]
                );
            }
        }
    }

    #[test]
    fn test_cross_cov_hand_computed() {
        // n=3, p=2, q=2 with hand-computed means.
        // X columns: [1,2,3] mean 2 ; [4,6,8] mean 6
        // Y columns: [2,4,6] mean 4 ; [10,10,13] mean 11
        let x = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0], 3, 2).unwrap();
        let y = FdMatrix::from_column_major(vec![2.0, 4.0, 6.0, 10.0, 10.0, 13.0], 3, 2).unwrap();
        let c = cross_covariance(&x, &y).unwrap();
        // xc col0 = [-1,0,1], col1 = [-2,0,2]; yc col0=[-2,0,2], col1=[-1,-1,2]
        // C[0,0] = ((-1)(-2)+0+ (1)(2))/2 = (2+2)/2 = 2
        // C[0,1] = ((-1)(-1)+0+(1)(2))/2 = (1+2)/2 = 1.5
        // C[1,0] = ((-2)(-2)+0+(2)(2))/2 = (4+4)/2 = 4
        // C[1,1] = ((-2)(-1)+0+(2)(2))/2 = (2+4)/2 = 3
        assert!(approx(c[(0, 0)], 2.0, 1e-12));
        assert!(approx(c[(0, 1)], 1.5, 1e-12));
        assert!(approx(c[(1, 0)], 4.0, 1e-12));
        assert!(approx(c[(1, 1)], 3.0, 1e-12));
    }

    #[test]
    fn test_cross_cov_errors() {
        let x = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        // mismatched sample size
        let y3 = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        assert!(cross_covariance(&x, &y3).is_err());
        // n < 2
        let x1 = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
        let y1 = FdMatrix::from_column_major(vec![3.0, 4.0], 1, 2).unwrap();
        assert!(cross_covariance(&x1, &y1).is_err());
        // zero columns
        let x0 = FdMatrix::zeros(3, 0);
        let y0 = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(cross_covariance(&x0, &y0).is_err());
    }

    // ---- fpca_der --------------------------------------------------------

    #[test]
    fn test_fpca_der_nderiv0() {
        // nderiv = 0 must equal fdata_to_pc_1d exactly.
        let data = FdMatrix::from_column_major(
            (0..40)
                .map(|i| (i as f64 * 0.13).sin() + (i as f64 * 0.02))
                .collect(),
            5,
            8,
        )
        .unwrap();
        let argvals: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
        let a = fpca_der(&data, 3, &argvals, 0).unwrap();
        let b = fdata_to_pc_1d(&data, 3, &argvals).unwrap();
        assert_eq!(a.singular_values.len(), b.singular_values.len());
        for k in 0..a.singular_values.len() {
            assert!(
                approx(a.singular_values[k], b.singular_values[k], 1e-12),
                "sv[{k}]: {} vs {}",
                a.singular_values[k],
                b.singular_values[k]
            );
        }
        let (m, nc) = a.rotation.shape();
        for j in 0..m {
            for k in 0..nc {
                assert!(approx(a.rotation[(j, k)], b.rotation[(j, k)], 1e-12));
            }
        }
    }

    #[test]
    fn test_fpca_der() {
        // Mode of variation: x_i(t) = a_i * sin(2πt), varying a_i.
        // Derivative: x_i'(t) = a_i * 2π cos(2πt). The leading derivative
        // component should reconstruct the differentiated curves well.
        let m = 40usize;
        let n = 6usize;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
        let amps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut vals = vec![0.0; n * m];
        for j in 0..m {
            for i in 0..n {
                let t = argvals[j];
                vals[i + j * n] = amps[i] * (2.0 * std::f64::consts::PI * t).sin();
            }
        }
        let data = FdMatrix::from_column_major(vals, n, m).unwrap();
        let res = fpca_der(&data, 1, &argvals, 1).unwrap();

        // Reconstruct the centered differentiated curves from the leading PC:
        // deriv_centered ≈ scores[:,0] outer rotation[:,0].
        let deriv = fdata::deriv_1d(&data, &argvals, 1);
        // center derivative columns
        let dc = fdata::center_1d(&deriv);
        let mut sse = 0.0;
        let mut sst = 0.0;
        for i in 0..n {
            for j in 0..m {
                let recon = res.scores[(i, 0)] * res.rotation[(j, 0)];
                let actual = dc[(i, j)];
                sse += (actual - recon).powi(2);
                sst += actual.powi(2);
            }
        }
        // Single mode of variation -> leading component explains essentially all.
        assert!(
            sse / sst < 1e-6,
            "relative reconstruction error {}",
            sse / sst
        );
    }

    #[test]
    fn test_fpca_der_errors() {
        let argvals: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
        let data = FdMatrix::from_column_major((0..40).map(|i| i as f64).collect(), 5, 8).unwrap();
        // empty matrix
        assert!(fpca_der(&FdMatrix::zeros(0, 0), 1, &[], 1).is_err());
        // argvals length mismatch
        assert!(fpca_der(&data, 1, &argvals[..7], 1).is_err());
        // ncomp < 1
        assert!(fpca_der(&data, 0, &argvals, 1).is_err());
        // nderiv > 0 with m < 2
        let thin = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(fpca_der(&thin, 1, &[0.0], 1).is_err());
    }

    // ---- dynamical_correlation -------------------------------------------

    fn sine_sample(n: usize, m: usize, seedish: f64) -> (FdMatrix, Vec<f64>) {
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
        let mut vals = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let t = argvals[j];
                vals[i + j * n] =
                    ((i as f64 + 1.0) * seedish * t).sin() + 0.3 * (i as f64 + 1.0) * t;
            }
        }
        (FdMatrix::from_column_major(vals, n, m).unwrap(), argvals)
    }

    #[test]
    fn test_dyncorr_identical() {
        let (data, argvals) = sine_sample(6, 20, 6.0);
        let r = dynamical_correlation(&data, &data, &argvals).unwrap();
        assert!(approx(r, 1.0, 1e-10), "dyncorr(x,x) = {r}");
    }

    #[test]
    fn test_dyncorr_negated() {
        let (data, argvals) = sine_sample(6, 20, 5.0);
        let (n, m) = data.shape();
        let mut neg = data.clone();
        for i in 0..n {
            for j in 0..m {
                neg[(i, j)] = -data[(i, j)];
            }
        }
        let r = dynamical_correlation(&data, &neg, &argvals).unwrap();
        assert!(approx(r, -1.0, 1e-10), "dyncorr(x,-x) = {r}");
    }

    #[test]
    fn test_dyncorr_range() {
        let (x, argvals) = sine_sample(7, 25, 4.0);
        let (y, _) = sine_sample(7, 25, 9.0);
        let r = dynamical_correlation(&x, &y, &argvals).unwrap();
        assert!(
            (-1.0 - 1e-9..=1.0 + 1e-9).contains(&r),
            "dyncorr out of range: {r}"
        );
    }

    #[test]
    fn test_dyncorr_errors() {
        let (x, argvals) = sine_sample(5, 10, 3.0);
        // mismatched sample size
        let (y6, _) = sine_sample(6, 10, 3.0);
        assert!(dynamical_correlation(&x, &y6, &argvals).is_err());
        // mismatched columns
        let (yc, _) = sine_sample(5, 12, 3.0);
        assert!(dynamical_correlation(&x, &yc, &argvals).is_err());
        // argvals length mismatch
        assert!(dynamical_correlation(&x, &x, &argvals[..9]).is_err());
        // n < 2
        let (x1, a1) = sine_sample(1, 10, 3.0);
        assert!(dynamical_correlation(&x1, &x1, &a1).is_err());
    }

    // ---- fsvd ------------------------------------------------------------

    #[test]
    fn test_fsvd_unit_norm() {
        let (x, ax) = sine_sample(6, 15, 4.0);
        let (y, ay) = sine_sample(6, 12, 7.0);
        let res = fsvd(&x, &ax, &y, &ay, 2).unwrap();
        let wx = simpsons_weights(&ax);
        let wy = simpsons_weights(&ay);
        for comp in 0..res.singular_values.len() {
            let ln: f64 = (0..15)
                .map(|s| res.left_functions[(s, comp)].powi(2) * wx[s])
                .sum();
            let rn: f64 = (0..12)
                .map(|t| res.right_functions[(t, comp)].powi(2) * wy[t])
                .sum();
            assert!(approx(ln, 1.0, 1e-8), "left norm[{comp}]={ln}");
            assert!(approx(rn, 1.0, 1e-8), "right norm[{comp}]={rn}");
        }
    }

    #[test]
    fn test_fsvd_rank1() {
        // X[i,j] = a_i · sin(argvals_x[j]); Y[i,j] = a_i · cos(argvals_y[j]).
        let n = 8usize;
        let px = 20usize;
        let qy = 18usize;
        let ax: Vec<f64> = (0..px).map(|i| i as f64 / (px as f64 - 1.0)).collect();
        let ay: Vec<f64> = (0..qy).map(|i| i as f64 / (qy as f64 - 1.0)).collect();
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 * 0.4).collect();
        let mut xv = vec![0.0; n * px];
        let mut yv = vec![0.0; n * qy];
        for i in 0..n {
            for j in 0..px {
                xv[i + j * n] = amps[i] * (std::f64::consts::PI * ax[j]).sin();
            }
            for j in 0..qy {
                yv[i + j * n] = amps[i] * (std::f64::consts::PI * ay[j]).cos();
            }
        }
        let x = FdMatrix::from_column_major(xv, n, px).unwrap();
        let y = FdMatrix::from_column_major(yv, n, qy).unwrap();
        let res = fsvd(&x, &ax, &y, &ay, 3).unwrap();

        // Rank-1 dominance: first singular value dwarfs the rest.
        assert!(
            res.singular_values[0] > 1e6 * res.singular_values[1].max(1e-14),
            "not rank-1 dominant: {:?}",
            res.singular_values
        );

        // The unit-L2 singular functions reconstruct the (unweighted) empirical
        // cross-covariance directly: C[s,t] = Σ_k σ_k · left_k[s] · right_k[t]
        // (the √-weights cancel between the weighted SVD and the unscaling).
        let c = cross_covariance(&x, &y).unwrap();
        let mut sse = 0.0;
        let mut sst = 0.0;
        for s in 0..px {
            for t in 0..qy {
                let mut recon = 0.0;
                for k in 0..res.singular_values.len() {
                    recon += res.singular_values[k]
                        * res.left_functions[(s, k)]
                        * res.right_functions[(t, k)];
                }
                sse += (c[(s, t)] - recon).powi(2);
                sst += c[(s, t)].powi(2);
            }
        }
        assert!(
            sse / sst < 1e-6,
            "cross-cov reconstruction rel err {}",
            sse / sst
        );
    }

    #[test]
    fn test_fsvd_wide_left_gram_branch() {
        // p < q exercises the Cw·Cwᵀ (gram-on-left) branch. Same rank-1 structure
        // with the grids swapped: X has fewer points than Y.
        let n = 8usize;
        let px = 14usize;
        let qy = 22usize;
        let ax: Vec<f64> = (0..px).map(|i| i as f64 / (px as f64 - 1.0)).collect();
        let ay: Vec<f64> = (0..qy).map(|i| i as f64 / (qy as f64 - 1.0)).collect();
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 * 0.4).collect();
        let mut xv = vec![0.0; n * px];
        let mut yv = vec![0.0; n * qy];
        for i in 0..n {
            for j in 0..px {
                xv[i + j * n] = amps[i] * (std::f64::consts::PI * ax[j]).sin();
            }
            for j in 0..qy {
                yv[i + j * n] = amps[i] * (std::f64::consts::PI * ay[j]).cos();
            }
        }
        let x = FdMatrix::from_column_major(xv, n, px).unwrap();
        let y = FdMatrix::from_column_major(yv, n, qy).unwrap();
        let res = fsvd(&x, &ax, &y, &ay, 2).unwrap();
        assert_eq!(res.left_functions.shape(), (px, 2));
        assert_eq!(res.right_functions.shape(), (qy, 2));
        // Unit-L2 norm on both grids (proves the gram-on-left unscaling is correct).
        let wx = simpsons_weights(&ax);
        let wy = simpsons_weights(&ay);
        assert!(approx(
            weighted_l2_sq(res.left_functions.column(0), &wx),
            1.0,
            1e-8
        ));
        assert!(approx(
            weighted_l2_sq(res.right_functions.column(0), &wy),
            1.0,
            1e-8
        ));
        // Rank-1 cross-covariance reconstruction.
        let c = cross_covariance(&x, &y).unwrap();
        let mut sse = 0.0;
        let mut sst = 0.0;
        for s in 0..px {
            for t in 0..qy {
                let mut recon = 0.0;
                for k in 0..res.singular_values.len() {
                    recon += res.singular_values[k]
                        * res.left_functions[(s, k)]
                        * res.right_functions[(t, k)];
                }
                sse += (c[(s, t)] - recon).powi(2);
                sst += c[(s, t)].powi(2);
            }
        }
        assert!(
            sse / sst < 1e-6,
            "wide reconstruction rel err {}",
            sse / sst
        );
    }

    #[test]
    fn test_fsvd_errors() {
        let (x, ax) = sine_sample(5, 10, 3.0);
        let (y, ay) = sine_sample(5, 8, 4.0);
        // mismatched sample size
        let (y6, ay6) = sine_sample(6, 8, 4.0);
        assert!(fsvd(&x, &ax, &y6, &ay6, 1).is_err());
        // ncomp < 1
        assert!(fsvd(&x, &ax, &y, &ay, 0).is_err());
        // argvals mismatch
        assert!(fsvd(&x, &ax[..9], &y, &ay, 1).is_err());
    }

    // ---- ssvd ------------------------------------------------------------

    #[test]
    fn test_ssvd_dense_limit() {
        let (data, argvals) = sine_sample(8, 20, 5.0);
        let a = ssvd(&data, 3, &argvals, 1e-12).unwrap();
        let b = fdata_to_pc_1d(&data, 3, &argvals).unwrap();
        assert_eq!(a.singular_values.len(), b.singular_values.len());
        for k in 0..a.singular_values.len() {
            let rel = (a.singular_values[k] - b.singular_values[k]).abs()
                / b.singular_values[k].abs().max(1e-12);
            assert!(
                rel < 1e-4,
                "sv[{k}] dense-limit mismatch: {} vs {} (rel {})",
                a.singular_values[k],
                b.singular_values[k],
                rel
            );
        }
    }

    #[test]
    fn test_ssvd_orthonormality() {
        let (data, argvals) = sine_sample(8, 20, 5.0);
        let res = ssvd(&data, 3, &argvals, 0.05).unwrap();
        let w = simpsons_weights(&argvals);
        let nc = res.rotation.shape().1;
        for a in 0..nc {
            for b in 0..nc {
                let ip: f64 = (0..20)
                    .map(|j| res.rotation[(j, a)] * res.rotation[(j, b)] * w[j])
                    .sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(approx(ip, expected, 1e-6), "⟨φ{a},φ{b}⟩={ip}");
            }
        }
    }

    #[test]
    fn test_ssvd_errors() {
        let (data, argvals) = sine_sample(5, 10, 3.0);
        // ncomp < 1
        assert!(ssvd(&data, 0, &argvals, 0.1).is_err());
        // empty matrix
        assert!(ssvd(&FdMatrix::zeros(0, 0), 1, &[], 0.1).is_err());
        // argvals mismatch
        assert!(ssvd(&data, 1, &argvals[..9], 0.1).is_err());
        // negative bandwidth
        assert!(ssvd(&data, 1, &argvals, -0.1).is_err());
    }

    // ---- reexport smoke (all five variants) ------------------------------

    #[test]
    fn smoke_reexports() {
        // Crate-root reachability + runs on tiny valid inputs.
        let (x, argvals) = sine_sample(4, 6, 4.0);
        let (y, ay) = sine_sample(4, 6, 6.0);
        let _c = crate::cross_covariance(&x, &y).unwrap();
        let _f = crate::fpca_der(&x, 1, &argvals, 1).unwrap();
        let _d = crate::dynamical_correlation(&x, &y, &argvals).unwrap();
        let _s = crate::fsvd(&x, &argvals, &y, &ay, 1).unwrap();
        let _v = crate::ssvd(&x, 1, &argvals, 0.1).unwrap();
        // FsvdResult reachable at the crate root.
        let _first: &crate::FsvdResult = &_s;
        // Touch the L2 helper (unit-norm check building block).
        let w = simpsons_weights(&argvals);
        let _ = weighted_l2_sq(x.column(0), &w);
    }
}
