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
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::simpsons_weights;

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

    // ---- reexport smoke (extended in Plan 02) ----------------------------

    #[test]
    fn smoke_reexports() {
        // Crate-root reachability (compile-level).
        let x = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let _c = crate::cross_covariance(&x, &x).unwrap();
        let argvals: Vec<f64> = (0..2).map(|i| i as f64).collect();
        let _f = crate::fpca_der(&x, 1, &argvals, 0).unwrap();
        // Touch the L2 helper so it is exercised for later fsvd unit-norm checks.
        let w = simpsons_weights(&argvals);
        let _ = weighted_l2_sq(x.column(0), &w);
    }
}
