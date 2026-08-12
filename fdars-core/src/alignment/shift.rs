//! Least-squares shift (rigid horizontal) registration.
//!
//! Provides [`least_squares_shift_registration`] which aligns each curve in a
//! functional data set to the cross-sectional sample mean by estimating a
//! per-curve rigid horizontal shift δᵢ that minimises the Simpson-weighted L2
//! distance ‖fᵢ(t − δᵢ) − mean(t)‖².
//!
//! The shift δᵢ is found by golden-section search over the closed interval
//! `[−max_shift, +max_shift]`. The objective is assumed to be unimodal in δ
//! for typical functional data within the default bracket — see
//! [`least_squares_shift_registration`] for caveats.

use crate::error::FdarError;
use crate::helpers::{linear_interp, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default fraction of the domain range to use as `max_shift`.
///
/// Recommended caller value: `DEFAULT_MAX_SHIFT_FRACTION * (argvals.last() - argvals.first())`.
///
/// Example: for `argvals` on `[0.0, 1.0]`, use `max_shift = 0.25`.
pub const DEFAULT_MAX_SHIFT_FRACTION: f64 = 0.25;

/// Convergence tolerance for the golden-section search (in domain units).
const GS_TOL: f64 = 1e-6;

/// Maximum iterations for the golden-section search.
const GS_MAX_ITER: usize = 100;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of least-squares shift (rigid horizontal) registration.
///
/// Each curve fᵢ is shifted by `shifts[i]` and re-evaluated at the original
/// `argvals` grid via linear interpolation with boundary clamping.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShiftRegistrationResult {
    /// Registered (shifted) functional data matrix (n × m).
    ///
    /// `registered_data[(i, j)]` equals `linear_interp(argvals, row_i, argvals[j] − shifts[i])`.
    pub registered_data: FdMatrix,

    /// Per-curve horizontal shifts δᵢ (length n).
    ///
    /// Positive δᵢ shifts the curve to the right (later in time);
    /// negative δᵢ shifts the curve to the left.
    pub shifts: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Golden-section search for the minimum of a unimodal function on `[lo, hi]`.
///
/// Returns the midpoint of the final bracket after convergence (width < `tol`)
/// or after `max_iter` iterations.
fn golden_section_search<F>(f: F, mut lo: f64, mut hi: f64, tol: f64, max_iter: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    const PHI: f64 = 1.618_033_988_749_895;
    let mut x1 = hi - (hi - lo) / PHI;
    let mut x2 = lo + (hi - lo) / PHI;
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..max_iter {
        if (hi - lo) < tol {
            break;
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - (hi - lo) / PHI;
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + (hi - lo) / PHI;
            f2 = f(x2);
        }
    }
    (lo + hi) / 2.0
}

/// Simpson-weighted L2 distance between the shifted curve fᵢ(t − δ) and `mean`.
///
/// For each grid point j, evaluates `linear_interp(argvals, row, argvals[j] − delta)`,
/// which applies boundary clamping for out-of-domain arguments.
fn l2_shift_objective(
    row: &[f64],
    argvals: &[f64],
    mean: &[f64],
    weights: &[f64],
    delta: f64,
) -> f64 {
    argvals
        .iter()
        .zip(mean.iter())
        .zip(weights.iter())
        .map(|((&t, &m_j), &w)| {
            let fi_shifted = linear_interp(argvals, row, t - delta);
            let diff = fi_shifted - m_j;
            diff * diff * w
        })
        .sum::<f64>()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Register a set of functional curves by a per-curve rigid horizontal shift.
///
/// For each curve fᵢ in `data`, finds the shift δᵢ ∈ `[−max_shift, +max_shift]`
/// that minimises the Simpson-weighted L2 distance to the cross-sectional sample mean:
///
/// ```text
/// δᵢ = argmin_δ ‖fᵢ(· − δ) − mean(·)‖²_L2
/// ```
///
/// The minimisation is performed by golden-section search, which assumes the
/// objective is **unimodal** in δ. This holds for typical functional data within
/// the default bracket but may not hold for multi-modal or highly oscillatory
/// curves shifted by more than half a period.
///
/// Shifted curves are re-evaluated at the original `argvals` grid via linear
/// interpolation. Points shifted outside the domain are clamped to the boundary
/// value (Boundary extrapolation policy, inherited from [`crate::helpers::linear_interp`]).
///
/// # Arguments
///
/// * `data` — Functional data matrix (n × m, column-major).
/// * `argvals` — Evaluation points, length m. Must be sorted in ascending order.
/// * `max_shift` — Half-width of the shift search interval (must be > 0).
///   Recommended value: `0.25 * (argvals.last() - argvals.first())`, i.e. `0.25 * domain_range`.
///
/// # Returns
///
/// [`ShiftRegistrationResult`] containing the registered curves and per-curve shifts.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `data` is empty (`n = 0` or `m = 0`).
/// * [`FdarError::InvalidDimension`] if `argvals.len() != m`.
/// * [`FdarError::InvalidParameter`] if `argvals.len() < 2`.
/// * [`FdarError::InvalidParameter`] if `max_shift <= 0.0`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::alignment::least_squares_shift_registration;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..60).map(|i| ((i as f64 * 0.1).sin())).collect(),
///     3, 20,
/// ).unwrap();
/// let max_shift = fdars_core::alignment::DEFAULT_MAX_SHIFT_FRACTION * (argvals[19] - argvals[0]);
/// let result = least_squares_shift_registration(&data, &argvals, max_shift).unwrap();
/// assert_eq!(result.registered_data.shape(), (3, 20));
/// assert_eq!(result.shifts.len(), 3);
/// ```
pub fn least_squares_shift_registration(
    data: &FdMatrix,
    argvals: &[f64],
    max_shift: f64,
) -> Result<ShiftRegistrationResult, FdarError> {
    let (n, m) = data.shape();

    // V5 input validation — all checks before any computation
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{}", n, m),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must have at least 2 evaluation points".to_string(),
        });
    }
    if max_shift <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_shift",
            message: format!("must be positive, got {max_shift}"),
        });
    }

    // Pre-compute shared mean and integration weights
    let weights = simpsons_weights(argvals);
    let mean = crate::fdata::mean_1d(data);

    // Parallel per-curve shift estimation + re-evaluation
    // Collect into Vec first (parallel order arbitrary), then assemble sequentially (#3099)
    let results: Vec<(f64, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let row = data.row(i);
            let delta = golden_section_search(
                |d| l2_shift_objective(&row, argvals, &mean, &weights, d),
                -max_shift,
                max_shift,
                GS_TOL,
                GS_MAX_ITER,
            );
            let shifted: Vec<f64> = argvals
                .iter()
                .map(|&t| linear_interp(argvals, &row, t - delta))
                .collect();
            (delta, shifted)
        })
        .collect();

    // Sequential assembly into FdMatrix (maintains row order regardless of parallel dispatch)
    let mut registered_data = FdMatrix::zeros(n, m);
    let mut shifts = Vec::with_capacity(n);

    for (i, (delta, shifted_row)) in results.into_iter().enumerate() {
        for j in 0..m {
            registered_data[(i, j)] = shifted_row[j];
        }
        shifts.push(delta);
    }

    Ok(ShiftRegistrationResult {
        registered_data,
        shifts,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    /// Uniform grid on [0, 1] with `n` points.
    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    /// Gaussian bump at position `mu` with standard deviation `sigma`.
    fn gaussian_bump(argvals: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        argvals
            .iter()
            .map(|&t| (-(t - mu).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect()
    }

    /// Build an n×m FdMatrix of Gaussian bumps, each shifted by `i * delta`.
    fn make_shifted_bumps(n: usize, m: usize, delta: f64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let shift = i as f64 * delta;
            let row = gaussian_bump(&argvals, 0.5 + shift, 0.05);
            for j in 0..m {
                data[(i, j)] = row[j];
            }
        }
        (data, argvals)
    }

    // FEAT-06-A: already-aligned curves → estimated shifts ≈ 0
    #[test]
    fn test_shift_already_aligned() {
        let m = 51;
        let argvals = uniform_grid(m);
        // All curves are the same Gaussian bump centred at 0.5 — already aligned
        let n = 4;
        let mut data = FdMatrix::zeros(n, m);
        let bump = gaussian_bump(&argvals, 0.5, 0.08);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = bump[j];
            }
        }
        let max_shift = 0.25 * (argvals[m - 1] - argvals[0]);
        let result = least_squares_shift_registration(&data, &argvals, max_shift).unwrap();
        for (i, &delta) in result.shifts.iter().enumerate() {
            assert!(
                delta.abs() < 1e-3,
                "curve {i}: expected shift ≈ 0, got {delta}"
            );
        }
    }

    // FEAT-06-B: injected offsets recovered within tolerance
    #[test]
    fn test_shift_recovers_injected_offset() {
        // 3 curves: centred at 0.5, 0.4, 0.6.
        // Mean of bumps ≈ Gaussian at 0.5 (average centre).
        // Convention: registered(t) = original(t - δ).
        // To bring peak at mu=0.4 to t=0.5 we need δ=+0.1: original(0.5-0.1)=original(0.4)=peak.
        // To bring peak at mu=0.6 to t=0.5 we need δ=-0.1: original(0.5-(-0.1))=original(0.6)=peak.
        let m = 101;
        let argvals = uniform_grid(m);
        let sigma = 0.05_f64;
        let centres = [0.5_f64, 0.4, 0.6];
        let n = centres.len();
        let mut data = FdMatrix::zeros(n, m);
        for (i, &mu) in centres.iter().enumerate() {
            let row = gaussian_bump(&argvals, mu, sigma);
            for j in 0..m {
                data[(i, j)] = row[j];
            }
        }
        // Expected shifts: δᵢ = mean_centre - mu_i = 0.5 - mu_i
        let true_shifts = [0.0_f64, 0.1, -0.1];
        let max_shift = 0.25 * (argvals[m - 1] - argvals[0]);
        let result = least_squares_shift_registration(&data, &argvals, max_shift).unwrap();

        // Allow generous tolerance: golden-section tol 1e-6 but some boundary effects.
        for (i, (&recovered, &expected)) in result.shifts.iter().zip(true_shifts.iter()).enumerate()
        {
            assert!(
                (recovered - expected).abs() < 0.05,
                "curve {i}: expected shift ≈ {expected}, got {recovered}"
            );
        }
    }

    // FEAT-06-C: registered curves are the correct shifted re-evaluation
    #[test]
    fn test_shift_registration_curve_values() {
        // Small deterministic case: 2 curves on a 5-point grid.
        // After registration, registered_data[(i,j)] must equal
        // linear_interp(argvals, row_i, argvals[j] - shifts[i]) exactly.
        let m = 5;
        let argvals = uniform_grid(m);
        let n = 2;
        // Curve 0: Gaussian bump at 0.3; Curve 1: Gaussian bump at 0.7
        let mut data = FdMatrix::zeros(n, m);
        for (i, &mu) in [0.3_f64, 0.7].iter().enumerate() {
            let row = gaussian_bump(&argvals, mu, 0.15);
            for j in 0..m {
                data[(i, j)] = row[j];
            }
        }
        let max_shift = 0.25;
        let result = least_squares_shift_registration(&data, &argvals, max_shift).unwrap();

        // Spot-check all (i, j) positions: registered value must equal the
        // shifted linear-interpolation evaluation, proving the matrix is
        // assembled from the correct re-evaluation calls.
        for i in 0..n {
            let row = data.row(i);
            let delta = result.shifts[i];
            for j in 0..m {
                let expected = linear_interp(&argvals, &row, argvals[j] - delta);
                let actual = result.registered_data[(i, j)];
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "registered_data[({i},{j})] = {actual}, expected {expected} (shift={delta})"
                );
            }
        }
    }

    // FEAT-06-D: empty data returns Err(InvalidDimension)
    #[test]
    fn test_shift_registration_empty_data() {
        let argvals = uniform_grid(5);
        // n = 0
        let data_n0 = FdMatrix::zeros(0, 5);
        let result = least_squares_shift_registration(&data_n0, &argvals, 0.1);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected Err(InvalidDimension) for n=0, got {result:?}"
        );

        // m = 0
        let data_m0 = FdMatrix::zeros(3, 0);
        let result_m0 = least_squares_shift_registration(&data_m0, &[], 0.1);
        assert!(
            matches!(result_m0, Err(FdarError::InvalidDimension { .. })),
            "expected Err(InvalidDimension) for m=0, got {result_m0:?}"
        );
    }

    // FEAT-06-E: argvals length mismatch returns Err(InvalidDimension)
    #[test]
    fn test_shift_registration_argvals_mismatch() {
        let m = 5;
        let argvals = uniform_grid(m);
        let data = FdMatrix::zeros(2, m);
        // Pass argvals with wrong length (m+1 instead of m)
        let wrong_argvals = uniform_grid(m + 1);
        let result = least_squares_shift_registration(&data, &wrong_argvals, 0.1);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected Err(InvalidDimension) for argvals length mismatch, got {result:?}"
        );
        // Also test argvals shorter than m
        let short_argvals = uniform_grid(m - 1);
        let result2 = least_squares_shift_registration(&data, &short_argvals, 0.1);
        assert!(
            matches!(result2, Err(FdarError::InvalidDimension { .. })),
            "expected Err(InvalidDimension) for argvals too short, got {result2:?}"
        );
    }
}
