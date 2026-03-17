//! Phase (warping) box plots for functional data.
//!
//! Constructs functional box plots on warping functions to visualize
//! phase variation. The median warp is the deepest warp by modified band
//! depth, the central region contains the 50% deepest warps, and
//! outliers are identified beyond the whisker fences.
//!
//! # References
//!
//! - Lopez-Pintado, S. & Romo, J. (2009). On the concept of depth for
//!   functional data. *Journal of the American Statistical Association*,
//!   104(486), 718-734.

use crate::depth::modified_band_1d;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Result of computing a phase (warping) box plot.
///
/// Contains the median warp, the 50% central envelope, whisker
/// fences, depth values, and outlier indices.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PhaseBoxplot {
    /// Median warping function (deepest by modified band depth).
    pub median: Vec<f64>,
    /// Index of the median warp in the original matrix.
    pub median_index: usize,
    /// Lower boundary of the 50% central region (pointwise min of central warps).
    pub central_lower: Vec<f64>,
    /// Upper boundary of the 50% central region (pointwise max of central warps).
    pub central_upper: Vec<f64>,
    /// Lower whisker fence (trimmed to non-outlier envelope).
    pub whisker_lower: Vec<f64>,
    /// Upper whisker fence (trimmed to non-outlier envelope).
    pub whisker_upper: Vec<f64>,
    /// Modified band depth of each warp.
    pub depths: Vec<f64>,
    /// Indices of outlier warps (beyond whisker fences).
    pub outlier_indices: Vec<usize>,
    /// Whisker factor used (analogous to 1.5 in classical box plots).
    pub factor: f64,
}

/// Compute a phase box plot from a set of warping functions.
///
/// Ranks warps by modified band depth, identifies the median (deepest warp),
/// builds the 50% central envelope, and flags outliers beyond the whisker
/// fences. Whiskers are trimmed to the actual envelope of non-outlier warps.
///
/// # Arguments
/// * `gammas` — Warping functions (n x m, one warp per row)
/// * `argvals` — Evaluation grid (length m)
/// * `factor` — Whisker extent as a multiple of the IQR envelope (e.g. 1.5)
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if n < 3 or column count differs
/// from `argvals`. Returns `FdarError::InvalidParameter` if `factor` is not
/// positive.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::alignment::phase_boxplot;
///
/// let m = 20;
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// // 5 slightly shifted warps
/// let mut data = vec![0.0; 5 * m];
/// for i in 0..5 {
///     let shift = 0.02 * (i as f64 - 2.0);
///     for j in 0..m {
///         let t = argvals[j];
///         data[j * 5 + i] = (t + shift * t * (1.0 - t)).clamp(0.0, 1.0);
///     }
/// }
/// let gammas = FdMatrix::from_column_major(data, 5, m).unwrap();
/// let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();
/// assert_eq!(bp.median.len(), m);
/// assert!(bp.depths.iter().all(|&d| d >= 0.0));
/// ```
#[must_use = "box plot result should not be discarded"]
pub fn phase_boxplot(
    gammas: &FdMatrix,
    argvals: &[f64],
    factor: f64,
) -> Result<PhaseBoxplot, FdarError> {
    let n = gammas.nrows();
    let m = gammas.ncols();

    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "gammas",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if factor <= 0.0 || !factor.is_finite() {
        return Err(FdarError::InvalidParameter {
            parameter: "factor",
            message: format!("must be positive and finite, got {factor}"),
        });
    }

    // Step 1: Compute modified band depths.
    let depths = modified_band_1d(gammas, gammas);

    // Step 2: Sort indices by depth (descending).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        depths[b]
            .partial_cmp(&depths[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let median_index = order[0];
    let median = gammas.row(median_index);

    // Step 3: Top ceil(n/2) warps form the central region.
    let n_central = n.div_ceil(2);
    let central_indices = &order[..n_central];

    // Step 4: Central envelope (pointwise min/max).
    let mut central_lower = vec![f64::INFINITY; m];
    let mut central_upper = vec![f64::NEG_INFINITY; m];
    for &idx in central_indices {
        for j in 0..m {
            let v = gammas[(idx, j)];
            if v < central_lower[j] {
                central_lower[j] = v;
            }
            if v > central_upper[j] {
                central_upper[j] = v;
            }
        }
    }

    // Step 5: IQR envelope and whisker fences.
    let domain_lo = argvals[0];
    let domain_hi = argvals[m - 1];
    let mut fence_lower = vec![0.0; m];
    let mut fence_upper = vec![0.0; m];
    for j in 0..m {
        let iqr = central_upper[j] - central_lower[j];
        fence_lower[j] = (central_lower[j] - factor * iqr).max(domain_lo);
        fence_upper[j] = (central_upper[j] + factor * iqr).min(domain_hi);
    }

    // Step 6: Identify outliers (any warp exceeding fences at any grid point).
    let mut outlier_indices = Vec::new();
    for i in 0..n {
        let mut is_outlier = false;
        for j in 0..m {
            let v = gammas[(i, j)];
            if v < fence_lower[j] || v > fence_upper[j] {
                is_outlier = true;
                break;
            }
        }
        if is_outlier {
            outlier_indices.push(i);
        }
    }

    // Step 7: Trim whiskers to actual non-outlier envelope.
    let mut whisker_lower = vec![f64::INFINITY; m];
    let mut whisker_upper = vec![f64::NEG_INFINITY; m];
    for i in 0..n {
        if outlier_indices.contains(&i) {
            continue;
        }
        for j in 0..m {
            let v = gammas[(i, j)];
            if v < whisker_lower[j] {
                whisker_lower[j] = v;
            }
            if v > whisker_upper[j] {
                whisker_upper[j] = v;
            }
        }
    }

    // If all warps are outliers (unlikely but defensive), fall back to fences.
    if outlier_indices.len() == n {
        whisker_lower = fence_lower;
        whisker_upper = fence_upper;
    }

    Ok(PhaseBoxplot {
        median,
        median_index,
        central_lower,
        central_upper,
        whisker_lower,
        whisker_upper,
        depths,
        outlier_indices,
        factor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    /// Build n identity-like warps with small perturbations.
    fn make_warps(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let shift = 0.03 * (i as f64 - (n as f64 - 1.0) / 2.0);
            for j in 0..m {
                let t = argvals[j];
                // Smooth perturbation preserving [0,1] boundary
                let v = t + shift * t * (1.0 - t);
                data[j * n + i] = v.clamp(0.0, 1.0);
            }
        }
        let gammas = FdMatrix::from_column_major(data, n, m).unwrap();
        (gammas, argvals)
    }

    #[test]
    fn too_few_curves_rejected() {
        let argvals = uniform_grid(10);
        let gammas = FdMatrix::zeros(2, 10);
        let err = phase_boxplot(&gammas, &argvals, 1.5).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn argvals_length_mismatch_rejected() {
        let gammas = FdMatrix::zeros(5, 10);
        let argvals = uniform_grid(8);
        let err = phase_boxplot(&gammas, &argvals, 1.5).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn negative_factor_rejected() {
        let (gammas, argvals) = make_warps(5, 20);
        let err = phase_boxplot(&gammas, &argvals, -1.0).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }));
    }

    #[test]
    fn zero_factor_rejected() {
        let (gammas, argvals) = make_warps(5, 20);
        let err = phase_boxplot(&gammas, &argvals, 0.0).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }));
    }

    #[test]
    fn basic_structure() {
        let (gammas, argvals) = make_warps(7, 30);
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        assert_eq!(bp.median.len(), 30);
        assert_eq!(bp.central_lower.len(), 30);
        assert_eq!(bp.central_upper.len(), 30);
        assert_eq!(bp.whisker_lower.len(), 30);
        assert_eq!(bp.whisker_upper.len(), 30);
        assert_eq!(bp.depths.len(), 7);
        assert_eq!(bp.factor, 1.5);
        assert!(bp.median_index < 7);
    }

    #[test]
    fn central_envelope_inside_whiskers() {
        let (gammas, argvals) = make_warps(10, 25);
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        for j in 0..25 {
            assert!(
                bp.central_lower[j] >= bp.whisker_lower[j] - 1e-12,
                "central_lower should be >= whisker_lower at j={j}"
            );
            assert!(
                bp.central_upper[j] <= bp.whisker_upper[j] + 1e-12,
                "central_upper should be <= whisker_upper at j={j}"
            );
            assert!(
                bp.central_lower[j] <= bp.central_upper[j] + 1e-12,
                "central_lower should be <= central_upper at j={j}"
            );
        }
    }

    #[test]
    fn median_inside_central_region() {
        let (gammas, argvals) = make_warps(9, 20);
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        for j in 0..20 {
            assert!(
                bp.median[j] >= bp.central_lower[j] - 1e-12,
                "median should be >= central_lower at j={j}"
            );
            assert!(
                bp.median[j] <= bp.central_upper[j] + 1e-12,
                "median should be <= central_upper at j={j}"
            );
        }
    }

    #[test]
    fn whiskers_within_domain() {
        let (gammas, argvals) = make_warps(8, 20);
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        let lo = argvals[0];
        let hi = argvals[19];
        for j in 0..20 {
            assert!(
                bp.whisker_lower[j] >= lo - 1e-12,
                "whisker_lower should be >= domain lo at j={j}"
            );
            assert!(
                bp.whisker_upper[j] <= hi + 1e-12,
                "whisker_upper should be <= domain hi at j={j}"
            );
        }
    }

    #[test]
    fn identical_warps_no_outliers() {
        let m = 15;
        let argvals = uniform_grid(m);
        // All curves are the identity warp
        let mut data = vec![0.0; 5 * m];
        for i in 0..5 {
            for j in 0..m {
                data[j * 5 + i] = argvals[j];
            }
        }
        let gammas = FdMatrix::from_column_major(data, 5, m).unwrap();
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        assert!(
            bp.outlier_indices.is_empty(),
            "identical warps should have no outliers"
        );
    }

    #[test]
    fn extreme_warp_is_outlier() {
        let m = 20;
        let argvals = uniform_grid(m);

        // 4 near-identity warps + 1 extreme warp
        let n = 5;
        let mut data = vec![0.0; n * m];
        for i in 0..4 {
            let shift = 0.01 * (i as f64 - 1.5);
            for j in 0..m {
                let t = argvals[j];
                data[j * n + i] = (t + shift * t * (1.0 - t)).clamp(0.0, 1.0);
            }
        }
        // Extreme warp: very fast early, very slow late
        for j in 0..m {
            let t = argvals[j];
            let extreme = (t * t).clamp(0.0, 1.0);
            data[j * n + 4] = extreme;
        }

        let gammas = FdMatrix::from_column_major(data, n, m).unwrap();
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        assert!(
            bp.outlier_indices.contains(&4),
            "extreme warp (index 4) should be an outlier, got {:?}",
            bp.outlier_indices
        );
    }

    #[test]
    fn depths_are_nonnegative() {
        let (gammas, argvals) = make_warps(6, 20);
        let bp = phase_boxplot(&gammas, &argvals, 1.5).unwrap();

        assert!(
            bp.depths.iter().all(|&d| d >= 0.0),
            "all depths should be non-negative"
        );
    }
}
