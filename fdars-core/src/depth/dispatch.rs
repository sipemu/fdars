//! Unified depth dispatcher and depth-fence functional boxplot.
//!
//! This module provides a single [`functional_depth`] entry point that computes
//! the **self-depth** of a sample (each curve's depth with respect to the sample
//! itself, i.e. `data_obj == data_ori`) by dispatching to the existing depth
//! functions via the [`DepthMethod`] selector. It also provides the canonical
//! López-Pintado–Romo depth-fence [`functional_boxplot`], which produces numeric
//! central-region / whisker / outlier-flag outputs (no plotting).
//!
//! The dispatcher only *wraps* the underlying depth functions — their signatures
//! are unchanged.

use crate::depth::{
    band_1d, epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d, fraiman_muniz_1d,
    half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d, modified_band_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d, random_projection_1d_seeded,
    total_variation_depth_1d,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Depth measure selector for [`functional_depth`] and [`functional_boxplot`].
///
/// Each variant maps 1:1 to an existing self-depth call:
/// - `FraimanMuniz { scale }` → `fraiman_muniz_1d(data, data, scale)`
/// - `Band` → `band_1d(data, data)`
/// - `ModifiedBand` → `modified_band_1d(data, data)`
/// - `RandomProjection { nproj, seed }` → `random_projection_1d_seeded(data, data, nproj, Some(seed))`
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DepthMethod {
    /// Fraiman-Muniz depth. `scale` toggles the scaled `2·min(Fn, 1−Fn)` form.
    FraimanMuniz {
        /// Whether to scale the depth values.
        scale: bool,
    },
    /// Band depth (BD).
    Band,
    /// Modified band depth (MBD).
    ModifiedBand,
    /// Random projection depth. `seed` makes results bit-reproducible.
    RandomProjection {
        /// Number of random projection directions.
        nproj: usize,
        /// RNG seed for deterministic projections.
        seed: u64,
    },
    /// Hypograph index (HI). Global indicator: fraction of reference curves globally
    /// at or below the object curve. Requires n >= 2. [CITED: roahd::HI]
    HypographIndex,
    /// Modified hypograph index (MHI). Pointwise average: fraction of time the object
    /// curve dominates each reference curve. [CITED: roahd::MHI]
    ModifiedHypographIndex,
    /// Epigraph index (EI, un-modified). Global indicator: fraction of reference curves
    /// globally at or above the object curve. Requires n >= 2. [CITED: roahd::EI]
    EpigraphIndex,
    /// Half-region depth (HRD) = min(EI, HI) over the global indicators. Requires n >= 2.
    /// [CITED: roahd::HRD]
    HalfRegion,
    /// Modified half-region depth (MHRD) = min(MEI, MHI) over the pointwise modified
    /// indices. [CITED: roahd::MHRD]
    ModifiedHalfRegion,
    /// Extremal depth (Narisetty & Nair 2016). Rank-ordering measure; requires n >= 3.
    /// [CITED: fdaoutlier::extremal_depth]
    Extremal,
    /// Extreme-rank-length depth. Lexicographic rank-vector ordering; requires n >= 2.
    /// [CITED: fdaoutlier::extreme_rank_length]
    ExtremeRankLength,
    /// L-infinity (sup-norm) depth. Valid for n >= 1. [CITED: fdaoutlier::linfinity_depth]
    LInfinity,
    /// Total variation depth + MSSI (Huang & Sun 2019). Dispatches the TVD (magnitude)
    /// component; requires n >= 3. [CITED: fdaoutlier::total_variation_depth]
    TotalVariation,
}

/// Compute the **self-depth** of every curve in `data` w.r.t. the sample.
///
/// Passes `data` as both the object and reference matrix and dispatches to the
/// underlying depth function selected by `method`. Returns one depth per curve
/// (`Vec<f64>` of length `data.nrows()`).
///
/// # Errors
/// - `InvalidDimension` if `data` has zero rows or zero columns, or if
///   `Band`/`ModifiedBand` is requested with fewer than 2 curves (a band needs
///   two reference curves).
/// - `InvalidParameter` if `RandomProjection { nproj: 0, .. }` is requested.
pub fn functional_depth(data: &FdMatrix, method: DepthMethod) -> Result<Vec<f64>, FdarError> {
    let (n, m) = (data.nrows(), data.ncols());
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0 and ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }

    let depths = match method {
        DepthMethod::FraimanMuniz { scale } => fraiman_muniz_1d(data, data, scale),
        DepthMethod::Band => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for band depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            band_1d(data, data)
        }
        DepthMethod::ModifiedBand => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for modified band depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            modified_band_1d(data, data)
        }
        DepthMethod::RandomProjection { nproj, seed } => {
            if nproj == 0 {
                return Err(FdarError::InvalidParameter {
                    parameter: "nproj",
                    message: "must be >= 1".to_string(),
                });
            }
            random_projection_1d_seeded(data, data, nproj, Some(seed))
        }
        DepthMethod::HypographIndex => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for hypograph index".to_string(),
                    actual: format!("{n}"),
                });
            }
            hypograph_index_1d(data, data)?
        }
        DepthMethod::ModifiedHypographIndex => modified_hypograph_index_1d(data, data)?,
        DepthMethod::EpigraphIndex => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for epigraph index".to_string(),
                    actual: format!("{n}"),
                });
            }
            epigraph_index_1d(data, data)?
        }
        DepthMethod::HalfRegion => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for half-region depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            half_region_depth_1d(data, data)?
        }
        DepthMethod::ModifiedHalfRegion => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for modified half-region depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            modified_half_region_depth_1d(data, data)?
        }
        DepthMethod::Extremal => {
            if n < 3 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 3 curves for extremal depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            extremal_depth_1d(data, data)?
        }
        DepthMethod::ExtremeRankLength => {
            if n < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 curves for extreme-rank-length depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            extreme_rank_length_depth_1d(data, data)?
        }
        DepthMethod::LInfinity => linfinity_depth_1d(data, data)?,
        DepthMethod::TotalVariation => {
            if n < 3 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 3 curves for total variation depth".to_string(),
                    actual: format!("{n}"),
                });
            }
            total_variation_depth_1d(data, data)?.tvd
        }
    };

    Ok(depths)
}

/// Numeric outputs of a depth-fence functional boxplot (no plotting).
///
/// All curve vectors have length `data.ncols()` (one value per evaluation point);
/// `depths` has length `data.nrows()` and `outliers` holds flagged row indices.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FunctionalBoxplotResult {
    /// The deepest (median) curve's values across all evaluation points.
    pub median: Vec<f64>,
    /// Pointwise lower bound of the 50% central region.
    pub central_lower: Vec<f64>,
    /// Pointwise upper bound of the 50% central region.
    pub central_upper: Vec<f64>,
    /// Pointwise lower whisker (central region inflated by `factor × width`).
    pub whisker_lower: Vec<f64>,
    /// Pointwise upper whisker (central region inflated by `factor × width`).
    pub whisker_upper: Vec<f64>,
    /// Row indices of curves that exceed the fence at any evaluation point.
    pub outliers: Vec<usize>,
    /// Per-curve self-depth used to rank curves.
    pub depths: Vec<f64>,
}

/// Canonical López-Pintado–Romo depth-fence functional boxplot (numeric only).
///
/// Ranks curves by [`functional_depth`], takes the deepest curve as the median,
/// builds the 50% central region as the pointwise envelope of the deepest half,
/// inflates it by `factor × (central width)` to form the whiskers/fence, and
/// flags any curve that exceeds the fence at any evaluation point as an outlier.
///
/// The recommended defaults are `method = DepthMethod::ModifiedBand` and
/// `factor = 1.5`; the caller passes both explicitly.
///
/// # Errors
/// - `InvalidDimension` if `data` is empty or has fewer than 2 curves.
/// - `InvalidParameter` if `factor` is negative or not finite.
/// - Propagates errors from [`functional_depth`].
pub fn functional_boxplot(
    data: &FdMatrix,
    method: DepthMethod,
    factor: f64,
) -> Result<FunctionalBoxplotResult, FdarError> {
    let (n, m) = (data.nrows(), data.ncols());
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0 and ncols > 0)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves for a functional boxplot".to_string(),
            actual: format!("{n}"),
        });
    }
    if !factor.is_finite() || factor < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "factor",
            message: "must be a finite value >= 0.0".to_string(),
        });
    }

    let depths = functional_depth(data, method)?;

    // Median = deepest curve (argmax of depth; ties broken by lowest index).
    let mut median_row = 0usize;
    for i in 1..n {
        if depths[i] > depths[median_row] {
            median_row = i;
        }
    }
    let median: Vec<f64> = (0..m).map(|t| data[(median_row, t)]).collect();

    // Deepest 50% of rows (ceil(n/2)), ties broken by index for determinism.
    let half = n.div_ceil(2);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        depths[b]
            .partial_cmp(&depths[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let central_rows = &order[..half];

    // Central region = pointwise min/max over the deepest-half rows.
    let mut central_lower = vec![f64::INFINITY; m];
    let mut central_upper = vec![f64::NEG_INFINITY; m];
    for &i in central_rows {
        for t in 0..m {
            let v = data[(i, t)];
            if v < central_lower[t] {
                central_lower[t] = v;
            }
            if v > central_upper[t] {
                central_upper[t] = v;
            }
        }
    }

    // Whiskers = central region inflated by factor × width at each t.
    let mut whisker_lower = vec![0.0; m];
    let mut whisker_upper = vec![0.0; m];
    for t in 0..m {
        let width = central_upper[t] - central_lower[t];
        whisker_lower[t] = central_lower[t] - factor * width;
        whisker_upper[t] = central_upper[t] + factor * width;
    }

    // Outliers = any curve exceeding the fence at any evaluation point.
    let mut outliers = Vec::new();
    for i in 0..n {
        let mut flagged = false;
        for t in 0..m {
            let v = data[(i, t)];
            if v < whisker_lower[t] || v > whisker_upper[t] {
                flagged = true;
                break;
            }
        }
        if flagged {
            outliers.push(i);
        }
    }

    Ok(FunctionalBoxplotResult {
        median,
        central_lower,
        central_upper,
        whisker_lower,
        whisker_upper,
        outliers,
        depths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic sample: `n` mild sinusoids on an `m`-point grid.
    fn sample(n: usize, m: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                // element (i, t) at index i + t*n (column-major)
                col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.05 * i as f64;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    #[test]
    fn fraiman_muniz_dispatch_equals_underlying() {
        let data = sample(6, 12);
        for scale in [true, false] {
            let got = functional_depth(&data, DepthMethod::FraimanMuniz { scale }).unwrap();
            let want = fraiman_muniz_1d(&data, &data, scale);
            assert_eq!(got, want);
            assert_eq!(got.len(), data.nrows());
        }
    }

    #[test]
    fn band_dispatch_equals_underlying() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::Band).unwrap();
        assert_eq!(got, band_1d(&data, &data));
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn modified_band_dispatch_equals_underlying() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::ModifiedBand).unwrap();
        assert_eq!(got, modified_band_1d(&data, &data));
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn random_projection_dispatch_equals_underlying_and_is_reproducible() {
        let data = sample(6, 12);
        let method = DepthMethod::RandomProjection {
            nproj: 20,
            seed: 42,
        };
        let got = functional_depth(&data, method).unwrap();
        let want = random_projection_1d_seeded(&data, &data, 20, Some(42));
        assert_eq!(got, want);
        // Two dispatch calls with the same seed are bit-identical.
        let got2 = functional_depth(&data, method).unwrap();
        assert_eq!(got, got2);
    }

    #[test]
    fn empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(functional_depth(&empty, DepthMethod::FraimanMuniz { scale: true }).is_err());
    }

    #[test]
    fn too_few_curves_for_band_returns_err() {
        let one = sample(1, 8);
        assert!(functional_depth(&one, DepthMethod::Band).is_err());
        assert!(functional_depth(&one, DepthMethod::ModifiedBand).is_err());
    }

    #[test]
    fn zero_nproj_returns_err() {
        let data = sample(6, 12);
        assert!(
            functional_depth(&data, DepthMethod::RandomProjection { nproj: 0, seed: 1 }).is_err()
        );
    }

    // --- functional_boxplot ---

    /// Inlier sinusoids plus one gross-outlier curve at row `outlier_idx`.
    fn sample_with_outlier(n: usize, m: usize, outlier_idx: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                let base = (x * std::f64::consts::PI).sin();
                let val = if i == outlier_idx {
                    base + 100.0 // gross vertical shift far outside the band
                } else {
                    base + 0.01 * i as f64 // tight inliers
                };
                col_major[i + t * n] = val;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    #[test]
    fn boxplot_flags_planted_outlier_and_spares_inliers() {
        let outlier_idx = 3;
        let data = sample_with_outlier(8, 15, outlier_idx);
        let res = functional_boxplot(&data, DepthMethod::ModifiedBand, 1.5).unwrap();
        assert!(res.outliers.contains(&outlier_idx));
        for i in 0..8 {
            if i != outlier_idx {
                assert!(!res.outliers.contains(&i), "inlier {i} wrongly flagged");
            }
        }
    }

    #[test]
    fn boxplot_median_equals_deepest_and_central_brackets_median() {
        let data = sample_with_outlier(8, 15, 3);
        let res = functional_boxplot(&data, DepthMethod::ModifiedBand, 1.5).unwrap();
        // Median row = argmax depth.
        let mut deepest = 0usize;
        for i in 1..res.depths.len() {
            if res.depths[i] > res.depths[deepest] {
                deepest = i;
            }
        }
        let expected_median: Vec<f64> = (0..data.ncols()).map(|t| data[(deepest, t)]).collect();
        assert_eq!(res.median, expected_median);
        for t in 0..data.ncols() {
            assert!(res.central_lower[t] <= res.median[t] + 1e-12);
            assert!(res.median[t] <= res.central_upper[t] + 1e-12);
        }
    }

    #[test]
    fn boxplot_fence_contains_central_region() {
        let data = sample_with_outlier(8, 15, 3);
        let res = functional_boxplot(&data, DepthMethod::ModifiedBand, 1.5).unwrap();
        for t in 0..data.ncols() {
            assert!(res.whisker_lower[t] <= res.central_lower[t] + 1e-12);
            assert!(res.whisker_upper[t] >= res.central_upper[t] - 1e-12);
        }
    }

    #[test]
    fn boxplot_random_projection_is_seed_reproducible() {
        let data = sample_with_outlier(8, 15, 3);
        let method = DepthMethod::RandomProjection { nproj: 25, seed: 7 };
        let a = functional_boxplot(&data, method, 1.5).unwrap();
        let b = functional_boxplot(&data, method, 1.5).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn boxplot_invalid_input_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(functional_boxplot(&empty, DepthMethod::ModifiedBand, 1.5).is_err());
        let single = sample(1, 8);
        assert!(functional_boxplot(&single, DepthMethod::ModifiedBand, 1.5).is_err());
        let data = sample(6, 12);
        assert!(functional_boxplot(&data, DepthMethod::ModifiedBand, -1.0).is_err());
    }

    // --- Phase 28 all-variant coverage ---

    /// The nine parameter-free depth measures added across plans 28-01..28-03.
    const NEW_VARIANTS: [DepthMethod; 9] = [
        DepthMethod::HalfRegion,
        DepthMethod::ModifiedHalfRegion,
        DepthMethod::HypographIndex,
        DepthMethod::ModifiedHypographIndex,
        DepthMethod::EpigraphIndex,
        DepthMethod::Extremal,
        DepthMethod::ExtremeRankLength,
        DepthMethod::LInfinity,
        DepthMethod::TotalVariation,
    ];

    #[test]
    fn all_nine_new_variants_round_trip() {
        let data = sample(6, 12); // n >= 3 so every guard is satisfied
        for method in NEW_VARIANTS {
            let got = functional_depth(&data, method).unwrap();
            assert_eq!(got.len(), data.nrows(), "wrong length for {method:?}");
        }
    }

    #[test]
    fn existing_variants_unchanged_regression() {
        // The additive change must not alter the four pre-existing variants.
        let data = sample(6, 12);
        assert!(functional_depth(&data, DepthMethod::FraimanMuniz { scale: false }).is_ok());
        assert!(functional_depth(&data, DepthMethod::Band).is_ok());
        assert!(functional_depth(&data, DepthMethod::ModifiedBand).is_ok());
        assert!(
            functional_depth(&data, DepthMethod::RandomProjection { nproj: 10, seed: 1 }).is_ok()
        );
    }

    #[test]
    fn min_n_guards_return_err_without_panic() {
        // Single curve: every measure needing >= 2 (and >= 3) rejects it.
        let single = sample(1, 8);
        for method in [
            DepthMethod::Band,
            DepthMethod::HalfRegion,
            DepthMethod::ModifiedHalfRegion,
            DepthMethod::HypographIndex,
            DepthMethod::EpigraphIndex,
            DepthMethod::ExtremeRankLength,
            DepthMethod::Extremal,
            DepthMethod::TotalVariation,
        ] {
            assert!(
                functional_depth(&single, method).is_err(),
                "{method:?} should reject n=1"
            );
        }
        // n=2 rejects the n>=3 measures.
        let two = sample(2, 8);
        assert!(functional_depth(&two, DepthMethod::Extremal).is_err());
        assert!(functional_depth(&two, DepthMethod::TotalVariation).is_err());
        // Empty matrix rejected by a representative new variant.
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(functional_depth(&empty, DepthMethod::LInfinity).is_err());
    }
}
