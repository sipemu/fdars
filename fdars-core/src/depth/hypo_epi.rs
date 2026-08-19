//! Hypograph, modified-hypograph, and epigraph indices (HI, MHI, EI).
//!
//! These are members of the roahd index-measure family, complementing the
//! already-shipped Modified Epigraph Index (MEI).

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute the Hypograph Index (HI) for 1D functional data.
///
/// HI is a **global indicator**: for each object curve `X_i`, count the fraction
/// of reference curves `X_j` that lie **entirely at or below** `X_i` across the
/// whole domain, i.e. `X_j(t) <= X_i(t)` for every evaluation point `t`.
///
/// ```text
/// HI(X_i) = (1/N) · #{j : X_j(t) ≤ X_i(t) for all t}
/// ```
///
/// Tie convention: `<=` (consistent with MEI and roahd). [CITED: roahd::HI]
///
/// Range: [0, 1]. Returned values are always integer multiples of `1/nori`
/// because the indicator is 0 or 1 for each reference curve — one crossing is
/// enough to exclude that reference. The deepest curve in the HI sense is the
/// one that is highest (all others below it); HRD = min(EI, HI) provides the
/// symmetric composite.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data_obj` or `data_ori` is empty,
/// or if `data_ori` has fewer than 2 rows.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn hypograph_index_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    if nobj == 0 || nori == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: "non-empty matrix".to_string(),
            actual: format!("{nobj}x{m}"),
        });
    }
    if nori < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "at least 2 reference curves for hypograph index".to_string(),
            actual: format!("{nori}"),
        });
    }

    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut count = 0.0_f64;
            'outer: for j in 0..nori {
                for t in 0..m {
                    // If X_j crosses above X_i, this reference is not in the hypograph.
                    if data_ori[(j, t)] > data_obj[(i, t)] {
                        continue 'outer;
                    }
                }
                count += 1.0;
            }
            count / nori as f64
        })
        .collect();

    Ok(depths)
}

/// Compute the Epigraph Index (EI) for 1D functional data.
///
/// EI is a **global indicator** and the complement of HI: for each object curve
/// `X_i`, count the fraction of reference curves `X_j` that lie **entirely at
/// or above** `X_i` across the whole domain, i.e. `X_j(t) >= X_i(t)` for every
/// evaluation point `t`.
///
/// ```text
/// EI(X_i) = (1/N) · #{j : X_j(t) ≥ X_i(t) for all t}
/// ```
///
/// Tie convention: `>=` (i.e., `<=` on the reversed side, consistent with MEI).
/// [CITED: roahd::EI]
///
/// Range: [0, 1]. Returned values are always integer multiples of `1/nori`.
/// Curves at the bottom of the sample score highest in EI; HRD = min(EI, HI)
/// resolves this asymmetry.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data_obj` or `data_ori` is empty,
/// or if `data_ori` has fewer than 2 rows.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn epigraph_index_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    if nobj == 0 || nori == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: "non-empty matrix".to_string(),
            actual: format!("{nobj}x{m}"),
        });
    }
    if nori < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "at least 2 reference curves for epigraph index".to_string(),
            actual: format!("{nori}"),
        });
    }

    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut count = 0.0_f64;
            'outer: for j in 0..nori {
                for t in 0..m {
                    // If X_j drops below X_i, this reference is not in the epigraph.
                    if data_ori[(j, t)] < data_obj[(i, t)] {
                        continue 'outer;
                    }
                }
                count += 1.0;
            }
            count / nori as f64
        })
        .collect();

    Ok(depths)
}

/// Compute the Modified Hypograph Index (MHI) for 1D functional data.
///
/// MHI is a **pointwise average**: for each object curve `X_i` and each reference
/// curve `X_j`, measure the fraction of evaluation points where `X_i(t) >= X_j(t)`,
/// then average over all reference curves.
///
/// ```text
/// MHI(X_i) = (1/N) · Σ_j (1/m) · #{t : X_i(t) ≥ X_j(t)}
/// ```
///
/// This is the exact mirror of MEI (`modified_epigraph_index_1d`) with the
/// comparison direction reversed. Without ties: MHI(X) + MEI(X) = 1.
/// [CITED: roahd::MHI]
///
/// Tie convention: `>=` for the MHI condition (consistent with roahd). Range: [0, 1].
/// The central curve of a sample gets MHI ≈ 0.5.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data_obj` or `data_ori` is empty.
/// Unlike HI/EI, a single reference curve (nori = 1) is mathematically valid for MHI.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modified_hypograph_index_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    if nobj == 0 || nori == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: "non-empty matrix".to_string(),
            actual: format!("{nobj}x{m}"),
        });
    }

    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0_f64;
            for j in 0..nori {
                let mut count = 0.0_f64;
                for t in 0..m {
                    // MHI: fraction of t where X_i(t) dominates X_j(t) from above.
                    if data_obj[(i, t)] >= data_ori[(j, t)] {
                        count += 1.0;
                    }
                }
                total += count / m as f64;
            }
            total / nori as f64
        })
        .collect();

    Ok(depths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::dispatch::{functional_depth, DepthMethod};
    use crate::matrix::FdMatrix;

    /// Build a sample of `n` sinusoidal curves on an `m`-point grid with a
    /// uniform vertical offset of 0.05 per curve index. Row 0 is the lowest
    /// curve, row n-1 is the highest.
    fn sample(n: usize, m: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                // column-major: element (i, t) at index i + t*n
                col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.05 * i as f64;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    // --- HI tests ---

    #[test]
    fn hi_values_are_multiples_of_1_over_nori() {
        // HI is a global indicator — values must be k/nori for some non-negative integer k.
        let n = 8usize;
        let data = sample(n, 20);
        let hi = hypograph_index_1d(&data, &data).unwrap();
        assert_eq!(hi.len(), n);
        let inv = 1.0 / n as f64;
        for &v in &hi {
            assert!((0.0..=1.0 + 1e-12).contains(&v), "HI out of range: {v}");
            // v * n must be an integer within floating-point rounding.
            let k = v / inv;
            assert!(
                (k - k.round()).abs() < 1e-9,
                "HI value {v} is not a multiple of 1/{n}"
            );
        }
    }

    #[test]
    fn hi_highest_curve_scores_deepest() {
        // In the sample fixture, the curve with the largest offset (row n-1) has
        // all others at or below it, so it should get the maximum HI.
        let n = 8usize;
        let data = sample(n, 20);
        let hi = hypograph_index_1d(&data, &data).unwrap();
        let max_idx = hi
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx,
            n - 1,
            "Expected the highest curve (row {}) to have max HI",
            n - 1
        );
    }

    #[test]
    fn hi_empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(hypograph_index_1d(&empty, &empty).is_err());
    }

    #[test]
    fn hi_single_curve_returns_err() {
        let one = sample(1, 10);
        assert!(hypograph_index_1d(&one, &one).is_err());
    }

    // --- EI tests ---

    #[test]
    fn ei_values_are_multiples_of_1_over_nori() {
        let n = 8usize;
        let data = sample(n, 20);
        let ei = epigraph_index_1d(&data, &data).unwrap();
        assert_eq!(ei.len(), n);
        let inv = 1.0 / n as f64;
        for &v in &ei {
            assert!((0.0..=1.0 + 1e-12).contains(&v), "EI out of range: {v}");
            let k = v / inv;
            assert!(
                (k - k.round()).abs() < 1e-9,
                "EI value {v} is not a multiple of 1/{n}"
            );
        }
    }

    #[test]
    fn ei_lowest_curve_scores_deepest() {
        // In the sample fixture, row 0 is the lowest curve — all others are above it.
        let n = 8usize;
        let data = sample(n, 20);
        let ei = epigraph_index_1d(&data, &data).unwrap();
        let max_idx = ei
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx, 0,
            "Expected the lowest curve (row 0) to have max EI"
        );
    }

    #[test]
    fn ei_empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(epigraph_index_1d(&empty, &empty).is_err());
    }

    #[test]
    fn ei_single_curve_returns_err() {
        let one = sample(1, 10);
        assert!(epigraph_index_1d(&one, &one).is_err());
    }

    // --- MHI tests ---

    #[test]
    fn mhi_is_monotone_in_curve_height_and_central_near_half() {
        // MHI is a ONE-SIDED index, not a depth: it measures how much a curve
        // dominates the others from above. With n=9 vertically-stacked curves
        // (offsets 0..8), MHI increases monotonically with row height, so the
        // TOP curve (row 8) is the maximum and the central curve (row 4) ≈ 0.5.
        let n = 9usize;
        let data = sample(n, 30);
        let mhi = modified_hypograph_index_1d(&data, &data).unwrap();
        assert_eq!(mhi.len(), n);

        // Monotone non-decreasing in row index (higher curve dominates more).
        for i in 1..n {
            assert!(
                mhi[i] >= mhi[i - 1] - 1e-12,
                "MHI should be non-decreasing in curve height: row {i} = {}, row {} = {}",
                mhi[i],
                i - 1,
                mhi[i - 1]
            );
        }

        // The top curve is the maximum; the central curve sits near 0.5.
        let max_idx = mhi
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx,
            n - 1,
            "top curve should maximize MHI, got row {max_idx}"
        );
        let center = n / 2;
        assert!(
            (mhi[center] - 0.5).abs() < 0.1,
            "central curve MHI should be near 0.5, got {}",
            mhi[center]
        );
    }

    #[test]
    fn mhi_central_value_approx_half() {
        let n = 9usize;
        let data = sample(n, 30);
        let mhi = modified_hypograph_index_1d(&data, &data).unwrap();
        let center = n / 2;
        assert!(
            (mhi[center] - 0.5).abs() < 0.1,
            "MHI of central curve should be near 0.5, got {}",
            mhi[center]
        );
    }

    #[test]
    fn mhi_values_in_unit_interval() {
        let data = sample(8, 20);
        let mhi = modified_hypograph_index_1d(&data, &data).unwrap();
        for &v in &mhi {
            assert!((0.0..=1.0 + 1e-12).contains(&v), "MHI out of range: {v}");
        }
    }

    #[test]
    fn mhi_empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(modified_hypograph_index_1d(&empty, &empty).is_err());
    }

    // MHI allows nori=1 (no n<2 guard) — single curve is valid.
    #[test]
    fn mhi_single_curve_is_ok() {
        let one = sample(1, 10);
        let result = modified_hypograph_index_1d(&one, &one);
        assert!(
            result.is_ok(),
            "MHI with a single curve should be Ok (no n<2 guard)"
        );
        let depths = result.unwrap();
        assert_eq!(depths.len(), 1);
        // A single curve compared against itself: all t satisfy xi >= xi, so MHI = 1.0.
        assert!((depths[0] - 1.0).abs() < 1e-12);
    }

    // --- Dispatcher round-trip tests ---

    #[test]
    fn dispatcher_hypograph_index_round_trips() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::HypographIndex).unwrap();
        let want = hypograph_index_1d(&data, &data).unwrap();
        assert_eq!(got, want);
        assert_eq!(got.len(), data.nrows());
    }

    #[test]
    fn dispatcher_epigraph_index_round_trips() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::EpigraphIndex).unwrap();
        let want = epigraph_index_1d(&data, &data).unwrap();
        assert_eq!(got, want);
        assert_eq!(got.len(), data.nrows());
    }

    #[test]
    fn dispatcher_modified_hypograph_index_round_trips() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::ModifiedHypographIndex).unwrap();
        let want = modified_hypograph_index_1d(&data, &data).unwrap();
        assert_eq!(got, want);
        assert_eq!(got.len(), data.nrows());
    }

    #[test]
    fn dispatcher_band_still_works_after_extension() {
        // Regression check: existing DepthMethod variants must still work.
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::Band).unwrap();
        assert_eq!(got.len(), data.nrows());
        assert!(got.iter().all(|&d| (0.0..=1.0 + 1e-12).contains(&d)));
    }
}
