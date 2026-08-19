//! Half-region depth and modified half-region depth (HRD, MHRD).
//!
//! These are the roahd half-region composites: HRD = min(EI, HI) over the global
//! hypograph/epigraph indicators, and MHRD = min(MEI, MHI) over the pointwise
//! modified indices. Both reward curves that are simultaneously hard to dominate
//! from above and from below — i.e. central curves.

use crate::depth::{modified_epigraph_index_1d, modified_hypograph_index_1d};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute the Half-Region Depth (HRD) for 1D functional data.
///
/// HRD is the roahd half-region composite of the two **global indicators**, the
/// epigraph index (EI) and hypograph index (HI):
///
/// ```text
/// HRD(X_i) = min(EI(X_i), HI(X_i))
/// ```
///
/// where `EI(X_i) = (1/N)·#{j : X_j(t) ≥ X_i(t) ∀t}` and
/// `HI(X_i) = (1/N)·#{j : X_j(t) ≤ X_i(t) ∀t}`. EI and HI are computed in a
/// single fused pass over the reference curves to avoid nested parallelism.
/// [CITED: roahd::HRD]
///
/// Range: [0, 1]. A curve that sits in the middle of the sample scores high (few
/// references lie entirely above or entirely below it); the highest and lowest
/// curves score near `1/N` because they bound the sample on one side. Ties use
/// `<=`/`>=` consistent with HI/EI.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data_obj` or `data_ori` is empty,
/// or if `data_ori` has fewer than 2 reference curves (the global indicators are
/// undefined for a single reference).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn half_region_depth_1d(
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
    if nori < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "at least 2 reference curves for half-region depth".to_string(),
            actual: format!("{nori}"),
        });
    }

    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut ei_count = 0.0_f64; // references entirely at or above X_i
            let mut hi_count = 0.0_f64; // references entirely at or below X_i
            for j in 0..nori {
                let mut j_above = true; // X_j(t) >= X_i(t) for all t so far
                let mut j_below = true; // X_j(t) <= X_i(t) for all t so far
                for t in 0..m {
                    let x_j = data_ori[(j, t)];
                    let x_i = data_obj[(i, t)];
                    if x_j > x_i {
                        j_below = false;
                    }
                    if x_j < x_i {
                        j_above = false;
                    }
                    if !j_above && !j_below {
                        break;
                    }
                }
                if j_above {
                    ei_count += 1.0;
                }
                if j_below {
                    hi_count += 1.0;
                }
            }
            f64::min(ei_count, hi_count) / nori as f64
        })
        .collect();

    Ok(depths)
}

/// Compute the Modified Half-Region Depth (MHRD) for 1D functional data.
///
/// MHRD is the half-region composite of the two **pointwise-average** modified
/// indices, the modified epigraph index (MEI) and modified hypograph index (MHI):
///
/// ```text
/// MHRD(X_i) = min(MEI(X_i), MHI(X_i))
/// ```
///
/// It reuses the already-shipped [`modified_epigraph_index_1d`] and
/// [`modified_hypograph_index_1d`]; the outer `zip` is sequential so the two
/// internally-parallel index computations do not nest. [CITED: roahd::MHRD]
///
/// Range: [0, 0.5]. The central curve of a sample scores near 0.5 (MEI ≈ MHI ≈ 0.5);
/// magnitude and shape outliers score lower on whichever side they escape to.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data_obj` or `data_ori` is empty.
/// A single reference curve (nori = 1) is mathematically valid for MHRD.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modified_half_region_depth_1d(
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

    // MEI is infallible (returns Vec); MHI is Result-returning (shares this file's guards).
    let mei = modified_epigraph_index_1d(data_obj, data_ori);
    let mhi = modified_hypograph_index_1d(data_obj, data_ori)?;

    let depths: Vec<f64> = mei
        .iter()
        .zip(mhi.iter())
        .map(|(&e, &h)| f64::min(e, h))
        .collect();

    Ok(depths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::dispatch::{functional_depth, DepthMethod};
    use crate::matrix::FdMatrix;

    /// `n` sinusoids on an `m`-point grid, offset 0.05 per index (row 0 lowest,
    /// row n-1 highest). The middle rows are the most central.
    fn sample(n: usize, m: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.05 * i as f64;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    /// Inliers plus one gross magnitude outlier at `outlier_idx`.
    fn sample_with_outlier(n: usize, m: usize, outlier_idx: usize) -> FdMatrix {
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                let base = (x * std::f64::consts::PI).sin();
                let val = if i == outlier_idx {
                    base + 100.0
                } else {
                    base + 0.01 * i as f64
                };
                col_major[i + t * n] = val;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    #[test]
    fn hrd_range_and_length() {
        let n = 8usize;
        let data = sample(n, 20);
        let hrd = half_region_depth_1d(&data, &data).unwrap();
        assert_eq!(hrd.len(), n);
        for &v in &hrd {
            assert!((0.0..=1.0 + 1e-12).contains(&v), "HRD out of range: {v}");
        }
    }

    #[test]
    fn hrd_equals_min_of_ei_and_hi() {
        let data = sample(7, 15);
        let ei = crate::depth::epigraph_index_1d(&data, &data).unwrap();
        let hi = crate::depth::hypograph_index_1d(&data, &data).unwrap();
        let hrd = half_region_depth_1d(&data, &data).unwrap();
        for i in 0..hrd.len() {
            assert!((hrd[i] - f64::min(ei[i], hi[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn hrd_ranks_central_deepest_and_extremes_shallow() {
        // Odd count → unique central row. Central curve should be deepest.
        let n = 9usize;
        let data = sample(n, 25);
        let hrd = half_region_depth_1d(&data, &data).unwrap();
        let mut deepest = 0usize;
        for i in 1..n {
            if hrd[i] > hrd[deepest] {
                deepest = i;
            }
        }
        assert_eq!(deepest, n / 2, "central curve should be deepest by HRD");
        // The two boundary curves are the shallowest (bounded on one side).
        assert!(hrd[0] <= hrd[deepest] && hrd[n - 1] <= hrd[deepest]);
    }

    #[test]
    fn mhrd_ranks_central_deepest_and_outlier_shallow() {
        // A far magnitude outlier is above everything → MHI≈1, MEI≈0 → MHRD≈0.
        // The lowest boundary inlier is below everything → MEI≈1, MHI≈0 → MHRD≈0
        // too, so the outlier is NOT uniquely the global minimum. The robust
        // properties are: the outlier is shallow (near 0) and is not the deepest,
        // and a central inlier is the deepest.
        let outlier_idx = 3usize;
        let data = sample_with_outlier(8, 20, outlier_idx);
        let mhrd = modified_half_region_depth_1d(&data, &data).unwrap();

        let mut deepest = 0usize;
        for i in 1..mhrd.len() {
            if mhrd[i] > mhrd[deepest] {
                deepest = i;
            }
        }
        assert_ne!(
            deepest, outlier_idx,
            "magnitude outlier must not be deepest"
        );
        assert!(
            mhrd[outlier_idx] < mhrd[deepest],
            "outlier MHRD {} should be below the deepest curve's {}",
            mhrd[outlier_idx],
            mhrd[deepest]
        );
        // The outlier is above every curve, so MEI(outlier) floors at 1/n
        // (only the self-comparison satisfies `<=`); MHRD = min(MEI, MHI) ≈ 1/n.
        // That is well below the central curve's ≈ 0.5.
        assert!(
            mhrd[outlier_idx] < 0.2,
            "magnitude outlier should be shallow (MHRD ≈ 1/n), got {}",
            mhrd[outlier_idx]
        );
    }

    #[test]
    fn mhrd_equals_min_of_mei_and_mhi() {
        let data = sample(6, 12);
        let mei = crate::depth::modified_epigraph_index_1d(&data, &data);
        let mhi = crate::depth::modified_hypograph_index_1d(&data, &data).unwrap();
        let mhrd = modified_half_region_depth_1d(&data, &data).unwrap();
        for i in 0..mhrd.len() {
            assert!((mhrd[i] - f64::min(mei[i], mhi[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn dispatch_round_trip() {
        let data = sample(6, 12);
        let hrd = functional_depth(&data, DepthMethod::HalfRegion).unwrap();
        assert_eq!(hrd, half_region_depth_1d(&data, &data).unwrap());
        let mhrd = functional_depth(&data, DepthMethod::ModifiedHalfRegion).unwrap();
        assert_eq!(mhrd, modified_half_region_depth_1d(&data, &data).unwrap());
    }

    #[test]
    fn empty_and_single_curve_return_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(half_region_depth_1d(&empty, &empty).is_err());
        assert!(modified_half_region_depth_1d(&empty, &empty).is_err());
        let single = sample(1, 8);
        // HRD needs >= 2 references; MHRD accepts a single reference.
        assert!(half_region_depth_1d(&single, &single).is_err());
        assert!(modified_half_region_depth_1d(&single, &single).is_ok());
        // Dispatcher guards both at n < 2.
        assert!(functional_depth(&single, DepthMethod::HalfRegion).is_err());
        assert!(functional_depth(&single, DepthMethod::ModifiedHalfRegion).is_err());
    }
}
