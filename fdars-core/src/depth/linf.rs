//! L∞ (sup-norm) depth.
//!
//! L∞ depth inverts a curve's *average sup-norm distance* to the reference sample:
//! curves that are close (in max-pointwise-deviation) to all others are deep, far
//! ones are shallow.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute the L∞ (sup-norm) Depth for 1D functional data.
///
/// For each object curve `X_i`, the depth is the inverse of one plus its mean
/// sup-norm distance to the reference curves:
///
/// ```text
/// L∞_depth(X_i) = 1 / (1 + (1/N) · Σ_j max_t |X_i(t) − X_j(t)|)     ∈ (0, 1]
/// ```
///
/// Note this is the average sup-norm distance to **all** reference curves, not the
/// distance to the pointwise median. Depth is monotonically decreasing in that
/// mean distance: the curve closest to the sample is deepest; a far magnitude
/// outlier is shallow. [CITED: fdaoutlier::linfinity_depth]
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if either matrix is empty or if the two
/// grids differ. A single reference curve (`nori = 1`) is valid — the self-distance
/// is 0, giving depth 1.0.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn linfinity_depth_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    if nobj == 0 || nori == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: "non-empty matrices".to_string(),
            actual: format!("{nobj}x{m}"),
        });
    }
    if data_ori.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: format!("same number of columns as data_obj ({m})"),
            actual: format!("{}", data_ori.ncols()),
        });
    }

    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0_f64;
            for j in 0..nori {
                let mut sup = 0.0_f64;
                for t in 0..m {
                    let d = (data_obj[(i, t)] - data_ori[(j, t)]).abs();
                    if d > sup {
                        sup = d;
                    }
                }
                total += sup;
            }
            let mean_dist = total / nori as f64;
            1.0 / (1.0 + mean_dist)
        })
        .collect();

    Ok(depths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::dispatch::{functional_depth, DepthMethod};

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
    fn depths_in_unit_interval() {
        let data = sample(8, 20);
        let ld = linfinity_depth_1d(&data, &data).unwrap();
        assert_eq!(ld.len(), 8);
        for &v in &ld {
            assert!(v > 0.0 && v <= 1.0 + 1e-12, "L∞ depth out of range: {v}");
        }
    }

    #[test]
    fn closest_to_sample_is_deepest_and_outlier_shallow() {
        let outlier_idx = 3usize;
        let data = sample_with_outlier(8, 20, outlier_idx);
        let ld = linfinity_depth_1d(&data, &data).unwrap();
        // The far magnitude outlier has the largest mean sup-norm distance → shallowest.
        let mut shallowest = 0usize;
        let mut deepest = 0usize;
        for i in 1..ld.len() {
            if ld[i] < ld[shallowest] {
                shallowest = i;
            }
            if ld[i] > ld[deepest] {
                deepest = i;
            }
        }
        assert_eq!(
            shallowest, outlier_idx,
            "outlier should be shallowest by L∞ depth"
        );
        assert_ne!(deepest, outlier_idx);
        assert!(ld[deepest] > ld[outlier_idx]);
    }

    #[test]
    fn monotone_decreasing_in_mean_distance() {
        // On the stacked no-outlier sample the central curve is closest to all others.
        let n = 9usize;
        let data = sample(n, 25);
        let ld = linfinity_depth_1d(&data, &data).unwrap();
        let mut deepest = 0usize;
        for i in 1..n {
            if ld[i] > ld[deepest] {
                deepest = i;
            }
        }
        assert_eq!(
            deepest,
            n / 2,
            "central curve should be deepest by L∞ depth"
        );
    }

    #[test]
    fn single_curve_self_depth_is_one() {
        let one = sample(1, 8);
        let ld = linfinity_depth_1d(&one, &one).unwrap();
        assert_eq!(ld.len(), 1);
        assert!(
            (ld[0] - 1.0).abs() < 1e-12,
            "n=1 self-depth should be 1.0, got {}",
            ld[0]
        );
    }

    #[test]
    fn dispatch_round_trip_and_empty_err() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::LInfinity).unwrap();
        assert_eq!(got, linfinity_depth_1d(&data, &data).unwrap());
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(linfinity_depth_1d(&empty, &empty).is_err());
    }
}
