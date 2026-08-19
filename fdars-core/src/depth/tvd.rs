//! Total variation depth with modified shape similarity index (TVD + MSSI).
//!
//! Two-component depth of Huang & Sun (2019): the **TVD** magnitude component
//! scores how central a curve's pointwise value ranks are, and the **MSS** shape
//! component scores how central its first-difference (derivative) ranks are,
//! weighted by each interval's contribution to the curve's total variation.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of a total-variation-depth + MSSI computation.
///
/// **Pinned interface** — Phase 29's `tvdmss` outlier detector consumes both
/// fields, so the field names and types are a forward contract and must not change.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssResult {
    /// Total variation depth (magnitude component). Length = `nrows` of the data.
    /// Range `(0, 0.25]`; higher = more central in magnitude.
    pub tvd: Vec<f64>,
    /// Modified shape similarity index (shape component). Length = `nrows`.
    /// Range `[0, 0.25]`; higher = more shape-central. Flat curves score `0.0`.
    pub mss: Vec<f64>,
}

/// Average-tie ranks of a slice of values, as `f64` in `1..=len`
/// (matching R's `rank(ties.method = "average")`).
fn rank_slice(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut indexed: Vec<(f64, usize)> = vals
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && indexed[j].0 == indexed[k].0 {
            j += 1;
        }
        let avg_rank = (k as f64 + 1.0 + j as f64) / 2.0;
        for item in &indexed[k..j] {
            ranks[item.1] = avg_rank;
        }
        k = j;
    }
    ranks
}

/// Compute Total Variation Depth (TVD) and the Modified Shape Similarity index (MSS).
///
/// **TVD** (magnitude): for each grid point rank the curves, normalize `p = rank/n`
/// (divided by `n`, not `n+1`), and average the pointwise variation `p·(1−p)`:
///
/// ```text
/// TVD(i) = (1/m) · Σ_t p(i,t)·(1 − p(i,t))     ∈ (0, 0.25]
/// ```
///
/// **MSS** (shape): rank each interval's first differences `Δ(i,k) = X_i(t_{k+1}) − X_i(t_k)`,
/// form `shape(i,k) = q·(1−q)` with `q = rank(Δ)/n`, and weight by each interval's share of the
/// curve's total variation:
///
/// ```text
/// MSS(i) = Σ_k shape(i,k) · |Δ(i,k)| / Σ_k |Δ(i,k)|     ∈ [0, 0.25]
/// ```
///
/// A flat curve (zero total variation) yields `MSS = 0.0` via a guarded division.
/// [CITED: fdaoutlier::total_variation_depth]. The MSS `shape_variation` derivation
/// (rank of first differences, same `q·(1−q)` transform) follows the Huang & Sun
/// (2019) description; the fdaoutlier C++ internals were not read directly, so this
/// is a documented reconstruction. [ASSUMED]
///
/// This is a self-depth measure on `data_ori`; `data_obj` must share the grid.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if either matrix is empty, if the grids
/// differ, or if the sample has fewer than 3 curves (the R reference requires n ≥ 3).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn total_variation_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<TvdMssResult, FdarError> {
    let (n, m) = (data_ori.nrows(), data_ori.ncols());
    if n == 0 || m == 0 || data_obj.nrows() == 0 || data_obj.ncols() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if data_obj.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: format!("same number of columns as data_ori ({m})"),
            actual: format!("{}", data_obj.ncols()),
        });
    }
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "at least 3 curves for total variation depth".to_string(),
            actual: format!("{n}"),
        });
    }

    // TVD rank pass: p[i][t] = rank(X_i(t)) / n.
    let mut p = vec![vec![0.0_f64; m]; n];
    let mut col = vec![0.0_f64; n];
    for t in 0..m {
        for i in 0..n {
            col[i] = data_ori[(i, t)];
        }
        let ranks = rank_slice(&col);
        for i in 0..n {
            p[i][t] = ranks[i] / n as f64;
        }
    }

    // Shape rank pass over the m-1 first-difference columns.
    let has_intervals = m >= 2;
    let mut shape = vec![vec![0.0_f64; m.saturating_sub(1)]; n];
    let mut diff = vec![vec![0.0_f64; m.saturating_sub(1)]; n];
    if has_intervals {
        let mut dcol = vec![0.0_f64; n];
        for k in 0..m - 1 {
            for i in 0..n {
                let d = data_ori[(i, k + 1)] - data_ori[(i, k)];
                diff[i][k] = d;
                dcol[i] = d;
            }
            let ranks = rank_slice(&dcol);
            for i in 0..n {
                let q = ranks[i] / n as f64;
                shape[i][k] = q * (1.0 - q);
            }
        }
    }

    // Per-curve accumulation (parallel over curves).
    let pairs: Vec<(f64, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let tvd_i = (0..m).map(|t| p[i][t] * (1.0 - p[i][t])).sum::<f64>() / m as f64;
            let mss_i = if has_intervals {
                let total_var: f64 = diff[i].iter().map(|d| d.abs()).sum();
                if total_var > 0.0 {
                    (0..m - 1)
                        .map(|k| shape[i][k] * diff[i][k].abs() / total_var)
                        .sum::<f64>()
                } else {
                    0.0 // flat curve → zero total variation → MSS 0 (guarded division)
                }
            } else {
                0.0
            };
            (tvd_i, mss_i)
        })
        .collect();

    let (tvd, mss): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    Ok(TvdMssResult { tvd, mss })
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

    #[test]
    fn tvd_hand_computed_toy_case() {
        // Three constant curves 0,1,2 → at every column ranks [1,2,3], p=[1/3,2/3,1],
        // TV=[2/9,2/9,0]. Averaged over columns → TVD=[2/9,2/9,0]. (Flat → MSS all 0.)
        let n = 3usize;
        let m = 5usize;
        let mut cm = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                cm[i + t * n] = i as f64;
            }
        }
        let data = FdMatrix::from_column_major(cm, n, m).unwrap();
        let res = total_variation_depth_1d(&data, &data).unwrap();
        let two_ninths = 2.0 / 9.0;
        assert!(
            (res.tvd[0] - two_ninths).abs() < 1e-9,
            "tvd[0] = {}",
            res.tvd[0]
        );
        assert!(
            (res.tvd[1] - two_ninths).abs() < 1e-9,
            "tvd[1] = {}",
            res.tvd[1]
        );
        assert!(res.tvd[2].abs() < 1e-9, "tvd[2] = {}", res.tvd[2]);
        // Constant curves have zero total variation → MSS guarded to 0.0 (not NaN).
        for &v in &res.mss {
            assert_eq!(v, 0.0);
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn median_curve_has_max_tvd_near_quarter() {
        // n=8 stacked distinct curves → ranks 1..8 constant; rank 4 → p=0.5 → TV=0.25.
        let n = 8usize;
        let data = sample(n, 20);
        let res = total_variation_depth_1d(&data, &data).unwrap();
        assert_eq!(res.tvd.len(), n);
        assert_eq!(res.mss.len(), n);
        let max = res.tvd.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (max - 0.25).abs() < 1e-9,
            "max TVD should be ~0.25, got {max}"
        );
    }

    #[test]
    fn magnitude_outlier_has_low_tvd() {
        let outlier_idx = 3usize;
        let n = 8usize;
        let m = 20usize;
        let mut cm = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                let base = (x * std::f64::consts::PI).sin();
                cm[i + t * n] = if i == outlier_idx {
                    base + 100.0
                } else {
                    base + 0.01 * i as f64
                };
            }
        }
        let data = FdMatrix::from_column_major(cm, n, m).unwrap();
        let res = total_variation_depth_1d(&data, &data).unwrap();
        // Outlier is rank n at every t → p=1 → TV=0 → TVD ~ 0.
        assert!(
            res.tvd[outlier_idx] < 0.02,
            "outlier TVD should be ~0, got {}",
            res.tvd[outlier_idx]
        );
        let max = res.tvd.iter().cloned().fold(f64::MIN, f64::max);
        assert!(res.tvd[outlier_idx] < max);
    }

    #[test]
    fn shape_outlier_has_low_mss() {
        // 7 smooth co-monotone inliers + 1 high-frequency shape outlier of similar magnitude.
        let n = 8usize;
        let m = 40usize;
        let shape_idx = 7usize;
        let mut cm = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                let val = if i == shape_idx {
                    0.5 * (x * 8.0 * std::f64::consts::PI).sin()
                } else {
                    (x * std::f64::consts::PI).sin() + 0.01 * i as f64
                };
                cm[i + t * n] = val;
            }
        }
        let data = FdMatrix::from_column_major(cm, n, m).unwrap();
        let res = total_variation_depth_1d(&data, &data).unwrap();
        // The shape outlier's derivative ranks are extreme at every interval → low MSS.
        let mut min_idx = 0usize;
        for i in 1..n {
            if res.mss[i] < res.mss[min_idx] {
                min_idx = i;
            }
        }
        assert_eq!(
            min_idx, shape_idx,
            "shape outlier should have the lowest MSS"
        );
    }

    #[test]
    fn dispatch_projects_tvd_field() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::TotalVariation).unwrap();
        let res = total_variation_depth_1d(&data, &data).unwrap();
        assert_eq!(got, res.tvd);
    }

    #[test]
    fn empty_and_too_few_curves_return_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(total_variation_depth_1d(&empty, &empty).is_err());
        let two = sample(2, 8);
        assert!(total_variation_depth_1d(&two, &two).is_err()); // n < 3
        assert!(functional_depth(&two, DepthMethod::TotalVariation).is_err());
    }
}
