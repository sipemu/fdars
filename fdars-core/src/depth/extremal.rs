//! Extremal depth (Narisetty & Nair 2016).
//!
//! Extremal depth ranks curves by the *worst* (minimum) pointwise centrality they
//! attain and how long they attain it, then converts that ordering into a depth in
//! `(0, 1]`. It is a **self-depth** measure evaluated on the reference sample
//! `data_ori`; `data_obj` is accepted for signature uniformity with the other depth
//! measures and must share the sample's grid.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Average-tie ranks for one column of a `FdMatrix`, as `f64` in `1..=n`
/// (matching R's `rank(ties.method = "average")`).
fn column_ranks(data: &FdMatrix, col: usize) -> Vec<f64> {
    let n = data.nrows();
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|i| (data[(i, col)], i)).collect();
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && indexed[j].0 == indexed[k].0 {
            j += 1;
        }
        // Tie group indices k..j → average of the 1-based ranks (k+1 .. j).
        let avg_rank = (k as f64 + 1.0 + j as f64) / 2.0;
        for item in &indexed[k..j] {
            ranks[item.1] = avg_rank;
        }
        k = j;
    }
    ranks
}

/// Compute the Extremal Depth (ED) for 1D functional data.
///
/// For each grid point `t`, the pointwise depth of curve `i` is
/// `D(i,t) = 1 − |2·rank(X_i(t)) − n − 1| / n` (median-ranked → ≈1, extreme → ≈0).
/// Each curve is then summarized by `d_level(i) = min_t D(i,t)` and
/// `mass(i) = fraction of t attaining that minimum`. Curves are ordered by
/// ascending `d_level`, then descending `mass` (more extreme first), and the depth
/// is the 1-based position in that order divided by `n`:
///
/// ```text
/// ED(i) = rank_in_ordering(i) / n     ∈ (0, 1]
/// ```
///
/// The ordering is made deterministic by breaking `(d_level, mass)` ties on the
/// original index. [CITED: fdaoutlier::extremal_depth]
///
/// This is a self-depth measure computed on `data_ori`; `data_obj` must share its
/// grid. Central curves score near 1.0; magnitude/shape outliers score near `1/n`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if either matrix is empty, if their grids
/// differ, or if the sample has fewer than 3 curves (the R reference requires n ≥ 3).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn extremal_depth_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Result<Vec<f64>, FdarError> {
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
            expected: "at least 3 curves for extremal depth".to_string(),
            actual: format!("{n}"),
        });
    }

    // Step 1: pointwise depth D(i,t) for every curve/point.
    let mut d = vec![vec![0.0_f64; m]; n];
    for t in 0..m {
        let ranks = column_ranks(data_ori, t);
        for i in 0..n {
            d[i][t] = 1.0 - (2.0 * ranks[i] - n as f64 - 1.0).abs() / n as f64;
        }
    }

    // Step 2: d_level (min pointwise depth) and mass (fraction attaining it).
    let eps = 1e-9;
    let mut d_level = vec![0.0_f64; n];
    let mut mass = vec![0.0_f64; n];
    for i in 0..n {
        let mut lvl = f64::INFINITY;
        for t in 0..m {
            if d[i][t] < lvl {
                lvl = d[i][t];
            }
        }
        let cnt = (0..m).filter(|&t| (d[i][t] - lvl).abs() < eps).count();
        d_level[i] = lvl;
        mass[i] = cnt as f64 / m as f64;
    }

    // Step 3: order by ascending d_level, then descending mass, then original index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        d_level[a]
            .partial_cmp(&d_level[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                mass[b]
                    .partial_cmp(&mass[a])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.cmp(&b))
    });

    // Step 4: depth = 1-based position in the ordering / n.
    let mut depth = vec![0.0_f64; n];
    for (k, &idx) in order.iter().enumerate() {
        depth[idx] = (k + 1) as f64 / n as f64;
    }

    Ok(depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::dispatch::{functional_depth, DepthMethod};

    /// `n` stacked sinusoids (offset 0.05 per index); middle rows are most central.
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

    /// Inliers plus a gross magnitude outlier at `outlier_idx`.
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
    fn depths_in_unit_interval_and_deepest_is_one() {
        let n = 9usize;
        let data = sample(n, 25);
        let ed = extremal_depth_1d(&data, &data).unwrap();
        assert_eq!(ed.len(), n);
        for &v in &ed {
            assert!(v > 0.0 && v <= 1.0 + 1e-12, "ED out of range: {v}");
        }
        let max = ed.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (max - 1.0).abs() < 1e-12,
            "deepest ED should be 1.0, got {max}"
        );
    }

    #[test]
    fn central_deepest_and_outlier_among_shallowest() {
        // Extremal depth is SYMMETRIC: a magnitude outlier at the top (rank n) and
        // the naturally-lowest boundary curve (rank 1) attain the same minimum
        // pointwise depth 1/n and therefore tie as the most extreme. So the outlier
        // is among the two shallowest, not uniquely so; the robust properties are
        // that the median curve is deepest (ED = 1.0) and the outlier is extreme.
        let outlier_idx = 3usize;
        let data = sample_with_outlier(9, 25, outlier_idx);
        let ed = extremal_depth_1d(&data, &data).unwrap();

        let mut deepest = 0usize;
        for i in 1..ed.len() {
            if ed[i] > ed[deepest] {
                deepest = i;
            }
        }
        assert!(
            (ed[deepest] - 1.0).abs() < 1e-12,
            "median curve should have ED 1.0, got {}",
            ed[deepest]
        );
        assert_ne!(deepest, outlier_idx, "outlier must not be deepest");
        // Ordered first or second among 9 → depth ≤ 2/9.
        assert!(
            ed[outlier_idx] <= 2.0 / 9.0 + 1e-12,
            "outlier should be among the most extreme, got {}",
            ed[outlier_idx]
        );
    }

    #[test]
    fn ordering_is_deterministic_across_runs() {
        // Two identical (tied) curves must resolve by original index, identically each call.
        let n = 5usize;
        let m = 10usize;
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                // Rows 1 and 2 are identical → tie.
                let offset = if i == 2 { 1.0 } else { i as f64 };
                col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.1 * offset;
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let a = extremal_depth_1d(&data, &data).unwrap();
        let b = extremal_depth_1d(&data, &data).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn dispatch_round_trip() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::Extremal).unwrap();
        assert_eq!(got, extremal_depth_1d(&data, &data).unwrap());
    }

    #[test]
    fn empty_and_too_few_curves_return_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(extremal_depth_1d(&empty, &empty).is_err());
        let two = sample(2, 8);
        assert!(extremal_depth_1d(&two, &two).is_err()); // n < 3
        assert!(functional_depth(&two, DepthMethod::Extremal).is_err());
    }
}
