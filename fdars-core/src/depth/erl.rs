//! Extreme-rank-length (ERL) depth.
//!
//! ERL depth orders curves by a reverse-lexicographic comparison of their sorted,
//! two-sided pointwise-rank vectors: a curve whose most-extreme ranks are more
//! extreme (and stay extreme longer) is ranked as more outlying. It is a
//! **self-depth** measure evaluated on the reference sample `data_ori`; `data_obj`
//! is accepted for signature uniformity and must share the sample's grid.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

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
        let avg_rank = (k as f64 + 1.0 + j as f64) / 2.0;
        for item in &indexed[k..j] {
            ranks[item.1] = avg_rank;
        }
        k = j;
    }
    ranks
}

/// Compute the Extreme-Rank-Length (ERL) Depth for 1D functional data.
///
/// For each grid point the curves are ranked (average ties), then each rank is
/// folded two-sided as `R[i,t] = min(r, n + 1 − r)` so that extremes (near 1 or n)
/// map to small values. Each curve's transformed rank vector is sorted ascending,
/// and curves are totally ordered by lexicographic comparison of these sorted
/// vectors (a curve with smaller leading entries is *more extreme*):
///
/// ```text
/// ERL(i) = #{ j : curve i is NOT more extreme than j } / n     ∈ (0, 1]
/// ```
///
/// A curve counts itself as not-more-extreme, so the least extreme (central) curve
/// scores `n/n = 1`. [CITED: fdaoutlier::extreme_rank_length]
///
/// This is a self-depth measure computed on `data_ori`; `data_obj` must share its
/// grid. Central curves score near 1.0; outliers score near 0.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if either matrix is empty, if their grids
/// differ, or if the sample has fewer than 2 curves.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn extreme_rank_length_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
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
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_ori",
            expected: "at least 2 curves for extreme-rank-length depth".to_string(),
            actual: format!("{n}"),
        });
    }

    // Steps 1-2: two-sided transformed ranks R[i][t].
    let mut r = vec![vec![0.0_f64; m]; n];
    for t in 0..m {
        let ranks = column_ranks(data_ori, t);
        for i in 0..n {
            r[i][t] = f64::min(ranks[i], (n + 1) as f64 - ranks[i]);
        }
    }

    // Step 3: each curve's transformed rank vector sorted ascending.
    let mut sorted_r: Vec<Vec<f64>> = r
        .into_iter()
        .map(|mut row| {
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row
        })
        .collect();
    // Move out of the Vec so the parallel closure can borrow it immutably.
    let sorted_r = std::mem::take(&mut sorted_r);

    // Step 4: for each curve i, count curves it is NOT more extreme than.
    const EPS: f64 = 1e-12;
    let depths: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut not_more_extreme = 0.0_f64;
            for j in 0..n {
                // i is "more extreme" than j iff at the first differing lex position
                // sorted_r[i] < sorted_r[j]. Otherwise i is NOT more extreme than j.
                let more_extreme = sorted_r[i]
                    .iter()
                    .zip(sorted_r[j].iter())
                    .find(|(a, b)| (**a - **b).abs() > EPS)
                    .map(|(a, b)| a < b)
                    .unwrap_or(false);
                if !more_extreme {
                    not_more_extreme += 1.0;
                }
            }
            not_more_extreme / n as f64
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
    fn depths_in_unit_interval_and_deepest_is_one() {
        let n = 9usize;
        let data = sample(n, 25);
        let erl = extreme_rank_length_depth_1d(&data, &data).unwrap();
        assert_eq!(erl.len(), n);
        for &v in &erl {
            assert!(v > 0.0 && v <= 1.0 + 1e-12, "ERL out of range: {v}");
        }
        let max = erl.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (max - 1.0).abs() < 1e-12,
            "deepest ERL should be 1.0, got {max}"
        );
    }

    #[test]
    fn central_deepest_and_outlier_among_shallowest() {
        // ERL folds ranks two-sided (min(r, n+1-r)), so the top magnitude outlier
        // (rank n) and the naturally-lowest boundary curve (rank 1) both fold to 1
        // and tie as most extreme. The robust properties: the median-rank curve is
        // the unique deepest (ERL = 1.0) and the outlier is among the shallowest.
        let outlier_idx = 3usize;
        let data = sample_with_outlier(9, 25, outlier_idx);
        let erl = extreme_rank_length_depth_1d(&data, &data).unwrap();

        let mut deepest = 0usize;
        for i in 1..erl.len() {
            if erl[i] > erl[deepest] {
                deepest = i;
            }
        }
        assert!(
            (erl[deepest] - 1.0).abs() < 1e-12,
            "median curve should have ERL 1.0, got {}",
            erl[deepest]
        );
        assert_ne!(deepest, outlier_idx, "outlier must not be deepest");
        // Outlier ties the bottom boundary curve at the minimum ERL 2/9.
        assert!(
            erl[outlier_idx] <= 2.0 / 9.0 + 1e-12,
            "outlier should be among the most extreme, got {}",
            erl[outlier_idx]
        );
    }

    #[test]
    fn dispatch_round_trip() {
        let data = sample(6, 12);
        let got = functional_depth(&data, DepthMethod::ExtremeRankLength).unwrap();
        assert_eq!(got, extreme_rank_length_depth_1d(&data, &data).unwrap());
    }

    #[test]
    fn empty_and_single_curve_return_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(extreme_rank_length_depth_1d(&empty, &empty).is_err());
        let one = sample(1, 8);
        assert!(extreme_rank_length_depth_1d(&one, &one).is_err()); // n < 2
        assert!(functional_depth(&one, DepthMethod::ExtremeRankLength).is_err());
    }
}
