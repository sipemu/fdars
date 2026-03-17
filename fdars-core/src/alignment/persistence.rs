//! Peak persistence diagram for choosing the alignment regularisation parameter.

use super::karcher::karcher_mean;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Result of the peak persistence analysis across a sweep of lambda values.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PersistenceDiagramResult {
    /// Lambda values evaluated.
    pub lambdas: Vec<f64>,
    /// Number of peaks in the Karcher mean at each lambda.
    pub peak_counts: Vec<usize>,
    /// Persistence pairs: (birth_lambda_index, death_lambda_index).
    ///
    /// Each pair describes a maximal interval where the peak count is constant.
    pub persistence_pairs: Vec<(usize, usize)>,
    /// Optimal lambda (center of the longest stable interval).
    pub optimal_lambda: f64,
    /// Index into `lambdas` for the optimal value.
    pub optimal_index: usize,
}

/// Count peaks in a curve with a small prominence threshold.
///
/// A local maximum at index j (1 <= j < m-1) is counted when
/// `mean[j-1] < mean[j] && mean[j] > mean[j+1]` and the prominence
/// (relative to the curve's range) exceeds a small threshold.
fn count_peaks(mean: &[f64], prominence_frac: f64) -> usize {
    let m = mean.len();
    if m < 3 {
        return 0;
    }

    let min_val = mean.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = mean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let threshold = prominence_frac * range;

    let mut count = 0;
    for j in 1..m - 1 {
        if mean[j] > mean[j - 1] && mean[j] > mean[j + 1] {
            // Check prominence: min height above the two neighbors
            let prom = (mean[j] - mean[j - 1]).min(mean[j] - mean[j + 1]);
            if prom > threshold {
                count += 1;
            }
        }
    }
    count
}

/// Build persistence pairs from peak counts: maximal constant intervals.
fn build_persistence_pairs(peak_counts: &[usize]) -> Vec<(usize, usize)> {
    if peak_counts.is_empty() {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    let mut start = 0;
    for i in 1..peak_counts.len() {
        if peak_counts[i] != peak_counts[start] {
            pairs.push((start, i - 1));
            start = i;
        }
    }
    pairs.push((start, peak_counts.len() - 1));
    pairs
}

/// Analyse the stability of peak count in the Karcher mean across a sweep
/// of alignment penalty values.
///
/// For each candidate lambda the Karcher mean is computed and its peaks are
/// counted. The longest interval of constant peak count is identified as the
/// most stable configuration, and the midpoint lambda of that interval is
/// returned as the optimal choice.
///
/// # Arguments
/// * `data`     - Functional data matrix (n x m).
/// * `argvals`  - Evaluation grid (length m).
/// * `lambdas`  - Candidate lambda values to sweep (must be non-empty, all >= 0).
/// * `max_iter` - Maximum Karcher iterations per lambda.
/// * `tol`      - Karcher convergence tolerance.
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if `data` has fewer than 2 rows or
/// `argvals` length mismatches.
/// Returns `FdarError::InvalidParameter` if `lambdas` is empty, any lambda
/// is negative, or `max_iter` is 0.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn peak_persistence(
    data: &FdMatrix,
    argvals: &[f64],
    lambdas: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<PersistenceDiagramResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();

    // ── Validation ──────────────────────────────────────────────────────
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if lambdas.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "lambdas",
            message: "must be non-empty".to_string(),
        });
    }
    if lambdas.iter().any(|&l| l < 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "lambdas",
            message: "all lambda values must be >= 0".to_string(),
        });
    }
    if max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "must be > 0".to_string(),
        });
    }

    // ── Lambda sweep ────────────────────────────────────────────────────
    let mut peak_counts = Vec::with_capacity(lambdas.len());

    for &lam in lambdas {
        let result = karcher_mean(data, argvals, max_iter, tol, lam);
        let count = count_peaks(&result.mean, 0.001);
        peak_counts.push(count);
    }

    // ── Build persistence pairs ─────────────────────────────────────────
    let persistence_pairs = build_persistence_pairs(&peak_counts);

    // ── Find optimal lambda (longest stable interval) ───────────────────
    let (best_start, best_end) = persistence_pairs
        .iter()
        .copied()
        .max_by_key(|&(s, e)| {
            // Use the span in lambda space as the primary criterion,
            // discretised to avoid floating-point comparison issues.
            let span = lambdas[e] - lambdas[s];
            // Convert to an integer score (nano-units) for stable ordering
            (span * 1e9) as u64
        })
        .unwrap_or((0, 0));

    let optimal_index = (best_start + best_end) / 2;
    let optimal_lambda = lambdas[optimal_index];

    Ok(PersistenceDiagramResult {
        lambdas: lambdas.to_vec(),
        peak_counts,
        persistence_pairs,
        optimal_lambda,
        optimal_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Build a small dataset with one clear sine peak per curve.
    fn single_peak_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let mut data_vec = vec![0.0; n * m];
        for i in 0..n {
            let shift = 0.05 * i as f64;
            for j in 0..m {
                // sin(pi * t) has exactly one peak on [0,1]
                data_vec[i + j * n] = (std::f64::consts::PI * (t[j] + shift)).sin();
            }
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        (data, t)
    }

    #[test]
    fn persistence_single_peak_stable() {
        let (data, t) = single_peak_data(6, 31);
        let lambdas = vec![0.0, 0.01, 0.1, 1.0];

        let result = peak_persistence(&data, &t, &lambdas, 5, 1e-2).unwrap();

        // All (or most) peak counts should be 1
        let count_one = result.peak_counts.iter().filter(|&&c| c == 1).count();
        assert!(
            count_one >= lambdas.len() / 2,
            "Expected most peak counts to be 1, got {:?}",
            result.peak_counts
        );
    }

    #[test]
    fn persistence_optimal_in_range() {
        let (data, t) = single_peak_data(6, 31);
        let lambdas = vec![0.0, 0.01, 0.1, 1.0, 10.0];

        let result = peak_persistence(&data, &t, &lambdas, 5, 1e-2).unwrap();

        assert!(
            result.optimal_lambda >= lambdas[0],
            "optimal_lambda {} below range",
            result.optimal_lambda
        );
        assert!(
            result.optimal_lambda <= *lambdas.last().unwrap(),
            "optimal_lambda {} above range",
            result.optimal_lambda
        );
    }

    #[test]
    fn persistence_peak_counts_length() {
        let (data, t) = single_peak_data(6, 31);
        let lambdas = vec![0.0, 0.5, 1.0];

        let result = peak_persistence(&data, &t, &lambdas, 3, 1e-2).unwrap();
        assert_eq!(result.peak_counts.len(), lambdas.len());
    }

    #[test]
    fn persistence_rejects_empty_lambdas() {
        let (data, t) = single_peak_data(4, 21);
        let result = peak_persistence(&data, &t, &[], 5, 1e-3);
        assert!(result.is_err(), "Empty lambdas should produce an error");
    }
}
