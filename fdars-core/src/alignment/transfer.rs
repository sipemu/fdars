//! Transfer alignment: align curves across populations using a shared reference.

use super::karcher::karcher_mean;
use super::pairwise::{elastic_align_pair, elastic_distance};
use super::srsf::{compose_warps, reparameterize_curve};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for transfer alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferAlignConfig {
    /// Roughness penalty for elastic alignment.
    pub lambda: f64,
    /// Maximum Karcher mean iterations.
    pub max_iter: usize,
    /// Convergence tolerance for the Karcher mean.
    pub tol: f64,
}

impl Default for TransferAlignConfig {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            max_iter: 15,
            tol: 1e-3,
        }
    }
}

/// Result of transfer alignment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TransferAlignResult {
    /// Source Karcher mean (population A's reference).
    pub source_mean: Vec<f64>,
    /// Target curves aligned to source coordinate system (n_target x m).
    pub aligned_data: FdMatrix,
    /// Warping functions mapping target curves to source frame (n_target x m).
    pub gammas: FdMatrix,
    /// Bridging warp from target mean to source mean.
    pub bridging_gamma: Vec<f64>,
    /// Per-curve elastic distances after alignment.
    pub distances: Vec<f64>,
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Align curves from a target population to a source population's coordinate system.
///
/// Computes Karcher means for both populations, finds the bridging warp that
/// aligns the target mean to the source mean, then composes this bridge with
/// each target curve's within-population warp to produce curves aligned in
/// the source coordinate frame.
///
/// # Arguments
/// * `source_data` - Source population (n_source x m).
/// * `target_data` - Target population to align (n_target x m).
/// * `argvals`     - Evaluation points (length m).
/// * `config`      - Transfer alignment configuration.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if matrices have different `ncols`,
/// `argvals` length does not match, or either matrix has 0 rows.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn transfer_alignment(
    source_data: &FdMatrix,
    target_data: &FdMatrix,
    argvals: &[f64],
    config: &TransferAlignConfig,
) -> Result<TransferAlignResult, FdarError> {
    let (n_source, m_source) = source_data.shape();
    let (n_target, m_target) = target_data.shape();

    // ── Validation ──
    if m_source != m_target {
        return Err(FdarError::InvalidDimension {
            parameter: "target_data",
            expected: format!("{m_source} columns (matching source_data)"),
            actual: format!("{m_target} columns"),
        });
    }
    let m = m_source;
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n_source < 1 {
        return Err(FdarError::InvalidDimension {
            parameter: "source_data",
            expected: "at least 1 row".to_string(),
            actual: format!("{n_source} rows"),
        });
    }
    if n_target < 1 {
        return Err(FdarError::InvalidDimension {
            parameter: "target_data",
            expected: "at least 1 row".to_string(),
            actual: format!("{n_target} rows"),
        });
    }

    // ── Compute source reference ──
    let source_karcher = karcher_mean(
        source_data,
        argvals,
        config.max_iter,
        config.tol,
        config.lambda,
    );

    // ── Compute target reference ──
    let target_karcher = karcher_mean(
        target_data,
        argvals,
        config.max_iter,
        config.tol,
        config.lambda,
    );

    // ── Bridging alignment: align target mean to source mean ──
    let bridge_result = elastic_align_pair(
        &source_karcher.mean,
        &target_karcher.mean,
        argvals,
        config.lambda,
    );

    // ── Align target curves ──
    // For each target curve: compose bridging warp with within-population warp,
    // then apply to original target curve.
    let results: Vec<(Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n_target)
        .map(|i| {
            // Within-population warp for curve i (from target Karcher computation)
            let within_gamma = target_karcher.gammas.row(i);

            // Compose: bridge_gamma(within_gamma(t))
            let gamma_total = compose_warps(&bridge_result.gamma, &within_gamma, argvals);

            // Apply to original target curve
            let aligned_i = reparameterize_curve(&target_data.row(i), argvals, &gamma_total);

            // Compute distance to source mean
            let dist_i = elastic_distance(&source_karcher.mean, &aligned_i, argvals, config.lambda);

            (gamma_total, aligned_i, dist_i)
        })
        .collect();

    // ── Assemble result ──
    let mut gammas = FdMatrix::zeros(n_target, m);
    let mut aligned_data = FdMatrix::zeros(n_target, m);
    let mut distances = Vec::with_capacity(n_target);

    for (i, (gamma, aligned, dist)) in results.into_iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = gamma[j];
            aligned_data[(i, j)] = aligned[j];
        }
        distances.push(dist);
    }

    Ok(TransferAlignResult {
        source_mean: source_karcher.mean,
        aligned_data,
        gammas,
        bridging_gamma: bridge_result.gamma,
        distances,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(
            n,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(seed),
        );
        (data, t)
    }

    #[test]
    fn transfer_same_population() {
        let (data, t) = make_data(8, 20, 42);
        let config = TransferAlignConfig {
            max_iter: 5,
            tol: 1e-2,
            ..Default::default()
        };
        let result = transfer_alignment(&data, &data, &t, &config).unwrap();

        // Bridging warp should be close to identity
        let max_dev: f64 = result
            .bridging_gamma
            .iter()
            .zip(t.iter())
            .map(|(&g, &ti)| (g - ti).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 0.3,
            "bridging warp should be near identity for same population, max_dev={max_dev}"
        );

        // Distances should be small
        for (i, &d) in result.distances.iter().enumerate() {
            assert!(
                d < 5.0,
                "distance[{i}]={d} should be small for same-population transfer"
            );
        }
    }

    #[test]
    fn transfer_shifted_population() {
        let (source, t) = make_data(8, 20, 42);
        let m = t.len();
        let n = source.nrows();

        // Create a shifted version of source
        let mut target = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                target[(i, j)] = source[(i, j)] + 2.0;
            }
        }

        let config = TransferAlignConfig {
            max_iter: 5,
            tol: 1e-2,
            ..Default::default()
        };
        let result = transfer_alignment(&source, &target, &t, &config).unwrap();

        // After alignment, the aligned target curves should be closer to the
        // source mean than the raw target curves
        let source_mean = &result.source_mean;
        let raw_mean_dist: f64 = (0..m)
            .map(|j| {
                let diff = target[(0, j)] - source_mean[j];
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        let aligned_mean_dist: f64 = (0..m)
            .map(|j| {
                let diff = result.aligned_data[(0, j)] - source_mean[j];
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        // The aligned version should not be worse than raw (with some tolerance
        // since the shift is in amplitude and alignment is mainly phase)
        assert!(
            aligned_mean_dist < raw_mean_dist + 1.0,
            "aligned dist ({aligned_mean_dist:.2}) should not be much worse than raw dist ({raw_mean_dist:.2})"
        );
    }

    #[test]
    fn transfer_output_dimensions() {
        let (source, t) = make_data(6, 20, 42);
        let (target, _) = make_data(10, 20, 99);
        let config = TransferAlignConfig {
            max_iter: 3,
            tol: 1e-2,
            ..Default::default()
        };
        let result = transfer_alignment(&source, &target, &t, &config).unwrap();

        assert_eq!(result.aligned_data.shape(), (10, 20));
        assert_eq!(result.gammas.shape(), (10, 20));
        assert_eq!(result.distances.len(), 10);
        assert_eq!(result.source_mean.len(), 20);
        assert_eq!(result.bridging_gamma.len(), 20);
    }

    #[test]
    fn transfer_config_default() {
        let config = TransferAlignConfig::default();
        assert!((config.lambda - 0.0).abs() < f64::EPSILON);
        assert_eq!(config.max_iter, 15);
        assert!((config.tol - 1e-3).abs() < f64::EPSILON);
    }
}
