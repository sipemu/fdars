//! Registration failure detection and alignment diagnostics.

use super::quality::{warp_complexity, warp_smoothness};
use super::{AlignmentResult, KarcherMeanResult};
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

// ─── Types ───────────────────────────────────────────────────────────────────

/// Diagnostic information for a single curve's alignment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentDiagnostic {
    /// Index of the curve in the original dataset (or 0 for pairwise).
    pub curve_index: usize,
    /// Geodesic distance from the warp to the identity.
    pub warp_complexity: f64,
    /// Bending energy of the warp.
    pub warp_smoothness: f64,
    /// True if the residual is barely reduced (possible under-alignment).
    pub is_under_aligned: bool,
    /// True if warp complexity exceeds the threshold (possible over-alignment).
    pub is_over_aligned: bool,
    /// True if the warp contains a non-monotone segment.
    pub has_non_monotone: bool,
    /// Post-alignment L2 residual (weighted).
    pub residual: f64,
    /// Ratio of post-alignment residual to pre-alignment distance.
    pub distance_ratio: f64,
    /// True if any issue was detected.
    pub flagged: bool,
    /// Human-readable issue descriptions.
    pub issues: Vec<String>,
}

/// Configuration for alignment diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticConfig {
    /// Warp complexity above which the curve is flagged as over-aligned.
    pub over_alignment_threshold: f64,
    /// Distance ratio below which the curve is flagged as under-aligned
    /// (i.e. the alignment barely improved the fit).
    pub under_alignment_threshold: f64,
    /// Maximum bending energy before the warp is considered too irregular.
    pub max_bending_energy: f64,
    /// Minimum improvement ratio (residual / pre-distance) to avoid flagging.
    pub min_improvement_ratio: f64,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            over_alignment_threshold: 1.0,
            under_alignment_threshold: 1e-6,
            max_bending_energy: 100.0,
            min_improvement_ratio: 0.5,
        }
    }
}

/// Summary of diagnostics across all curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentDiagnosticSummary {
    /// Per-curve diagnostics.
    pub diagnostics: Vec<AlignmentDiagnostic>,
    /// Indices of flagged curves.
    pub flagged_indices: Vec<usize>,
    /// Number of flagged curves.
    pub n_flagged: usize,
    /// Overall health score in [0, 1]: fraction of curves that are *not* flagged.
    pub health_score: f64,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Weighted L2 distance between two slices using pre-computed Simpson weights.
fn weighted_l2(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d * weights[i];
    }
    sum.sqrt()
}

/// Check monotonicity of a warp: returns true if any gamma[j+1] < gamma[j].
fn is_non_monotone(gamma: &[f64]) -> bool {
    gamma.windows(2).any(|w| w[1] < w[0])
}

/// Build a diagnostic for one curve given its warp, pre-distance, and residual.
fn build_diagnostic(
    curve_index: usize,
    gamma: &[f64],
    argvals: &[f64],
    pre_distance: f64,
    residual: f64,
    config: &DiagnosticConfig,
) -> AlignmentDiagnostic {
    let wc = warp_complexity(gamma, argvals);
    let ws = warp_smoothness(gamma, argvals);
    let non_mono = is_non_monotone(gamma);

    let distance_ratio = if pre_distance > 1e-15 {
        residual / pre_distance
    } else {
        0.0
    };

    let is_over = wc > config.over_alignment_threshold;
    let is_under = distance_ratio > config.min_improvement_ratio
        && pre_distance > config.under_alignment_threshold;

    let mut issues = Vec::new();
    if is_over {
        issues.push(format!(
            "warp complexity {wc:.4} exceeds threshold {}",
            config.over_alignment_threshold
        ));
    }
    if is_under {
        issues.push(format!(
            "distance ratio {distance_ratio:.4} exceeds improvement threshold {}",
            config.min_improvement_ratio
        ));
    }
    if non_mono {
        issues.push("warp contains non-monotone segments".to_string());
    }
    if ws > config.max_bending_energy {
        issues.push(format!(
            "bending energy {ws:.2} exceeds threshold {}",
            config.max_bending_energy
        ));
    }

    let flagged = !issues.is_empty();

    AlignmentDiagnostic {
        curve_index,
        warp_complexity: wc,
        warp_smoothness: ws,
        is_under_aligned: is_under,
        is_over_aligned: is_over,
        has_non_monotone: non_mono,
        residual,
        distance_ratio,
        flagged,
        issues,
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Diagnose alignment quality for every curve after a Karcher mean computation.
///
/// For each curve the function computes warp complexity, smoothness, pre- and
/// post-alignment residuals, and checks for non-monotone warps and insufficient
/// improvement. Curves with any issue are flagged.
///
/// # Arguments
/// * `data`    — Original (unaligned) functional data (n x m).
/// * `karcher` — Result of [`super::karcher::karcher_mean`].
/// * `argvals` — Evaluation grid (length m).
/// * `config`  — Diagnostic thresholds.
///
/// # Errors
/// Returns `FdarError::InvalidDimension` on shape mismatches.
pub fn diagnose_alignment(
    data: &FdMatrix,
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    config: &DiagnosticConfig,
) -> Result<AlignmentDiagnosticSummary, FdarError> {
    let (n, m) = data.shape();

    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if karcher.gammas.nrows() != n || karcher.gammas.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "karcher.gammas",
            expected: format!("{n} x {m}"),
            actual: format!("{} x {}", karcher.gammas.nrows(), karcher.gammas.ncols()),
        });
    }
    if karcher.aligned_data.nrows() != n || karcher.aligned_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "karcher.aligned_data",
            expected: format!("{n} x {m}"),
            actual: format!(
                "{} x {}",
                karcher.aligned_data.nrows(),
                karcher.aligned_data.ncols()
            ),
        });
    }
    if karcher.mean.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "karcher.mean",
            expected: format!("{m}"),
            actual: format!("{}", karcher.mean.len()),
        });
    }

    let weights = simpsons_weights(argvals);

    let mut diagnostics = Vec::with_capacity(n);
    let mut flagged_indices = Vec::new();

    for i in 0..n {
        let gamma_i: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();

        // Pre-alignment distance: ||f_i - mean||
        let fi = data.row(i);
        let pre_distance = weighted_l2(&fi, &karcher.mean, &weights);

        // Post-alignment residual: ||f_i_aligned - mean||
        let fi_aligned = karcher.aligned_data.row(i);
        let residual = weighted_l2(&fi_aligned, &karcher.mean, &weights);

        let diag = build_diagnostic(i, &gamma_i, argvals, pre_distance, residual, config);
        if diag.flagged {
            flagged_indices.push(i);
        }
        diagnostics.push(diag);
    }

    let n_flagged = flagged_indices.len();
    let health_score = if n > 0 {
        1.0 - n_flagged as f64 / n as f64
    } else {
        1.0
    };

    Ok(AlignmentDiagnosticSummary {
        diagnostics,
        flagged_indices,
        n_flagged,
        health_score,
    })
}

/// Diagnose a single pairwise alignment.
///
/// Examines the warp produced by [`super::pairwise::elastic_align_pair`] and
/// checks for over-alignment, under-alignment, non-monotonicity, and excessive
/// bending energy.
pub fn diagnose_pairwise(
    f1: &[f64],
    f2: &[f64],
    result: &AlignmentResult,
    argvals: &[f64],
    config: &DiagnosticConfig,
) -> AlignmentDiagnostic {
    let weights = simpsons_weights(argvals);

    // Pre-alignment L2 distance
    let pre_distance = weighted_l2(f1, f2, &weights);

    // Post-alignment residual: ||f1 - f2_aligned||
    let residual = weighted_l2(f1, &result.f_aligned, &weights);

    build_diagnostic(0, &result.gamma, argvals, pre_distance, residual, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::karcher_mean;
    use crate::alignment::pairwise::elastic_align_pair;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(99));
        (data, t)
    }

    #[test]
    fn diagnose_alignment_smoke() {
        let (data, t) = make_data(8, 30);
        let km = karcher_mean(&data, &t, 5, 1e-2, 0.0);
        let config = DiagnosticConfig::default();
        let summary = diagnose_alignment(&data, &km, &t, &config).unwrap();
        assert_eq!(summary.diagnostics.len(), 8);
        assert!(summary.health_score >= 0.0 && summary.health_score <= 1.0);
        assert_eq!(summary.n_flagged, summary.flagged_indices.len());
    }

    #[test]
    fn diagnose_alignment_identical_returns_low_complexity() {
        // When data is identical, warp complexity should be small even though
        // post-centering numerics may not yield exactly the identity warp.
        let t = uniform_grid(30);
        let curve: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let mut vals = Vec::with_capacity(5 * 30);
        for _ in 0..5 {
            vals.extend_from_slice(&curve);
        }
        let data = FdMatrix::from_column_major(vals, 5, 30).unwrap();
        let km = karcher_mean(&data, &t, 5, 1e-3, 0.0);
        let config = DiagnosticConfig::default();
        let summary = diagnose_alignment(&data, &km, &t, &config).unwrap();
        assert_eq!(summary.diagnostics.len(), 5);
        // All warp complexities should be small (near identity)
        for d in &summary.diagnostics {
            assert!(
                d.warp_complexity < 0.5,
                "curve {} warp_complexity {} should be small for identical data",
                d.curve_index,
                d.warp_complexity,
            );
        }
    }

    #[test]
    fn diagnose_alignment_rejects_shape_mismatch() {
        let (data, t) = make_data(6, 30);
        let km = karcher_mean(&data, &t, 3, 1e-2, 0.0);
        let bad_t = uniform_grid(20);
        let config = DiagnosticConfig::default();
        assert!(diagnose_alignment(&data, &km, &bad_t, &config).is_err());
    }

    #[test]
    fn diagnose_pairwise_smoke() {
        let t = uniform_grid(30);
        let f1: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();
        let f2: Vec<f64> = t.iter().map(|&x| ((x + 0.15) * 6.0).sin()).collect();
        let alignment = elastic_align_pair(&f1, &f2, &t, 0.0);
        let config = DiagnosticConfig::default();
        let diag = diagnose_pairwise(&f1, &f2, &alignment, &t, &config);
        assert!(diag.warp_complexity >= 0.0);
        assert!(diag.residual >= 0.0);
    }

    #[test]
    fn diagnose_pairwise_identical() {
        let t = uniform_grid(30);
        let f: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let alignment = elastic_align_pair(&f, &f, &t, 0.0);
        let config = DiagnosticConfig::default();
        let diag = diagnose_pairwise(&f, &f, &alignment, &t, &config);
        assert!(
            diag.residual < 1e-3,
            "identical curves should have near-zero residual"
        );
        assert!(!diag.has_non_monotone);
    }
}
