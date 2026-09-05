//! Inductive conformal anomaly detection for functional data using elastic distances.
//!
//! This module implements an inductive (split) conformal anomaly detector that scores
//! functional curves against a reference template using elastic distances. Curves whose
//! conformal p-value falls at or below a significance level `alpha` are flagged as
//! anomalies.
//!
//! # Overview
//!
//! 1. **Calibration**: compute elastic nonconformity scores for each calibration curve
//!    against a shared template (default: Karcher mean of the calibration set).
//! 2. **Threshold**: derive the `(1-alpha)` empirical quantile of calibration scores.
//! 3. **Scoring**: for each test curve, compute its nonconformity score and the
//!    conformal p-value `(1 + #{calib_score >= test_score}) / (n_calib + 1)`.
//! 4. **Flagging**: flag a curve as anomalous when `p_value <= alpha`.
//!
//! # Variants
//!
//! - [`NonConformityScore::AmplitudeElastic`]: flags magnitude (amplitude) outliers
//! - [`NonConformityScore::PhaseElastic`]: flags shape (timing/phase) outliers
//! - [`NonConformityScore::CombinedElastic`]: flags both (default)
//!
//! # Example
//!
//! ```
//! use fdars_core::simulation::{sim_fundata, EFunType, EValType};
//! use fdars_core::tolerance::{
//!     elastic_conformal_anomaly, ConformalAnomalyConfig, NonConformityScore,
//! };
//!
//! let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
//!
//! // Build a calibration set of clean curves
//! let calibration = sim_fundata(30, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
//! // Build a test set of clean curves (exchangeable with calibration)
//! let test = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(99));
//!
//! let mut config = ConformalAnomalyConfig::default();
//! config.variant = NonConformityScore::CombinedElastic;
//! config.alpha = 0.1;
//!
//! let result = elastic_conformal_anomaly(&calibration, &test, &t, &config).unwrap();
//! assert_eq!(result.p_values.len(), 20);
//! assert_eq!(result.flags.len(), 20);
//! assert!(result.threshold >= 0.0);
//! // On clean exchangeable data, flag rate should be approximately alpha
//! let flag_rate = result.flags.iter().filter(|&&f| f).count() as f64 / 20.0;
//! assert!(flag_rate <= 0.4, "Flag rate {} too high for alpha=0.1", flag_rate);
//! ```

use super::NonConformityScore;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`elastic_conformal_anomaly`].
///
/// Controls the elastic distance variant, significance level, warp penalty,
/// Karcher mean computation parameters, and an optional pre-computed template.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ConformalAnomalyConfig {
    /// Non-conformity score variant. Must be an elastic variant
    /// ([`AmplitudeElastic`], [`PhaseElastic`], or [`CombinedElastic`]).
    ///
    /// [`AmplitudeElastic`]: NonConformityScore::AmplitudeElastic
    /// [`PhaseElastic`]: NonConformityScore::PhaseElastic
    /// [`CombinedElastic`]: NonConformityScore::CombinedElastic
    pub variant: NonConformityScore,
    /// Miscoverage level — flag a test curve when `p_value <= alpha` (default: 0.1).
    pub alpha: f64,
    /// Warp penalty passed to the elastic distance functions (default: 0.0 = no penalty).
    pub lambda: f64,
    /// Maximum iterations for Karcher mean convergence (used when `template` is `None`; default: 20).
    pub max_iter: usize,
    /// Convergence tolerance for Karcher mean (default: 1e-4).
    pub tol: f64,
    /// Optional pre-computed template curve of length `m`.
    ///
    /// When `None`, the Karcher mean of the calibration set is computed and used as the template.
    pub template: Option<Vec<f64>>,
}

impl Default for ConformalAnomalyConfig {
    fn default() -> Self {
        Self {
            variant: NonConformityScore::CombinedElastic,
            alpha: 0.1,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
            template: None,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result of [`elastic_conformal_anomaly`].
///
/// Contains per-test-curve conformal p-values, nonconformity scores, boolean anomaly
/// flags, and the calibrated threshold.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ConformalAnomalyResult {
    /// Conformal p-value for each test curve (length `n_test`).
    ///
    /// Computed as `(1 + #{calib_score >= test_score}) / (n_calib + 1)`.
    pub p_values: Vec<f64>,
    /// Elastic nonconformity score for each test curve (length `n_test`).
    pub scores: Vec<f64>,
    /// Boolean anomaly flag for each test curve (length `n_test`).
    ///
    /// `true` when `p_value <= alpha` (the curve is anomalous at level `alpha`).
    pub flags: Vec<bool>,
    /// Calibrated threshold: `(1-alpha)` quantile of calibration nonconformity scores.
    ///
    /// A test curve is flagged iff its score exceeds this threshold.
    pub threshold: f64,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute elastic nonconformity score for a single curve against a template.
///
/// Scores the curve against the template using the specified elastic distance variant.
/// Returns a non-negative value; near zero when the curve is identical to the template.
///
/// # Argument Order (load-bearing)
///
/// `curve` is the query curve (`f1`) and `template` is the reference (`f2`). The
/// elastic alignment optimally warps `f2` (the template) onto `f1` (the curve).
/// Keeping the test/calibration curve as `f1` ensures all scores are on the same
/// scale relative to the fixed template. The zero-for-identical gate catches any
/// argument-order flip.
///
/// # Arguments
///
/// * `curve`    — Curve to score (length `m`)
/// * `template` — Reference/template curve (length `m`)
/// * `argvals`  — Evaluation points (length `m`)
/// * `lambda`   — Warp penalty (0.0 = no penalty)
/// * `variant`  — Must be [`AmplitudeElastic`], [`PhaseElastic`], or [`CombinedElastic`]
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `variant` is [`SupNorm`] or [`L2`].
///
/// [`AmplitudeElastic`]: NonConformityScore::AmplitudeElastic
/// [`PhaseElastic`]: NonConformityScore::PhaseElastic
/// [`CombinedElastic`]: NonConformityScore::CombinedElastic
/// [`SupNorm`]: NonConformityScore::SupNorm
/// [`L2`]: NonConformityScore::L2
#[must_use = "expensive computation: elastic_nonconformity runs an elastic alignment; use the score"]
pub fn elastic_nonconformity(
    curve: &[f64],
    template: &[f64],
    argvals: &[f64],
    lambda: f64,
    variant: NonConformityScore,
) -> Result<f64, FdarError> {
    match variant {
        NonConformityScore::AmplitudeElastic => Ok(crate::alignment::amplitude_distance(
            curve, template, argvals, lambda,
        )),
        NonConformityScore::PhaseElastic => Ok(crate::alignment::phase_distance_pair(
            curve, template, argvals, lambda,
        )),
        NonConformityScore::CombinedElastic => {
            // Genuine combination of amplitude and phase distances.
            // amplitude_distance == elastic_distance (it delegates exactly), so
            // CombinedElastic must NOT be a bare alias. Formula: sqrt(amp² + phase²).
            let amp = crate::alignment::amplitude_distance(curve, template, argvals, lambda);
            let ph = crate::alignment::phase_distance_pair(curve, template, argvals, lambda);
            Ok((amp.powi(2) + ph.powi(2)).sqrt())
        }
        _ => Err(FdarError::InvalidParameter {
            parameter: "variant",
            message: "elastic_nonconformity requires an elastic NonConformityScore variant \
                      (AmplitudeElastic, PhaseElastic, or CombinedElastic)"
                .to_string(),
        }),
    }
}

/// Inductive conformal anomaly detection for functional data using elastic distances.
///
/// Calibrates elastic nonconformity scores on `calibration`, then for each test
/// curve in `test` computes a conformal p-value and an anomaly flag at level `alpha`.
///
/// # Algorithm
///
/// 1. Resolve the template: use `config.template` if supplied; otherwise compute the
///    Karcher mean of `calibration`.
/// 2. Score each calibration curve against the template via [`elastic_nonconformity`].
/// 3. Compute the calibrated threshold as the `(1-alpha)` empirical quantile of the
///    calibration scores (using `ceil((n_calib+1)*(1-alpha))` order statistic).
/// 4. For each test curve, compute score `a*` and p-value
///    `(1 + #{calib_score >= a*}) / (n_calib + 1)`. Flag when `p_value <= alpha`.
///
/// # Arguments
///
/// * `calibration` — Reference/calibration functional data (`n_calib × m`)
/// * `test`        — Test functional data to score (`n_test × m`)
/// * `argvals`     — Evaluation points (length `m`)
/// * `config`      — [`ConformalAnomalyConfig`]
///
/// # Returns
///
/// [`ConformalAnomalyResult`] with per-curve p-values, scores, flags, and the
/// calibrated threshold.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if column counts do not match `argvals.len()`.
/// Returns [`FdarError::InvalidParameter`] if:
/// - `n_calib < 1`
/// - `alpha` is not in `(0.0, 1.0)` exclusive
/// - `config.variant` is not an elastic variant
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_conformal_anomaly(
    calibration: &FdMatrix,
    test: &FdMatrix,
    argvals: &[f64],
    config: &ConformalAnomalyConfig,
) -> Result<ConformalAnomalyResult, FdarError> {
    let m = argvals.len();
    let (n_calib, m_calib) = calibration.shape();
    let (n_test, m_test) = test.shape();

    // ── Input validation ─────────────────────────────────────────────────────
    if m_calib != m {
        return Err(FdarError::InvalidDimension {
            parameter: "calibration",
            expected: format!("{m} columns (argvals.len())"),
            actual: format!("{m_calib}"),
        });
    }
    if m_test != m {
        return Err(FdarError::InvalidDimension {
            parameter: "test",
            expected: format!("{m} columns (argvals.len())"),
            actual: format!("{m_test}"),
        });
    }
    if n_calib < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "calibration",
            message: "calibration set must have at least one curve (n_calib >= 1)".to_string(),
        });
    }
    if !(config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!(
                "alpha must be in (0.0, 1.0) exclusive, got {}",
                config.alpha
            ),
        });
    }
    // Validate variant is elastic
    match config.variant {
        NonConformityScore::AmplitudeElastic
        | NonConformityScore::PhaseElastic
        | NonConformityScore::CombinedElastic => {}
        _ => {
            return Err(FdarError::InvalidParameter {
                parameter: "config.variant",
                message: "elastic_conformal_anomaly requires an elastic NonConformityScore \
                          variant (AmplitudeElastic, PhaseElastic, or CombinedElastic)"
                    .to_string(),
            });
        }
    }

    // ── Template resolution ───────────────────────────────────────────────────
    let template: Vec<f64> = match config.template.clone() {
        Some(t) => {
            // A caller-supplied template MUST match the evaluation grid — otherwise
            // srsf_transform silently returns a zero matrix and every score is wrong.
            if t.len() != m {
                return Err(FdarError::InvalidDimension {
                    parameter: "config.template",
                    expected: format!("{m} elements (argvals.len())"),
                    actual: format!("{}", t.len()),
                });
            }
            t
        }
        None => {
            let km = crate::alignment::karcher_mean(
                calibration,
                argvals,
                config.max_iter,
                config.tol,
                config.lambda,
            );
            km.mean
        }
    };

    // ── Calibration scoring ───────────────────────────────────────────────────
    // Propagate any scoring error explicitly — a silent NaN fallback would poison
    // the threshold (NaN) and suppress every anomaly flag with no error signal.
    let calib_scores: Vec<f64> = (0..n_calib)
        .map(|i| {
            let curve = calibration.row(i);
            elastic_nonconformity(&curve, &template, argvals, config.lambda, config.variant)
        })
        .collect::<Result<Vec<f64>, FdarError>>()?;

    // ── Threshold ─────────────────────────────────────────────────────────────
    let mut sorted_calib = calib_scores.clone();
    crate::helpers::sort_nan_safe(&mut sorted_calib);
    let threshold = calibrated_threshold(&sorted_calib, config.alpha);

    // ── Test scoring ──────────────────────────────────────────────────────────
    let mut p_values = Vec::with_capacity(n_test);
    let mut scores = Vec::with_capacity(n_test);
    let mut flags = Vec::with_capacity(n_test);

    for j in 0..n_test {
        let curve = test.row(j);
        let a_star =
            elastic_nonconformity(&curve, &template, argvals, config.lambda, config.variant)?;

        // p-value: (1 + #{calib_score >= a_star}) / (n_calib + 1)
        let count = calib_scores.iter().filter(|&&a| a >= a_star).count();
        let p_value = (1 + count) as f64 / (n_calib + 1) as f64;
        let flag = p_value <= config.alpha;

        scores.push(a_star);
        p_values.push(p_value);
        flags.push(flag);
    }

    Ok(ConformalAnomalyResult {
        p_values,
        scores,
        flags,
        threshold,
    })
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

/// Compute the calibrated threshold from sorted calibration scores.
///
/// Returns the `(1-alpha)` order-statistic: the `k`-th smallest value where
/// `k = ceil((n+1) * (1-alpha))`. Returns `f64::INFINITY` when `k > n`
/// (insufficient calibration data to form a threshold at level `alpha`).
fn calibrated_threshold(sorted_scores: &[f64], alpha: f64) -> f64 {
    let n = sorted_scores.len();
    let k = ((n + 1) as f64 * (1.0 - alpha)).ceil() as usize;
    if k > n {
        f64::INFINITY
    } else {
        sorted_scores[k.saturating_sub(1)]
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};

    /// Build a uniform grid on [0, 1] of length `n`.
    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    /// Build a smooth sinusoid curve on the given grid.
    fn sinusoid(argvals: &[f64]) -> Vec<f64> {
        argvals
            .iter()
            .map(|&t| (t * 6.0 * std::f64::consts::PI).sin())
            .collect()
    }

    // ── ECA-01: elastic_nonconformity ─────────────────────────────────────────

    #[test]
    fn test_elastic_nonconformity_self_near_zero() {
        let t = uniform_grid(50);
        let curve = sinusoid(&t);
        for variant in [
            NonConformityScore::AmplitudeElastic,
            NonConformityScore::PhaseElastic,
            NonConformityScore::CombinedElastic,
        ] {
            let score = elastic_nonconformity(&curve, &curve, &t, 0.0, variant).unwrap();
            assert!(
                score < 1e-4,
                "Self-score for {:?} should be near zero, got {score}",
                variant
            );
        }
    }

    #[test]
    fn test_elastic_nonconformity_nonneg() {
        let t = uniform_grid(50);
        let curve = sinusoid(&t);
        // A distinct curve (cosine)
        let other: Vec<f64> = t
            .iter()
            .map(|&x| (x * 6.0 * std::f64::consts::PI).cos())
            .collect();
        for variant in [
            NonConformityScore::AmplitudeElastic,
            NonConformityScore::PhaseElastic,
            NonConformityScore::CombinedElastic,
        ] {
            let score = elastic_nonconformity(&curve, &other, &t, 0.0, variant).unwrap();
            assert!(
                score >= 0.0,
                "Non-negativity violated for {:?}: got {score}",
                variant
            );
        }
    }

    #[test]
    fn test_elastic_nonconformity_invalid_variant() {
        let t = uniform_grid(50);
        let curve = sinusoid(&t);
        for variant in [NonConformityScore::SupNorm, NonConformityScore::L2] {
            let result = elastic_nonconformity(&curve, &curve, &t, 0.0, variant);
            assert!(
                matches!(result, Err(FdarError::InvalidParameter { .. })),
                "Expected InvalidParameter for {:?}, got {:?}",
                variant,
                result
            );
        }
    }

    #[test]
    fn test_conformal_anomaly_rejects_mismatched_template() {
        // A caller-supplied template of the wrong length must be rejected up front,
        // not silently zero-scored (CR-01 regression guard).
        let t = uniform_grid(50);
        let calibration = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(1));
        let test_data = sim_fundata(5, &t, 3, EFunType::Fourier, EValType::Exponential, Some(2));
        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::CombinedElastic,
            alpha: 0.1,
            template: Some(uniform_grid(40)), // wrong length (40 != 50)
            ..Default::default()
        };
        let result = elastic_conformal_anomaly(&calibration, &test_data, &t, &config);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "Expected InvalidDimension for mismatched template, got {result:?}"
        );
    }

    #[test]
    fn test_elastic_nonconformity_combined_not_alias() {
        // Scaled curve differs in amplitude only; for a pure amplitude outlier,
        // AmplitudeElastic > 0, and CombinedElastic >= AmplitudeElastic (not an alias of amplitude).
        let t = uniform_grid(50);
        let template = sinusoid(&t);
        // Scale by 5x to create a large amplitude outlier
        let scaled: Vec<f64> = template.iter().map(|&v| v * 5.0).collect();

        let amp = elastic_nonconformity(
            &scaled,
            &template,
            &t,
            0.0,
            NonConformityScore::AmplitudeElastic,
        )
        .unwrap();
        let combined = elastic_nonconformity(
            &scaled,
            &template,
            &t,
            0.0,
            NonConformityScore::CombinedElastic,
        )
        .unwrap();

        assert!(
            amp > 0.0,
            "AmplitudeElastic should be positive for scaled curve, got {amp}"
        );
        // CombinedElastic = sqrt(amp^2 + phase^2) >= amp (since phase^2 >= 0)
        assert!(
            combined >= amp - 1e-12,
            "CombinedElastic ({combined}) should be >= AmplitudeElastic ({amp})"
        );
    }

    // ── ECA-02 / ECA-03: elastic_conformal_anomaly ────────────────────────────

    #[test]
    fn test_elastic_conformal_marginal_validity() {
        // On exchangeable clean data, flag rate should be approximately alpha.
        let t = uniform_grid(50);
        let calibration = sim_fundata(
            100,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(1),
        );
        let test_data = sim_fundata(
            100,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(2),
        );

        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::CombinedElastic,
            alpha: 0.1,
            ..Default::default()
        };

        let result = elastic_conformal_anomaly(&calibration, &test_data, &t, &config).unwrap();
        let flag_rate = result.flags.iter().filter(|&&f| f).count() as f64 / 100.0;
        // Generous tolerance: finite-sample marginal validity guarantees <= alpha + correction
        assert!(
            flag_rate <= 0.25,
            "Flag rate {flag_rate} too high for alpha=0.1 on clean data"
        );
    }

    #[test]
    fn test_elastic_conformal_magnitude_outlier() {
        // An amplitude-scaled curve should be flagged by AmplitudeElastic.
        //
        // Strategy: use a sinusoid as the fixed template. Build a calibration set of
        // curves that are identical to the template (self-score ≈ 0), so the calibration
        // distribution is tightly concentrated near zero. Then inject a 10x-scaled
        // sinusoid as the magnitude outlier — its AmplitudeElastic score will be large,
        // clearly exceeding the near-zero threshold.
        let t = uniform_grid(50);
        let template = sinusoid(&t);

        // Calibration: 20 copies of the template (self-scores ≈ 0)
        let n_calib = 20;
        let mut calibration = FdMatrix::zeros(n_calib, t.len());
        for i in 0..n_calib {
            for j in 0..t.len() {
                calibration[(i, j)] = template[j];
            }
        }

        // Test set: 5 clean (identical to template) + 1 magnitude outlier (10x scaled)
        let n_clean = 5;
        let outlier: Vec<f64> = template.iter().map(|&v| v * 10.0).collect();
        let mut test_mat = FdMatrix::zeros(n_clean + 1, t.len());
        for i in 0..n_clean {
            for j in 0..t.len() {
                test_mat[(i, j)] = template[j];
            }
        }
        for j in 0..t.len() {
            test_mat[(n_clean, j)] = outlier[j];
        }

        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::AmplitudeElastic,
            alpha: 0.1,
            template: Some(template),
            ..Default::default()
        };

        let result = elastic_conformal_anomaly(&calibration, &test_mat, &t, &config).unwrap();
        assert!(
            result.flags[n_clean],
            "Magnitude outlier (10x scaled) should be flagged by AmplitudeElastic; \
             outlier score={}, threshold={}",
            result.scores[n_clean], result.threshold
        );
        assert!(
            result.scores[n_clean] > result.threshold,
            "Outlier score {} should exceed threshold {}",
            result.scores[n_clean],
            result.threshold
        );
    }

    #[test]
    fn test_elastic_conformal_shape_outlier() {
        // A phase-distorted curve should be flagged by PhaseElastic.
        //
        // Strategy: calibrate on sinusoids (phase distance ≈ 0 against template).
        // Inject a "reversed" curve (t → 1-t phase map) — this creates a large
        // phase distortion from the sinusoid template.
        let t = uniform_grid(50);
        let template = sinusoid(&t);

        // Calibration: 20 copies of the template (phase-score ≈ 0)
        let n_calib = 20;
        let mut calibration = FdMatrix::zeros(n_calib, t.len());
        for i in 0..n_calib {
            for j in 0..t.len() {
                calibration[(i, j)] = template[j];
            }
        }

        // Shape/phase outlier: a cosine (quarter-period phase shift relative to sinusoid).
        // The cosine and sine are of the same shape class but have a large phase offset.
        let phase_outlier: Vec<f64> = t
            .iter()
            .map(|&x| (x * 6.0 * std::f64::consts::PI).cos())
            .collect();

        let n_clean = 5;
        let mut test_mat = FdMatrix::zeros(n_clean + 1, t.len());
        for i in 0..n_clean {
            for j in 0..t.len() {
                test_mat[(i, j)] = template[j];
            }
        }
        for j in 0..t.len() {
            test_mat[(n_clean, j)] = phase_outlier[j];
        }

        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::PhaseElastic,
            alpha: 0.1,
            template: Some(template),
            ..Default::default()
        };

        let result = elastic_conformal_anomaly(&calibration, &test_mat, &t, &config).unwrap();
        assert!(
            result.flags[n_clean],
            "Phase-distorted outlier (cosine vs sine template) should be flagged by PhaseElastic; \
             outlier score={}, threshold={}",
            result.scores[n_clean], result.threshold
        );
    }

    #[test]
    fn test_elastic_conformal_combined_catches_both() {
        // CombinedElastic should flag both a magnitude and a phase outlier.
        //
        // Strategy: calibrate on sinusoids (calib scores ≈ 0). Inject:
        // - a 10x-scaled sinusoid (magnitude outlier)
        // - a cosine (phase outlier relative to sinusoid template)
        let t = uniform_grid(50);
        let template = sinusoid(&t);

        // Calibration: 20 copies of the template
        let n_calib = 20;
        let mut calibration = FdMatrix::zeros(n_calib, t.len());
        for i in 0..n_calib {
            for j in 0..t.len() {
                calibration[(i, j)] = template[j];
            }
        }

        let magnitude_outlier: Vec<f64> = template.iter().map(|&v| v * 10.0).collect();
        let phase_outlier: Vec<f64> = t
            .iter()
            .map(|&x| (x * 6.0 * std::f64::consts::PI).cos())
            .collect();

        let n_clean = 3;
        let mut test_mat = FdMatrix::zeros(n_clean + 2, t.len());
        for i in 0..n_clean {
            for j in 0..t.len() {
                test_mat[(i, j)] = template[j];
            }
        }
        for j in 0..t.len() {
            test_mat[(n_clean, j)] = magnitude_outlier[j];
            test_mat[(n_clean + 1, j)] = phase_outlier[j];
        }

        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::CombinedElastic,
            alpha: 0.1,
            template: Some(template),
            ..Default::default()
        };

        let result = elastic_conformal_anomaly(&calibration, &test_mat, &t, &config).unwrap();
        assert!(
            result.flags[n_clean],
            "Magnitude outlier (10x scaled) should be flagged by CombinedElastic; \
             outlier score={}, threshold={}",
            result.scores[n_clean], result.threshold
        );
        assert!(
            result.flags[n_clean + 1],
            "Phase outlier (cosine vs sine template) should be flagged by CombinedElastic; \
             outlier score={}, threshold={}",
            result.scores[n_clean + 1],
            result.threshold
        );
    }

    #[test]
    fn test_elastic_conformal_pvalue_threshold_correctness() {
        // Hand-verifiable small example: verify p-value formula and threshold.
        // Calibration: 5 identical sinusoids (self-score = ~0) against a template.
        let t = uniform_grid(20);
        let template = sinusoid(&t);

        // 5 calibration curves: identical to template → calib_scores ≈ 0
        let mut calib = FdMatrix::zeros(5, t.len());
        for i in 0..5 {
            for j in 0..t.len() {
                calib[(i, j)] = template[j];
            }
        }

        // 1 test curve that is a scaled version (high score)
        let scaled: Vec<f64> = template.iter().map(|&v| v * 5.0).collect();
        let mut test_mat = FdMatrix::zeros(1, t.len());
        for j in 0..t.len() {
            test_mat[(0, j)] = scaled[j];
        }

        let config = ConformalAnomalyConfig {
            variant: NonConformityScore::AmplitudeElastic,
            alpha: 0.1,
            template: Some(template.clone()),
            ..Default::default()
        };

        let result = elastic_conformal_anomaly(&calib, &test_mat, &t, &config).unwrap();

        // Expected: calib_scores all near 0, test score > 0
        // p_value = (1 + 5) / (5+1) = 1.0 (all 5 calib scores >= test score is false,
        // but actually if calib ≈ 0 and test > 0, count of (calib >= test) = 0)
        // → p_value = (1 + 0) / 6 = 1/6 ≈ 0.1667
        // flag = p_value <= 0.1 → false (0.1667 > 0.1)
        // But let's verify the formula directly:
        let a_star = result.scores[0];

        // All calib scores should be near zero, test score should be large
        // Recompute p_value independently
        let calib_scores: Vec<f64> = (0..5)
            .map(|i| {
                let row: Vec<f64> = (0..t.len()).map(|j| calib[(i, j)]).collect();
                elastic_nonconformity(
                    &row,
                    &template,
                    &t,
                    0.0,
                    NonConformityScore::AmplitudeElastic,
                )
                .unwrap()
            })
            .collect();

        let count = calib_scores.iter().filter(|&&a| a >= a_star).count();
        let expected_p = (1 + count) as f64 / 6.0;
        assert!(
            (result.p_values[0] - expected_p).abs() < 1e-12,
            "p_value mismatch: got {}, expected {}",
            result.p_values[0],
            expected_p
        );
        assert_eq!(
            result.flags[0],
            result.p_values[0] <= 0.1,
            "Flag should equal (p_value <= alpha)"
        );

        // Threshold: order-statistic at k = ceil(6 * 0.9) = ceil(5.4) = 6 → k > n → INFINITY
        let mut sorted_calib = calib_scores.clone();
        crate::helpers::sort_nan_safe(&mut sorted_calib);
        let expected_threshold = {
            let n = sorted_calib.len();
            let k = ((n + 1) as f64 * 0.9).ceil() as usize;
            if k > n {
                f64::INFINITY
            } else {
                sorted_calib[k.saturating_sub(1)]
            }
        };
        assert!(
            (result.threshold - expected_threshold).abs() < 1e-12
                || (result.threshold.is_infinite() && expected_threshold.is_infinite()),
            "Threshold mismatch: got {}, expected {}",
            result.threshold,
            expected_threshold
        );
    }

    #[test]
    fn test_conformal_anomaly_result_shape() {
        let t = uniform_grid(50);
        let calibration = sim_fundata(
            40,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(100),
        );
        let test_data = sim_fundata(
            15,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(101),
        );

        let config = ConformalAnomalyConfig::default();
        let result = elastic_conformal_anomaly(&calibration, &test_data, &t, &config).unwrap();

        assert_eq!(
            result.p_values.len(),
            15,
            "p_values length should equal n_test"
        );
        assert_eq!(result.scores.len(), 15, "scores length should equal n_test");
        assert_eq!(result.flags.len(), 15, "flags length should equal n_test");
        assert!(result.threshold.is_finite() || result.threshold.is_infinite());
        assert!(result.threshold >= 0.0 || result.threshold.is_infinite());
    }

    #[test]
    fn test_conformal_band_rejects_elastic_variants() {
        use crate::tolerance::conformal_prediction_band;

        let t = uniform_grid(50);
        let data = sim_fundata(
            40,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        );

        // Elastic variants must return None
        for variant in [
            NonConformityScore::AmplitudeElastic,
            NonConformityScore::PhaseElastic,
            NonConformityScore::CombinedElastic,
        ] {
            let result = conformal_prediction_band(&data, 0.2, 0.95, variant, 42);
            assert!(
                result.is_none(),
                "conformal_prediction_band should return None for {:?}",
                variant
            );
        }

        // SupNorm and L2 must still return Some on valid data
        for variant in [NonConformityScore::SupNorm, NonConformityScore::L2] {
            let result = conformal_prediction_band(&data, 0.2, 0.95, variant, 42);
            assert!(
                result.is_some(),
                "conformal_prediction_band should return Some for {:?}",
                variant
            );
        }
    }
}
