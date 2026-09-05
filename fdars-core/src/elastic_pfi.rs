//! Model-agnostic permutation feature importance (PFI) over jfPCA PC scores,
//! principal-direction reconstruction, and the end-to-end `veesa_pipeline`.
//!
//! This module implements the VEESA explainability layer (VEE-03, VEE-04, VEE-05)
//! on top of the [`JfpcaModel`] fit/transform seam from Phase 72.
//!
//! # Quick Start
//!
//! ```
//! use fdars_core::{jfpca_fit, elastic_pfi, PfiMetric};
//! use fdars_core::matrix::FdMatrix;
//! use std::f64::consts::PI;
//!
//! // Build a small spanning multi-frequency fixture (n=8, m=12)
//! let n = 8usize;
//! let m = 12usize;
//! let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
//! let mut data = FdMatrix::zeros(n, m);
//! for i in 0..n {
//!     let fi = i as f64;
//!     for j in 0..m {
//!         let t = argvals[j];
//!         data[(i, j)] = (1.0 + 0.3 * fi) * (2.0 * PI * t).sin()
//!                      + (0.5 - 0.05 * fi) * (4.0 * PI * t).cos()
//!                      + (0.3 + 0.04 * fi) * (6.0 * PI * t).sin();
//!     }
//! }
//! let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
//!
//! let model = jfpca_fit(&data, &argvals, 3, None, 0.0, 10)?;
//! let tr = model.score_training()?;
//! // Model-agnostic: any closure over scores
//! let pfi = elastic_pfi(
//!     &tr.scores, &y,
//!     |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
//!     &PfiMetric::Mse,
//!     5, 42,
//! )?;
//! assert_eq!(pfi.importance.len(), model.ncomp);
//! # Ok::<(), fdars_core::FdarError>(())
//! ```

use crate::explain::helpers::{clone_scores_matrix, shuffle_global};
use crate::jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};
use crate::matrix::FdMatrix;
use crate::FdarError;
use rand::prelude::*;
use std::sync::Arc;

/// Metric used to evaluate prediction quality in [`elastic_pfi`].
///
/// Importance is always computed as `baseline_metric - permuted_metric`, which
/// assumes a **higher-is-better** metric. Sign interpretation therefore depends
/// on the metric:
///
/// - For higher-is-better metrics ([`PfiMetric::Accuracy`], or a [`PfiMetric::Custom`]
///   score), a **positive** importance means the component is informative
///   (permuting it degrades the prediction).
/// - For the loss metrics [`PfiMetric::Mse`] / [`PfiMetric::Mae`] (lower-is-better),
///   an informative component yields a **negative** importance (permuting it raises
///   the loss, so `baseline - permuted < 0`). Wrap the loss in a `Custom` closure
///   that negates it if you want the positive-is-informative convention.
///
/// Note: `PfiMetric` does NOT derive `serde::{Serialize, Deserialize}` because the
/// `Custom` variant holds a non-serializable closure. `Debug`, `Clone`, and
/// `PartialEq` are provided via manual impls (two `Custom` variants always compare
/// unequal — closures are not comparable).
#[non_exhaustive]
pub enum PfiMetric {
    /// Mean squared error (a loss — lower is better).
    ///
    /// Importance is `baseline_mse - permuted_mse`; since permuting an informative
    /// component *increases* MSE, informative components have **negative** importance
    /// under this metric. Use [`PfiMetric::Accuracy`] or a negated [`PfiMetric::Custom`]
    /// for a directly-interpretable positive-is-informative sign.
    Mse,
    /// Mean absolute error (a loss — lower is better; same negative-for-informative
    /// sign convention as [`PfiMetric::Mse`]).
    Mae,
    /// Classification accuracy: fraction of predictions where `round(pred) == y_true`.
    /// Higher is better; permuting an informative component drops accuracy, so
    /// informative components have positive importance.
    Accuracy,
    /// Custom metric closure: `(y_true, y_pred) -> metric_value`.
    ///
    /// Convention: higher values mean a *better* prediction (importance = baseline - permuted).
    /// If your metric is a loss (lower = better), negate it inside the closure.
    Custom(Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for PfiMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PfiMetric::Mse => write!(f, "Mse"),
            PfiMetric::Mae => write!(f, "Mae"),
            PfiMetric::Accuracy => write!(f, "Accuracy"),
            PfiMetric::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

impl Clone for PfiMetric {
    fn clone(&self) -> Self {
        match self {
            PfiMetric::Mse => PfiMetric::Mse,
            PfiMetric::Mae => PfiMetric::Mae,
            PfiMetric::Accuracy => PfiMetric::Accuracy,
            PfiMetric::Custom(f) => PfiMetric::Custom(Arc::clone(f)),
        }
    }
}

impl PartialEq for PfiMetric {
    /// Non-`Custom` variants compare by discriminant; two `Custom` variants are
    /// never equal (closures cannot be compared).
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PfiMetric::Mse, PfiMetric::Mse)
                | (PfiMetric::Mae, PfiMetric::Mae)
                | (PfiMetric::Accuracy, PfiMetric::Accuracy)
        )
    }
}

/// Result of [`elastic_pfi`].
///
/// `importance[k] = baseline_metric - mean_permuted_metric[k]`. Sign
/// interpretation depends on the metric direction — see [`PfiMetric`]: positive
/// means informative for higher-is-better metrics ([`PfiMetric::Accuracy`] /
/// score-style `Custom`), while for the loss metrics [`PfiMetric::Mse`] /
/// [`PfiMetric::Mae`] informative PCs are negative.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElasticPfiResult {
    /// Metric drop per PC (length ncomp). Sign per metric direction (see [`PfiMetric`]):
    /// positive = informative for higher-is-better metrics; negative = informative for Mse/Mae.
    pub importance: Vec<f64>,
    /// Baseline metric (no permutation).
    pub baseline_metric: f64,
    /// Mean metric after permuting each PC (length ncomp).
    pub permuted_metric: Vec<f64>,
}

/// Result of [`veesa_pipeline`].
///
/// Bundles the trained [`JfpcaModel`], the exact training-set PC scores, and the
/// permutation feature importance result into a single return value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VeesaPipelineResult {
    /// Trained joint-FPCA model.
    pub model: JfpcaModel,
    /// Exact training-set PC scores (via [`JfpcaModel::score_training`]).
    pub training_scores: JfpcaTransform,
    /// Permutation feature importance result over the training scores.
    pub pfi: ElasticPfiResult,
}

// ─── Internal metric helper ───────────────────────────────────────────────────

fn compute_metric(y: &[f64], pred: &[f64], metric: &PfiMetric) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    match metric {
        PfiMetric::Mse => {
            y.iter()
                .zip(pred.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                / n as f64
        }
        PfiMetric::Mae => {
            y.iter()
                .zip(pred.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>()
                / n as f64
        }
        PfiMetric::Accuracy => {
            y.iter()
                .zip(pred.iter())
                .filter(|(&a, &b)| (a - b.round()).abs() < 1e-10)
                .count() as f64
                / n as f64
        }
        PfiMetric::Custom(f) => f(y, pred),
    }
}

// ─── elastic_pfi ─────────────────────────────────────────────────────────────

/// Model-agnostic permutation feature importance over jfPCA PC scores (VEE-03).
///
/// Permutes each PC-score column in turn and measures the drop in prediction
/// quality using the caller's closure — so any external predictor (elastic-PCR,
/// logistic, random forest, etc.) can be explained without coupling to its
/// internals.
///
/// # Determinism
///
/// A single `StdRng::seed_from_u64(seed)` is advanced across the entire
/// component loop, matching the advancing-RNG convention of the existing
/// [`fpc_permutation_importance`] family.  Two calls with the same `seed`
/// and `n_repeats` return bit-identical `importance` vectors.
///
/// # Arguments
///
/// * `scores` — PC score matrix (n × ncomp), e.g. from [`JfpcaModel::score_training`].
/// * `y` — Response vector (length n).
/// * `predict` — Closure mapping a score matrix → predictions (length n).
/// * `metric` — Prediction metric; see [`PfiMetric`].
/// * `n_repeats` — Number of permutation repeats per component (≥ 1).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `scores.nrows() != y.len()` or the
///   matrix is empty.
/// * [`FdarError::InvalidParameter`] if `n_repeats == 0`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pfi(
    scores: &FdMatrix,
    y: &[f64],
    predict: impl Fn(&FdMatrix) -> Vec<f64>,
    metric: &PfiMetric,
    n_repeats: usize,
    seed: u64,
) -> Result<ElasticPfiResult, FdarError> {
    let (n, ncomp) = scores.shape();

    // Input validation (V5 — T-73-01)
    if n == 0 || ncomp == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "scores",
            expected: "at least 1 row and 1 column".to_string(),
            actual: format!("({}, {})", n, ncomp),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("length {} (== scores.nrows())", n),
            actual: format!("length {}", y.len()),
        });
    }
    if n_repeats == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_repeats",
            message: "must be >= 1 (zero repeats produce no permutation samples)".to_string(),
        });
    }

    // Baseline metric — computed ONCE before the permutation loop.
    let baseline_pred = predict(scores);
    // Validate the caller's closure returns one prediction per observation; a short
    // vector would otherwise be silently truncated by the metric zip and divided by
    // the full n, yielding wrong importance values.
    if baseline_pred.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "predict() output",
            expected: format!("length {} (== scores.nrows())", n),
            actual: format!("length {}", baseline_pred.len()),
        });
    }
    let baseline_metric = compute_metric(y, &baseline_pred, metric);

    // Single advancing RNG — mirrors the existing fpc_permutation_importance
    // advancing-RNG pattern (importance.rs:130).  Do NOT reseed per component.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_metric = 0.0;
        for _ in 0..n_repeats {
            let mut perm = clone_scores_matrix(scores, n, ncomp);
            shuffle_global(&mut perm, scores, k, n, &mut rng);
            let pred = predict(&perm);
            sum_metric += compute_metric(y, &pred, metric);
        }
        let mean_perm = sum_metric / n_repeats as f64;
        permuted_metric[k] = mean_perm;
        importance[k] = baseline_metric - mean_perm;
    }

    Ok(ElasticPfiResult {
        importance,
        baseline_metric,
        permuted_metric,
    })
}

// ─── veesa_pipeline ───────────────────────────────────────────────────────────

/// End-to-end VEESA pipeline: fit jfPCA → score training data → compute PFI (VEE-05).
///
/// Convenience wrapper that chains [`jfpca_fit`] → [`JfpcaModel::score_training`] →
/// [`elastic_pfi`] into a single call, returning a [`VeesaPipelineResult`] that
/// bundles the trained model, exact training-set scores, and importance values.
///
/// # Arguments
///
/// * `data` — Functional data matrix (n × m); rows are curves.
/// * `argvals` — Evaluation grid (length m).
/// * `ncomp` — Number of jfPCA components to extract.
/// * `balance_c` — Phase-vs-amplitude balance; `None` triggers golden-section search.
/// * `lambda` — Warp regularization penalty (0.0 = no penalty).
/// * `max_iter` — Max Karcher-mean iterations (20 is a safe default).
/// * `y` — Response vector (length n) for PFI.
/// * `predict` — Closure mapping PC scores → predictions (length n).
/// * `metric` — Prediction metric; see [`PfiMetric`].
/// * `n_repeats` — Permutation repeats per component (≥ 1).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// Propagates errors from [`jfpca_fit`] and [`elastic_pfi`].
///
/// # Example
///
/// ```
/// use fdars_core::{veesa_pipeline, PfiMetric};
/// use fdars_core::matrix::FdMatrix;
/// use std::f64::consts::PI;
///
/// let n = 8usize;
/// let m = 12usize;
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let mut data = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let fi = i as f64;
///     for j in 0..m {
///         let t = argvals[j];
///         data[(i, j)] = (1.0 + 0.3 * fi) * (2.0 * PI * t).sin()
///                      + (0.5 - 0.05 * fi) * (4.0 * PI * t).cos()
///                      + (0.3 + 0.04 * fi) * (6.0 * PI * t).sin();
///     }
/// }
/// let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
///
/// let result = veesa_pipeline(
///     &data, &argvals, 3, None, 0.0, 10,
///     &y,
///     |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
///     &PfiMetric::Mse,
///     5, 42,
/// )?;
/// assert_eq!(result.pfi.importance.len(), result.model.ncomp);
/// # Ok::<(), fdars_core::FdarError>(())
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn veesa_pipeline(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
    lambda: f64,
    max_iter: usize,
    y: &[f64],
    predict: impl Fn(&FdMatrix) -> Vec<f64>,
    metric: &PfiMetric,
    n_repeats: usize,
    seed: u64,
) -> Result<VeesaPipelineResult, FdarError> {
    let model = jfpca_fit(data, argvals, ncomp, balance_c, lambda, max_iter)?;
    let training_scores = model.score_training()?;
    let pfi = elastic_pfi(&training_scores.scores, y, predict, metric, n_repeats, seed)?;
    Ok(VeesaPipelineResult {
        model,
        training_scores,
        pfi,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build a spanning multi-frequency fixture.
    /// Uses 3 harmonics with per-curve distinct amplitudes for full-rank coverage.
    fn spanning_fixture(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let fi = i as f64;
            let a1 = 1.0 + 0.4 * fi;
            let a2 = 0.6 - 0.08 * fi;
            let a3 = 0.35 + 0.07 * fi;
            let a4 = 0.2 - 0.03 * fi;
            for j in 0..m {
                let t = argvals[j];
                data[(i, j)] = a1 * (2.0 * PI * t).sin()
                    + a2 * (4.0 * PI * t).cos()
                    + a3 * (6.0 * PI * t).sin()
                    + a4 * (8.0 * PI * t).cos();
            }
        }
        (data, argvals)
    }

    /// Smoke test: end-to-end veesa_pipeline on a small spanning fixture.
    /// Verifies the full fit → score_training → elastic_pfi path returns Ok
    /// and the importance vector has the expected length.
    #[test]
    fn smoke_veesa_pipeline() {
        let n = 10usize;
        let m = 14usize;
        let (data, argvals) = spanning_fixture(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let result = veesa_pipeline(
            &data,
            &argvals,
            3,
            None,
            0.0,
            10,
            &y,
            |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
            &PfiMetric::Mse,
            5,
            42,
        )
        .expect("veesa_pipeline should succeed on spanning fixture");

        assert_eq!(
            result.pfi.importance.len(),
            result.model.ncomp,
            "importance length should equal ncomp"
        );
        assert!(
            !result.pfi.importance.is_empty(),
            "importance should be non-empty"
        );
    }

    /// VEE-03 gate: two runs with the same seed produce bit-identical importance vectors.
    #[test]
    fn pfi_seed_determinism() {
        let n = 10usize;
        let m = 14usize;
        let (data, argvals) = spanning_fixture(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let model = jfpca_fit(&data, &argvals, 3, None, 0.0, 10).expect("jfpca_fit should succeed");
        let tr = model
            .score_training()
            .expect("score_training should succeed");

        let run1 = elastic_pfi(
            &tr.scores,
            &y,
            |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
            &PfiMetric::Mse,
            10,
            42,
        )
        .expect("elastic_pfi run 1 should succeed");

        let run2 = elastic_pfi(
            &tr.scores,
            &y,
            |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
            &PfiMetric::Mse,
            10,
            42,
        )
        .expect("elastic_pfi run 2 should succeed");

        assert_eq!(
            run1.importance, run2.importance,
            "two runs with the same seed must produce identical importance vectors"
        );
    }

    /// VEE-03 gate: the informative PC (PC 0, tied to the response) ranks strictly
    /// above both noise PCs on a full-rank spanning fixture.
    ///
    /// Uses a spanning multi-frequency fixture (n=10, m=14) and fits ncomp=3.
    /// The response is exactly the PC-0 score * 2.0, and the predictor closure
    /// returns the same function — so PC 0 is perfectly informative while PCs 1
    /// and 2 are pure noise relative to the response.
    ///
    /// Uses a Custom metric (negative MSE) so that higher=better convention yields
    /// positive importance for the informative PC: importance = baseline - permuted_mean,
    /// where permuting PC 0 lowers the (negated) metric more than permuting noise PCs.
    #[test]
    fn pfi_known_signal_ranking() {
        let n = 10usize;
        let m = 14usize;
        let (data, argvals) = spanning_fixture(n, m);

        let model = jfpca_fit(&data, &argvals, 3, None, 0.0, 10).expect("jfpca_fit should succeed");
        let tr = model
            .score_training()
            .expect("score_training should succeed");

        // Response tied exactly to PC 0
        let y: Vec<f64> = (0..n).map(|i| tr.scores[(i, 0)] * 2.0).collect();

        // Negative MSE: higher = better, so importance = neg_baseline - neg_permuted
        // = -(baseline_mse) + permuted_mse > 0 when permuting hurts.
        let result = elastic_pfi(
            &tr.scores,
            &y,
            // Predictor uses only PC 0: s[(i,0)]*2.0
            |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)] * 2.0).collect() },
            &PfiMetric::Custom(Arc::new(|y_true: &[f64], y_pred: &[f64]| {
                let n = y_true.len() as f64;
                let mse = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    / n;
                -mse // negate so higher = better
            })),
            20,
            42,
        )
        .expect("elastic_pfi should succeed on known-signal design");

        assert!(
            result.importance[0] > result.importance[1],
            "PC 0 importance ({}) must be strictly above PC 1 importance ({})",
            result.importance[0],
            result.importance[1]
        );
        assert!(
            result.importance[0] > result.importance[2],
            "PC 0 importance ({}) must be strictly above PC 2 importance ({})",
            result.importance[0],
            result.importance[2]
        );
    }

    /// VEE-03 gate: n_repeats == 0 is rejected with InvalidParameter.
    #[test]
    fn pfi_rejects_zero_repeats() {
        let n = 8usize;
        let m = 12usize;
        let (data, argvals) = spanning_fixture(n, m);
        let y: Vec<f64> = vec![0.0; n];

        let model = jfpca_fit(&data, &argvals, 2, None, 0.0, 10).expect("jfpca_fit should succeed");
        let tr = model
            .score_training()
            .expect("score_training should succeed");

        let res = elastic_pfi(
            &tr.scores,
            &y,
            |s: &FdMatrix| -> Vec<f64> { vec![0.0; s.nrows()] },
            &PfiMetric::Mse,
            0, // zero repeats — must fail
            42,
        );

        assert!(
            matches!(res, Err(FdarError::InvalidParameter { .. })),
            "n_repeats=0 should return Err(InvalidParameter), got: {:?}",
            res
        );
    }
}
