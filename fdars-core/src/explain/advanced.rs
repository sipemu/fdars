//! Calibration, conformal prediction, regression depth, stability, and anchors.

use super::helpers::*;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{
    fregre_lm, functional_logistic, sigmoid, FregreLmResult, FunctionalLogisticResult,
};
use rand::prelude::*;

// ===========================================================================
// Calibration Diagnostics (logistic only)
// ===========================================================================

/// Calibration diagnostics for a functional logistic regression model.
pub struct CalibrationDiagnosticsResult {
    /// Brier score: (1/n) Σ (p_i - y_i)².
    pub brier_score: f64,
    /// Log loss: -(1/n) Σ [y log p + (1-y) log(1-p)].
    pub log_loss: f64,
    /// Hosmer-Lemeshow chi² statistic.
    pub hosmer_lemeshow_chi2: f64,
    /// Degrees of freedom: n_groups - 2.
    pub hosmer_lemeshow_df: usize,
    /// Number of calibration groups.
    pub n_groups: usize,
    /// Reliability bins: (mean_predicted, mean_observed) per group.
    pub reliability_bins: Vec<(f64, f64)>,
    /// Number of observations in each group.
    pub bin_counts: Vec<usize>,
}

/// Calibration diagnostics for a functional logistic regression model.
pub fn calibration_diagnostics(
    fit: &FunctionalLogisticResult,
    y: &[f64],
    n_groups: usize,
) -> Option<CalibrationDiagnosticsResult> {
    let n = fit.probabilities.len();
    if n == 0 || n != y.len() || n_groups < 2 {
        return None;
    }

    // Brier score
    let brier_score: f64 = fit
        .probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| (p - yi).powi(2))
        .sum::<f64>()
        / n as f64;

    // Log loss
    let log_loss: f64 = -fit
        .probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| {
            let p_clip = p.clamp(1e-15, 1.0 - 1e-15);
            yi * p_clip.ln() + (1.0 - yi) * (1.0 - p_clip).ln()
        })
        .sum::<f64>()
        / n as f64;

    let (hosmer_lemeshow_chi2, reliability_bins, bin_counts) =
        hosmer_lemeshow_computation(&fit.probabilities, y, n, n_groups);

    let actual_groups = bin_counts.len();
    let hosmer_lemeshow_df = if actual_groups > 2 {
        actual_groups - 2
    } else {
        1
    };

    Some(CalibrationDiagnosticsResult {
        brier_score,
        log_loss,
        hosmer_lemeshow_chi2,
        hosmer_lemeshow_df,
        n_groups: actual_groups,
        reliability_bins,
        bin_counts,
    })
}

// ===========================================================================
// Expected Calibration Error (ECE)
// ===========================================================================

/// Result of expected calibration error analysis.
pub struct EceResult {
    /// Expected calibration error: Σ (n_b/n) |acc_b - conf_b|.
    pub ece: f64,
    /// Maximum calibration error: max |acc_b - conf_b|.
    pub mce: f64,
    /// Adaptive calibration error (equal-mass bins).
    pub ace: f64,
    /// Number of bins used.
    pub n_bins: usize,
    /// Per-bin ECE contributions (length n_bins).
    pub bin_ece_contributions: Vec<f64>,
}

/// Compute expected, maximum, and adaptive calibration errors for a logistic model.
///
/// # Arguments
/// * `fit` — A fitted [`FunctionalLogisticResult`]
/// * `y` — Binary labels (0/1), length n
/// * `n_bins` — Number of bins for equal-width binning
pub fn expected_calibration_error(
    fit: &FunctionalLogisticResult,
    y: &[f64],
    n_bins: usize,
) -> Option<EceResult> {
    let n = fit.probabilities.len();
    if n == 0 || n != y.len() || n_bins == 0 {
        return None;
    }

    let (ece, mce, bin_ece_contributions) =
        compute_equal_width_ece(&fit.probabilities, y, n, n_bins);

    // ACE: equal-mass (quantile) bins
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        fit.probabilities[a]
            .partial_cmp(&fit.probabilities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let group_size = n / n_bins.max(1);
    let mut ace = 0.0;
    let mut start = 0;
    for g in 0..n_bins {
        if start >= n {
            break;
        }
        let end = if g < n_bins - 1 {
            (start + group_size).min(n)
        } else {
            n
        };
        ace += calibration_gap_weighted(&sorted_idx[start..end], y, &fit.probabilities, n);
        start = end;
    }

    Some(EceResult {
        ece,
        mce,
        ace,
        n_bins,
        bin_ece_contributions,
    })
}

// ===========================================================================
// Conformal Prediction Residuals
// ===========================================================================

/// Result of split-conformal prediction.
pub struct ConformalPredictionResult {
    /// Predictions on test data (length n_test).
    pub predictions: Vec<f64>,
    /// Lower bounds of prediction intervals (length n_test).
    pub lower: Vec<f64>,
    /// Upper bounds of prediction intervals (length n_test).
    pub upper: Vec<f64>,
    /// Quantile of calibration residuals.
    pub residual_quantile: f64,
    /// Empirical coverage on the calibration set.
    pub coverage: f64,
    /// Absolute residuals on calibration set.
    pub calibration_scores: Vec<f64>,
}

/// Split-conformal prediction intervals for a linear functional regression.
///
/// Randomly splits training data into proper-train and calibration subsets,
/// refits the model, and constructs distribution-free prediction intervals.
///
/// # Arguments
/// * `fit` — Original [`FregreLmResult`] (used for ncomp)
/// * `train_data` — Training functional data (n × m)
/// * `train_y` — Training response (length n)
/// * `test_data` — Test functional data (n_test × m)
/// * `scalar_covariates_train` — Optional scalar covariates for training
/// * `scalar_covariates_test` — Optional scalar covariates for test
/// * `cal_fraction` — Fraction of training data for calibration (0, 1)
/// * `alpha` — Miscoverage level (e.g. 0.1 for 90% intervals)
/// * `seed` — Random seed
pub fn conformal_prediction_residuals(
    fit: &FregreLmResult,
    train_data: &FdMatrix,
    train_y: &[f64],
    test_data: &FdMatrix,
    scalar_covariates_train: Option<&FdMatrix>,
    scalar_covariates_test: Option<&FdMatrix>,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalPredictionResult> {
    let (n, m) = train_data.shape();
    let (n_test, m_test) = test_data.shape();
    let ncomp = fit.ncomp;
    let (_n_cal, n_proper) = validate_conformal_inputs(
        n,
        m,
        n_test,
        m_test,
        train_y.len(),
        ncomp,
        cal_fraction,
        alpha,
    )?;

    // Random split
    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_idx: Vec<usize> = (0..n).collect();
    all_idx.shuffle(&mut rng);
    let proper_idx = &all_idx[..n_proper];
    let cal_idx = &all_idx[n_proper..];

    // Subsample data
    let proper_data = subsample_rows(train_data, proper_idx);
    let proper_y: Vec<f64> = proper_idx.iter().map(|&i| train_y[i]).collect();
    let proper_sc = scalar_covariates_train.map(|sc| subsample_rows(sc, proper_idx));

    // Refit on proper-train
    let refit = fregre_lm(&proper_data, &proper_y, proper_sc.as_ref(), ncomp)?;

    // Predict on calibration set
    let cal_data = subsample_rows(train_data, cal_idx);
    let cal_sc = scalar_covariates_train.map(|sc| subsample_rows(sc, cal_idx));
    let cal_scores = project_scores(&cal_data, &refit.fpca.mean, &refit.fpca.rotation, ncomp);
    let cal_preds = predict_from_scores(
        &cal_scores,
        &refit.coefficients,
        &refit.gamma,
        cal_sc.as_ref(),
        ncomp,
    );
    let cal_n = cal_idx.len();

    let calibration_scores: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (train_y[orig] - cal_preds[i]).abs())
        .collect();

    let (residual_quantile, coverage) =
        conformal_quantile_and_coverage(&calibration_scores, cal_n, alpha);

    // Predict on test data
    let test_scores = project_scores(test_data, &refit.fpca.mean, &refit.fpca.rotation, ncomp);
    let predictions = predict_from_scores(
        &test_scores,
        &refit.coefficients,
        &refit.gamma,
        scalar_covariates_test,
        ncomp,
    );

    let lower: Vec<f64> = predictions.iter().map(|&p| p - residual_quantile).collect();
    let upper: Vec<f64> = predictions.iter().map(|&p| p + residual_quantile).collect();

    Some(ConformalPredictionResult {
        predictions,
        lower,
        upper,
        residual_quantile,
        coverage,
        calibration_scores,
    })
}

// ===========================================================================
// Regression Depth
// ===========================================================================

/// Type of functional depth measure for regression diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthType {
    FraimanMuniz,
    ModifiedBand,
    FunctionalSpatial,
}

/// Result of regression depth analysis.
pub struct RegressionDepthResult {
    /// Depth of β̂ in bootstrap distribution.
    pub beta_depth: f64,
    /// Depth of each observation's FPC scores (length n).
    pub score_depths: Vec<f64>,
    /// Mean of score_depths.
    pub mean_score_depth: f64,
    /// Depth method used.
    pub depth_type: DepthType,
    /// Number of successful bootstrap refits.
    pub n_boot_success: usize,
}

/// Regression depth diagnostics for a linear functional regression.
///
/// Computes depth of each observation's FPC scores and depth of the
/// regression coefficients in a bootstrap distribution.
///
/// # Arguments
/// * `fit` — Fitted [`FregreLmResult`]
/// * `data` — Functional data (n × m)
/// * `y` — Response (length n)
/// * `scalar_covariates` — Optional scalar covariates
/// * `n_boot` — Number of bootstrap iterations
/// * `depth_type` — Which depth measure to use
/// * `seed` — Random seed
pub fn regression_depth(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_boot: usize,
    depth_type: DepthType,
    seed: u64,
) -> Option<RegressionDepthResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let score_depths = compute_score_depths(&scores, depth_type);
    if score_depths.is_empty() {
        return None;
    }
    let mean_score_depth = score_depths.iter().sum::<f64>() / score_depths.len() as f64;

    let orig_coefs: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boot_coefs = Vec::new();
    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        if let Some(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
            boot_coefs.push((0..ncomp).map(|k| refit.coefficients[1 + k]).collect());
        }
    }

    let beta_depth = beta_depth_from_bootstrap(&boot_coefs, &orig_coefs, ncomp, depth_type);

    Some(RegressionDepthResult {
        beta_depth,
        score_depths,
        mean_score_depth,
        depth_type,
        n_boot_success: boot_coefs.len(),
    })
}

/// Regression depth diagnostics for a functional logistic regression.
pub fn regression_depth_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_boot: usize,
    depth_type: DepthType,
    seed: u64,
) -> Option<RegressionDepthResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let score_depths = compute_score_depths(&scores, depth_type);
    if score_depths.is_empty() {
        return None;
    }
    let mean_score_depth = score_depths.iter().sum::<f64>() / score_depths.len() as f64;

    let orig_coefs: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let boot_coefs =
        bootstrap_logistic_coefs(data, y, scalar_covariates, n, ncomp, n_boot, &mut rng);

    let beta_depth = beta_depth_from_bootstrap(&boot_coefs, &orig_coefs, ncomp, depth_type);

    Some(RegressionDepthResult {
        beta_depth,
        score_depths,
        mean_score_depth,
        depth_type,
        n_boot_success: boot_coefs.len(),
    })
}

// ===========================================================================
// Stability / Robustness Analysis
// ===========================================================================

/// Result of bootstrap stability analysis.
pub struct StabilityAnalysisResult {
    /// Pointwise std of β(t) across bootstraps (length m).
    pub beta_t_std: Vec<f64>,
    /// Std of FPC coefficients γ_k across bootstraps (length ncomp).
    pub coefficient_std: Vec<f64>,
    /// Std of R² or accuracy across bootstraps.
    pub metric_std: f64,
    /// Coefficient of variation of β(t): std / |mean| (length m).
    pub beta_t_cv: Vec<f64>,
    /// Mean Spearman rank correlation of FPC importance rankings.
    pub importance_stability: f64,
    /// Number of successful bootstrap refits.
    pub n_boot_success: usize,
}

/// Bootstrap stability analysis of a linear functional regression.
///
/// Refits the model on `n_boot` bootstrap samples and reports variability
/// of β(t), FPC coefficients, R², and importance rankings.
pub fn explanation_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
) -> Option<StabilityAnalysisResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot < 2 || ncomp == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_beta_t: Vec<Vec<f64>> = Vec::new();
    let mut all_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_metrics: Vec<f64> = Vec::new();
    let mut all_abs_coefs: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        if let Some(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
            all_beta_t.push(refit.beta_t.clone());
            let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
            all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
            all_coefs.push(coefs);
            all_metrics.push(refit.r_squared);
        }
    }

    build_stability_result(
        &all_beta_t,
        &all_coefs,
        &all_abs_coefs,
        &all_metrics,
        m,
        ncomp,
    )
}

/// Bootstrap stability analysis of a functional logistic regression.
pub fn explanation_stability_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
) -> Option<StabilityAnalysisResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot < 2 || ncomp == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let (all_beta_t, all_coefs, all_abs_coefs, all_metrics) =
        bootstrap_logistic_stability(data, y, scalar_covariates, n, ncomp, n_boot, &mut rng);

    build_stability_result(
        &all_beta_t,
        &all_coefs,
        &all_abs_coefs,
        &all_metrics,
        m,
        ncomp,
    )
}

// ===========================================================================
// Anchors / Rule Extraction
// ===========================================================================

/// A single condition in an anchor rule.
pub struct AnchorCondition {
    /// FPC component index.
    pub component: usize,
    /// Lower bound on FPC score.
    pub lower_bound: f64,
    /// Upper bound on FPC score.
    pub upper_bound: f64,
}

/// An anchor rule consisting of FPC score conditions.
pub struct AnchorRule {
    /// Conditions forming the rule (conjunction).
    pub conditions: Vec<AnchorCondition>,
    /// Precision: fraction of matching observations with same prediction.
    pub precision: f64,
    /// Coverage: fraction of all observations matching the rule.
    pub coverage: f64,
    /// Number of observations matching the rule.
    pub n_matching: usize,
}

/// Result of anchor explanation for one observation.
pub struct AnchorResult {
    /// The anchor rule.
    pub rule: AnchorRule,
    /// Which observation was explained.
    pub observation: usize,
    /// Predicted value for the observation.
    pub predicted_value: f64,
}

/// Anchor explanation for a linear functional regression.
///
/// Uses beam search in FPC score space to find a minimal set of conditions
/// (score bin memberships) that locally "anchor" the prediction.
///
/// # Arguments
/// * `fit` — Fitted [`FregreLmResult`]
/// * `data` — Functional data (n × m)
/// * `scalar_covariates` — Optional scalar covariates
/// * `observation` — Index of observation to explain
/// * `precision_threshold` — Minimum precision (e.g. 0.95)
/// * `n_bins` — Number of quantile bins per FPC dimension
pub fn anchor_explanation(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Option<AnchorResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || observation >= n || n_bins < 2 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let obs_pred = fit.fitted_values[observation];
    let tol = fit.residual_se;

    // "Same prediction" for regression: within ±1 residual_se
    let same_pred = |i: usize| -> bool {
        let mut yhat = fit.coefficients[0];
        for k in 0..ncomp {
            yhat += fit.coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..fit.gamma.len() {
                yhat += fit.gamma[j] * sc[(i, j)];
            }
        }
        (yhat - obs_pred).abs() <= tol
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &same_pred,
    );

    Some(AnchorResult {
        rule,
        observation,
        predicted_value: obs_pred,
    })
}

/// Anchor explanation for a functional logistic regression.
///
/// "Same prediction" = same predicted class.
pub fn anchor_explanation_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Option<AnchorResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || observation >= n || n_bins < 2 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let obs_class = fit.predicted_classes[observation];
    let obs_prob = fit.probabilities[observation];
    let p_scalar = fit.gamma.len();

    // "Same prediction" = same class
    let same_pred = |i: usize| -> bool {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * sc[(i, j)];
            }
        }
        let pred_class = if sigmoid(eta) >= 0.5 { 1usize } else { 0usize };
        pred_class == obs_class
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &same_pred,
    );

    Some(AnchorResult {
        rule,
        observation,
        predicted_value: obs_prob,
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Hosmer-Lemeshow computation.
fn hosmer_lemeshow_computation(
    probabilities: &[f64],
    y: &[f64],
    n: usize,
    n_groups: usize,
) -> (f64, Vec<(f64, f64)>, Vec<usize>) {
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        probabilities[a]
            .partial_cmp(&probabilities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let group_size = n / n_groups;
    let remainder = n % n_groups;
    let mut start = 0;

    let mut chi2 = 0.0;
    let mut reliability_bins = Vec::with_capacity(n_groups);
    let mut bin_counts = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let sz = group_size + if g < remainder { 1 } else { 0 };
        let group = &sorted_idx[start..start + sz];
        start += sz;

        let ng = group.len();
        if ng == 0 {
            continue;
        }
        let o_g: f64 = group.iter().map(|&i| y[i]).sum();
        let e_g: f64 = group.iter().map(|&i| probabilities[i]).sum();
        let p_bar = e_g / ng as f64;
        let mean_obs = o_g / ng as f64;

        reliability_bins.push((p_bar, mean_obs));
        bin_counts.push(ng);

        let denom = ng as f64 * p_bar * (1.0 - p_bar);
        if denom > 1e-15 {
            chi2 += (o_g - e_g).powi(2) / denom;
        }
    }

    (chi2, reliability_bins, bin_counts)
}

/// Compute equal-width ECE, MCE, and per-bin contributions.
fn compute_equal_width_ece(
    probabilities: &[f64],
    y: &[f64],
    n: usize,
    n_bins: usize,
) -> (f64, f64, Vec<f64>) {
    let mut bin_sum_y = vec![0.0; n_bins];
    let mut bin_sum_p = vec![0.0; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for i in 0..n {
        let b = ((probabilities[i] * n_bins as f64).floor() as usize).min(n_bins - 1);
        bin_sum_y[b] += y[i];
        bin_sum_p[b] += probabilities[i];
        bin_count[b] += 1;
    }

    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    let mut bin_ece_contributions = vec![0.0; n_bins];

    for b in 0..n_bins {
        if bin_count[b] == 0 {
            continue;
        }
        let gap = (bin_sum_y[b] / bin_count[b] as f64 - bin_sum_p[b] / bin_count[b] as f64).abs();
        let contrib = bin_count[b] as f64 / n as f64 * gap;
        bin_ece_contributions[b] = contrib;
        ece += contrib;
        if gap > mce {
            mce = gap;
        }
    }

    (ece, mce, bin_ece_contributions)
}

/// Bootstrap logistic stability: collect beta_t, coefs, abs_coefs, and metrics.
fn bootstrap_logistic_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    n_boot: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut all_beta_t: Vec<Vec<f64>> = Vec::new();
    let mut all_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_abs_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_metrics: Vec<f64> = Vec::new();

    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        let has_both = boot_y.iter().any(|&v| v < 0.5) && boot_y.iter().any(|&v| v >= 0.5);
        if !has_both {
            continue;
        }
        if let Some(refit) =
            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, 25, 1e-6)
        {
            all_beta_t.push(refit.beta_t.clone());
            let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
            all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
            all_coefs.push(coefs);
            all_metrics.push(refit.accuracy);
        }
    }

    (all_beta_t, all_coefs, all_abs_coefs, all_metrics)
}

/// Bootstrap logistic coefficients for regression depth.
fn bootstrap_logistic_coefs(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    n_boot: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let mut boot_coefs = Vec::new();
    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        let has_both = boot_y.iter().any(|&v| v < 0.5) && boot_y.iter().any(|&v| v >= 0.5);
        if !has_both {
            continue;
        }
        if let Some(refit) =
            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, 25, 1e-6)
        {
            boot_coefs.push((0..ncomp).map(|k| refit.coefficients[1 + k]).collect());
        }
    }
    boot_coefs
}
