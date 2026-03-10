//! Example 25: Model Explainability
//!
//! Interprets scalar-on-function regression models using:
//! - Bootstrap confidence intervals (`bootstrap_ci_fregre_lm`)
//! - PDP/ICE curves (`functional_pdp`)
//! - Beta decomposition (`beta_decomposition`)
//! - Permutation importance (`fpc_permutation_importance`)
//! - Influence diagnostics (`influence_diagnostics`, `dfbetas_dffits`)
//! - SHAP values (`fpc_shap_values`)
//! - Pointwise variable importance (`pointwise_importance`)
//! - VIF (`fpc_vif`)
//! - Prediction intervals (`prediction_intervals`)
//! - ALE plots (`fpc_ale`)
//! - Expected calibration error (`expected_calibration_error`)
//! - Conformal prediction intervals (`conformal_prediction_residuals`)
//! - Regression depth diagnostics (`regression_depth`)
//! - Stability / robustness analysis (`explanation_stability`)
//! - Anchor explanations (`anchor_explanation`)

use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn generate_data(n: usize, m: usize, t: &[f64]) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let mut col_major = vec![0.0; n * m];
    let mut y = vec![0.0; n];
    let mut y_binary = vec![0.0; n];
    for i in 0..n {
        let a = (i as f64 * 7.0 + 3.0) % 10.0 / 5.0 - 1.0;
        let b = (i as f64 * 11.0 + 1.0) % 10.0 / 5.0 - 1.0;
        y[i] = 2.0 * a + 0.5 * b + 0.1 * ((i * 13) % 100) as f64 / 100.0;
        y_binary[i] = if y[i] > 0.0 { 1.0 } else { 0.0 };
        for (j, &tj) in t.iter().enumerate() {
            let noise = 0.05 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            col_major[i + j * n] = a * (2.0 * PI * tj).sin() + b * (4.0 * PI * tj).cos() + noise;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    (data, y, y_binary)
}

fn demo_core_explainability(
    fit: &fdars_core::scalar_on_function::FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    t: &[f64],
    n: usize,
) {
    println!("\n=== 1. Bootstrap Confidence Intervals ===");
    let ci = fdars_core::bootstrap_ci_fregre_lm(data, y, None, 3, 200, 0.05, 42).unwrap();
    let n_sig = fdars_core::significant_regions(&ci.lower, &ci.upper)
        .unwrap()
        .len();
    println!("  Significant regions: {}", n_sig);

    println!("\n=== 2. Beta Decomposition ===");
    let dec = fdars_core::beta_decomposition(fit).unwrap();
    for (k, &prop) in dec.variance_proportion.iter().enumerate() {
        println!(
            "  FPC {}: coef={:+.4}, variance proportion={:.1}%",
            k,
            dec.coefficients[k],
            prop * 100.0
        );
    }

    println!("\n=== 3. Partial Dependence (PDP/ICE) ===");
    for comp in 0..3 {
        let pdp = fdars_core::functional_pdp(fit, data, None, comp, 10).unwrap();
        let pdp_range = pdp.pdp_curve.last().unwrap() - pdp.pdp_curve[0];
        println!("  FPC {}: PDP range={:+.4}", comp, pdp_range);
    }

    println!("\n=== 4. Permutation Importance ===");
    let perm = fdars_core::fpc_permutation_importance(fit, data, y, 50, 42).unwrap();
    println!("  Baseline R² = {:.4}", perm.baseline_metric);
    for (k, &imp) in perm.importance.iter().enumerate() {
        println!("  FPC {}: importance={:.4} (R² drop)", k, imp);
    }

    println!("\n=== 5. Pointwise Variable Importance ===");
    let pw = fdars_core::pointwise_importance(fit).unwrap();
    let top_idx = pw
        .importance_normalized
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!(
        "  Peak importance at t[{}] = {:.3} ({:.1}%)",
        top_idx,
        t[top_idx],
        pw.importance_normalized[top_idx] * 100.0
    );

    println!("\n=== 6. Influence Diagnostics ===");
    let infl = fdars_core::influence_diagnostics(fit, data, None).unwrap();
    let max_cook = infl
        .cooks_distance
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "  Most influential obs: i={} (Cook's D={:.4})",
        max_cook.0, max_cook.1
    );

    println!("\n=== 7. DFBETAS / DFFITS ===");
    let db = fdars_core::dfbetas_dffits(fit, data, None).unwrap();
    let n_flagged = db
        .dffits
        .iter()
        .filter(|d| d.abs() > db.dffits_cutoff)
        .count();
    println!("  Obs exceeding DFFITS cutoff: {}/{}", n_flagged, n);

    println!("\n=== 8. Variance Inflation Factors ===");
    let vif = fdars_core::fpc_vif(fit, data, None).unwrap();
    for (&v, label) in vif.vif.iter().zip(&vif.labels) {
        println!("  {}: VIF={:.4}", label, v);
    }

    println!("\n=== 9. SHAP Values ===");
    let shap = fdars_core::fpc_shap_values(fit, data, None).unwrap();
    println!("  Base value (E[ŷ]): {:.4}", shap.base_value);

    println!("\n=== 10. Prediction Intervals (95%) ===");
    let pi = fdars_core::prediction_intervals(fit, data, None, data, None, 0.95).unwrap();
    let covered: usize = (0..n)
        .filter(|&i| y[i] >= pi.lower[i] && y[i] <= pi.upper[i])
        .count();
    println!("  Coverage on training data: {}/{}", covered, n);

    println!("\n=== 11. ALE ===");
    for comp in 0..3 {
        let ale = fdars_core::fpc_ale(fit, data, None, comp, 10).unwrap();
        let ale_range = ale.ale_values.last().unwrap() - ale.ale_values[0];
        println!("  FPC {}: ALE range={:+.4}", comp, ale_range);
    }

    println!("\n=== 12. Friedman H-statistic ===");
    for (j, k) in [(0, 1), (0, 2), (1, 2)] {
        let h = fdars_core::friedman_h_statistic(fit, data, j, k, 10).unwrap();
        println!("  H²(FPC{}, FPC{}) = {:.6}", j, k, h.h_squared);
    }
}

fn demo_logistic(data: &FdMatrix, y_binary: &[f64]) {
    println!("\n=== 13. Logistic Model Explainability ===");
    let fit_log = functional_logistic(data, y_binary, None, 3, 25, 1e-6).unwrap();
    println!("  Accuracy: {:.1}%", fit_log.accuracy * 100.0);

    let shap_log = fdars_core::fpc_shap_values_logistic(&fit_log, data, None, 500, 42).unwrap();
    println!("  SHAP base (mean P(Y=1)): {:.4}", shap_log.base_value);

    let perm_log =
        fdars_core::fpc_permutation_importance_logistic(&fit_log, data, y_binary, 50, 42).unwrap();
    for (k, &imp) in perm_log.importance.iter().enumerate() {
        println!("  FPC {}: permutation importance={:.4}", k, imp);
    }

    println!("\n=== 14. Expected Calibration Error ===");
    let ece = fdars_core::expected_calibration_error(&fit_log, y_binary, 10).unwrap();
    println!(
        "  ECE={:.4}, MCE={:.4}, ACE={:.4}",
        ece.ece, ece.mce, ece.ace
    );
}

fn demo_advanced(
    fit: &fdars_core::scalar_on_function::FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    n: usize,
) {
    println!("\n=== 15. Conformal Prediction ===");
    let cp =
        fdars_core::conformal_prediction_residuals(fit, data, y, data, None, None, 0.3, 0.1, 42)
            .unwrap();
    println!(
        "  Residual quantile q̂ = {:.4}, coverage = {:.1}%",
        cp.residual_quantile,
        cp.coverage * 100.0
    );

    println!("\n=== 16. Regression Depth (Fraiman-Muniz) ===");
    let rd = fdars_core::regression_depth(
        fit,
        data,
        y,
        None,
        30,
        fdars_core::DepthType::FraimanMuniz,
        42,
    )
    .unwrap();
    println!(
        "  β̂ depth = {:.4}, mean score depth = {:.4}",
        rd.beta_depth, rd.mean_score_depth
    );

    println!("\n=== 17. Stability Analysis ===");
    let sa = fdars_core::explanation_stability(data, y, None, 3, 50, 42).unwrap();
    println!(
        "  R² std = {:.4}, importance stability = {:.4}",
        sa.metric_std, sa.importance_stability
    );

    println!("\n=== 18. Anchor Explanations ===");
    for obs in [0, n / 2, n - 1] {
        let ar = fdars_core::anchor_explanation(fit, data, None, obs, 0.9, 5).unwrap();
        let conds: Vec<String> = ar
            .rule
            .conditions
            .iter()
            .map(|c| {
                format!(
                    "FPC{}∈[{:.2},{:.2}]",
                    c.component, c.lower_bound, c.upper_bound
                )
            })
            .collect();
        let rule_str = if conds.is_empty() {
            "∅".to_string()
        } else {
            conds.join(" ∧ ")
        };
        println!(
            "  Obs {}: precision={:.2}, coverage={:.2}, rule: {}",
            obs, ar.rule.precision, ar.rule.coverage, rule_str
        );
    }
}

fn main() {
    let n = 60;
    let m = 51;
    let t = uniform_grid(m);
    let (data, y, y_binary) = generate_data(n, m, &t);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    println!("=== Model Explainability for Scalar-on-Function Regression ===");
    println!(
        "  n={}, m={}, ncomp={}, R²={:.4}",
        n, m, fit.ncomp, fit.r_squared
    );

    demo_core_explainability(&fit, &data, &y, &t, n);
    demo_logistic(&data, &y_binary);
    demo_advanced(&fit, &data, &y, n);
    println!("\nDone.");
}
