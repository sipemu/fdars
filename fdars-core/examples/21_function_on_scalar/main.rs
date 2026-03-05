//! Example 21: Function-on-Scalar Regression
//!
//! Predicts a functional response from scalar predictors using:
//! - Penalized pointwise OLS (`fosr`) with automatic smoothing
//! - FPC-based regression (`fosr_fpc`) matching R's fda.usc approach
//! - Functional ANOVA (`fanova`) for group comparisons

use fdars_core::function_on_scalar::{fanova, fosr, fosr_fpc, predict_fosr};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let n = 40;
    let m = 51;
    let t = uniform_grid(m);

    // Generate data: Y_i(t) = sin(2πt) + age_i * t + group_i * cos(4πt) + noise
    let mut col_major = vec![0.0; n * m];
    let mut pred_data = vec![0.0; n * 2];
    let mut groups = vec![0usize; n];

    for i in 0..n {
        let age = i as f64 / n as f64;
        let group = if i % 2 == 0 { 1.0 } else { 0.0 };
        pred_data[i] = age; // column 0
        pred_data[i + n] = group; // column 1
        groups[i] = if i < n / 2 { 0 } else { 1 };

        for (j, &tj) in t.iter().enumerate() {
            let mu = (2.0 * PI * tj).sin();
            let beta1 = tj;
            let beta2 = (4.0 * PI * tj).cos();
            let noise = 0.05 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            col_major[i + j * n] = mu + age * beta1 + group * beta2 + noise;
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let predictors = FdMatrix::from_column_major(pred_data, n, 2).unwrap();

    // ── 1. Penalized FOSR (auto-lambda via GCV) ───────────────────────────
    println!("=== Penalized Function-on-Scalar Regression ===");
    let fit = fosr(&data, &predictors, -1.0).unwrap();
    println!("  Global R² = {:.4}", fit.r_squared);
    println!("  Lambda (GCV) = {:.4}", fit.lambda);
    println!("  GCV = {:.6}", fit.gcv);

    // Predict for a new subject (age=0.5, group=1)
    let new_pred = FdMatrix::from_column_major(vec![0.5, 1.0], 1, 2).unwrap();
    let predicted = predict_fosr(&fit, &new_pred);
    println!(
        "  Predicted curve range: [{:.3}, {:.3}]",
        (0..m)
            .map(|j| predicted[(0, j)])
            .fold(f64::INFINITY, f64::min),
        (0..m)
            .map(|j| predicted[(0, j)])
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // ── 2. FPC-based FOSR ─────────────────────────────────────────────────
    println!("\n=== FPC-based FOSR (3 components) ===");
    let fpc_fit = fosr_fpc(&data, &predictors, 3).unwrap();
    println!("  Global R² = {:.4}", fpc_fit.r_squared);
    println!("  Components used: {}", fpc_fit.ncomp);
    for (j, bs) in fpc_fit.beta_scores.iter().enumerate() {
        println!(
            "  Predictor {} beta_scores: [{}]",
            j,
            bs.iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ── 3. Functional ANOVA ───────────────────────────────────────────────
    println!("\n=== Functional ANOVA (2 groups, 500 permutations) ===");
    let anova = fanova(&data, &groups, 500).unwrap();
    println!("  Number of groups: {}", anova.n_groups);
    println!("  Global F-statistic: {:.4}", anova.global_statistic);
    println!("  Permutation p-value: {:.4}", anova.p_value);
    println!(
        "  Significant: {}",
        if anova.p_value < 0.05 { "yes" } else { "no" }
    );
}
