//! Example 20: Scalar-on-Function Regression
//!
//! Predicts a scalar response from functional predictors using:
//! - FPC-based linear model (`fregre_lm`)
//! - Cross-validation for component selection (`fregre_cv`)
//! - Nonparametric mixed kernel regression (`fregre_np_mixed`)
//! - Functional logistic regression (`functional_logistic`)

use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{
    fregre_cv, fregre_lm, fregre_np_mixed, functional_logistic, predict_fregre_lm,
};
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let n = 50;
    let m = 51;
    let t = uniform_grid(m);

    // Generate functional data: X_i(t) = a_i*sin(2πt) + b_i*cos(4πt)
    // Scalar response: y_i = 2*a_i + 0.5*b_i + noise
    let mut col_major = vec![0.0; n * m];
    let mut y = vec![0.0; n];
    let mut y_binary = vec![0.0; n];

    for i in 0..n {
        let a = (i as f64 * 7.0 + 3.0) % 10.0 / 5.0 - 1.0;
        let b = (i as f64 * 11.0 + 1.0) % 10.0 / 5.0 - 1.0;
        y[i] = 2.0 * a + 0.5 * b + 0.1 * ((i * 13) % 100) as f64 / 100.0;
        y_binary[i] = if y[i] > 0.0 { 1.0 } else { 0.0 };
        for (j, &tj) in t.iter().enumerate() {
            let noise = 0.1 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            col_major[i + j * n] = a * (2.0 * PI * tj).sin() + b * (4.0 * PI * tj).cos() + noise;
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    // ── 1. FPC-based linear model ──────────────────────────────────────────
    println!("=== Scalar-on-Function: FPC Linear Model ===");
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    println!("  R²     = {:.4}", fit.r_squared);
    println!("  R²_adj = {:.4}", fit.r_squared_adj);
    println!("  Residual SE = {:.4}", fit.residual_se);
    println!("  Components used: {}", fit.ncomp);

    // Predict on training data
    let preds = predict_fregre_lm(&fit, &data, None);
    let pred_err: f64 = (0..n).map(|i| (preds[i] - y[i]).powi(2)).sum::<f64>() / n as f64;
    println!("  Training MSE = {:.6}", pred_err);

    // ── 2. Cross-validation for ncomp ──────────────────────────────────────
    println!("\n=== Cross-Validation for Component Selection ===");
    let cv = fregre_cv(&data, &y, None, 1, 6, 5).unwrap();
    println!("  K values tested: {:?}", cv.k_values);
    println!(
        "  CV errors: {:?}",
        cv.cv_errors
            .iter()
            .map(|e| format!("{e:.4}"))
            .collect::<Vec<_>>()
    );
    println!("  Optimal K = {}", cv.optimal_k);
    println!("  Min CV error = {:.4}", cv.min_cv_error);

    // ── 3. Nonparametric regression ────────────────────────────────────────
    println!("\n=== Nonparametric Mixed Kernel Regression ===");
    let np = fregre_np_mixed(&data, &y, &t, None, 0.5, 0.5).unwrap();
    println!("  R² = {:.4}", np.r_squared);
    println!("  CV error = {:.4}", np.cv_error);

    // ── 4. Functional logistic regression ──────────────────────────────────
    println!("\n=== Functional Logistic Regression ===");
    let logit = functional_logistic(&data, &y_binary, None, 3, 100, 1e-6).unwrap();
    println!("  Accuracy = {:.2}%", logit.accuracy * 100.0);
    println!("  Iterations = {}", logit.iterations);
    println!("  Log-likelihood = {:.4}", logit.log_likelihood);
}
