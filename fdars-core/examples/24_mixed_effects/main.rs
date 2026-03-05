//! Example 24: Functional Mixed Effects Models (FAMM)
//!
//! Fits models for repeated functional measurements with subject-level effects:
//! - Functional mixed model via FPCA + iterative GLS/REML (`fmm`)
//! - Prediction for new subjects (`fmm_predict`)
//! - Hypothesis testing on fixed effects (`fmm_test_fixed`)

use fdars_core::famm::{fmm, fmm_predict, fmm_test_fixed};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let n_subjects = 15;
    let n_visits = 3;
    let m = 41;
    let t = uniform_grid(m);
    let n_total = n_subjects * n_visits;

    // Generate repeated functional measurements:
    // Y_sv(t) = sin(2πt) + x_s * cos(2πt) + b_s + noise
    let mut col_major = vec![0.0; n_total * m];
    let mut subject_ids = vec![0usize; n_total];
    let mut cov_data = vec![0.0; n_total];

    for s in 0..n_subjects {
        let x_s = s as f64 / (n_subjects - 1) as f64; // covariate in [0, 1]
        let b_s = 0.4 * (s as f64 - n_subjects as f64 / 2.0) / n_subjects as f64; // random effect

        for v in 0..n_visits {
            let obs = s * n_visits + v;
            subject_ids[obs] = s;
            cov_data[obs] = x_s;

            for (j, &tj) in t.iter().enumerate() {
                let noise = 0.05 * ((obs * 7 + j * 3) % 100) as f64 / 100.0;
                col_major[obs + j * n_total] =
                    (2.0 * PI * tj).sin() + x_s * (2.0 * PI * tj).cos() + b_s + noise;
            }
        }
    }

    let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
    let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();

    // ── 1. Fit functional mixed model ─────────────────────────────────────
    println!("=== Functional Mixed Effects Model ===");
    println!("  {n_subjects} subjects x {n_visits} visits = {n_total} curves, {m} time points\n");

    let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

    println!("  Components used: {}", result.ncomp);
    println!("  Subjects: {}", result.n_subjects);
    println!(
        "  sigma2_u (per component): [{}]",
        result
            .sigma2_u
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  sigma2_eps: {:.6}", result.sigma2_eps);

    // Fitted quality
    let mut sum_all = 0.0;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..n_total {
        for j in 0..m {
            sum_all += data[(i, j)];
        }
    }
    let overall_mean = sum_all / (n_total * m) as f64;
    for i in 0..n_total {
        for j in 0..m {
            ss_tot += (data[(i, j)] - overall_mean).powi(2);
            ss_res += result.residuals[(i, j)].powi(2);
        }
    }
    let r2 = 1.0 - ss_res / ss_tot;
    let mse = ss_res / (n_total * m) as f64;
    println!("  Fitted R² = {:.4}", r2);
    println!("  Fitted MSE = {:.6}", mse);

    // ── 2. Predict for new subject ────────────────────────────────────────
    println!("\n=== Prediction for New Subject (x=0.5) ===");
    let new_cov = FdMatrix::from_column_major(vec![0.5], 1, 1).unwrap();
    let predicted = fmm_predict(&result, Some(&new_cov));
    println!(
        "  Predicted curve range: [{:.3}, {:.3}]",
        (0..m)
            .map(|j| predicted[(0, j)])
            .fold(f64::INFINITY, f64::min),
        (0..m)
            .map(|j| predicted[(0, j)])
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // ── 3. Hypothesis test on fixed effect ────────────────────────────────
    println!("\n=== Permutation Test for Fixed Effect (99 permutations) ===");
    let test = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 99, 42).unwrap();
    println!("  Test statistic: {:.4}", test.f_statistics[0]);
    println!("  p-value: {:.4}", test.p_values[0]);
    println!(
        "  Significant: {}",
        if test.p_values[0] < 0.05 { "yes" } else { "no" }
    );
}
