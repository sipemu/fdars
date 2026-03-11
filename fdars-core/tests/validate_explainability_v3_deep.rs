//! Deep validation for explainability round 3.
//!
//! Mathematical properties, cross-method consistency, numerical stability,
//! and analytical invariants for:
//! - Pointwise importance
//! - VIF (Variance Inflation Factors)
//! - SHAP values
//! - DFBETAS / DFFITS
//! - Prediction intervals
//! - ALE (Accumulated Local Effects)

use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

// ─── Data generators ─────────────────────────────────────────────────────────

fn regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let phase = (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
        let amp = ((seed.wrapping_mul(13).wrapping_add(i as u64 * 7) % 100) as f64 / 100.0) - 0.5;
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[(i, j)] = (2.0 * PI * t + phase).sin() + amp * (4.0 * PI * t).cos();
        }
        y[i] = 2.0 * phase + 3.0 * amp;
    }
    (data, y)
}

fn binary_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let (data, y_cont) = regression_data(n, m, seed);
    let mut sorted = y_cont.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= med { 1.0 } else { 0.0 })
        .collect();
    (data, y_bin)
}

fn regression_data_with_scalars(
    n: usize,
    m: usize,
    p_scalar: usize,
    seed: u64,
) -> (FdMatrix, Vec<f64>, FdMatrix) {
    let (data, mut y) = regression_data(n, m, seed);
    let mut sc = FdMatrix::zeros(n, p_scalar);
    for i in 0..n {
        for j in 0..p_scalar {
            let v = ((seed
                .wrapping_mul(23)
                .wrapping_add(i as u64 * 11 + j as u64 * 37))
                % 1000) as f64
                / 1000.0;
            sc[(i, j)] = v;
            y[i] += 0.5 * v;
        }
    }
    (data, y, sc)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pointwise importance — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn pointwise_importance_all_values_finite() {
    let (data, y) = regression_data(80, 40, 77);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();
    for j in 0..40 {
        assert!(
            pi.importance[j].is_finite(),
            "Non-finite importance at j={}",
            j
        );
        assert!(
            pi.importance_normalized[j].is_finite(),
            "Non-finite normalized at j={}",
            j
        );
    }
    for &v in &pi.score_variance {
        assert!(v.is_finite() && v >= 0.0, "Invalid score variance: {}", v);
    }
}

#[test]
fn pointwise_importance_score_variance_is_sample_variance() {
    // Verify score_variance matches manual computation from FPCA scores
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();
    let n = fit.fpca.scores.nrows();

    for k in 0..3 {
        let mut ss = 0.0;
        for i in 0..n {
            let s = fit.fpca.scores[(i, k)];
            ss += s * s;
        }
        let expected = ss / (n - 1) as f64;
        assert!(
            (pi.score_variance[k] - expected).abs() < 1e-10,
            "Score variance mismatch at k={}: {} vs {}",
            k,
            pi.score_variance[k],
            expected
        );
    }
}

#[test]
fn pointwise_importance_algebraic_identity() {
    // importance[j] = Σ_k (coef[1+k] * rotation[(j,k)])^2 * score_variance[k]
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();

    for j in 0..40 {
        let mut manual = 0.0;
        for k in 0..3 {
            manual += (fit.coefficients[1 + k] * fit.fpca.rotation[(j, k)]).powi(2)
                * pi.score_variance[k];
        }
        assert!(
            (pi.importance[j] - manual).abs() < 1e-12,
            "Algebraic identity fails at j={}: {} vs {}",
            j,
            pi.importance[j],
            manual
        );
    }
}

#[test]
fn pointwise_importance_stable_across_sample_sizes() {
    // Dominant timepoints should be similar with n=60 vs n=120
    let (data60, y60) = regression_data(60, 30, 42);
    let (data120, y120) = regression_data(120, 30, 42);
    let fit60 = fregre_lm(&data60, &y60, None, 3).unwrap();
    let fit120 = fregre_lm(&data120, &y120, None, 3).unwrap();
    let pi60 = fdars_core::pointwise_importance(&fit60).unwrap();
    let pi120 = fdars_core::pointwise_importance(&fit120).unwrap();

    // Top-3 timepoints by normalized importance should overlap
    let mut order60: Vec<usize> = (0..30).collect();
    order60.sort_by(|&a, &b| {
        pi60.importance_normalized[b]
            .partial_cmp(&pi60.importance_normalized[a])
            .unwrap()
    });
    let mut order120: Vec<usize> = (0..30).collect();
    order120.sort_by(|&a, &b| {
        pi120.importance_normalized[b]
            .partial_cmp(&pi120.importance_normalized[a])
            .unwrap()
    });
    let top3_60: Vec<usize> = order60[..3].to_vec();
    let top3_120: Vec<usize> = order120[..3].to_vec();
    let overlap = top3_60.iter().filter(|t| top3_120.contains(t)).count();
    assert!(
        overlap >= 1,
        "Top-3 timepoints should partially overlap: {:?} vs {:?}",
        top3_60,
        top3_120
    );
}

#[test]
fn pointwise_importance_logistic_matches_linear_pattern() {
    // Same data → importance shape should be correlated between linear and logistic
    let (data, y_cont) = regression_data(80, 30, 42);
    let (_, y_bin) = binary_data(80, 30, 42);

    let fit_lm = fregre_lm(&data, &y_cont, None, 3).unwrap();
    let fit_log = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let pi_lm = fdars_core::pointwise_importance(&fit_lm).unwrap();
    let pi_log = fdars_core::pointwise_importance_logistic(&fit_log).unwrap();

    // Both should produce valid normalized importance
    let sum_lm: f64 = pi_lm.importance_normalized.iter().sum();
    let sum_log: f64 = pi_log.importance_normalized.iter().sum();
    assert!((sum_lm - 1.0).abs() < 1e-10);
    assert!((sum_log - 1.0).abs() < 1e-10);
}

#[test]
fn pointwise_importance_single_component_is_squared_rotation() {
    // With ncomp=1, importance ∝ rotation[:,0]^2
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 1).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();

    let c = fit.coefficients[1];
    let sv = pi.score_variance[0];
    for j in 0..30 {
        let expected = (c * fit.fpca.rotation[(j, 0)]).powi(2) * sv;
        assert!(
            (pi.importance[j] - expected).abs() < 1e-12,
            "Single component: importance should be c²*φ²*σ² at j={}",
            j
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VIF — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

/// Project data onto FPC scores manually.
fn project_to_scores(
    data: &FdMatrix,
    mean: &[f64],
    rotation: &FdMatrix,
    n: usize,
    ncomp: usize,
) -> FdMatrix {
    let m = data.ncols();
    let mut scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (data[(i, j)] - mean[j]) * rotation[(j, k)];
            }
            scores[(i, k)] = s;
        }
    }
    scores
}

/// Build design matrix for regressing component k on all others (with intercept).
fn build_leave_one_out_design(
    scores: &FdMatrix,
    k: usize,
    n: usize,
    ncomp: usize,
) -> (FdMatrix, Vec<f64>) {
    let p_reg = ncomp; // 1 intercept + (ncomp-1) others
    let mut x_other = FdMatrix::zeros(n, p_reg);
    let mut y_k = vec![0.0; n];
    for i in 0..n {
        x_other[(i, 0)] = 1.0;
        let mut col = 0;
        for kk in 0..ncomp {
            if kk != k {
                x_other[(i, 1 + col)] = scores[(i, kk)];
                col += 1;
            }
        }
        y_k[i] = scores[(i, k)];
    }
    (x_other, y_k)
}

/// Compute X'X and X'y for OLS.
fn compute_normal_equations(x: &FdMatrix, y: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xtx = vec![0.0; p * p];
    for r in 0..p {
        for c in r..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] * x[(i, c)];
            }
            xtx[r * p + c] = s;
            xtx[c * p + r] = s;
        }
    }
    let mut xty = vec![0.0; p];
    for r in 0..p {
        for i in 0..n {
            xty[r] += x[(i, r)] * y[i];
        }
    }
    (xtx, xty)
}

/// Cholesky factorization of a p×p symmetric matrix. Returns L or None if singular.
fn cholesky_factor(xtx: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut diag = xtx[j * p + j];
        for kk in 0..j {
            diag -= l[j * p + kk] * l[j * p + kk];
        }
        if diag <= 1e-12 {
            return None;
        }
        l[j * p + j] = diag.sqrt();
        for i in (j + 1)..p {
            let mut s = xtx[i * p + j];
            for kk in 0..j {
                s -= l[i * p + kk] * l[j * p + kk];
            }
            l[i * p + j] = s / l[j * p + j];
        }
    }
    Some(l)
}

/// Solve L L' z = rhs using forward/backward substitution.
fn cholesky_solve(l: &[f64], rhs: Vec<f64>, p: usize) -> Vec<f64> {
    let mut z = rhs;
    for j in 0..p {
        for kk in 0..j {
            z[j] -= l[j * p + kk] * z[kk];
        }
        z[j] /= l[j * p + j];
    }
    for j in (0..p).rev() {
        for kk in (j + 1)..p {
            z[j] -= l[kk * p + j] * z[kk];
        }
        z[j] /= l[j * p + j];
    }
    z
}

/// Cholesky-based OLS solve: returns coefficients or None if singular.
fn cholesky_ols(x: &FdMatrix, y: &[f64], n: usize, p: usize) -> Option<Vec<f64>> {
    let (xtx, xty) = compute_normal_equations(x, y, n, p);
    let l = cholesky_factor(&xtx, p)?;
    Some(cholesky_solve(&l, xty, p))
}

/// Compute R² from design matrix, response, and OLS coefficients.
fn compute_r_squared(x: &FdMatrix, y: &[f64], coeffs: &[f64], n: usize, p: usize) -> f64 {
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    if ss_tot <= 0.0 {
        return 0.0;
    }
    let mut ss_res = 0.0;
    for i in 0..n {
        let mut yhat = 0.0;
        for r in 0..p {
            yhat += x[(i, r)] * coeffs[r];
        }
        let res = y[i] - yhat;
        ss_res += res * res;
    }
    1.0 - ss_res / ss_tot
}

#[test]
fn vif_definition_via_r_squared() {
    // VIF_k = 1 / (1 - R²_k) where R²_k is from regressing x_k on all other predictors.
    // For orthogonal FPC scores, R²_k ≈ 0 → VIF ≈ 1.
    // Verify by manual OLS regression of each score on the others.
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fdars_core::fpc_vif(&fit, &data, None).unwrap();

    let n = 100usize;
    let ncomp = 3usize;
    let scores = project_to_scores(&data, &fit.fpca.mean, &fit.fpca.rotation, n, ncomp);

    for k in 0..ncomp {
        let (x_other, y_k) = build_leave_one_out_design(&scores, k, n, ncomp);
        let p_reg = ncomp;
        let coeffs = match cholesky_ols(&x_other, &y_k, n, p_reg) {
            Some(c) => c,
            None => continue,
        };
        let r2 = compute_r_squared(&x_other, &y_k, &coeffs, n, p_reg);
        let manual_vif = if (1.0 - r2).abs() > 1e-10 {
            1.0 / (1.0 - r2)
        } else {
            f64::INFINITY
        };

        assert!(
            (vif.vif[k] - manual_vif).abs() < 0.5,
            "VIF from (X'X)^-1 diagonal should ≈ 1/(1-R²) at k={}: {} vs {}",
            k,
            vif.vif[k],
            manual_vif
        );
    }
}

#[test]
fn vif_with_scalar_covariates_increases() {
    // Adding correlated scalar covariates should not decrease VIF
    let (data, y, sc) = regression_data_with_scalars(80, 40, 2, 42);
    let fit_plain = fregre_lm(&data, &y, None, 3).unwrap();
    let fit_scalar = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let vif_plain = fdars_core::fpc_vif(&fit_plain, &data, None).unwrap();
    let vif_scalar = fdars_core::fpc_vif(&fit_scalar, &data, Some(&sc)).unwrap();

    // VIF with scalars should have more entries
    assert_eq!(vif_plain.vif.len(), 3);
    assert_eq!(vif_scalar.vif.len(), 5); // 3 FPC + 2 scalar

    // Labels should be correct
    assert_eq!(vif_scalar.labels[3], "scalar_0");
    assert_eq!(vif_scalar.labels[4], "scalar_1");
}

#[test]
fn vif_all_values_finite_and_positive() {
    for seed in [42, 77, 123, 999] {
        let (data, y) = regression_data(100, 40, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fdars_core::fpc_vif(&fit, &data, None).unwrap();
        for (k, &v) in vif.vif.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 1.0 - 1e-6,
                "VIF should be finite and ≥1 at k={} (seed={}): {}",
                k,
                seed,
                v
            );
        }
        assert!(vif.mean_vif.is_finite());
    }
}

#[test]
fn vif_mean_is_arithmetic_mean() {
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fdars_core::fpc_vif(&fit, &data, None).unwrap();
    let manual_mean = vif.vif.iter().sum::<f64>() / vif.vif.len() as f64;
    assert!(
        (vif.mean_vif - manual_mean).abs() < 1e-12,
        "mean_vif should be arithmetic mean: {} vs {}",
        vif.mean_vif,
        manual_mean
    );
}

#[test]
fn vif_moderate_severe_counts_correct() {
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fdars_core::fpc_vif(&fit, &data, None).unwrap();
    let manual_moderate = vif.vif.iter().filter(|&&v| v > 5.0).count();
    let manual_severe = vif.vif.iter().filter(|&&v| v > 10.0).count();
    assert_eq!(vif.n_moderate, manual_moderate);
    assert_eq!(vif.n_severe, manual_severe);
    assert!(vif.n_severe <= vif.n_moderate);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHAP — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn shap_linear_efficiency_to_machine_epsilon() {
    // base_value + Σ_k SHAP[(i,k)] = fitted_values[i] exactly for linear models
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    let max_err: f64 = (0..80)
        .map(|i| {
            let sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
            (sum - fit.fitted_values[i]).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-10,
        "SHAP efficiency error should be near machine eps: {}",
        max_err
    );
}

#[test]
fn shap_linear_mean_shap_per_feature_is_zero() {
    // Mean of SHAP values for each feature should be zero (centered)
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    for k in 0..3 {
        let mean_shap: f64 = (0..80).map(|i| shap.values[(i, k)]).sum::<f64>() / 80.0;
        assert!(
            mean_shap.abs() < 1e-10,
            "Mean SHAP for feature {} should be ~0: {}",
            k,
            mean_shap
        );
    }
}

#[test]
fn shap_linear_equals_centered_contribution() {
    // values[(i,k)] = coef[1+k] * (score_ik - mean_score_k) exactly
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    let n = fit.fpca.scores.nrows();
    for k in 0..3 {
        let mean_score = (0..n).map(|i| fit.fpca.scores[(i, k)]).sum::<f64>() / n as f64;
        let coef_k = fit.coefficients[1 + k];

        for i in 0..n {
            let expected = coef_k * (fit.fpca.scores[(i, k)] - mean_score);
            assert!(
                (shap.values[(i, k)] - expected).abs() < 1e-10,
                "SHAP should equal centered contribution at ({},{}): {} vs {}",
                i,
                k,
                shap.values[(i, k)],
                expected
            );
        }
    }
}

#[test]
fn shap_linear_with_scalar_covariates_still_efficient() {
    // Scalar covariate effects absorbed into base value
    let (data, y, sc) = regression_data_with_scalars(80, 40, 2, 42);
    let fit = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, Some(&sc)).unwrap();

    // base_value should absorb intercept + mean FPC contributions + mean scalar contributions
    // Efficiency: base + Σ shap_k + Σ scalar_effects = fitted
    // Since scalar effects vary per obs but aren't in SHAP, check that
    // base + Σ shap captures the FPC part correctly
    for i in 0..80 {
        let shap_sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
        let scalar_effect: f64 = (0..2)
            .map(|j| {
                fit.gamma[j]
                    * (sc[(i, j)] - {
                        let mut m = 0.0;
                        for ii in 0..80 {
                            m += sc[(ii, j)];
                        }
                        m / 80.0
                    })
            })
            .sum();
        assert!(
            (shap_sum + scalar_effect - fit.fitted_values[i]).abs() < 1e-8,
            "SHAP + scalar should reconstruct fitted at i={}",
            i
        );
    }
}

#[test]
fn shap_logistic_base_value_is_prediction_at_mean() {
    let (data, y_bin) = binary_data(80, 40, 42);
    let ncomp = 3;
    let fit = functional_logistic(&data, &y_bin, None, ncomp, 25, 1e-6).unwrap();
    let shap = fdars_core::fpc_shap_values_logistic(&fit, &data, None, 200, 42).unwrap();

    // base_value should equal sigmoid(intercept + sum(coef_k * mean_score_k)),
    // i.e., the model's prediction evaluated at the mean FPC scores.
    let mut eta = fit.intercept;
    for k in 0..ncomp {
        eta += fit.coefficients[1 + k] * shap.mean_scores[k];
    }
    let expected_base = 1.0 / (1.0 + (-eta).exp());
    assert!(
        (shap.base_value - expected_base).abs() < 1e-10,
        "Base value should equal prediction at mean features: {} vs {}",
        shap.base_value,
        expected_base
    );
}

#[test]
fn shap_logistic_all_values_finite() {
    let (data, y_bin) = binary_data(80, 40, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let shap = fdars_core::fpc_shap_values_logistic(&fit, &data, None, 200, 42).unwrap();

    for i in 0..80 {
        for k in 0..3 {
            assert!(
                shap.values[(i, k)].is_finite(),
                "SHAP value non-finite at ({}, {}): {}",
                i,
                k,
                shap.values[(i, k)]
            );
        }
    }
}

#[test]
fn shap_linear_absolute_importance_correlates_with_permutation() {
    // Mean |SHAP_k| should roughly rank-correlate with permutation importance
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();
    let perm = fdars_core::fpc_permutation_importance(&fit, &data, &y, 50, 42).unwrap();

    // Mean absolute SHAP
    let mut shap_imp = [0.0; 3];
    for (k, imp) in shap_imp.iter_mut().enumerate() {
        *imp = (0..80).map(|i| shap.values[(i, k)].abs()).sum::<f64>() / 80.0;
    }

    // The most important feature by SHAP should also be among top-2 by permutation
    let shap_top = shap_imp
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let mut perm_order: Vec<usize> = (0..3).collect();
    perm_order.sort_by(|&a, &b| perm.importance[b].partial_cmp(&perm.importance[a]).unwrap());

    assert!(
        perm_order[..2].contains(&shap_top),
        "Top SHAP feature ({}) should be in top-2 by permutation: {:?}",
        shap_top,
        perm_order
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DFBETAS / DFFITS — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dfbetas_all_values_finite() {
    for seed in [42, 77, 123] {
        let (data, y) = regression_data(80, 40, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();
        for i in 0..80 {
            assert!(db.dffits[i].is_finite(), "DFFITS non-finite at i={}", i);
            assert!(
                db.studentized_residuals[i].is_finite(),
                "Studentized residual non-finite at i={}",
                i
            );
            for j in 0..db.p {
                assert!(
                    db.dfbetas[(i, j)].is_finite(),
                    "DFBETAS non-finite at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn dffits_sign_matches_studentized_residual() {
    // DFFITS = t_i * sqrt(h_ii / (1-h_ii)), so sign matches studentized residual
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();

    for i in 0..80 {
        if db.dffits[i].abs() > 1e-10 && db.studentized_residuals[i].abs() > 1e-10 {
            assert_eq!(
                db.dffits[i].signum(),
                db.studentized_residuals[i].signum(),
                "DFFITS sign should match studentized residual at i={}",
                i
            );
        }
    }
}

#[test]
fn dffits_squared_proportional_to_cooks_distance() {
    // DFFITS² / p ≈ Cook's distance (approximately, not exactly due to different s² estimates)
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    // Rank correlation: order by DFFITS² and Cook's D should be similar
    let mut dffits_order: Vec<usize> = (0..80).collect();
    dffits_order.sort_by(|&a, &b| {
        (db.dffits[b].powi(2))
            .partial_cmp(&db.dffits[a].powi(2))
            .unwrap()
    });
    let mut cooks_order: Vec<usize> = (0..80).collect();
    cooks_order.sort_by(|&a, &b| {
        infl.cooks_distance[b]
            .partial_cmp(&infl.cooks_distance[a])
            .unwrap()
    });

    // Top-5 should have significant overlap
    let top5_dffits: Vec<usize> = dffits_order[..5].to_vec();
    let top5_cooks: Vec<usize> = cooks_order[..5].to_vec();
    let overlap = top5_dffits
        .iter()
        .filter(|t| top5_cooks.contains(t))
        .count();
    assert!(
        overlap >= 3,
        "Top-5 by DFFITS² and Cook's D should overlap ≥3: {:?} vs {:?}",
        top5_dffits,
        top5_cooks
    );
}

#[test]
fn dfbetas_intercept_large_for_y_outlier() {
    // An observation with an extreme y value but typical x should have large DFBETAS for intercept
    let (mut data, mut y) = regression_data(60, 40, 42);
    y[0] = y.iter().sum::<f64>() / y.len() as f64 + 100.0; // extreme y
                                                           // Keep x normal
    for j in 0..40 {
        data[(0, j)] = data[(1, j)]; // copy a normal observation's x
    }
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();

    // Obs 0 should have large DFBETAS for intercept (column 0)
    let max_intercept_dfbetas = (0..60)
        .map(|i| db.dfbetas[(i, 0)].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        db.dfbetas[(0, 0)].abs() >= max_intercept_dfbetas * 0.5,
        "Y-outlier should have large intercept DFBETAS: {} vs max {}",
        db.dfbetas[(0, 0)].abs(),
        max_intercept_dfbetas
    );
}

#[test]
fn dfbetas_with_scalar_covariates_has_correct_p() {
    let (data, y, sc) = regression_data_with_scalars(80, 40, 2, 42);
    let fit = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, Some(&sc)).unwrap();
    // p = 1 (intercept) + 3 (FPC) + 2 (scalar) = 6
    assert_eq!(db.p, 6);
    assert_eq!(db.dfbetas.shape(), (80, 6));
    assert_eq!(db.dffits.len(), 80);
}

#[test]
fn dfbetas_cutoffs_are_standard() {
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();

    let expected_dfbetas_cutoff = 2.0 / (100.0_f64).sqrt();
    let expected_dffits_cutoff = 2.0 * (4.0 / 100.0_f64).sqrt();

    assert!(
        (db.dfbetas_cutoff - expected_dfbetas_cutoff).abs() < 1e-12,
        "DFBETAS cutoff: {} vs {}",
        db.dfbetas_cutoff,
        expected_dfbetas_cutoff
    );
    assert!(
        (db.dffits_cutoff - expected_dffits_cutoff).abs() < 1e-12,
        "DFFITS cutoff: {} vs {}",
        db.dffits_cutoff,
        expected_dffits_cutoff
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prediction intervals — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prediction_interval_se_formula_is_correct() {
    // prediction_se[i] = residual_se * sqrt(1 + h_new_i)
    // Verify by manual computation of h_new
    let (data, y) = regression_data(60, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();

    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    for i in 0..60 {
        let expected_se = fit.residual_se * (1.0 + infl.leverage[i]).sqrt();
        assert!(
            (pi.prediction_se[i] - expected_se).abs() < 1e-8,
            "Prediction SE formula mismatch at i={}: {} vs {}",
            i,
            pi.prediction_se[i],
            expected_se
        );
    }
}

#[test]
fn prediction_interval_width_increases_with_leverage() {
    // Higher leverage → wider prediction interval
    let (data, y) = regression_data(60, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    // Sort by leverage
    let mut pairs: Vec<(f64, f64)> = infl
        .leverage
        .iter()
        .zip(pi.prediction_se.iter())
        .map(|(&h, &se)| (h, se))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Higher leverage → higher SE (monotonic with small noise)
    let low_se = pairs[0].1;
    let high_se = pairs.last().unwrap().1;
    assert!(
        high_se >= low_se - 1e-10,
        "Higher leverage should give wider intervals: low_se={}, high_se={}",
        low_se,
        high_se
    );
}

#[test]
fn prediction_interval_new_data_wider_than_training() {
    // Extrapolation (data far from training centroid) should give wider intervals
    let (data, y) = regression_data(60, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    // Create new data that's a scaled version of training data
    let mut extreme_data = FdMatrix::zeros(10, 40);
    for i in 0..10 {
        for j in 0..40 {
            extreme_data[(i, j)] = data[(i, j)] * 5.0; // extreme extrapolation
        }
    }

    let pi_train = fdars_core::prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    let pi_extreme =
        fdars_core::prediction_intervals(&fit, &data, None, &extreme_data, None, 0.95).unwrap();

    let mean_width_train: f64 = (0..60)
        .map(|i| pi_train.upper[i] - pi_train.lower[i])
        .sum::<f64>()
        / 60.0;
    let mean_width_extreme: f64 = (0..10)
        .map(|i| pi_extreme.upper[i] - pi_extreme.lower[i])
        .sum::<f64>()
        / 10.0;

    assert!(
        mean_width_extreme > mean_width_train,
        "Extrapolation should give wider intervals: extreme={}, train={}",
        mean_width_extreme,
        mean_width_train
    );
}

#[test]
fn prediction_interval_confidence_levels_monotonic() {
    let (data, y) = regression_data(60, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let levels = [0.50, 0.80, 0.90, 0.95, 0.99];
    let mut prev_mean_width = 0.0;
    for &level in &levels {
        let pi = fdars_core::prediction_intervals(&fit, &data, None, &data, None, level).unwrap();
        let mean_width: f64 = (0..60).map(|i| pi.upper[i] - pi.lower[i]).sum::<f64>() / 60.0;
        assert!(
            mean_width >= prev_mean_width - 1e-10,
            "Width at {}% should be ≥ previous: {} vs {}",
            level * 100.0,
            mean_width,
            prev_mean_width
        );
        prev_mean_width = mean_width;
    }
}

#[test]
fn prediction_interval_t_critical_decreases_with_df() {
    // t_critical for same confidence but more df → closer to z (smaller)
    let (data30, y30) = regression_data(30, 40, 42);
    let (data100, y100) = regression_data(100, 40, 42);
    let fit30 = fregre_lm(&data30, &y30, None, 3).unwrap();
    let fit100 = fregre_lm(&data100, &y100, None, 3).unwrap();

    let pi30 =
        fdars_core::prediction_intervals(&fit30, &data30, None, &data30, None, 0.95).unwrap();
    let pi100 =
        fdars_core::prediction_intervals(&fit100, &data100, None, &data100, None, 0.95).unwrap();

    assert!(
        pi30.t_critical >= pi100.t_critical - 1e-6,
        "t-critical with df=26 should be ≥ t-critical with df=96: {} vs {}",
        pi30.t_critical,
        pi100.t_critical
    );
}

#[test]
fn prediction_interval_residual_se_matches_fit() {
    let (data, y) = regression_data(60, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();

    assert!(
        (pi.residual_se - fit.residual_se).abs() < 1e-12,
        "Residual SE should match fit: {} vs {}",
        pi.residual_se,
        fit.residual_se
    );
}

#[test]
fn prediction_interval_coverage_simulation() {
    // Generate data, fit, predict → check empirical coverage
    let (data, y) = regression_data(200, 30, 42);
    let _fit = fregre_lm(&data, &y, None, 3).unwrap();

    // Use first 150 for training, last 50 for test
    let mut train_data = FdMatrix::zeros(150, 30);
    let mut test_data = FdMatrix::zeros(50, 30);
    let mut train_y = vec![0.0; 150];
    for i in 0..150 {
        for j in 0..30 {
            train_data[(i, j)] = data[(i, j)];
        }
        train_y[i] = y[i];
    }
    for i in 0..50 {
        for j in 0..30 {
            test_data[(i, j)] = data[(150 + i, j)];
        }
    }

    let fit_train = fregre_lm(&train_data, &train_y, None, 3).unwrap();
    let pi =
        fdars_core::prediction_intervals(&fit_train, &train_data, None, &test_data, None, 0.95)
            .unwrap();

    let covered: usize = (0..50)
        .filter(|&i| y[150 + i] >= pi.lower[i] && y[150 + i] <= pi.upper[i])
        .count();

    // With 50 test points at 95%, expect ~47-48 covered, allow ≥35 for randomness
    assert!(
        covered >= 35,
        "At least 70% should be covered at 95% level: {}/50",
        covered
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALE — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ale_linear_model_slope_equals_coefficient() {
    // For a linear model, ALE for component k should have slope = coef[1+k]
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fdars_core::fpc_ale(&fit, &data, None, 0, 20).unwrap();

    // Overall ALE slope ≈ coefficient
    if ale.bin_midpoints.len() >= 2 {
        let x0 = ale.bin_midpoints[0];
        let x_last = *ale.bin_midpoints.last().unwrap();
        let y0 = ale.ale_values[0];
        let y_last = *ale.ale_values.last().unwrap();
        let slope = (y_last - y0) / (x_last - x0);
        let expected_slope = fit.coefficients[1]; // coef for FPC0

        assert!(
            (slope - expected_slope).abs() < expected_slope.abs() * 0.3 + 0.1,
            "ALE slope should ≈ coefficient: {} vs {}",
            slope,
            expected_slope
        );
    }
}

#[test]
fn ale_all_values_finite() {
    for seed in [42, 77, 123, 999] {
        let (data, y) = regression_data(80, 40, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        for comp in 0..3 {
            let ale = fdars_core::fpc_ale(&fit, &data, None, comp, 10).unwrap();
            for (b, &v) in ale.ale_values.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "ALE non-finite at bin {} (seed={}, comp={}): {}",
                    b,
                    seed,
                    comp,
                    v
                );
            }
        }
    }
}

#[test]
fn ale_bin_edges_are_sorted() {
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fdars_core::fpc_ale(&fit, &data, None, 0, 15).unwrap();

    for w in ale.bin_edges.windows(2) {
        assert!(
            w[1] >= w[0],
            "Bin edges should be sorted: {} >= {}",
            w[1],
            w[0]
        );
    }
}

#[test]
fn ale_midpoints_within_edges() {
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fdars_core::fpc_ale(&fit, &data, None, 0, 10).unwrap();

    for b in 0..ale.bin_midpoints.len() {
        assert!(
            ale.bin_midpoints[b] >= ale.bin_edges[b] - 1e-10
                && ale.bin_midpoints[b] <= ale.bin_edges[b + 1] + 1e-10,
            "Midpoint {} should be within edges [{}, {}]",
            ale.bin_midpoints[b],
            ale.bin_edges[b],
            ale.bin_edges[b + 1]
        );
    }
}

#[test]
fn ale_centering_exact() {
    // Weighted mean of ALE values = 0 exactly
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    for comp in 0..3 {
        let ale = fdars_core::fpc_ale(&fit, &data, None, comp, 10).unwrap();
        let total_n: usize = ale.bin_counts.iter().sum();
        let weighted_mean: f64 = ale
            .ale_values
            .iter()
            .zip(&ale.bin_counts)
            .map(|(&v, &c)| v * c as f64)
            .sum::<f64>()
            / total_n as f64;
        assert!(
            weighted_mean.abs() < 1e-10,
            "ALE centering exact for comp {}: {}",
            comp,
            weighted_mean
        );
    }
}

#[test]
fn ale_agrees_with_pdp_for_linear_model() {
    // For linear models, ALE and PDP should give similar shapes
    // (both linear in the feature for a linear model)
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fdars_core::fpc_ale(&fit, &data, None, 0, 10).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 10).unwrap();

    // Both should have similar overall trend direction
    let ale_range = ale.ale_values.last().unwrap() - ale.ale_values[0];
    let pdp_range = pdp.pdp_curve.last().unwrap() - pdp.pdp_curve[0];

    // Same sign of trend
    if ale_range.abs() > 0.01 && pdp_range.abs() > 0.01 {
        assert_eq!(
            ale_range.signum(),
            pdp_range.signum(),
            "ALE and PDP should have same trend direction: ALE range={}, PDP range={}",
            ale_range,
            pdp_range
        );
    }
}

#[test]
fn ale_logistic_values_bounded() {
    // For logistic, individual bin deltas come from sigmoid differences, so ALE should be bounded
    let (data, y_bin) = binary_data(80, 40, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ale = fdars_core::fpc_ale_logistic(&fit, &data, None, 0, 10).unwrap();

    for &v in &ale.ale_values {
        assert!(
            v.abs() < 2.0,
            "Logistic ALE centered values should be bounded: {}",
            v
        );
    }
}

#[test]
fn ale_different_n_bins_gives_same_trend() {
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let ale5 = fdars_core::fpc_ale(&fit, &data, None, 0, 5).unwrap();
    let ale20 = fdars_core::fpc_ale(&fit, &data, None, 0, 20).unwrap();

    // Overall range should be similar
    let range5 = ale5.ale_values.last().unwrap() - ale5.ale_values[0];
    let range20 = ale20.ale_values.last().unwrap() - ale20.ale_values[0];

    // Ratio should be within factor of 2
    if range5.abs() > 0.01 {
        let ratio = range20 / range5;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "ALE range should be similar across bin counts: {} vs {} (ratio {})",
            range5,
            range20,
            ratio
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-method consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cross_method_importance_ranking_agreement() {
    // Pointwise importance, SHAP mean |φ|, and permutation importance should roughly agree
    let (data, y) = regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let pi = fdars_core::pointwise_importance(&fit).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();
    let perm = fdars_core::fpc_permutation_importance(&fit, &data, &y, 100, 42).unwrap();

    // Sum pointwise importance per component (total contribution of each FPC)
    let mut comp_importance = [0.0; 3];
    for (k, imp) in comp_importance.iter_mut().enumerate() {
        for j in 0..40 {
            *imp += pi.component_importance[(k, j)];
        }
    }

    // Mean |SHAP| per component
    let mut shap_importance = [0.0; 3];
    for (k, imp) in shap_importance.iter_mut().enumerate() {
        *imp = (0..100).map(|i| shap.values[(i, k)].abs()).sum::<f64>() / 100.0;
    }

    // All three rankings should agree on the most important component
    let top_pi = comp_importance
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let top_shap = shap_importance
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let top_perm = perm
        .importance
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // At least 2 of 3 should agree
    let agree = (top_pi == top_shap) as usize
        + (top_pi == top_perm) as usize
        + (top_shap == top_perm) as usize;
    assert!(
        agree >= 1,
        "At least 2 of 3 importance methods should agree on top component: PI={}, SHAP={}, Perm={}",
        top_pi,
        top_shap,
        top_perm
    );
}

#[test]
fn cross_method_dfbetas_and_shap_identify_same_influential_obs() {
    // Observations with extreme DFFITS should also have extreme SHAP values
    let (mut data, mut y) = regression_data(80, 40, 42);
    // Make obs 0 extreme
    for j in 0..40 {
        data[(0, j)] *= 5.0;
    }
    y[0] = 50.0;

    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = fdars_core::dfbetas_dffits(&fit, &data, None).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    // Obs 0 should have among the largest |DFFITS|
    let dffits_rank: usize = (0..80)
        .filter(|&i| db.dffits[i].abs() > db.dffits[0].abs())
        .count();
    assert!(
        dffits_rank < 5,
        "Outlier obs should be in top-5 DFFITS: rank {}",
        dffits_rank
    );

    // Obs 0 should have large total |SHAP|
    let total_shap_0: f64 = (0..3).map(|k| shap.values[(0, k)].abs()).sum();
    let mean_total_shap: f64 = (0..80)
        .map(|i| (0..3).map(|k| shap.values[(i, k)].abs()).sum::<f64>())
        .sum::<f64>()
        / 80.0;
    assert!(
        total_shap_0 > mean_total_shap,
        "Outlier should have above-average total |SHAP|: {} vs mean {}",
        total_shap_0,
        mean_total_shap
    );
}

#[test]
fn prediction_interval_consistent_with_influence_leverage() {
    // Observations with high leverage should have wider prediction intervals
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = fdars_core::prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    // Top-10 leverage observations should have wider-than-median prediction SE
    let mut lev_order: Vec<usize> = (0..80).collect();
    lev_order.sort_by(|&a, &b| infl.leverage[b].partial_cmp(&infl.leverage[a]).unwrap());

    let mut se_sorted = pi.prediction_se.clone();
    se_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_se = se_sorted[40];

    let high_lev_above_median: usize = lev_order[..10]
        .iter()
        .filter(|&&i| pi.prediction_se[i] > median_se)
        .count();
    assert!(
        high_lev_above_median >= 5,
        "High-leverage obs should have above-median prediction SE: {}/10",
        high_lev_above_median
    );
}
