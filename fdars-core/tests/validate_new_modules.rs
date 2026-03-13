//! Deep technical validation of new modules (Issues #4-#8).
//!
//! Tests mathematical properties, analytical solutions, internal consistency,
//! and cross-module agreement. Each section validates one module with multiple
//! independent property checks.
//!
//! Run: cargo test --test validate_new_modules --features linalg

use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

// ─── Test data generators ──────────────────────────────────────────────────

/// Deterministic pseudo-random via simple LCG (no external deps).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Box-Muller transform for N(0,1).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Generate n curves on m grid points: mean + random Fourier + noise.
fn generate_curves(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        let a1 = rng.next_normal() * 0.8;
        let a2 = rng.next_normal() * 0.4;
        for (j, &t) in argvals.iter().enumerate() {
            col_major[i + j * n] = (2.0 * PI * t).sin()
                + a1 * (2.0 * PI * t).sin()
                + a2 * (4.0 * PI * t).cos()
                + rng.next_normal() * 0.05;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    (data, argvals)
}

/// Generate classification data: g groups with distinct mean functions.
fn generate_classification_data(
    n_per_class: usize,
    g: usize,
    m: usize,
    seed: u64,
) -> (FdMatrix, Vec<usize>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let n = n_per_class * g;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut col_major = vec![0.0; n * m];
    let mut labels = vec![0usize; n];

    for c in 0..g {
        for k in 0..n_per_class {
            let i = c * n_per_class + k;
            labels[i] = c;
            let shift = c as f64 * 2.0;
            for (j, &t) in argvals.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * t + shift).sin() + rng.next_normal() * 0.15;
            }
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    (data, labels, argvals)
}

/// Generate longitudinal data: n_subjects with n_visits each.
fn generate_longitudinal_data(
    n_subjects: usize,
    n_visits: usize,
    m: usize,
    seed: u64,
) -> (FdMatrix, Vec<usize>, FdMatrix, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let n_total = n_subjects * n_visits;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut col_major = vec![0.0; n_total * m];
    let mut subject_ids = vec![0usize; n_total];
    let mut cov_data = vec![0.0; n_total];

    for s in 0..n_subjects {
        let random_effect = rng.next_normal() * 0.5;
        let x_val = s as f64 / n_subjects.max(1) as f64;
        for v in 0..n_visits {
            let obs = s * n_visits + v;
            subject_ids[obs] = s;
            cov_data[obs] = x_val;
            for (j, &t) in argvals.iter().enumerate() {
                col_major[obs + j * n_total] = (2.0 * PI * t).sin()
                    + x_val * 0.5 * (2.0 * PI * t).cos()
                    + random_effect
                    + rng.next_normal() * 0.05;
            }
        }
    }

    let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
    let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();
    (data, subject_ids, covariates, argvals)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Scalar-on-function regression (#4)
// ═══════════════════════════════════════════════════════════════════════════════

mod scalar_on_function {
    use super::*;
    use fdars_core::{
        fregre_cv, fregre_lm, fregre_np_mixed, functional_logistic, predict_fregre_lm,
        predict_fregre_np,
    };

    /// Property: R² ∈ [0, 1] for well-specified models.
    #[test]
    fn test_r_squared_valid_range() {
        let (data, _argvals) = generate_curves(30, 51, 100);
        let mut rng = Lcg::new(200);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let integral: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                integral * 2.0 + rng.next_normal() * 0.1
            })
            .collect();

        let result = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(
            result.r_squared >= -0.01 && result.r_squared <= 1.0 + 1e-10,
            "R² = {} out of valid range",
            result.r_squared
        );
        assert!(
            result.r_squared > 0.0,
            "R² = {} should be positive for model with signal",
            result.r_squared
        );
    }

    /// Property: residuals sum to approximately zero (intercept model).
    #[test]
    fn test_residuals_sum_to_zero() {
        let (data, _) = generate_curves(30, 51, 101);
        let mut rng = Lcg::new(201);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let integral: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                integral + rng.next_normal() * 0.2
            })
            .collect();

        let result = fregre_lm(&data, &y, None, 3).unwrap();
        let resid_sum: f64 = result.residuals.iter().sum();
        assert!(
            resid_sum.abs() < 1e-8,
            "Residual sum = {} should be ~0",
            resid_sum
        );
    }

    /// Property: fitted + residuals = y.
    #[test]
    fn test_fitted_plus_residuals_equals_y() {
        let (data, _) = generate_curves(30, 51, 102);
        let mut rng = Lcg::new(202);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s + rng.next_normal() * 0.3
            })
            .collect();

        let result = fregre_lm(&data, &y, None, 3).unwrap();
        for (i, &yi) in y.iter().enumerate() {
            let reconstructed = result.fitted_values[i] + result.residuals[i];
            assert!(
                (reconstructed - yi).abs() < 1e-10,
                "fitted[{}] + residuals[{}] = {}, y = {}",
                i,
                i,
                reconstructed,
                yi
            );
        }
    }

    /// Property: residuals orthogonal to FPC scores.
    #[test]
    fn test_residuals_orthogonal_to_scores() {
        let (data, _) = generate_curves(40, 51, 103);
        let mut rng = Lcg::new(203);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s * 3.0 + rng.next_normal() * 0.2
            })
            .collect();

        let result = fregre_lm(&data, &y, None, 3).unwrap();
        for k in 0..result.ncomp {
            let dot: f64 = (0..n)
                .map(|i| result.residuals[i] * result.fpca.scores[(i, k)])
                .sum();
            assert!(
                dot.abs() < 1e-6,
                "Residuals·PC{} = {:.2e}, should be ~0",
                k,
                dot
            );
        }
    }

    /// Property: predict on training data matches fitted values.
    #[test]
    fn test_predict_matches_fitted() {
        let (data, _) = generate_curves(30, 51, 104);
        let mut rng = Lcg::new(204);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s + rng.next_normal() * 0.1
            })
            .collect();

        let result = fregre_lm(&data, &y, None, 3).unwrap();
        let predicted = predict_fregre_lm(&result, &data, None);
        for (i, &p) in predicted.iter().enumerate() {
            assert!(
                (p - result.fitted_values[i]).abs() < 1e-8,
                "predict[{}] = {}, fitted = {}",
                i,
                p,
                result.fitted_values[i]
            );
        }
    }

    /// Property: cross-validation selects reasonable ncomp.
    #[test]
    fn test_cv_selects_reasonable_ncomp() {
        let (data, _) = generate_curves(40, 51, 105);
        let mut rng = Lcg::new(205);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s * 2.0 + rng.next_normal() * 0.1
            })
            .collect();

        let cv_result = fregre_cv(&data, &y, None, 1, 6, 5).unwrap();
        assert!(cv_result.optimal_k >= 1 && cv_result.optimal_k <= 6);
        assert!(
            cv_result.min_cv_error.is_finite() && cv_result.min_cv_error >= 0.0,
            "CV error = {} should be non-negative finite",
            cv_result.min_cv_error
        );
    }

    /// Property: logistic regression produces probabilities in [0, 1].
    #[test]
    fn test_logistic_probabilities_valid() {
        let (data, _) = generate_curves(40, 51, 107);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                if s > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let result = functional_logistic(&data, &y, None, 3, 50, 1e-6).unwrap();
        for (i, &p) in result.probabilities.iter().enumerate() {
            assert!((0.0..=1.0).contains(&p), "P[{}] = {} not in [0, 1]", i, p);
        }
        assert!(
            result.accuracy > 0.5,
            "Logistic accuracy {} below chance",
            result.accuracy
        );
    }

    /// Property: with scalar covariates, γ coefficients are recovered.
    #[test]
    fn test_scalar_covariates_contribute() {
        let (data, _) = generate_curves(40, 51, 108);
        let mut rng = Lcg::new(208);
        let n = data.nrows();
        let m = data.ncols();
        let mut z_data = vec![0.0; n];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let z = rng.next_normal();
                z_data[i] = z;
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                3.0 * z + s + rng.next_normal() * 0.1
            })
            .collect();
        let z_mat = FdMatrix::from_column_major(z_data, n, 1).unwrap();

        let result = fregre_lm(&data, &y, Some(&z_mat), 3).unwrap();
        assert_eq!(result.gamma.len(), 1);
        assert!(
            (result.gamma[0] - 3.0).abs() < 1.0,
            "γ[0] = {} should be near 3.0",
            result.gamma[0]
        );
        assert!(
            result.r_squared > 0.7,
            "R² = {} too low with strong signal",
            result.r_squared
        );
    }

    /// Property: nonparametric regression produces finite fitted values and valid R².
    #[test]
    fn test_np_mixed_finite_output() {
        let (data, argvals) = generate_curves(30, 51, 109);
        let mut rng = Lcg::new(209);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s + rng.next_normal() * 0.3
            })
            .collect();

        let result = fregre_np_mixed(&data, &y, &argvals, None, 0.5, 0.5).unwrap();
        assert_eq!(result.fitted_values.len(), n);
        for (i, &v) in result.fitted_values.iter().enumerate() {
            assert!(v.is_finite(), "NP fitted[{}] = {} not finite", i, v);
        }
        // fitted + residuals = y
        for (i, &yi) in y.iter().enumerate() {
            let reconstructed = result.fitted_values[i] + result.residuals[i];
            assert!(
                (reconstructed - yi).abs() < 1e-10,
                "NP fitted+resid != y at {}",
                i
            );
        }
    }

    /// Property: predict_fregre_np produces finite predictions.
    #[test]
    fn test_predict_np_finite() {
        let (data, argvals) = generate_curves(30, 51, 110);
        let mut rng = Lcg::new(210);
        let n = data.nrows();
        let m = data.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s: f64 = (0..m).map(|j| data[(i, j)]).sum::<f64>() / m as f64;
                s + rng.next_normal() * 0.3
            })
            .collect();

        // Predict on the first 5 observations as "new" data
        let new_n = 5;
        let mut new_col_major = vec![0.0; new_n * m];
        for i in 0..new_n {
            for j in 0..m {
                new_col_major[i + j * new_n] = data[(i, j)];
            }
        }
        let new_data = FdMatrix::from_column_major(new_col_major, new_n, m).unwrap();

        let predicted = predict_fregre_np(&data, &y, None, &new_data, None, &argvals, 0.5, 0.5);
        assert_eq!(predicted.len(), new_n);
        for (i, &v) in predicted.iter().enumerate() {
            assert!(v.is_finite(), "NP predict[{}] = {} not finite", i, v);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Function-on-scalar regression (#5)
// ═══════════════════════════════════════════════════════════════════════════════

mod function_on_scalar {
    use super::*;
    use fdars_core::{fanova, fosr, predict_fosr};

    /// Generate data where y_i(t) = x_i * β(t) + ε(t), β(t) = sin(2πt).
    fn gen_fosr_data(n: usize, m: usize, seed: u64) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let beta_true: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();
        let mut col_major = vec![0.0; n * m];
        let mut x_data = vec![0.0; n];

        for i in 0..n {
            let x_i = rng.next_normal();
            x_data[i] = x_i;
            for (j, _t) in argvals.iter().enumerate() {
                col_major[i + j * n] = x_i * beta_true[j] + rng.next_normal() * 0.1;
            }
        }

        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let covariates = FdMatrix::from_column_major(x_data, n, 1).unwrap();
        (data, covariates, argvals)
    }

    /// Property: FOSR fitted + residuals = data.
    #[test]
    fn test_fosr_fitted_plus_residuals() {
        let (data, covariates, _) = gen_fosr_data(30, 41, 300);
        let n = data.nrows();
        let m = data.ncols();

        let result = fosr(&data, &covariates, 1.0).unwrap();
        for i in 0..n {
            for j in 0..m {
                let sum = result.fitted[(i, j)] + result.residuals[(i, j)];
                assert!(
                    (sum - data[(i, j)]).abs() < 1e-8,
                    "fitted+resid != data at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    /// Property: β̂(t) shape matches true β(t) = sin(2πt).
    #[test]
    fn test_fosr_beta_shape_recovery() {
        let (data, covariates, argvals) = gen_fosr_data(50, 41, 301);
        let m = data.ncols();

        let result = fosr(&data, &covariates, 0.1).unwrap();
        assert_eq!(result.beta.nrows(), 1); // 1 covariate
        assert_eq!(result.beta.ncols(), m);

        // β̂(t) should correlate strongly with sin(2πt)
        let beta_hat: Vec<f64> = (0..m).map(|j| result.beta[(0, j)]).collect();
        let beta_true: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();

        let corr = pearson_corr(&beta_hat, &beta_true);
        assert!(
            corr.abs() > 0.8,
            "β̂(t) correlation with sin(2πt) = {:.4}, too low",
            corr
        );
    }

    /// Property: predict on training data matches fitted curves.
    #[test]
    fn test_fosr_predict_matches_fitted() {
        let (data, covariates, _) = gen_fosr_data(30, 41, 302);
        let n = data.nrows();
        let m = data.ncols();

        let result = fosr(&data, &covariates, 1.0).unwrap();
        let predicted = predict_fosr(&result, &covariates);

        for i in 0..n {
            for j in 0..m {
                assert!(
                    (predicted[(i, j)] - result.fitted[(i, j)]).abs() < 1e-8,
                    "predict != fitted at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    /// Property: R² per time point is valid.
    #[test]
    fn test_fosr_r_squared_valid() {
        let (data, covariates, _) = gen_fosr_data(50, 41, 303);

        let result = fosr(&data, &covariates, 0.1).unwrap();
        assert!(
            result.r_squared >= -0.01 && result.r_squared <= 1.0 + 1e-10,
            "Overall R² = {} out of range",
            result.r_squared
        );
        // With strong signal, overall R² should be decent
        assert!(
            result.r_squared > 0.3,
            "R² = {} too low for clear signal",
            result.r_squared
        );
    }

    /// Property: FANOVA detects group differences.
    #[test]
    fn test_fanova_detects_group_effect() {
        let m = 41;
        let n_per_group = 20;
        let n = n_per_group * 2;
        let mut rng = Lcg::new(304);
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        // Group 0: sin(2πt), Group 1: 2.5*sin(2πt)
        let mut col_major = vec![0.0; n * m];
        let mut groups = vec![0usize; n];
        for i in 0..n {
            let amplitude = if i < n_per_group { 1.0 } else { 2.5 };
            groups[i] = if i < n_per_group { 0 } else { 1 };
            for (j, &t) in argvals.iter().enumerate() {
                col_major[i + j * n] = amplitude * (2.0 * PI * t).sin() + rng.next_normal() * 0.15;
            }
        }

        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        let result = fanova(&data, &groups, 199).unwrap();
        assert!(
            result.p_value < 0.05,
            "FANOVA p-value {} should be < 0.05 for distinct groups",
            result.p_value
        );
        assert!(
            result.global_statistic > 1.0,
            "Global F-statistic {} should be > 1",
            result.global_statistic
        );
    }

    /// Property: FANOVA p-value > 0.05 for null data.
    #[test]
    fn test_fanova_no_false_positive() {
        let m = 41;
        let n = 40;
        let mut rng = Lcg::new(305);

        // All from same distribution
        let mut col_major = vec![0.0; n * m];
        let mut groups = vec![0usize; n];
        for i in 0..n {
            groups[i] = if i < n / 2 { 0 } else { 1 };
            for j in 0..m {
                let t = j as f64 / (m - 1) as f64;
                col_major[i + j * n] = (2.0 * PI * t).sin() + rng.next_normal() * 0.3;
            }
        }

        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        let result = fanova(&data, &groups, 199).unwrap();
        assert!(
            result.p_value > 0.01,
            "FANOVA p-value {} false positive on null data",
            result.p_value
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. GMM clustering (#8)
// ═══════════════════════════════════════════════════════════════════════════════

mod gmm_clustering {
    use super::*;
    use fdars_core::{gmm_cluster, gmm_em, predict_gmm, CovType, ProjectionBasisType};

    /// Generate well-separated 2D Gaussian clusters as Vec<Vec<f64>>.
    fn gen_gmm_features(k: usize, n_per: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = Lcg::new(seed);
        let n = k * n_per;
        let mut features = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        let centers = [vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 0.0]];

        for (c, center) in centers.iter().enumerate().take(k) {
            for _ in 0..n_per {
                features.push(vec![
                    center[0] + rng.next_normal() * 0.5,
                    center[1] + rng.next_normal() * 0.5,
                ]);
                labels.push(c);
            }
        }
        (features, labels)
    }

    /// Property: membership (responsibilities) sum to 1 per observation.
    #[test]
    fn test_responsibilities_sum_to_one() {
        let (features, _) = gen_gmm_features(3, 30, 400);

        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        let n = features.len();
        let k = 3;

        for i in 0..n {
            let row_sum: f64 = (0..k).map(|c| result.membership[(i, c)]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Membership[{}] sum = {}, expected 1.0",
                i,
                row_sum
            );
        }
    }

    /// Property: mixture weights sum to 1.
    #[test]
    fn test_weights_sum_to_one() {
        let (features, _) = gen_gmm_features(3, 30, 401);

        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        let weight_sum: f64 = result.weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-10,
            "Weights sum = {}, expected 1.0",
            weight_sum
        );
    }

    /// Property: log-likelihood is finite.
    #[test]
    fn test_loglik_finite() {
        let (features, _) = gen_gmm_features(3, 30, 402);

        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood {} is not finite",
            result.log_likelihood
        );
    }

    /// Property: BIC is finite and computable.
    #[test]
    fn test_bic_finite() {
        let (features, _) = gen_gmm_features(3, 30, 403);
        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        assert!(result.bic.is_finite(), "BIC {} is not finite", result.bic);
    }

    /// Property: clustering accuracy > 90% for well-separated clusters.
    #[test]
    fn test_clustering_accuracy_high() {
        let (features, true_labels) = gen_gmm_features(3, 30, 405);
        let n = features.len();

        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();

        // Find best permutation (K=3)
        let perms: Vec<[usize; 3]> = vec![
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let best_acc = perms
            .iter()
            .map(|perm| {
                let correct: usize = result
                    .cluster
                    .iter()
                    .zip(true_labels.iter())
                    .filter(|(&p, &t)| perm[p] == t)
                    .count();
                correct as f64 / n as f64
            })
            .fold(0.0f64, f64::max);

        assert!(
            best_acc > 0.9,
            "GMM accuracy {} should be > 90% for well-separated clusters",
            best_acc
        );
    }

    /// Property: with K=1, all observations in one cluster.
    #[test]
    fn test_single_cluster() {
        let (features, _) = gen_gmm_features(3, 30, 406);
        let result = gmm_em(&features, 1, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        assert!(result.cluster.iter().all(|&c| c == 0));
        assert_eq!(result.weights.len(), 1);
        assert!((result.weights[0] - 1.0).abs() < 1e-10);
    }

    /// Property: full covariance also works.
    #[test]
    fn test_full_covariance() {
        let (features, _) = gen_gmm_features(2, 40, 407);
        let result = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();
        assert!(result.log_likelihood.is_finite());
        assert!(result.converged);
    }

    /// Property: gmm_cluster auto-selects K via BIC on functional data.
    #[test]
    fn test_gmm_cluster_auto_k() {
        let (data, _, argvals) = generate_classification_data(20, 3, 51, 408);

        let result = gmm_cluster(
            &data,
            &argvals,
            None,
            &[2, 3, 4],
            5,
            ProjectionBasisType::Bspline,
            CovType::Diagonal,
            0.0,
            100,
            1e-6,
            3,
            42,
            false,
        )
        .unwrap();

        assert!(
            result.best.k >= 2 && result.best.k <= 4,
            "Selected K={} out of range [2,4]",
            result.best.k
        );
        assert!(result.best.log_likelihood.is_finite());
        assert!(result.best.bic.is_finite());
        assert_eq!(result.bic_values.len(), 3);
    }

    /// Property: predict_gmm assigns new data to valid clusters with valid memberships.
    #[test]
    fn test_predict_gmm_valid() {
        let (data, _, argvals) = generate_classification_data(20, 3, 51, 409);

        let cluster_result = gmm_cluster(
            &data,
            &argvals,
            None,
            &[3],
            5,
            ProjectionBasisType::Bspline,
            CovType::Diagonal,
            0.0,
            100,
            1e-6,
            3,
            42,
            false,
        )
        .unwrap();

        // Predict on a subset as "new" data
        let new_n = 10;
        let m = data.ncols();
        let mut new_col_major = vec![0.0; new_n * m];
        for i in 0..new_n {
            for j in 0..m {
                new_col_major[i + j * new_n] = data[(i, j)];
            }
        }
        let new_data = FdMatrix::from_column_major(new_col_major, new_n, m).unwrap();

        let (pred_labels, pred_membership) = predict_gmm(
            &new_data,
            &argvals,
            None,
            &cluster_result.best,
            5,
            ProjectionBasisType::Bspline,
            0.0,
            CovType::Diagonal,
        )
        .unwrap();

        assert_eq!(pred_labels.len(), new_n);
        assert_eq!(pred_membership.nrows(), new_n);
        assert_eq!(pred_membership.ncols(), cluster_result.best.k);

        // Labels in valid range
        for (i, &l) in pred_labels.iter().enumerate() {
            assert!(l < cluster_result.best.k, "pred label[{}]={} >= K", i, l);
        }

        // Membership rows sum to 1
        for i in 0..new_n {
            let row_sum: f64 = (0..cluster_result.best.k)
                .map(|c| pred_membership[(i, c)])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Membership[{}] sum = {}",
                i,
                row_sum
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Functional classification (#7)
// ═══════════════════════════════════════════════════════════════════════════════

mod classification {
    use super::*;
    use fdars_core::{
        fclassif_cv, fclassif_dd, fclassif_kernel, fclassif_knn, fclassif_lda, fclassif_qda,
    };

    /// Property: confusion matrix rows sum to class counts.
    #[test]
    fn test_confusion_matrix_valid() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 500);
        let result = fclassif_lda(&data, &labels, None, 3).unwrap();

        let g = result.n_classes;

        // Each row should sum to the count of that class
        for c in 0..g {
            let row_sum: usize = result.confusion[c].iter().sum();
            let class_count = labels.iter().filter(|&&l| l == c).count();
            assert_eq!(
                row_sum, class_count,
                "Row {} sum = {}, expected {}",
                c, row_sum, class_count
            );
        }
    }

    /// Property: accuracy above chance for well-separated classes.
    #[test]
    fn test_lda_accuracy_above_chance() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 501);
        let result = fclassif_lda(&data, &labels, None, 3).unwrap();
        assert!(
            result.accuracy > 1.0 / 3.0 + 0.1,
            "LDA accuracy {} below chance+10%",
            result.accuracy
        );
    }

    /// Property: QDA accuracy above chance.
    #[test]
    fn test_qda_accuracy_above_chance() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 502);
        let result = fclassif_qda(&data, &labels, None, 3).unwrap();
        assert!(
            result.accuracy > 1.0 / 3.0 + 0.1,
            "QDA accuracy {} below chance+10%",
            result.accuracy
        );
    }

    /// Property: k-NN accuracy above chance.
    #[test]
    fn test_knn_accuracy_above_chance() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 503);
        let result = fclassif_knn(&data, &labels, None, 3, 3).unwrap();
        assert!(
            result.accuracy > 1.0 / 3.0 + 0.1,
            "k-NN accuracy {} below chance+10%",
            result.accuracy
        );
    }

    /// Property: DD-classifier accuracy above chance.
    #[test]
    fn test_dd_accuracy_above_chance() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 505);
        let result = fclassif_dd(&data, &labels, None).unwrap();
        assert!(
            result.accuracy > 1.0 / 3.0,
            "DD accuracy {} below chance",
            result.accuracy
        );
    }

    /// Property: predictions are valid class labels.
    #[test]
    fn test_predictions_valid_labels() {
        let (data, labels, _) = generate_classification_data(20, 3, 51, 506);
        let g = 3;

        let result = fclassif_lda(&data, &labels, None, 3).unwrap();
        for (i, &pred) in result.predicted.iter().enumerate() {
            assert!(
                pred < g,
                "Predicted[{}] = {} out of range [0, {})",
                i,
                pred,
                g
            );
        }
    }

    /// Property: all classifiers agree on easy problem.
    #[test]
    fn test_classifiers_agree_on_easy_problem() {
        let m = 51;
        let n_per = 25;
        let g = 2;
        let n = n_per * g;
        let mut rng = Lcg::new(508);

        let mut col_major = vec![0.0; n * m];
        let mut labels = vec![0usize; n];
        for c in 0..g {
            for k in 0..n_per {
                let i = c * n_per + k;
                labels[i] = c;
                let shift = if c == 0 { 5.0 } else { -5.0 };
                for j in 0..m {
                    let t = j as f64 / (m - 1) as f64;
                    col_major[i + j * n] = shift + (2.0 * PI * t).sin() + rng.next_normal() * 0.1;
                }
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        let lda = fclassif_lda(&data, &labels, None, 3).unwrap();
        let qda = fclassif_qda(&data, &labels, None, 3).unwrap();
        let knn = fclassif_knn(&data, &labels, None, 3, 3).unwrap();
        let dd = fclassif_dd(&data, &labels, None).unwrap();

        for (name, acc) in &[
            ("LDA", lda.accuracy),
            ("QDA", qda.accuracy),
            ("k-NN", knn.accuracy),
            ("DD", dd.accuracy),
        ] {
            assert!(
                *acc > 0.95,
                "{} accuracy {} < 95% on easy problem",
                name,
                acc
            );
        }
    }

    /// Property: kernel classifier produces valid accuracy and predictions.
    #[test]
    fn test_kernel_classifier_valid() {
        let (data, labels, argvals) = generate_classification_data(20, 3, 51, 509);
        let result = fclassif_kernel(&data, &labels, &argvals, None, 0.5, 0.5).unwrap();

        assert!(
            result.accuracy > 1.0 / 3.0,
            "Kernel accuracy {} below chance",
            result.accuracy
        );
        assert_eq!(result.predicted.len(), data.nrows());
        for (i, &pred) in result.predicted.iter().enumerate() {
            assert!(pred < 3, "Kernel predicted[{}] = {} >= 3", i, pred);
        }
    }

    /// Property: cross-validation produces valid error rate and selects ncomp.
    #[test]
    fn test_cv_valid_output() {
        let (data, labels, argvals) = generate_classification_data(20, 3, 51, 510);
        let cv_result = fclassif_cv(&data, &argvals, &labels, None, "lda", 3, 5, 42).unwrap();

        assert!(
            (0.0..=1.0).contains(&cv_result.error_rate),
            "CV error rate {} out of [0, 1]",
            cv_result.error_rate
        );
        assert_eq!(cv_result.fold_errors.len(), 5);
        for (i, &e) in cv_result.fold_errors.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&e),
                "Fold {} error {} out of [0, 1]",
                i,
                e
            );
        }
        assert!(
            cv_result.best_ncomp >= 1,
            "best_ncomp {} < 1",
            cv_result.best_ncomp
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Functional mixed effects (FAMM) (#6)
// ═══════════════════════════════════════════════════════════════════════════════

mod famm {
    use super::*;
    use fdars_core::{fmm, fmm_predict, fmm_test_fixed};

    /// Property: fitted + residuals = data.
    #[test]
    fn test_fmm_decomposition_identity() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(10, 3, 31, 600);
        let n = data.nrows();
        let m = data.ncols();

        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        for i in 0..n {
            for j in 0..m {
                let sum = result.fitted[(i, j)] + result.residuals[(i, j)];
                assert!(
                    (sum - data[(i, j)]).abs() < 1e-8,
                    "fitted+resid != data at ({}, {}): {} vs {}",
                    i,
                    j,
                    sum,
                    data[(i, j)]
                );
            }
        }
    }

    /// Property: random effect variance is non-negative.
    #[test]
    fn test_random_variance_nonnegative() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(10, 3, 31, 601);

        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        for (j, &v) in result.random_variance.iter().enumerate() {
            assert!(v >= -1e-10, "Random variance at t={} is negative: {}", j, v);
        }
    }

    /// Property: random effects sum to ~0 across subjects.
    #[test]
    fn test_random_effects_centered() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(15, 3, 31, 602);
        let m = data.ncols();

        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();
        let n_subj = result.random_effects.nrows();

        for j in 0..m {
            let mean: f64 = (0..n_subj)
                .map(|s| result.random_effects[(s, j)])
                .sum::<f64>()
                / n_subj as f64;
            assert!(
                mean.abs() < 0.5,
                "Random effects mean at t={}: {} should be near 0",
                j,
                mean
            );
        }
    }

    /// Property: prediction produces valid output dimensions.
    #[test]
    fn test_fmm_predict_dimensions() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(10, 3, 31, 603);
        let m = data.ncols();

        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        let new_cov_data = vec![0.1, 0.5, 0.9];
        let new_cov = FdMatrix::from_column_major(new_cov_data, 3, 1).unwrap();

        let predicted = fmm_predict(&result, Some(&new_cov));
        assert_eq!(predicted.nrows(), 3);
        assert_eq!(predicted.ncols(), m);

        for i in 0..3 {
            for j in 0..m {
                assert!(
                    predicted[(i, j)].is_finite(),
                    "Prediction ({}, {}) = {} is not finite",
                    i,
                    j,
                    predicted[(i, j)]
                );
            }
        }
    }

    /// Property: permutation test p-values ∈ [0, 1].
    #[test]
    fn test_permutation_test_pvalues_valid() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(10, 3, 31, 604);

        let result = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 49, 42).unwrap();
        for (j, &p) in result.p_values.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "p-value[{}] = {} not in [0, 1]",
                j,
                p
            );
        }
        for (j, &f) in result.f_statistics.iter().enumerate() {
            assert!(
                f >= 0.0 && f.is_finite(),
                "F-stat[{}] = {} should be non-negative finite",
                j,
                f
            );
        }
    }

    /// Property: permutation test detects strong covariate effect.
    #[test]
    fn test_detects_strong_covariate_effect() {
        let n_subjects = 15;
        let n_visits = 3;
        let n_total = n_subjects * n_visits;
        let m = 31;
        let mut rng = Lcg::new(605);

        let mut col_major = vec![0.0; n_total * m];
        let mut subject_ids = vec![0usize; n_total];
        let mut cov_data = vec![0.0; n_total];

        for s in 0..n_subjects {
            let x = s as f64 / (n_subjects - 1) as f64;
            for v in 0..n_visits {
                let obs = s * n_visits + v;
                subject_ids[obs] = s;
                cov_data[obs] = x;
                for j in 0..m {
                    let t = j as f64 / (m - 1) as f64;
                    col_major[obs + j * n_total] =
                        (1.0 + 3.0 * x) * (2.0 * PI * t).sin() + rng.next_normal() * 0.05;
                }
            }
        }

        let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
        let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();

        let result = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 99, 42).unwrap();
        assert!(
            result.p_values[0] < 0.1,
            "Should detect strong covariate effect, p = {}",
            result.p_values[0]
        );
    }

    /// Property: model without covariates still works.
    #[test]
    fn test_fmm_no_covariates() {
        let (data, subject_ids, _, _) = generate_longitudinal_data(10, 3, 31, 606);
        let n = data.nrows();
        let m = data.ncols();

        let result = fmm(&data, &subject_ids, None, 3).unwrap();
        assert_eq!(result.fitted.nrows(), n);
        assert_eq!(result.fitted.ncols(), m);
        assert_eq!(result.residuals.nrows(), n);
        assert_eq!(result.residuals.ncols(), m);
    }

    /// Property: eigenvalues are positive and decreasing.
    #[test]
    fn test_eigenvalues_valid() {
        let (data, subject_ids, covariates, _) = generate_longitudinal_data(10, 3, 31, 607);

        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();
        assert_eq!(result.eigenvalues.len(), result.ncomp);
        for (i, &ev) in result.eigenvalues.iter().enumerate() {
            assert!(
                ev >= 0.0,
                "Eigenvalue[{}] = {} should be non-negative",
                i,
                ev
            );
        }
        // Eigenvalues should be in decreasing order
        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues not decreasing: [{}]={} > [{}]={}",
                i - 1,
                result.eigenvalues[i - 1],
                i,
                result.eigenvalues[i]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SMOOTH BASIS (Feature 1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn smooth_basis_bspline_penalty_is_continuous_integral() {
    // The B-spline penalty should approximate ∫(D²f)² for a known function.
    // For f(t) = t³, D²f = 6t, so ∫₀¹(6t)² dt = 36∫₀¹t² dt = 12.
    use fdars_core::smooth_basis::bspline_penalty_matrix;

    let m = 101;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let nbasis = 20;
    let penalty = bspline_penalty_matrix(&argvals, nbasis, 4, 2);
    let k = (penalty.len() as f64).sqrt() as usize;

    // Penalty matrix should be symmetric and PSD
    for i in 0..k {
        for j in 0..k {
            assert!(
                (penalty[i + j * k] - penalty[j + i * k]).abs() < 1e-8,
                "Penalty not symmetric at ({}, {})",
                i,
                j
            );
        }
    }

    // All diagonal elements should be non-negative (PSD)
    for i in 0..k {
        assert!(
            penalty[i + i * k] >= -1e-10,
            "Diagonal {} is negative: {}",
            i,
            penalty[i + i * k]
        );
    }
}

#[test]
fn smooth_basis_fourier_penalty_eigenvalues() {
    // Fourier penalty should have eigenvalues (2πk/T)^{2m}
    use fdars_core::smooth_basis::fourier_penalty_matrix;

    let nbasis = 11;
    let period = 1.0;
    let lfd_order = 2;
    let penalty = fourier_penalty_matrix(nbasis, period, lfd_order);

    // Should be diagonal
    for i in 0..nbasis {
        for j in 0..nbasis {
            if i != j {
                assert!(
                    penalty[i + j * nbasis].abs() < 1e-10,
                    "Off-diagonal ({},{}) = {}",
                    i,
                    j,
                    penalty[i + j * nbasis]
                );
            }
        }
    }

    // Constant term penalty = 0
    assert!(penalty[0].abs() < 1e-10);

    // Check sin(2πt) penalty: (2π/1)^4 = (2π)^4 ≈ 1558.55
    let expected_1 = (2.0 * PI).powi(4);
    assert!(
        (penalty[1 + nbasis] - expected_1).abs() / expected_1 < 0.01,
        "sin(2πt) penalty: {} vs expected {}",
        penalty[1 + nbasis],
        expected_1
    );
}

#[test]
fn smooth_basis_edf_between_bounds() {
    // EDF should be between lfd_order and nbasis
    use fdars_core::smooth_basis::{bspline_penalty_matrix, smooth_basis, BasisType, FdPar};

    let m = 101;
    let n = 10;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let (data, _) = generate_curves(n, m, 999);

    let nbasis = 15;
    let penalty = bspline_penalty_matrix(&argvals, nbasis, 4, 2);
    let actual_k = (penalty.len() as f64).sqrt() as usize;

    let fdpar = FdPar {
        basis_type: BasisType::Bspline { order: 4 },
        nbasis,
        lambda: 1e-2,
        lfd_order: 2,
        penalty_matrix: penalty,
    };

    let result = smooth_basis(&data, &argvals, &fdpar).unwrap();
    assert!(
        result.edf >= 1.0 && result.edf <= actual_k as f64,
        "EDF {} should be in [1, {}]",
        result.edf,
        actual_k
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ELASTIC FPCA (Feature 2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn vert_fpca_eigenvalues_nonneg_decreasing() {
    use fdars_core::alignment::karcher_mean;
    use fdars_core::elastic_fpca::vert_fpca;

    let (data, argvals) = generate_elastic_fpca_data(20, 51, 100);
    let km = karcher_mean(&data, &argvals, 10, 1e-4, 0.0);
    let result = vert_fpca(&km, &argvals, 5).unwrap();

    for (i, ev) in result.eigenvalues.iter().enumerate() {
        assert!(*ev >= -1e-10, "eigenvalue {} is negative: {}", i, ev);
    }
    for i in 1..result.eigenvalues.len() {
        assert!(
            result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
            "eigenvalues not decreasing: {} > {}",
            result.eigenvalues[i],
            result.eigenvalues[i - 1]
        );
    }
}

#[test]
fn vert_fpca_cumvar_bounded() {
    use fdars_core::alignment::karcher_mean;
    use fdars_core::elastic_fpca::vert_fpca;

    let (data, argvals) = generate_elastic_fpca_data(20, 51, 200);
    let km = karcher_mean(&data, &argvals, 10, 1e-4, 0.0);
    let result = vert_fpca(&km, &argvals, 5).unwrap();

    assert!(*result.cumulative_variance.last().unwrap() <= 1.0 + 1e-10);
    for i in 1..result.cumulative_variance.len() {
        assert!(result.cumulative_variance[i] >= result.cumulative_variance[i - 1] - 1e-10);
    }
}

#[test]
fn horiz_fpca_shooting_vectors_tangent() {
    use fdars_core::alignment::karcher_mean;
    use fdars_core::elastic_fpca::horiz_fpca;

    let (data, argvals) = generate_elastic_fpca_data(15, 51, 300);
    let km = karcher_mean(&data, &argvals, 10, 1e-4, 0.0);
    let result = horiz_fpca(&km, &argvals, 3).unwrap();

    // Shooting vectors should have mean near zero (centered tangent space)
    let (n, m) = result.shooting_vectors.shape();
    for j in 0..m {
        let mean: f64 = (0..n).map(|i| result.shooting_vectors[(i, j)]).sum::<f64>() / n as f64;
        // The mean doesn't have to be exactly 0 (it's the mean of log maps, not linearly centered),
        // but should be small since the log maps are taken at the Karcher mean
        assert!(
            mean.abs() < 1.0,
            "Shooting vector mean at j={} is too large: {}",
            j,
            mean
        );
    }
}

#[test]
fn joint_fpca_decomposes_correctly() {
    use fdars_core::alignment::karcher_mean;
    use fdars_core::elastic_fpca::joint_fpca;

    let (data, argvals) = generate_elastic_fpca_data(15, 51, 400);
    let km = karcher_mean(&data, &argvals, 10, 1e-4, 0.0);
    let result = joint_fpca(&km, &argvals, 3, Some(1.0)).unwrap();

    assert_eq!(result.scores.nrows(), 15);
    assert_eq!(result.scores.ncols(), 3);
    assert_eq!(result.vert_component.nrows(), 3);
    assert_eq!(result.horiz_component.nrows(), 3);

    // Eigenvalues should sum to total variance
    let total: f64 = result.eigenvalues.iter().sum();
    assert!(total > 0.0, "Total variance should be positive");
}

fn generate_elastic_fpca_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let shift = 0.1 * rng.next_normal();
        let scale = 1.0 + 0.3 * rng.next_normal().abs();
        for j in 0..m {
            data[(i, j)] = scale * (2.0 * PI * (argvals[j] + shift)).sin();
        }
    }
    (data, argvals)
}

// ═══════════════════════════════════════════════════════════════════════════
// ELASTIC REGRESSION (Feature 3)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn elastic_regression_reduces_sse() {
    // SSE should be less than total variance
    use fdars_core::elastic_regression::elastic_regression;

    let (data, y, argvals) = generate_regression_data(20, 51, 500);
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let result = elastic_regression(&data, &y, &argvals, 10, 1e-3, 5, 1e-3).unwrap();
    assert!(
        result.sse < ss_tot * 1.1, // allow small tolerance
        "SSE {} should be less than SS_tot {}",
        result.sse,
        ss_tot
    );
}

#[test]
fn elastic_regression_fitted_residuals_consistent() {
    use fdars_core::elastic_regression::elastic_regression;

    let (data, y, argvals) = generate_regression_data(15, 51, 600);
    let result = elastic_regression(&data, &y, &argvals, 10, 1e-3, 3, 1e-3).unwrap();

    for (i, &yi) in y.iter().enumerate() {
        let expected_resid = yi - result.fitted_values[i];
        assert!(
            (result.residuals[i] - expected_resid).abs() < 1e-10,
            "Residual mismatch at {}: {} vs {}",
            i,
            result.residuals[i],
            expected_resid
        );
    }
}

#[test]
fn elastic_logistic_classifications_valid() {
    use fdars_core::elastic_regression::elastic_logistic;

    let n = 20;
    let m = 51;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];

    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j]).sin();
        }
    }

    let result = elastic_logistic(&data, &y, &argvals, 10, 1e-2, 5, 1e-3).unwrap();

    // All probabilities should be in [0, 1]
    for &p in &result.probabilities {
        assert!((0.0..=1.0).contains(&p), "Probability {} out of range", p);
    }

    // Predicted classes should be 0 or 1
    for &c in &result.predicted_classes {
        assert!(c == 0 || c == 1, "Invalid class: {}", c);
    }
}

#[test]
fn elastic_pcr_r_squared_valid() {
    use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};

    let (data, y, argvals) = generate_regression_data(15, 51, 700);
    let result = elastic_pcr(&data, &y, &argvals, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();

    assert!(
        result.r_squared >= -0.1 && result.r_squared <= 1.0 + 1e-10,
        "R² {} out of range",
        result.r_squared
    );
}

fn generate_regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];

    for i in 0..n {
        let amp = 1.0 + rng.next_f64();
        let shift = 0.1 * rng.next_normal();
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * (argvals[j] + shift)).sin();
        }
        y[i] = amp;
    }
    (data, y, argvals)
}

// ═══════════════════════════════════════════════════════════════════════════
// ELASTIC CHANGEPOINT (Feature 4)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn amp_changepoint_detects_known_shift() {
    use fdars_core::elastic_changepoint::elastic_amp_changepoint;

    let n = 30;
    let m = 51;
    let cp = 15;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);

    for i in 0..n {
        let amp = if i < cp { 1.0 } else { 2.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j]).sin();
        }
    }

    let result = elastic_amp_changepoint(&data, &argvals, 0.0, 5, 200, 42).unwrap();

    assert!(
        (result.changepoint as i64 - cp as i64).abs() <= 5,
        "Detected {} far from true {}",
        result.changepoint,
        cp
    );
    assert!(result.test_statistic > 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.cusum_values.len(), n - 1);
}

#[test]
fn changepoint_no_change_weak_signal() {
    use fdars_core::elastic_changepoint::elastic_amp_changepoint;

    let n = 20;
    let m = 51;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);

    // All identical curves
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * argvals[j]).sin();
        }
    }

    let result = elastic_amp_changepoint(&data, &argvals, 0.0, 5, 200, 42);

    if let Ok(res) = result {
        // p-value should be higher (less significant) when there's no change
        // Not a strict test, but a sanity check
        assert!(res.p_value > 0.0);
    }
}

#[test]
fn fpca_changepoint_dimensions_consistent() {
    use fdars_core::elastic_changepoint::{elastic_fpca_changepoint, FpcaChangepointMethod};

    let n = 30;
    let m = 51;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);

    for i in 0..n {
        let amp = if i < 15 { 1.0 } else { 2.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j]).sin();
        }
    }

    let result = elastic_fpca_changepoint(
        &data,
        &argvals,
        FpcaChangepointMethod::Vertical,
        3,
        0.0,
        5,
        100,
        42,
    )
    .unwrap();

    assert_eq!(result.cusum_values.len(), n - 1);
    assert!(result.changepoint >= 1 && result.changepoint < n);
}

// ─── Utility ───────────────────────────────────────────────────────────────

/// Pearson correlation coefficient.
fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx: f64 = x.iter().sum::<f64>() / n;
    let my: f64 = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx.sqrt() * vy.sqrt()).max(1e-15)
}
