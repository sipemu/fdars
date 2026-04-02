//! Example 28: Berkeley Growth Data -- Cross-Validation Experiment
//!
//! Predicts adult height (final measurement at age 18) from growth curves
//! using a functional linear model with GCV-based component selection.
//!
//! Workflow:
//! 1. Simulate realistic growth data (Preece-Baines model)
//! 2. Smooth curves with P-splines (GCV lambda selection)
//! 3. Select ncomp via GCV / AIC / BIC
//! 4. Fit `fregre_lm` on training set
//! 5. 10-fold cross-validation for prediction error
//! 6. Compare PLS regression

use fdars_core::basis::pspline::{pspline_evaluate, pspline_fit_gcv};
use fdars_core::cv::create_folds;
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{
    fregre_lm, fregre_pls, model_selection_ncomp, predict_fregre_lm, predict_fregre_pls,
    SelectionCriterion,
};

/// Preece-Baines Model 1 growth curve.
/// h(t) = h1 - 2*(h1 - htheta) / (exp(s0*(t-theta)) + exp(s1*(t-theta)))
fn preece_baines(t: f64, h1: f64, htheta: f64, s0: f64, s1: f64, theta: f64) -> f64 {
    let e0 = (s0 * (t - theta)).exp();
    let e1 = (s1 * (t - theta)).exp();
    h1 - 2.0 * (h1 - htheta) / (e0 + e1)
}

/// Simple LCG pseudo-random number generator (deterministic, no dependencies).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn main() {
    println!("=== Berkeley Growth Data: Cross-Validation Experiment ===\n");

    // ── 1. Generate realistic growth data ──────────────────────────────────
    // Ages 1 to 18, 31 measurement points (as in original Berkeley data)
    let n = 93;
    let ages: Vec<f64> = (0..31).map(|i| 1.0 + 17.0 * i as f64 / 30.0).collect();
    let m_full = ages.len();

    let mut rng = Lcg::new(42);

    // Preece-Baines parameters with individual variation
    // Population means: h1=175cm, htheta=130cm, s0=0.15, s1=1.5, theta=12
    let mut heights_raw = vec![0.0; n * m_full]; // column-major
    let mut y_final = vec![0.0; n]; // final height at age 18 (noiseless)

    for i in 0..n {
        let h1 = 165.0 + rng.next_normal(0.0, 8.0); // adult height ~ N(165, 8)
        let htheta = 125.0 + rng.next_normal(0.0, 6.0); // height at puberty onset
        let s0 = 0.14 + rng.next_normal(0.0, 0.02); // pre-pubertal rate
        let s1 = 1.4 + rng.next_normal(0.0, 0.3); // pubertal rate
        let theta = 11.5 + rng.next_normal(0.0, 1.2); // puberty timing

        for (j, &age) in ages.iter().enumerate() {
            let h = preece_baines(age, h1, htheta, s0, s1, theta);
            let noise = rng.next_normal(0.0, 0.5); // measurement noise
            heights_raw[i + j * n] = h + noise;
        }
        // Response: true adult height (noiseless Preece-Baines at age 18)
        y_final[i] = preece_baines(18.0, h1, htheta, s0, s1, theta);
    }

    // Use only ages 1--14 as predictor (predict adult height from early growth)
    let cutoff_age = 14.0;
    let m_pred = ages.iter().filter(|&&a| a <= cutoff_age).count();
    let ages_pred: Vec<f64> = ages[..m_pred].to_vec();
    let t_norm: Vec<f64> = (0..m_pred)
        .map(|j| j as f64 / (m_pred - 1) as f64)
        .collect();

    let mut pred_vals = vec![0.0; n * m_pred];
    for i in 0..n {
        for j in 0..m_pred {
            pred_vals[i + j * n] = heights_raw[i + j * n];
        }
    }
    let data_raw = FdMatrix::from_column_major(pred_vals, n, m_pred).unwrap();
    let m = m_pred;
    println!(
        "Data: {} growth curves, {} ages ({:.0}--{:.0}), predicting adult height at 18",
        n,
        m,
        ages_pred[0],
        ages_pred[m - 1]
    );
    println!(
        "Response: adult height at age 18 (mean={:.1} cm, sd={:.1} cm)",
        y_final.iter().sum::<f64>() / n as f64,
        {
            let mean = y_final.iter().sum::<f64>() / n as f64;
            (y_final.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        }
    );

    // ── 2. Smooth with P-splines (GCV lambda) ─────────────────────────────
    println!("\n--- Smoothing with P-splines (GCV lambda) ---");
    let smooth_result = pspline_fit_gcv(&data_raw, &ages_pred, 15, 2).unwrap();
    println!(
        "  Lambda (GCV-optimal): {:.2e}",
        smooth_result.gcv // GCV score, lambda is implicit
    );
    println!("  Basis functions: {}", smooth_result.n_basis);

    // Evaluate smoothed curves on the same grid
    let data_smooth = pspline_evaluate(&smooth_result, &ages_pred);

    // ── 3. Model selection: GCV, AIC, BIC ──────────────────────────────────
    println!("\n--- Model Selection (ncomp) ---");
    let max_k = 10;
    let sel_gcv =
        model_selection_ncomp(&data_smooth, &y_final, None, max_k, SelectionCriterion::Gcv)
            .unwrap();
    let sel_aic =
        model_selection_ncomp(&data_smooth, &y_final, None, max_k, SelectionCriterion::Aic)
            .unwrap();
    let sel_bic =
        model_selection_ncomp(&data_smooth, &y_final, None, max_k, SelectionCriterion::Bic)
            .unwrap();

    println!("  {:>4}  {:>10}  {:>10}  {:>10}", "K", "AIC", "BIC", "GCV");
    for &(k, aic, bic, gcv) in &sel_gcv.criteria {
        let markers = [
            if k == sel_aic.best_ncomp { "<-AIC" } else { "" },
            if k == sel_bic.best_ncomp { "<-BIC" } else { "" },
            if k == sel_gcv.best_ncomp { "<-GCV" } else { "" },
        ]
        .join("");
        println!(
            "  {:>4}  {:>10.2}  {:>10.2}  {:>10.4}  {}",
            k, aic, bic, gcv, markers
        );
    }

    let best_k = sel_gcv.best_ncomp.min(8); // cap to avoid overfitting in CV folds
    println!(
        "\n  Selected: K={} (GCV={}, capped at 8)",
        best_k, sel_gcv.best_ncomp
    );

    // ── 4. Fit on full data ────────────────────────────────────────────────
    println!("\n--- Full-Sample Fit (fregre_lm, K={}) ---", best_k);
    let fit = fregre_lm(&data_smooth, &y_final, None, best_k).unwrap();
    println!("  R²     = {:.4}", fit.r_squared);
    println!("  R²_adj = {:.4}", fit.r_squared_adj);
    println!("  Residual SE = {:.2} cm", fit.residual_se);

    // ── 5. 10-fold cross-validation ────────────────────────────────────────
    println!("\n--- 10-Fold Cross-Validation ---");
    let n_folds = 10;
    let folds = create_folds(n, n_folds, 42);

    let mut cv_errors_lm = vec![f64::NAN; n];
    let mut cv_errors_pls = vec![f64::NAN; n];

    for fold in 0..n_folds {
        // Split
        let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
        let n_train = train_idx.len();
        let n_test = test_idx.len();

        let mut train_vals = vec![0.0; n_train * m];
        let mut train_y = vec![0.0; n_train];
        for (ti, &i) in train_idx.iter().enumerate() {
            train_y[ti] = y_final[i];
            for j in 0..m {
                train_vals[ti + j * n_train] = data_smooth[(i, j)];
            }
        }
        let train_data = FdMatrix::from_column_major(train_vals, n_train, m).unwrap();

        let mut test_vals = vec![0.0; n_test * m];
        for (ti, &i) in test_idx.iter().enumerate() {
            for j in 0..m {
                test_vals[ti + j * n_test] = data_smooth[(i, j)];
            }
        }
        let test_data = FdMatrix::from_column_major(test_vals, n_test, m).unwrap();

        // Linear model
        if let Ok(fold_fit) = fregre_lm(&train_data, &train_y, None, best_k) {
            let preds = predict_fregre_lm(&fold_fit, &test_data, None);
            for (ti, &i) in test_idx.iter().enumerate() {
                cv_errors_lm[i] = (y_final[i] - preds[ti]).powi(2);
            }
        }

        // PLS regression
        if let Ok(fold_pls) = fregre_pls(&train_data, &train_y, &t_norm, best_k, None) {
            let preds = predict_fregre_pls(&fold_pls, &test_data, None).unwrap();
            for (ti, &i) in test_idx.iter().enumerate() {
                cv_errors_pls[i] = (y_final[i] - preds[ti]).powi(2);
            }
        }
    }

    let valid_lm: Vec<f64> = cv_errors_lm
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();
    let valid_pls: Vec<f64> = cv_errors_pls
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();
    let cv_mse_lm = valid_lm.iter().sum::<f64>() / valid_lm.len() as f64;
    let cv_rmse_lm = cv_mse_lm.sqrt();
    let cv_mse_pls = valid_pls.iter().sum::<f64>() / valid_pls.len().max(1) as f64;
    let cv_rmse_pls = cv_mse_pls.sqrt();

    // Total variance of y for R²_CV
    let y_mean = y_final.iter().sum::<f64>() / n as f64;
    let ss_tot = y_final.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>();
    let r2_cv_lm = 1.0 - cv_mse_lm * valid_lm.len() as f64 / ss_tot;
    let r2_cv_pls = if valid_pls.is_empty() {
        f64::NAN
    } else {
        1.0 - cv_mse_pls * valid_pls.len() as f64 / ss_tot
    };

    println!(
        "  {:>20}  {:>8}  {:>8}  {:>8}",
        "Method", "CV-RMSE", "CV-MSE", "CV-R²"
    );
    println!(
        "  {:>20}  {:>8.2}  {:>8.2}  {:>8.4}",
        format!("fregre_lm (K={})", best_k),
        cv_rmse_lm,
        cv_mse_lm,
        r2_cv_lm
    );
    println!(
        "  {:>20}  {:>8.2}  {:>8.2}  {:>8.4}",
        format!("fregre_pls (K={})", best_k),
        cv_rmse_pls,
        cv_mse_pls,
        r2_cv_pls
    );

    println!("\n=== Summary ===");
    println!(
        "Predicting adult height from growth curves (n={}, {} ages)",
        n, m
    );
    println!(
        "Best model: fregre_lm with K={} FPC components (GCV-selected)",
        best_k
    );
    println!("  In-sample R²  = {:.4}", fit.r_squared);
    println!("  CV R²          = {:.4}", r2_cv_lm);
    println!("  CV RMSE        = {:.2} cm", cv_rmse_lm);
}
