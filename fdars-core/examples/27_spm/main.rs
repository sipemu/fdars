//! Example 27: Statistical Process Monitoring (SPM)
//!
//! FPCA-based control charts for monitoring functional data streams:
//! - Phase I: build control chart from in-control training data
//! - Phase II: monitor new observations with T² and SPE statistics
//! - EWMA smoothing for detecting small persistent shifts
//! - CUSUM for detecting sustained mean shifts
//! - Western Electric / Nelson rules for pattern detection
//! - Robust control limits (empirical, bootstrap)
//! - Contribution diagnostics for fault identification

use fdars_core::matrix::FdMatrix;
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::spm::{
    evaluate_rules, spm_cusum_monitor, spm_ewma_monitor, spm_monitor, spm_phase1, t2_limit_robust,
    western_electric_rules, ChartRule, ControlLimitMethod, CusumConfig, EwmaConfig, SpmConfig,
};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let m = 50;
    let t = uniform_grid(m);

    // ── 1. Generate in-control (Phase I) data ──────────────────────────────
    println!("=== Phase I: Building control chart ===");
    let n_train = 80;
    let train_data = sim_fundata(
        n_train,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        ..Default::default()
    };
    let chart = spm_phase1(&train_data, &t, &config).unwrap();

    println!(
        "  T² limit: {:.3}, SPE limit: {:.3}",
        chart.t2_limit.ucl, chart.spe_limit.ucl
    );
    let n_t2_alarm = chart
        .t2_phase1
        .iter()
        .filter(|&&v| v > chart.t2_limit.ucl)
        .count();
    let n_spe_alarm = chart
        .spe_phase1
        .iter()
        .filter(|&&v| v > chart.spe_limit.ucl)
        .count();
    println!(
        "  Phase I alarms: {} T², {} SPE (out of {})",
        n_t2_alarm, n_spe_alarm, n_train
    );

    // ── 2. Monitor new in-control data (Phase II) ──────────────────────────
    println!("\n=== Phase II: Monitoring in-control data ===");
    let n_new = 30;
    let new_data = sim_fundata(
        n_new,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(99),
    );

    let result = spm_monitor(&chart, &new_data, &t).unwrap();
    let n_alarms: usize = result
        .t2_alarm
        .iter()
        .zip(result.spe_alarm.iter())
        .filter(|(&t2, &spe)| t2 || spe)
        .count();
    println!("  Observations: {}, Alarms: {}", n_new, n_alarms);
    println!(
        "  T² range: [{:.3}, {:.3}]",
        result.t2.iter().cloned().fold(f64::INFINITY, f64::min),
        result.t2.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── 3. Monitor shifted data (detect fault) ─────────────────────────────
    println!("\n=== Phase II: Monitoring shifted data ===");
    let mut shifted_flat = vec![0.0; n_new * m];
    for i in 0..n_new {
        for j in 0..m {
            shifted_flat[i + j * n_new] = new_data[(i, j)] + 3.0 * t[j]; // add a systematic shift
        }
    }
    let shifted_data = FdMatrix::from_column_major(shifted_flat, n_new, m).unwrap();

    let shifted_result = spm_monitor(&chart, &shifted_data, &t).unwrap();
    let n_t2_detect: usize = shifted_result.t2_alarm.iter().filter(|&&a| a).count();
    let n_spe_detect: usize = shifted_result.spe_alarm.iter().filter(|&&a| a).count();
    println!(
        "  T² alarms: {}/{}, SPE alarms: {}/{}",
        n_t2_detect, n_new, n_spe_detect, n_new
    );

    // ── 4. EWMA monitoring for small shifts ────────────────────────────────
    println!("\n=== EWMA Monitoring ===");
    let ewma_config = EwmaConfig {
        lambda: 0.2,
        ncomp: 3,
        alpha: 0.05,
        exact_covariance: false,
    };

    // Small shift — harder for Shewhart, easier for EWMA
    let mut small_shift_flat = vec![0.0; n_new * m];
    for i in 0..n_new {
        for j in 0..m {
            small_shift_flat[i + j * n_new] = new_data[(i, j)] + 0.5 * t[j];
        }
    }
    let small_shift_data = FdMatrix::from_column_major(small_shift_flat, n_new, m).unwrap();

    let shewhart = spm_monitor(&chart, &small_shift_data, &t).unwrap();
    let ewma = spm_ewma_monitor(&chart, &small_shift_data, &t, &ewma_config).unwrap();

    let shewhart_alarms: usize = shewhart.t2_alarm.iter().filter(|&&a| a).count();
    let ewma_alarms: usize = ewma.t2_alarm.iter().filter(|&&a| a).count();
    println!(
        "  Small shift detection — Shewhart T² alarms: {}, EWMA T² alarms: {}",
        shewhart_alarms, ewma_alarms
    );
    println!("  EWMA T² limit: {:.3}", ewma.t2_limit);

    // ── 5. CUSUM monitoring ────────────────────────────────────────────────
    println!("\n=== CUSUM Monitoring ===");
    let cusum_config = CusumConfig {
        k: 0.5,
        h: 5.0,
        ncomp: 3,
        ..Default::default()
    };
    let cusum = spm_cusum_monitor(&chart, &small_shift_data, &t, &cusum_config).unwrap();
    let cusum_alarms: usize = cusum.alarm.iter().filter(|&&a| a).count();
    println!(
        "  CUSUM alarms: {}/{} (k={}, h={})",
        cusum_alarms, n_new, cusum_config.k, cusum_config.h
    );

    // ── 6. Western Electric and Nelson rules ───────────────────────────────
    println!("\n=== Control Chart Rules ===");
    // Apply rules to the Phase I T² values
    let center: f64 = chart.t2_phase1.iter().sum::<f64>() / chart.t2_phase1.len() as f64;
    let variance: f64 = chart
        .t2_phase1
        .iter()
        .map(|v| (v - center).powi(2))
        .sum::<f64>()
        / (chart.t2_phase1.len() - 1) as f64;
    let sigma = variance.sqrt();

    let we_violations = western_electric_rules(&chart.t2_phase1, center, sigma).unwrap();
    println!(
        "  Western Electric violations on Phase I T²: {}",
        we_violations.len()
    );
    for v in &we_violations {
        println!(
            "    Rule {:?} at indices {:?}",
            v.rule,
            &v.indices[..v.indices.len().min(5)]
        );
    }

    // Custom rule: 3 consecutive points beyond 1.5 sigma
    let custom_rules = vec![ChartRule::CustomRun {
        n_points: 3,
        k_sigma: 1.5,
    }];
    let custom_violations = evaluate_rules(&chart.t2_phase1, center, sigma, &custom_rules).unwrap();
    println!(
        "  Custom rule (3 pts > 1.5 sigma) violations: {}",
        custom_violations.len()
    );

    // ── 7. Robust control limits ───────────────────────────────────────────
    println!("\n=== Robust Control Limits ===");

    let parametric = t2_limit_robust(
        &chart.t2_phase1,
        config.ncomp,
        0.05,
        &ControlLimitMethod::Parametric,
    )
    .unwrap();
    let empirical = t2_limit_robust(
        &chart.t2_phase1,
        config.ncomp,
        0.05,
        &ControlLimitMethod::Empirical,
    )
    .unwrap();
    let bootstrap = t2_limit_robust(
        &chart.t2_phase1,
        config.ncomp,
        0.05,
        &ControlLimitMethod::Bootstrap {
            n_bootstrap: 500,
            seed: 42,
        },
    )
    .unwrap();

    println!("  T² limits (alpha=0.05):");
    println!("    Parametric:  {:.3}", parametric.ucl);
    println!("    Empirical:   {:.3}", empirical.ucl);
    println!("    Bootstrap:   {:.3}", bootstrap.ucl);

    println!("\nDone.");
}
