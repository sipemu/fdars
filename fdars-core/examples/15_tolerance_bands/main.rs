//! Example 15: Tolerance Bands
//!
//! Demonstrates constructing tolerance bands for functional data using
//! FPCA + bootstrap and conformal prediction methods.

use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::tolerance::{
    conformal_prediction_band, fpca_tolerance_band, scb_mean_degras, BandType,
    MultiplierDistribution, NonConformityScore,
};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 15: Tolerance Bands ===\n");

    let n = 50;
    let m = 60;
    let t = uniform_grid(m);
    let data = sim_fundata(n, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));

    // --- Section 1: FPCA Pointwise Band ---
    println!("--- FPCA Tolerance Band (Pointwise, 95%) ---");
    let pw_band = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Pointwise, 42)
        .expect("FPCA pointwise band failed");
    let pw_mean_hw: f64 = pw_band.half_width.iter().sum::<f64>() / m as f64;
    println!("  Mean half-width: {pw_mean_hw:.4}");
    println!(
        "  Band at t=0.5: [{:.4}, {:.4}]",
        pw_band.lower[m / 2],
        pw_band.upper[m / 2]
    );

    // --- Section 2: FPCA Simultaneous Band ---
    println!("\n--- FPCA Tolerance Band (Simultaneous, 95%) ---");
    let sim_band = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Simultaneous, 42)
        .expect("FPCA simultaneous band failed");
    let sim_mean_hw: f64 = sim_band.half_width.iter().sum::<f64>() / m as f64;
    println!("  Mean half-width: {sim_mean_hw:.4}");
    println!(
        "  Band at t=0.5: [{:.4}, {:.4}]",
        sim_band.lower[m / 2],
        sim_band.upper[m / 2]
    );
    println!(
        "  Simultaneous/Pointwise ratio: {:.2}",
        sim_mean_hw / pw_mean_hw
    );

    // --- Section 3: Conformal Prediction Band ---
    println!("\n--- Conformal Prediction Band (SupNorm, 95%) ---");
    let conf_band = conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::SupNorm, 42)
        .expect("Conformal band failed");
    println!("  Constant half-width: {:.4}", conf_band.half_width[0]);
    println!(
        "  Band at t=0.5: [{:.4}, {:.4}]",
        conf_band.lower[m / 2],
        conf_band.upper[m / 2]
    );

    // --- Section 4: SCB for the Mean (Degras) ---
    println!("\n--- SCB for the Mean (Degras, Gaussian multiplier, 95%) ---");
    let scb_band = scb_mean_degras(&data, &t, 0.15, 200, 0.95, MultiplierDistribution::Gaussian)
        .expect("SCB mean failed");
    let scb_mean_hw: f64 = scb_band.half_width.iter().sum::<f64>() / m as f64;
    println!("  Mean half-width: {scb_mean_hw:.4}");
    println!(
        "  Band at t=0.5: [{:.4}, {:.4}]",
        scb_band.lower[m / 2],
        scb_band.upper[m / 2]
    );

    // --- Section 5: Empirical coverage check ---
    println!("\n--- Empirical Coverage Check ---");
    let test_data = sim_fundata(
        500,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(99),
    );
    let n_test = test_data.nrows();

    let pw_covered = (0..n_test)
        .filter(|&i| {
            (0..m).all(|j| {
                test_data[(i, j)] >= pw_band.lower[j] && test_data[(i, j)] <= pw_band.upper[j]
            })
        })
        .count();
    let sim_covered = (0..n_test)
        .filter(|&i| {
            (0..m).all(|j| {
                test_data[(i, j)] >= sim_band.lower[j] && test_data[(i, j)] <= sim_band.upper[j]
            })
        })
        .count();

    println!(
        "  FPCA pointwise coverage: {}/{} ({:.1}%)",
        pw_covered,
        n_test,
        100.0 * pw_covered as f64 / n_test as f64
    );
    println!(
        "  FPCA simultaneous coverage: {}/{} ({:.1}%)",
        sim_covered,
        n_test,
        100.0 * sim_covered as f64 / n_test as f64
    );

    println!("\n=== Done ===");
}
