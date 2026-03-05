//! Example 18: Landmark Registration
//!
//! Demonstrates aligning functional data by detecting features (peaks)
//! and warping curves so that corresponding features are at common positions.

use fdars_core::landmark::{detect_and_register, detect_landmarks, LandmarkKind};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 18: Landmark Registration ===\n");

    let n = 8;
    let m = 200;
    let t = uniform_grid(m);

    // Generate phase-shifted sinusoids with varying amplitude
    println!("--- Generating phase-shifted sinusoids ---");
    let shifts = [0.0, 0.03, -0.02, 0.05, -0.04, 0.01, -0.03, 0.02];
    let amplitudes = [1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 0.85, 1.15];

    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = amplitudes[i] * (2.0 * PI * (t[j] - shifts[i])).sin();
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    // Detect peaks before registration
    println!("\n--- Peak positions before registration ---");
    for i in 0..n {
        let curve = data.row(i);
        let peaks = detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.5);
        let positions: Vec<String> = peaks.iter().map(|p| format!("{:.4}", p.position)).collect();
        println!("  Curve {i}: peaks at [{}]", positions.join(", "));
    }

    // Register using peak detection
    println!("\n--- Performing landmark registration ---");
    let result = detect_and_register(&data, &t, LandmarkKind::Peak, 0.5, 1);
    println!(
        "  Target landmarks: {:?}",
        result
            .target_landmarks
            .iter()
            .map(|t| format!("{t:.4}"))
            .collect::<Vec<_>>()
    );

    // Show peak positions after registration
    println!("\n--- Peak positions after registration ---");
    for i in 0..n {
        let curve = result.registered.row(i);
        let peaks = detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.5);
        let positions: Vec<String> = peaks.iter().map(|p| format!("{:.4}", p.position)).collect();
        println!("  Curve {i}: peaks at [{}]", positions.join(", "));
    }

    // Measure alignment improvement
    let mut orig_peak_var = 0.0;
    let mut reg_peak_var = 0.0;
    let mut orig_peaks = Vec::new();
    let mut reg_peaks = Vec::new();

    for i in 0..n {
        let orig_curve = data.row(i);
        let orig_lms = detect_landmarks(&orig_curve, &t, LandmarkKind::Peak, 0.5);
        if let Some(p) = orig_lms.first() {
            orig_peaks.push(p.position);
        }

        let reg_curve = result.registered.row(i);
        let reg_lms = detect_landmarks(&reg_curve, &t, LandmarkKind::Peak, 0.5);
        if let Some(p) = reg_lms.first() {
            reg_peaks.push(p.position);
        }
    }

    if !orig_peaks.is_empty() {
        let mean: f64 = orig_peaks.iter().sum::<f64>() / orig_peaks.len() as f64;
        orig_peak_var =
            orig_peaks.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / orig_peaks.len() as f64;
    }
    if !reg_peaks.is_empty() {
        let mean: f64 = reg_peaks.iter().sum::<f64>() / reg_peaks.len() as f64;
        reg_peak_var =
            reg_peaks.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / reg_peaks.len() as f64;
    }

    println!("\n--- Alignment Quality ---");
    println!("  Peak position variance before: {orig_peak_var:.6}");
    println!("  Peak position variance after:  {reg_peak_var:.6}");
    println!(
        "  Variance reduction: {:.1}%",
        if orig_peak_var > 0.0 {
            (1.0 - reg_peak_var / orig_peak_var) * 100.0
        } else {
            0.0
        }
    );

    // Show warping function properties
    println!("\n--- Warping Functions ---");
    for i in 0..n.min(4) {
        let gamma_start = result.gammas[(i, 0)];
        let gamma_end = result.gammas[(i, m - 1)];
        let gamma_mid = result.gammas[(i, m / 2)];
        println!("  Curve {i}: gamma(0)={gamma_start:.4}, gamma(0.5)={gamma_mid:.4}, gamma(1)={gamma_end:.4}");
    }

    println!("\n=== Done ===");
}
