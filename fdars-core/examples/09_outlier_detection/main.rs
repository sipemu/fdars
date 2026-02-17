//! Example 09: Outlier Detection
//!
//! Demonstrates detecting outlying functional observations using the
//! likelihood ratio test (LRT) method with bootstrap thresholding,
//! and confirms findings with depth-based measures.

use fdars_core::depth::fraiman_muniz_1d;
use fdars_core::matrix::FdMatrix;
use fdars_core::outliers::{detect_outliers_lrt, outliers_threshold_lrt};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 09: Outlier Detection ===\n");

    let n_normal = 30;
    let n_outliers = 3;
    let n = n_normal + n_outliers;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);

    // --- Generate data with injected outliers ---
    println!("--- Generating Data ---");
    let normal = sim_fundata(
        n_normal,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    // Create combined dataset with outliers at the end
    let mut mat = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n_normal {
            mat[(i, j)] = normal[(i, j)];
        }
        // Outlier 1: Magnitude shift
        mat[(n_normal, j)] = normal[(0, j)] + 4.0;
        // Outlier 2: Shape anomaly (high-frequency oscillation)
        mat[(n_normal + 1, j)] = normal[(1, j)] + 2.0 * (20.0 * std::f64::consts::PI * t[j]).sin();
        // Outlier 3: Partial shift (only in second half)
        mat[(n_normal + 2, j)] = normal[(2, j)] + if t[j] > 0.5 { 3.5 } else { 0.0 };
    }

    println!("  Normal curves: {n_normal}");
    println!("  Injected outliers: {n_outliers}");
    println!("    Index {}: magnitude shift (+4.0)", n_normal);
    println!(
        "    Index {}: shape anomaly (high-freq oscillation)",
        n_normal + 1
    );
    println!("    Index {}: partial shift (t > 0.5)", n_normal + 2);

    // --- Section 1: Bootstrap threshold ---
    println!("\n--- Bootstrap LRT Threshold ---");
    let nb = 100; // bootstrap replicates
    let smo = 0.05; // smoothing parameter
    let trim = 0.10; // trimming proportion
    let percentile = 0.99;
    let threshold = outliers_threshold_lrt(&mat, nb, smo, trim, 42, percentile);
    println!("  Bootstrap replicates: {nb}");
    println!("  Smoothing: {smo}, Trim: {trim}");
    println!("  Percentile: {percentile}");
    println!("  Computed threshold: {threshold:.6}");

    // --- Section 2: Detect outliers ---
    println!("\n--- Outlier Detection ---");
    let is_outlier = detect_outliers_lrt(&mat, threshold, trim);
    println!("  Results:");
    let mut detected_indices = Vec::new();
    for (i, &outlier) in is_outlier.iter().enumerate() {
        if outlier {
            detected_indices.push(i);
            let label = if i >= n_normal {
                format!("(INJECTED outlier {})", i - n_normal + 1)
            } else {
                "(false positive)".to_string()
            };
            println!("    Curve {i}: OUTLIER {label}");
        }
    }
    let n_detected = detected_indices.len();
    let true_positives = detected_indices.iter().filter(|&&i| i >= n_normal).count();
    let false_positives = n_detected - true_positives;
    println!("  Summary: {n_detected} detected, {true_positives} true positives, {false_positives} false positives");

    // --- Section 3: Depth-based confirmation ---
    println!("\n--- Depth-Based Confirmation (Fraiman-Muniz) ---");
    let depths = fraiman_muniz_1d(&mat, &mat, true);
    let mut ranked: Vec<(usize, f64)> = depths.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("  5 shallowest curves (most outlying):");
    for &(idx, depth) in ranked.iter().take(5) {
        let label = if idx >= n_normal {
            format!("INJECTED #{}", idx - n_normal + 1)
        } else {
            "normal".to_string()
        };
        println!("    Curve {idx}: depth={depth:.4} ({label})");
    }

    println!("  5 deepest curves (most central):");
    for &(idx, depth) in ranked.iter().rev().take(5) {
        println!("    Curve {idx}: depth={depth:.4}");
    }

    // --- Section 4: Agreement between methods ---
    println!("\n--- Method Agreement ---");
    let depth_outliers: Vec<usize> = ranked.iter().take(n_outliers).map(|&(i, _)| i).collect();
    let lrt_outliers: Vec<usize> = detected_indices.clone();
    let agreement: Vec<usize> = depth_outliers
        .iter()
        .filter(|i| lrt_outliers.contains(i))
        .copied()
        .collect();
    println!("  LRT outliers:   {lrt_outliers:?}");
    println!("  Depth-bottom-{n_outliers}: {depth_outliers:?}");
    println!(
        "  Agreement:      {agreement:?} ({} / {n_outliers})",
        agreement.len()
    );

    println!("\n=== Done ===");
}
