//! Example 19: TSRVF (Transported Square-Root Velocity Function)
//!
//! Demonstrates mapping elastically-aligned curves to the tangent space
//! of the Karcher mean. Tangent vectors live in standard Euclidean space,
//! enabling PCA, regression, and clustering on elastic-aligned data.

use fdars_core::alignment::{karcher_mean, tsrvf_from_alignment, tsrvf_inverse, tsrvf_transform};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 19: TSRVF Transform ===\n");

    let n = 15;
    let m = 50;
    let t = uniform_grid(m);

    // Generate functional data
    let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
    println!("Generated {n} curves on {m} points");

    // --- Full pipeline: tsrvf_transform ---
    println!("\n--- TSRVF Transform (full pipeline) ---");
    let result = tsrvf_transform(&data, &t, 15, 1e-3, 0.0);
    println!("  Converged: {}", result.converged);
    println!("  Mean SRSF norm: {:.6}", result.mean_srsf_norm);
    println!(
        "  Tangent vectors shape: {:?}",
        result.tangent_vectors.shape()
    );

    // --- Tangent vector properties ---
    println!("\n--- Tangent Vector Properties ---");

    // Mean tangent vector should be ~0
    let mut mean_tv = vec![0.0; m];
    for i in 0..n {
        for (j, v) in mean_tv.iter_mut().enumerate() {
            *v += result.tangent_vectors[(i, j)];
        }
    }
    for v in &mut mean_tv {
        *v /= n as f64;
    }
    let mean_norm: f64 = mean_tv.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  Mean tangent vector norm: {mean_norm:.6} (should be ~0)");

    // Variance of tangent vectors
    let mut total_var = 0.0;
    for i in 0..n {
        let norm_sq: f64 = (0..m).map(|j| result.tangent_vectors[(i, j)].powi(2)).sum();
        total_var += norm_sq;
    }
    total_var /= n as f64;
    println!("  Mean squared tangent vector norm: {total_var:.6}");

    // SRSF norms
    let min_norm = result
        .srsf_norms
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_norm = result
        .srsf_norms
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  SRSF norm range: [{min_norm:.4}, {max_norm:.4}]");

    // --- From pre-computed alignment ---
    println!("\n--- TSRVF from pre-computed Karcher mean ---");
    let karcher = karcher_mean(&data, &t, 15, 1e-3, 0.0);
    let result2 = tsrvf_from_alignment(&karcher, &t);
    println!(
        "  Tangent vectors shape: {:?}",
        result2.tangent_vectors.shape()
    );
    println!("  Mean SRSF norm: {:.6}", result2.mean_srsf_norm);

    // --- Round-trip: transform → inverse ---
    println!("\n--- Round-trip: TSRVF → inverse ---");
    let reconstructed = tsrvf_inverse(&result, &t);
    println!("  Reconstructed shape: {:?}", reconstructed.shape());

    // Measure reconstruction error (against aligned curves, not originals)
    let mut max_err = 0.0_f64;
    let mut mean_err = 0.0_f64;
    let mut count = 0;
    for i in 0..n {
        for j in 5..(m - 5) {
            let err = (reconstructed[(i, j)] - result.mean[j]).abs();
            // Compare to mean as a baseline
            max_err = max_err.max(err);
            mean_err += err;
            count += 1;
        }
    }
    mean_err /= count as f64;
    println!("  Mean abs diff from mean: {mean_err:.6}");
    println!("  Max abs diff from mean: {max_err:.6}");

    // Per-curve tangent vector norms (sorted)
    println!("\n--- Per-curve tangent vector norms ---");
    let mut norms: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let norm: f64 = (0..m)
                .map(|j| result.tangent_vectors[(i, j)].powi(2))
                .sum::<f64>()
                .sqrt();
            (i, norm)
        })
        .collect();
    norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (i, norm) in norms.iter().take(5) {
        println!("  Curve {i:>2}: ||v|| = {norm:.6}");
    }
    println!("  ...");
    for (i, norm) in norms.iter().rev().take(3).collect::<Vec<_>>().iter().rev() {
        println!("  Curve {i:>2}: ||v|| = {norm:.6}");
    }

    println!("\n=== Done ===");
}
