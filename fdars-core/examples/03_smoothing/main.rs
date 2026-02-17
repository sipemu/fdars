//! Example 03: Kernel Smoothing Methods
//!
//! Demonstrates smoothing of noisy functional data using kernel-based
//! methods: Nadaraya-Watson, local linear, local polynomial, and k-NN.
//! Compares bandwidth sensitivity and MSE against a known true function.

use fdars_core::smoothing::{knn_smoother, local_linear, local_polynomial, nadaraya_watson};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

/// True underlying function: sin(2πt) + 0.5*cos(4πt)
fn true_function(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

fn mse(predicted: &[f64], truth: &[f64]) -> f64 {
    predicted
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predicted.len() as f64
}

fn main() {
    println!("=== Example 03: Kernel Smoothing Methods ===\n");

    let m = 100;
    let t = uniform_grid(m);
    let noise_sd = 0.3;

    // Generate noisy observations of the true function
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, noise_sd).unwrap();
    let truth: Vec<f64> = t.iter().map(|&ti| true_function(ti)).collect();
    let noisy: Vec<f64> = truth.iter().map(|&y| y + normal.sample(&mut rng)).collect();

    println!("--- Data ---");
    println!("  Grid points: {m}");
    println!("  Noise sd: {noise_sd}");
    println!("  Noisy MSE (no smoothing): {:.6}", mse(&noisy, &truth));

    // --- Section 1: Nadaraya-Watson with different bandwidths ---
    println!("\n--- Nadaraya-Watson Smoother ---");
    for bw in [0.02, 0.05, 0.10, 0.20] {
        let smoothed = nadaraya_watson(&t, &noisy, &t, bw, "gaussian");
        let err = mse(&smoothed, &truth);
        println!("  bandwidth={bw:.2}: MSE={err:.6}");
    }

    // --- Section 2: Local linear regression ---
    println!("\n--- Local Linear Smoother ---");
    for bw in [0.02, 0.05, 0.10, 0.20] {
        let smoothed = local_linear(&t, &noisy, &t, bw, "gaussian");
        let err = mse(&smoothed, &truth);
        println!("  bandwidth={bw:.2}: MSE={err:.6}");
    }

    // --- Section 3: Local polynomial (degree 2) ---
    println!("\n--- Local Polynomial Smoother (degree=2) ---");
    for bw in [0.02, 0.05, 0.10, 0.20] {
        let smoothed = local_polynomial(&t, &noisy, &t, bw, 2, "gaussian");
        let err = mse(&smoothed, &truth);
        println!("  bandwidth={bw:.2}: MSE={err:.6}");
    }

    // --- Section 4: k-NN smoother ---
    println!("\n--- k-NN Smoother ---");
    for k in [3, 5, 10, 20] {
        let smoothed = knn_smoother(&t, &noisy, &t, k);
        let err = mse(&smoothed, &truth);
        println!("  k={k:2}: MSE={err:.6}");
    }

    // --- Section 5: Best from each method ---
    println!("\n--- Method Comparison (best bandwidth/k for each) ---");
    let nw = nadaraya_watson(&t, &noisy, &t, 0.05, "gaussian");
    let ll = local_linear(&t, &noisy, &t, 0.05, "gaussian");
    let lp = local_polynomial(&t, &noisy, &t, 0.05, 2, "gaussian");
    let kn = knn_smoother(&t, &noisy, &t, 5);

    println!("  Nadaraya-Watson (bw=0.05): MSE={:.6}", mse(&nw, &truth));
    println!("  Local linear    (bw=0.05): MSE={:.6}", mse(&ll, &truth));
    println!("  Local poly deg2 (bw=0.05): MSE={:.6}", mse(&lp, &truth));
    println!("  k-NN            (k=5):     MSE={:.6}", mse(&kn, &truth));

    // Show a few smoothed values
    println!("\n--- Sample Smoothed Values (Nadaraya-Watson, bw=0.05) ---");
    for &idx in &[0, m / 4, m / 2, 3 * m / 4, m - 1] {
        println!(
            "  t={:.2}: truth={:.4}, noisy={:.4}, smoothed={:.4}",
            t[idx], truth[idx], noisy[idx], nw[idx]
        );
    }

    println!("\n=== Done ===");
}
