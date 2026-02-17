//! Example 01: Simulating Functional Data
//!
//! Demonstrates how to generate synthetic functional data using the
//! Karhunen-Loève expansion, different eigenfunction/eigenvalue types,
//! and how to add measurement noise.

use fdars_core::helpers::extract_curves;
use fdars_core::simulation::{
    add_error_curve, add_error_pointwise, eigenfunctions, eigenvalues, sim_fundata, EFunType,
    EValType,
};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn print_curve_summary(label: &str, values: &[f64]) {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    println!("  {label}: min={min:.4}, max={max:.4}, mean={mean:.4}");
}

fn main() {
    println!("=== Example 01: Simulating Functional Data ===\n");

    let n = 30; // number of curves
    let m = 50; // grid points per curve
    let big_m = 5; // number of basis components in the KL expansion
    let t = uniform_grid(m);

    // --- Section 1: Eigenfunction types ---
    println!("--- Eigenfunction Systems ---");
    for efun in [
        EFunType::Fourier,
        EFunType::Poly,
        EFunType::PolyHigh,
        EFunType::Wiener,
    ] {
        let phi = eigenfunctions(&t, big_m, efun);
        println!(
            "  {:?} eigenfunctions: {} values (m={m} x big_m={big_m})",
            efun,
            phi.len()
        );
    }

    // --- Section 2: Eigenvalue decay types ---
    println!("\n--- Eigenvalue Decay Patterns ---");
    for eval in [EValType::Linear, EValType::Exponential, EValType::Wiener] {
        let lam = eigenvalues(big_m, eval);
        println!("  {:?}: {:?}", eval, lam);
    }

    // --- Section 3: Simulate functional data via KL expansion ---
    println!("\n--- Simulated Functional Data (Fourier + Exponential) ---");
    let data = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    println!(
        "  Data matrix: {} elements ({n} curves x {m} grid points)",
        data.len()
    );

    let curves = extract_curves(&data);
    for (i, curve) in curves.iter().enumerate().take(3) {
        print_curve_summary(&format!("Curve {i}"), curve);
    }
    println!("  ... ({} more curves)", n - 3);

    // --- Section 4: Different generative models ---
    println!("\n--- Comparing Generative Models ---");
    let models = [
        (
            EFunType::Fourier,
            EValType::Exponential,
            "Fourier/Exponential",
        ),
        (EFunType::Poly, EValType::Linear, "Legendre/Linear"),
        (EFunType::Wiener, EValType::Wiener, "Wiener/Wiener"),
    ];
    for (efun, eval, label) in models {
        let d = sim_fundata(n, &t, big_m, efun, eval, Some(42));
        let curves = extract_curves(&d);
        let range: f64 = curves
            .iter()
            .map(|c| {
                let mn = c.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                mx - mn
            })
            .sum::<f64>()
            / n as f64;
        println!("  {label}: avg curve range = {range:.4}");
    }

    // --- Section 5: Adding noise ---
    println!("\n--- Adding Measurement Noise ---");
    let clean = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let noisy_point = add_error_pointwise(&clean, 0.1, Some(42));
    let noisy_curve = add_error_curve(&clean, 0.2, Some(42));

    // Compute noise magnitude
    let clean_slice = clean.as_slice();
    let noisy_point_slice = noisy_point.as_slice();
    let noisy_curve_slice = noisy_curve.as_slice();
    let pointwise_noise: f64 = clean_slice
        .iter()
        .zip(noisy_point_slice.iter())
        .map(|(c, n)| (c - n).powi(2))
        .sum::<f64>()
        / (n * m) as f64;
    let curve_noise: f64 = clean_slice
        .iter()
        .zip(noisy_curve_slice.iter())
        .map(|(c, n)| (c - n).powi(2))
        .sum::<f64>()
        / (n * m) as f64;

    println!("  Pointwise noise (sd=0.1): MSE = {pointwise_noise:.6}");
    println!("  Curve-level noise (sd=0.2): MSE = {curve_noise:.6}");

    // --- Section 6: Column-major layout demonstration ---
    println!("\n--- Column-Major Data Layout ---");
    println!("  data[(i, j)] accesses curve i at grid point j");
    println!("  Curve 0, point 0: {:.4}", data[(0, 0)]);
    println!("  Curve 0, point 1: {:.4}", data[(0, 1)]);
    println!("  Curve 1, point 0: {:.4}", data[(1, 0)]);
    println!("  Curve 1, point 1: {:.4}", data[(1, 1)]);

    println!("\n=== Done ===");
}
