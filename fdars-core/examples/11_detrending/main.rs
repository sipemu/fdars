//! Example 11: Detrending and Decomposition
//!
//! Demonstrates removing trends from functional data using linear,
//! polynomial, and LOESS detrending. Also shows automatic method
//! selection, additive seasonal decomposition, and STL decomposition.

use fdars_core::detrend::{
    auto_detrend, decompose_additive, detrend_linear, detrend_loess, detrend_polynomial,
    stl_decompose,
};
use fdars_core::FdMatrix;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 11: Detrending and Decomposition ===\n");

    let m = 200;
    let t = uniform_grid(m);

    // Synthetic signal: linear trend + seasonal + noise
    let true_trend: Vec<f64> = t.iter().map(|&ti| 2.0 * ti + 1.0).collect();
    let true_seasonal: Vec<f64> = t
        .iter()
        .map(|&ti| 0.5 * (2.0 * std::f64::consts::PI * ti * 5.0).sin())
        .collect();
    let data_vec: Vec<f64> = true_trend
        .iter()
        .zip(true_seasonal.iter())
        .map(|(tr, se)| tr + se)
        .collect();
    let data = FdMatrix::from_column_major(data_vec.clone(), 1, m).unwrap();

    println!("--- Synthetic Signal ---");
    println!("  y(t) = 2t + 1 + 0.5*sin(10\u{03c0}t)");
    println!("  {m} points on [0, 1]");

    // --- Section 1: Linear detrending ---
    println!("\n--- Linear Detrending ---");
    let lin = detrend_linear(&data, &t);
    println!("  Method: {}", lin.method);
    if let Some(ref coefs) = lin.coefficients {
        println!(
            "  Coefficients: intercept={:.4}, slope={:.4} (true: 1.0, 2.0)",
            coefs[(0, 0)],
            coefs[(0, 1)]
        );
    }
    let trend_err: f64 = lin
        .trend
        .as_slice()
        .iter()
        .zip(true_trend.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    println!("  Trend MSE: {trend_err:.6}");

    // --- Section 2: Polynomial detrending ---
    println!("\n--- Polynomial Detrending ---");
    for degree in [1, 2, 3] {
        let poly = detrend_polynomial(&data, &t, degree);
        let poly_err: f64 = poly
            .trend
            .as_slice()
            .iter()
            .zip(true_trend.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / m as f64;
        println!(
            "  Degree {degree}: method={}, MSE={poly_err:.6}",
            poly.method
        );
    }

    // --- Section 3: LOESS detrending ---
    println!("\n--- LOESS Detrending ---");
    for bw in [0.1, 0.2, 0.3, 0.5] {
        let loess = detrend_loess(&data, &t, bw, 1);
        let loess_err: f64 = loess
            .trend
            .as_slice()
            .iter()
            .zip(true_trend.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / m as f64;
        println!("  bandwidth={bw:.1}: MSE={loess_err:.6}");
    }

    // --- Section 4: Auto detrend ---
    println!("\n--- Automatic Method Selection ---");
    let auto = auto_detrend(&data, &t);
    println!("  Selected method: {}", auto.method);
    let auto_err: f64 = auto
        .trend
        .as_slice()
        .iter()
        .zip(true_trend.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    println!("  Trend MSE: {auto_err:.6}");

    // --- Section 5: Additive decomposition ---
    println!("\n--- Additive Decomposition ---");
    let period = 1.0 / 5.0; // period = 0.2 (5 cycles in [0,1])
    let decomp = decompose_additive(&data, &t, period, "loess", 0.2, 3);
    println!("  Method: {}", decomp.method);
    println!("  Period: {:.4}", decomp.period);

    // Compare recovered components
    let trend_mse: f64 = decomp
        .trend
        .as_slice()
        .iter()
        .zip(true_trend.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    let seasonal_mse: f64 = decomp
        .seasonal
        .as_slice()
        .iter()
        .zip(true_seasonal.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    let remainder_var: f64 = decomp
        .remainder
        .as_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        / m as f64;
    println!("  Trend MSE: {trend_mse:.6}");
    println!("  Seasonal MSE: {seasonal_mse:.6}");
    println!("  Remainder variance: {remainder_var:.6}");

    // --- Section 6: STL decomposition ---
    println!("\n--- STL Decomposition ---");
    // STL expects period as integer (number of grid points per cycle)
    let stl_period = (m as f64 * period).round() as usize;
    let stl = stl_decompose(&data, stl_period, None, None, None, false, None, None);
    let stl_trend_mse: f64 = stl
        .trend
        .as_slice()
        .iter()
        .zip(true_trend.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    let stl_seasonal_mse: f64 = stl
        .seasonal
        .as_slice()
        .iter()
        .zip(true_seasonal.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    println!("  Period: {stl_period} points");
    println!("  Trend MSE: {stl_trend_mse:.6}");
    println!("  Seasonal MSE: {stl_seasonal_mse:.6}");

    // Verify decomposition sums to original
    let recomposed: Vec<f64> = (0..m)
        .map(|i| stl.trend[(0, i)] + stl.seasonal[(0, i)] + stl.remainder[(0, i)])
        .collect();
    let recomp_err: f64 = data_vec
        .iter()
        .zip(recomposed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / m as f64;
    println!("  Recomposition error: {recomp_err:.2e} (should be ~0)");

    println!("\n=== Done ===");
}
