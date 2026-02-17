//! Example 02: Functional Data Operations
//!
//! Demonstrates core operations on functional data: mean, centering,
//! derivatives, Lp norms, geometric median, inner products, and
//! Simpson's rule integration.

use fdars_core::fdata::{center_1d, deriv_1d, geometric_median_1d, mean_1d, norm_lp_1d};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::utility::{inner_product, inner_product_matrix, integrate_simpson};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 02: Functional Data Operations ===\n");

    let n = 25;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);

    // Generate sample data
    let mat = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    // --- Section 1: Mean function ---
    println!("--- Mean Function ---");
    let mean = mean_1d(&mat);
    println!("  Mean function length: {}", mean.len());
    println!(
        "  Mean at t=0.0: {:.4}, t=0.5: {:.4}, t=1.0: {:.4}",
        mean[0],
        mean[m / 2],
        mean[m - 1]
    );

    // --- Section 2: Centering ---
    println!("\n--- Centering ---");
    let centered = center_1d(&mat);
    let centered_mean = mean_1d(&centered);
    let max_residual = centered_mean
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    println!("  Max absolute value of centered mean: {max_residual:.2e} (should be ~0)");

    // --- Section 3: Derivatives ---
    println!("\n--- Numerical Derivatives ---");
    let first_deriv = deriv_1d(&mat, &t, 1);
    let second_deriv = deriv_1d(&mat, &t, 2);
    // First derivative of first curve
    let d1_curve0: Vec<f64> = (0..first_deriv.ncols())
        .map(|j| first_deriv[(0, j)])
        .collect();
    let d2_curve0: Vec<f64> = (0..second_deriv.ncols())
        .map(|j| second_deriv[(0, j)])
        .collect();
    println!(
        "  1st derivative: {} points, range [{:.4}, {:.4}]",
        d1_curve0.len(),
        d1_curve0.iter().cloned().fold(f64::INFINITY, f64::min),
        d1_curve0.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "  2nd derivative: {} points, range [{:.4}, {:.4}]",
        d2_curve0.len(),
        d2_curve0.iter().cloned().fold(f64::INFINITY, f64::min),
        d2_curve0.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // --- Section 4: Lp norms ---
    println!("\n--- Lp Norms ---");
    let l1_norms = norm_lp_1d(&mat, &t, 1.0);
    let l2_norms = norm_lp_1d(&mat, &t, 2.0);
    let linf_norms = norm_lp_1d(&mat, &t, f64::INFINITY);
    println!(
        "  L1 norms (first 5): {:?}",
        &l1_norms[..5]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
    );
    println!(
        "  L2 norms (first 5): {:?}",
        &l2_norms[..5]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
    );
    println!(
        "  L∞ norms (first 5): {:?}",
        &linf_norms[..5]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
    );

    // --- Section 5: Geometric median ---
    // The geometric median minimizes the sum of L2 distances to all curves,
    // making it more robust to outliers than the pointwise mean.
    println!("\n--- Geometric Median ---");
    let gmed = geometric_median_1d(&mat, &t, 100, 1e-6);
    let mean_diff: f64 = mean
        .iter()
        .zip(gmed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("  L2 distance between mean and geometric median: {mean_diff:.4}");
    println!(
        "  Geometric median at t=0.0: {:.4}, t=0.5: {:.4}, t=1.0: {:.4}",
        gmed[0],
        gmed[m / 2],
        gmed[m - 1]
    );

    // --- Section 6: Simpson's rule integration ---
    println!("\n--- Numerical Integration (Simpson's Rule) ---");
    // Integrate the mean function over [0, 1]
    let integral = integrate_simpson(&mean, &t);
    println!("  Integral of mean function over [0,1]: {integral:.6}");
    // Integrate a known function: sin(pi*t) over [0,1] should be 2/pi ≈ 0.6366
    let sin_vals: Vec<f64> = t
        .iter()
        .map(|&ti| (std::f64::consts::PI * ti).sin())
        .collect();
    let sin_integral = integrate_simpson(&sin_vals, &t);
    println!(
        "  Integral of sin(πt) over [0,1]: {sin_integral:.6} (exact: {:.6})",
        2.0 / std::f64::consts::PI
    );

    // --- Section 7: Inner products ---
    println!("\n--- Inner Products ---");
    let curves = mat.rows();
    let ip_01 = inner_product(&curves[0], &curves[1], &t);
    let ip_00 = inner_product(&curves[0], &curves[0], &t);
    println!("  <curve_0, curve_0> = {ip_00:.6} (self inner product)");
    println!("  <curve_0, curve_1> = {ip_01:.6}");

    // Inner product (Gram) matrix
    let gram = inner_product_matrix(&mat, &t);
    println!("  Gram matrix: {} elements ({n} x {n})", gram.len());
    println!(
        "  Gram[0,0]={:.4}, Gram[0,1]={:.4}, Gram[1,1]={:.4}",
        gram[(0, 0)],
        gram[(0, 1)],
        gram[(1, 1)]
    );

    println!("\n=== Done ===");
}
