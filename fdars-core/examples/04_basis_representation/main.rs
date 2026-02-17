//! Example 04: Basis Representation and Smoothing
//!
//! Demonstrates representing functional data in different basis systems:
//! B-splines and Fourier bases. Shows projection, reconstruction,
//! P-spline smoothing with penalty selection (GCV/AIC/BIC), and
//! Fourier fitting with automatic basis selection.

use fdars_core::basis::{
    basis_to_fdata_1d, bspline_basis, fdata_to_basis_1d, fourier_basis, fourier_fit_1d,
    pspline_fit_1d, select_fourier_nbasis_gcv,
};
use fdars_core::simulation::{add_error_pointwise, sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

fn main() {
    println!("=== Example 04: Basis Representation ===\n");

    let n = 20;
    let m = 80;
    let big_m = 5;
    let t = uniform_grid(m);

    // Generate clean and noisy data
    let clean_mat = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let noisy = add_error_pointwise(&clean_mat, 0.15, Some(42));
    let clean = clean_mat.into_vec();

    // --- Section 1: B-spline basis evaluation ---
    println!("--- B-spline Basis ---");
    let nknots = 10;
    let order = 4; // cubic B-splines
    let bspline = bspline_basis(&t, nknots, order);
    let nbasis_bspline = nknots + order - 2;
    println!("  Knots: {nknots}, Order: {order} (cubic)");
    println!("  Number of basis functions: {nbasis_bspline}");
    println!(
        "  Basis matrix: {} elements ({m} x {nbasis_bspline})",
        bspline.len()
    );

    // --- Section 2: Fourier basis evaluation ---
    println!("\n--- Fourier Basis ---");
    let nbasis_fourier = 11; // 1 constant + 5 sin/cos pairs
    let fourier = fourier_basis(&t, nbasis_fourier);
    println!("  Number of basis functions: {nbasis_fourier}");
    println!(
        "  Basis matrix: {} elements ({m} x {nbasis_fourier})",
        fourier.len()
    );

    // --- Section 3: Project to basis and reconstruct ---
    println!("\n--- Basis Projection and Reconstruction ---");

    // B-spline projection (basis_type=0)
    if let Some(proj) = fdata_to_basis_1d(&noisy, &t, nbasis_bspline, 0) {
        println!("  B-spline coefficients: {} per curve", proj.n_basis);
        let reconstructed = basis_to_fdata_1d(&proj.coefficients, &t, proj.n_basis, 0);
        let err = rmse(reconstructed.as_slice(), &clean);
        println!("  B-spline reconstruction RMSE vs clean: {err:.6}");
    }

    // Fourier projection (basis_type=1)
    if let Some(proj) = fdata_to_basis_1d(&noisy, &t, nbasis_fourier, 1) {
        println!("  Fourier coefficients: {} per curve", proj.n_basis);
        let reconstructed = basis_to_fdata_1d(&proj.coefficients, &t, proj.n_basis, 1);
        let err = rmse(reconstructed.as_slice(), &clean);
        println!("  Fourier reconstruction RMSE vs clean: {err:.6}");
    }

    // --- Section 4: P-spline smoothing with different penalties ---
    println!("\n--- P-spline Smoothing ---");
    let pspline_nbasis = 20;
    for lambda in [0.001, 0.01, 0.1, 1.0, 10.0] {
        if let Some(result) = pspline_fit_1d(&noisy, &t, pspline_nbasis, lambda, 2) {
            let err = rmse(result.fitted.as_slice(), &clean);
            println!(
                "  λ={lambda:6.3}: RMSE={err:.6}, EDF={:.1}, GCV={:.6}, AIC={:.1}, BIC={:.1}",
                result.edf, result.gcv, result.aic, result.bic
            );
        }
    }

    // --- Section 5: Fourier fitting ---
    println!("\n--- Fourier Fitting ---");
    for nb in [5, 9, 15, 21] {
        if let Some(result) = fourier_fit_1d(&noisy, &t, nb) {
            let err = rmse(result.fitted.as_slice(), &clean);
            println!("  nbasis={nb:2}: RMSE={err:.6}, GCV={:.6}", result.gcv);
        }
    }

    // --- Section 6: Automatic Fourier basis selection ---
    println!("\n--- Automatic Fourier Basis Selection (GCV) ---");
    let best_nb = select_fourier_nbasis_gcv(&noisy, &t, 3, 25);
    println!("  Selected nbasis: {best_nb}");
    if let Some(result) = fourier_fit_1d(&noisy, &t, best_nb) {
        let err = rmse(result.fitted.as_slice(), &clean);
        println!("  RMSE with best nbasis: {err:.6}");
    }

    println!("\n=== Done ===");
}
