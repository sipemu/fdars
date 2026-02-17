//! Example 08: Functional PCA and PLS Regression
//!
//! Demonstrates dimensionality reduction of functional data via
//! Functional Principal Component Analysis (FPCA) and Partial Least
//! Squares (PLS) regression relating functional predictors to a
//! scalar response.

use fdars_core::helpers::extract_curves;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::{fdata_to_pc_1d, fdata_to_pls_1d, FpcaResult};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::utility::integrate_simpson;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn print_variance_explained(fpca: &FpcaResult) {
    let total_var: f64 = fpca.singular_values.iter().map(|s| s * s).sum();
    println!(
        "  Singular values: {:?}",
        fpca.singular_values
            .iter()
            .map(|s| format!("{s:.4}"))
            .collect::<Vec<_>>()
    );

    let mut cumvar = 0.0;
    println!("  Variance explained:");
    for (k, sv) in fpca.singular_values.iter().enumerate() {
        let var_k = sv * sv;
        cumvar += var_k;
        let prop = var_k / total_var * 100.0;
        let cumprop = cumvar / total_var * 100.0;
        println!("    PC{}: {prop:.1}% (cumulative: {cumprop:.1}%)", k + 1);
    }
}

fn print_loadings(fpca: &FpcaResult, ncomp: usize) {
    println!("\n  PC loadings (first 5 values of each):");
    for k in 0..ncomp {
        let loading: Vec<f64> = (0..5).map(|j| fpca.rotation[(j, k)]).collect();
        println!(
            "    PC{}: {:?}",
            k + 1,
            loading
                .iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
        );
    }
}

fn print_scores(fpca: &FpcaResult, ncomp: usize) {
    println!("\n  PC scores (first 5 observations):");
    println!(
        "  {:>5} {:>10} {:>10} {:>10} {:>10}",
        "Obs", "PC1", "PC2", "PC3", "PC4"
    );
    for i in 0..5 {
        print!("  {:>5}", i);
        for k in 0..ncomp {
            print!(" {:>10.4}", fpca.scores[(i, k)]);
        }
        println!();
    }
}

fn print_reconstruction_error(
    fpca: &FpcaResult,
    data_mat: &FdMatrix,
    n: usize,
    m: usize,
    ncomp: usize,
) {
    println!("\n  Reconstruction error by number of components:");
    for nc in 1..=ncomp {
        let mse = compute_reconstruction_mse(fpca, data_mat, n, m, nc);
        println!("    {nc} components: MSE = {mse:.6}");
    }
}

fn compute_reconstruction_mse(
    fpca: &FpcaResult,
    data_mat: &FdMatrix,
    n: usize,
    m: usize,
    nc: usize,
) -> f64 {
    let mut mse = 0.0;
    for j in 0..m {
        for i in 0..n {
            let mut val = fpca.mean[j];
            for k in 0..nc {
                val += fpca.scores[(i, k)] * fpca.rotation[(j, k)];
            }
            let orig = data_mat[(i, j)];
            mse += (orig - val).powi(2);
        }
    }
    mse / (n * m) as f64
}

fn main() {
    println!("=== Example 08: Functional PCA and PLS Regression ===\n");

    let n = 40;
    let m = 60;
    let big_m = 5;
    let t = uniform_grid(m);

    let data_mat = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    // --- Section 1: FPCA ---
    println!("--- Functional PCA ---");
    let ncomp = 4;
    if let Some(fpca) = fdata_to_pc_1d(&data_mat, ncomp) {
        print_variance_explained(&fpca);
        print_loadings(&fpca, ncomp);
        print_scores(&fpca, ncomp);
        print_reconstruction_error(&fpca, &data_mat, n, m, ncomp);
    }

    // --- Section 2: PLS regression ---
    println!("\n--- PLS Regression ---");
    let curves = extract_curves(&data_mat);
    let y: Vec<f64> = curves.iter().map(|c| integrate_simpson(c, &t)).collect();

    println!("  Response (integral of each curve):");
    println!(
        "  y (first 5): {:?}",
        y[..5].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>()
    );

    let ncomp_pls = 3;
    if let Some(pls) = fdata_to_pls_1d(&data_mat, &y, ncomp_pls) {
        println!("\n  PLS weights (first 5 values of each component):");
        for k in 0..ncomp_pls {
            let w: Vec<f64> = (0..5).map(|j| pls.weights[(j, k)]).collect();
            println!(
                "    Comp {}: {:?}",
                k + 1,
                w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>()
            );
        }

        println!("\n  PLS scores (first 5 observations):");
        println!(
            "  {:>5} {:>10} {:>10} {:>10}",
            "Obs", "Comp1", "Comp2", "Comp3"
        );
        for i in 0..5 {
            print!("  {:>5}", i);
            for k in 0..ncomp_pls {
                print!(" {:>10.4}", pls.scores[(i, k)]);
            }
            println!();
        }
    }

    println!("\n=== Done ===");
}
