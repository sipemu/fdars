//! Example 26: Elastic Analysis
//!
//! Elastic (shape-aware) analysis methods that jointly handle amplitude and phase:
//! - Elastic FPCA — vertical, horizontal, and joint (`vert_fpca`, `horiz_fpca`, `joint_fpca`)
//! - Elastic regression — alignment-integrated scalar-on-function (`elastic_regression`)
//! - Elastic PCR — principal component regression after alignment (`elastic_pcr`)
//! - Elastic logistic — binary classification with alignment (`elastic_logistic`)
//! - Elastic changepoint detection (`elastic_amp_changepoint`, `elastic_ph_changepoint`)
//! - Elastic attribution — amplitude vs phase decomposition (`elastic_pcr_attribution`)

use fdars_core::alignment::karcher_mean;
use fdars_core::elastic_changepoint::CovKernel;
use fdars_core::elastic_regression::PcaMethod;
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn generate_data(n: usize, m: usize, t: &[f64]) -> (FdMatrix, Vec<f64>, Vec<i8>) {
    let mut col_major = vec![0.0; n * m];
    let mut y = vec![0.0; n];
    let mut y_class: Vec<i8> = vec![0; n];
    for i in 0..n {
        let amp = 1.0 + 0.5 * (i as f64 / n as f64 - 0.5);
        let shift = 0.05 * (i as f64 - n as f64 / 2.0) / n as f64;
        y[i] = amp;
        y_class[i] = if amp > 1.0 { 1 } else { -1 };
        for (j, &tj) in t.iter().enumerate() {
            let noise = 0.02 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            col_major[i + j * n] = amp * (2.0 * PI * (tj + shift)).sin() + noise;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    (data, y, y_class)
}

fn print_fpca_result(label: &str, eigenvalues_len: usize, cumvar: &[f64]) {
    println!("  Components: {}", eigenvalues_len);
    for (k, &v) in cumvar.iter().enumerate() {
        println!("  FPC {}: cumulative variance={:.1}%", k, v * 100.0);
    }
    let _ = label;
}

fn demo_fpca(data: &FdMatrix, t: &[f64]) {
    println!("=== 1. Elastic FPCA ===");
    let km = karcher_mean(data, t, 15, 0.01, 1e-4);
    println!("  Karcher mean iterations: {}", km.n_iter);

    println!("\n  --- Vertical FPCA (amplitude variation) ---");
    if let Ok(vfpca) = fdars_core::vert_fpca(&km, t, 3) {
        print_fpca_result(
            "vertical",
            vfpca.eigenvalues.len(),
            &vfpca.cumulative_variance,
        );
    }
    println!("\n  --- Horizontal FPCA (phase variation) ---");
    if let Ok(hfpca) = fdars_core::horiz_fpca(&km, t, 2) {
        print_fpca_result(
            "horizontal",
            hfpca.eigenvalues.len(),
            &hfpca.cumulative_variance,
        );
    }
    println!("\n  --- Joint FPCA (combined) ---");
    if let Ok(jfpca) = fdars_core::joint_fpca(&km, t, 3, None) {
        println!("  Balance parameter c: {:.4}", jfpca.balance_c);
        print_fpca_result("joint", jfpca.eigenvalues.len(), &jfpca.cumulative_variance);
    }
}

fn demo_regression(data: &FdMatrix, y: &[f64], y_class: &[i8], t: &[f64], n: usize) {
    println!("\n=== 2. Elastic Regression ===");
    if let Ok(ereg) = fdars_core::elastic_regression(data, y, t, 4, 0.01, 20, 1e-4) {
        println!(
            "  R² = {:.4}, Iterations: {}, SSE = {:.4}",
            ereg.r_squared, ereg.n_iter, ereg.sse
        );
    } else {
        println!("  (did not converge)");
    }

    println!("\n=== 3. Elastic PCR ===");
    if let Ok(epcr) = fdars_core::elastic_pcr(data, y, t, 3, PcaMethod::Vertical, 0.01, 15, 1e-4) {
        println!(
            "  R² = {:.4}, Components: {}",
            epcr.r_squared,
            epcr.coefficients.len()
        );
        for (k, &c) in epcr.coefficients.iter().enumerate() {
            println!("  beta_{} = {:+.4}", k, c);
        }
        println!("\n=== 4. Elastic PCR Attribution ===");
        if let Ok(attr) = fdars_core::elastic_pcr_attribution(&epcr, y, 3, 100, 42) {
            println!("  Amplitude importance: {:.4}", attr.amplitude_importance);
            println!("  Phase importance:     {:.4}", attr.phase_importance);
        }
    }

    println!("\n=== 5. Elastic Logistic Classification ===");
    if let Ok(elog) = fdars_core::elastic_logistic(data, y_class, t, 4, 0.01, 20, 1e-4) {
        println!("  Accuracy: {:.1}%", elog.accuracy * 100.0);
        let n_pos = elog.predicted_classes.iter().filter(|&&c| c == 1).count();
        println!("  Predicted: {} positive, {} negative", n_pos, n - n_pos);
    } else {
        println!("  (did not converge)");
    }
}

fn demo_changepoint(t: &[f64], n: usize, m: usize) {
    println!("\n=== 6. Elastic Changepoint Detection ===");
    let mut cp_col = vec![0.0; n * m];
    for i in 0..n {
        let amp = if i < n / 2 { 1.0 } else { 2.0 };
        for (j, &tj) in t.iter().enumerate() {
            cp_col[i + j * n] = amp * (2.0 * PI * tj).sin();
        }
    }
    let cp_data = FdMatrix::from_column_major(cp_col, n, m).unwrap();

    if let Ok(cp) = fdars_core::elastic_amp_changepoint(
        &cp_data,
        t,
        0.01,
        15,
        200,
        CovKernel::Bartlett,
        None,
        42,
    ) {
        println!("  Amplitude changepoint at index {}", cp.changepoint);
        println!(
            "  Test statistic: {:.4}, p-value: {:.4}",
            cp.test_statistic, cp.p_value
        );
    }
    if let Ok(cp) = fdars_core::elastic_ph_changepoint(
        &cp_data,
        t,
        0.01,
        15,
        200,
        CovKernel::Bartlett,
        None,
        42,
    ) {
        println!("  Phase changepoint at index {}", cp.changepoint);
        println!("  Test statistic: {:.4}", cp.test_statistic);
    }
}

fn main() {
    let n = 30;
    let m = 51;
    let t = uniform_grid(m);
    let (data, y, y_class) = generate_data(n, m, &t);

    demo_fpca(&data, &t);
    demo_regression(&data, &y, &y_class, &t, n);
    demo_changepoint(&t, n, m);
    println!("\nDone.");
}
