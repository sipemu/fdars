//! Benchmarks for regression and FPCA methods
//!
//! Compares performance of:
//! - FPCA (fdata_to_pc_1d) with varying n and m
//! - Functional linear regression (fregre_lm) with varying ncomp
//! - Functional logistic regression with varying ncomp

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

/// Generate synthetic functional data (n curves, m time points) and a scalar response.
///
/// The response is a linear combination of the first few FPC scores + noise,
/// so that fregre_lm has something meaningful to fit.
fn generate_regression_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];

    for i in 0..n {
        let c1 = ((i as f64 * 3.7).sin()) * 2.0;
        let c2 = ((i as f64 * 5.1 + 0.3).sin()) * 1.5;
        let c3 = ((i as f64 * 7.9 + 0.7).sin()) * 1.0;
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] =
                c1 * (2.0 * PI * t).sin() + c2 * (4.0 * PI * t).sin() + c3 * (6.0 * PI * t).sin();
        }
    }

    // Response is a function of the underlying coefficients
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let c1 = ((i as f64 * 3.7).sin()) * 2.0;
            let c2 = ((i as f64 * 5.1 + 0.3).sin()) * 1.5;
            let noise = ((i as f64 * 13.7).sin()) * 0.1;
            2.0 * c1 + 1.0 * c2 + noise
        })
        .collect();

    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, y, argvals)
}

/// Generate binary response data for logistic regression.
fn generate_logistic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let (mat, y_continuous, argvals) = generate_regression_data(n, m);
    // Convert to binary based on sign of y
    let y_binary: Vec<f64> = y_continuous
        .iter()
        .map(|&yi| if yi >= 0.0 { 1.0 } else { 0.0 })
        .collect();
    (mat, y_binary, argvals)
}

fn bench_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdata_to_pc_1d");

    for &n in &[50, 200] {
        for &m in &[50, 100] {
            let (data, _y, _argvals) = generate_regression_data(n, m);
            let ncomp = 5;
            let label = format!("n{}_m{}", n, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(ncomp)));
            });
        }
    }

    group.finish();
}

fn bench_fregre_lm(c: &mut Criterion) {
    let mut group = c.benchmark_group("fregre_lm");
    let n = 100;
    let m = 50;
    let (data, y, _argvals) = generate_regression_data(n, m);

    for &ncomp in &[3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("ncomp", ncomp), &ncomp, |b, &ncomp| {
            b.iter(|| fregre_lm(black_box(&data), black_box(&y), None, black_box(ncomp)));
        });
    }

    // Also vary n with fixed ncomp
    for &n in &[50, 200] {
        let (data_n, y_n, _) = generate_regression_data(n, m);
        let label = format!("n{}_nc5", n);
        group.bench_with_input(BenchmarkId::new("scale", &label), &label, |b, _| {
            b.iter(|| fregre_lm(black_box(&data_n), black_box(&y_n), None, black_box(5)));
        });
    }

    group.finish();
}

fn bench_functional_logistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("functional_logistic");
    let n = 100;
    let m = 50;
    let (data, y, _argvals) = generate_logistic_data(n, m);

    for &ncomp in &[3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("ncomp", ncomp), &ncomp, |b, &ncomp| {
            b.iter(|| {
                functional_logistic(
                    black_box(&data),
                    black_box(&y),
                    None,
                    black_box(ncomp),
                    black_box(100),
                    black_box(1e-6),
                )
            });
        });
    }

    // Vary n with fixed ncomp
    for &n in &[50, 200] {
        let (data_n, y_n, _) = generate_logistic_data(n, m);
        let label = format!("n{}_nc5", n);
        group.bench_with_input(BenchmarkId::new("scale", &label), &label, |b, _| {
            b.iter(|| {
                functional_logistic(
                    black_box(&data_n),
                    black_box(&y_n),
                    None,
                    black_box(5),
                    black_box(100),
                    black_box(1e-6),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fpca,
    bench_fregre_lm,
    bench_functional_logistic
);
criterion_main!(benches);
