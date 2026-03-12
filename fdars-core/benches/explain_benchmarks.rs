//! Benchmarks for explainability methods
//!
//! Compares performance of:
//! - Generic SHAP values with varying n and n_samples
//! - Generic permutation importance with varying n
//! - Generic PDP with varying n
//!
//! All methods operate on a fitted fregre_lm model via the FpcPredictor trait.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::explain_generic::{generic_pdp, generic_permutation_importance, generic_shap_values};
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::fregre_lm;
use std::f64::consts::PI;

/// Generate synthetic functional data and a scalar response for regression.
fn generate_regression_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let mut data = vec![0.0; n * m];

    for i in 0..n {
        let c1 = ((i as f64 * 3.7).sin()) * 2.0;
        let c2 = ((i as f64 * 5.1 + 0.3).sin()) * 1.5;
        let c3 = ((i as f64 * 7.9 + 0.7).sin()) * 1.0;
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[i + j * n] = c1 * (2.0 * PI * t).sin()
                + c2 * (4.0 * PI * t).sin()
                + c3 * (6.0 * PI * t).sin();
        }
    }

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let c1 = ((i as f64 * 3.7).sin()) * 2.0;
            let c2 = ((i as f64 * 5.1 + 0.3).sin()) * 1.5;
            let noise = ((i as f64 * 13.7).sin()) * 0.1;
            2.0 * c1 + 1.0 * c2 + noise
        })
        .collect();

    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, y)
}

fn bench_shap_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_shap_values");
    let m = 50;
    let ncomp = 5;
    let seed = 42;

    // Vary n_samples with fixed n
    let n = 50;
    let (data, y) = generate_regression_data(n, m);
    let model = fregre_lm(&data, &y, None, ncomp).unwrap();

    for &n_samples in &[50, 100, 200] {
        let label = format!("n{}_ns{}", n, n_samples);
        group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
            b.iter(|| {
                generic_shap_values(
                    black_box(&model),
                    black_box(&data),
                    None,
                    black_box(n_samples),
                    black_box(seed),
                )
            });
        });
    }

    // Vary n with fixed n_samples to show scaling
    let n_samples = 100;
    for &n in &[30, 50, 100] {
        let (data_n, y_n) = generate_regression_data(n, m);
        let model_n = fregre_lm(&data_n, &y_n, None, ncomp).unwrap();
        let label = format!("n{}_ns{}", n, n_samples);
        group.bench_with_input(BenchmarkId::new("scale", &label), &label, |b, _| {
            b.iter(|| {
                generic_shap_values(
                    black_box(&model_n),
                    black_box(&data_n),
                    None,
                    black_box(n_samples),
                    black_box(seed),
                )
            });
        });
    }

    group.finish();
}

fn bench_permutation_importance(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_permutation_importance");
    let m = 50;
    let ncomp = 5;
    let n_perm = 50;
    let seed = 42;

    for &n in &[50, 100, 200] {
        let (data, y) = generate_regression_data(n, m);
        let model = fregre_lm(&data, &y, None, ncomp).unwrap();

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                generic_permutation_importance(
                    black_box(&model),
                    black_box(&data),
                    black_box(&y),
                    black_box(n_perm),
                    black_box(seed),
                )
            });
        });
    }

    group.finish();
}

fn bench_pdp(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_pdp");
    let m = 50;
    let ncomp = 5;
    let n_grid = 20;

    for &n in &[50, 100, 200] {
        let (data, y) = generate_regression_data(n, m);
        let model = fregre_lm(&data, &y, None, ncomp).unwrap();

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                generic_pdp(
                    black_box(&model),
                    black_box(&data),
                    None,
                    black_box(0), // component 0
                    black_box(n_grid),
                )
            });
        });
    }

    // Vary component index
    let n = 50;
    let (data, y) = generate_regression_data(n, m);
    let model = fregre_lm(&data, &y, None, ncomp).unwrap();

    for component in 0..ncomp.min(3) {
        group.bench_with_input(
            BenchmarkId::new("component", component),
            &component,
            |b, &comp| {
                b.iter(|| {
                    generic_pdp(
                        black_box(&model),
                        black_box(&data),
                        None,
                        black_box(comp),
                        black_box(n_grid),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_shap_values,
    bench_permutation_importance,
    bench_pdp,
);
criterion_main!(benches);
