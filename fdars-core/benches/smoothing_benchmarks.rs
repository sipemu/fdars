//! Benchmarks for kernel smoothing operations
//!
//! Compares performance of:
//! - Nadaraya-Watson kernel smoothing
//! - Local linear smoothing
//! - Local polynomial smoothing (degree 2)
//! - k-NN smoother (k = 5, k = 20)
//! - Bandwidth optimization (optim_bandwidth)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::smoothing::{
    knn_smoother, local_linear, local_polynomial, nadaraya_watson, optim_bandwidth, CvCriterion,
};
use std::f64::consts::PI;

/// Generate noisy sine-curve data for smoothing benchmarks.
fn generate_smoothing_data(n: usize, m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let noise = ((i as f64 * 17.3 + 0.5).sin()) * 0.3;
            (2.0 * PI * xi).sin() + 0.5 * (4.0 * PI * xi).cos() + noise
        })
        .collect();
    let x_new: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    (x, y, x_new)
}

fn bench_nadaraya_watson(c: &mut Criterion) {
    let mut group = c.benchmark_group("nadaraya_watson");

    for &n in &[50, 200, 1000] {
        for &m in &[50, 200] {
            let (x, y, x_new) = generate_smoothing_data(n, m);
            let bandwidth = 0.1;
            let label = format!("n{}_m{}", n, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    nadaraya_watson(
                        black_box(&x),
                        black_box(&y),
                        black_box(&x_new),
                        black_box(bandwidth),
                        black_box("gaussian"),
                    )
                    .unwrap()
                });
            });
        }
    }

    group.finish();
}

fn bench_local_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_linear");

    for &n in &[50, 200, 1000] {
        for &m in &[50, 200] {
            let (x, y, x_new) = generate_smoothing_data(n, m);
            let bandwidth = 0.1;
            let label = format!("n{}_m{}", n, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    local_linear(
                        black_box(&x),
                        black_box(&y),
                        black_box(&x_new),
                        black_box(bandwidth),
                        black_box("gaussian"),
                    )
                    .unwrap()
                });
            });
        }
    }

    group.finish();
}

fn bench_local_polynomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_polynomial");

    for &n in &[50, 200, 1000] {
        for &m in &[50, 200] {
            let (x, y, x_new) = generate_smoothing_data(n, m);
            let bandwidth = 0.1;
            let label = format!("n{}_m{}", n, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    local_polynomial(
                        black_box(&x),
                        black_box(&y),
                        black_box(&x_new),
                        black_box(bandwidth),
                        black_box(2),
                        black_box("gaussian"),
                    )
                    .unwrap()
                });
            });
        }
    }

    group.finish();
}

fn bench_knn_smoother(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_smoother");

    for &n in &[50, 200, 1000] {
        for &m in &[50, 200] {
            let (x, y, x_new) = generate_smoothing_data(n, m);

            for &k in &[5, 20] {
                let label = format!("n{}_m{}_k{}", n, m, k);

                group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                    b.iter(|| {
                        knn_smoother(
                            black_box(&x),
                            black_box(&y),
                            black_box(&x_new),
                            black_box(k),
                        )
                        .unwrap()
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_optim_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim_bandwidth");
    // Use shorter measurement for this expensive operation
    group.sample_size(20);

    for &n in &[50, 200, 1000] {
        let m = n; // optim_bandwidth evaluates on the same grid
        let (x, y, _) = generate_smoothing_data(n, m);
        let label = format!("n{}", n);

        group.bench_with_input(BenchmarkId::new("gcv", &label), &label, |b, _| {
            b.iter(|| {
                optim_bandwidth(
                    black_box(&x),
                    black_box(&y),
                    black_box(None),
                    black_box(CvCriterion::Gcv),
                    black_box("gaussian"),
                    black_box(20),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_nadaraya_watson,
    bench_local_linear,
    bench_local_polynomial,
    bench_knn_smoother,
    bench_optim_bandwidth
);
criterion_main!(benches);
