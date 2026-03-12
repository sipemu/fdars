//! Benchmarks for elastic alignment methods
//!
//! Compares performance of:
//! - Pairwise elastic alignment with varying curve length m
//! - Self-distance matrix with varying n and m
//! - Karcher mean computation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::alignment::{elastic_align_pair, elastic_self_distance_matrix, karcher_mean};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

/// Generate synthetic functional data (n curves, m time points).
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        // Deterministic phase/amplitude variation per curve
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}

fn bench_elastic_align_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_align_pair");

    for &m in &[50, 100, 200] {
        let (mat, argvals) = generate_curves(2, m);
        let f1 = mat.row(0);
        let f2 = mat.row(1);

        group.bench_with_input(BenchmarkId::new("m", m), &m, |b, _| {
            b.iter(|| {
                elastic_align_pair(
                    black_box(&f1),
                    black_box(&f2),
                    black_box(&argvals),
                    black_box(0.0),
                )
            });
        });
    }

    // Also benchmark with lambda > 0
    let (mat, argvals) = generate_curves(2, 100);
    let f1 = mat.row(0);
    let f2 = mat.row(1);
    group.bench_function("m100_lambda0.1", |b| {
        b.iter(|| {
            elastic_align_pair(
                black_box(&f1),
                black_box(&f2),
                black_box(&argvals),
                black_box(0.1),
            )
        });
    });

    group.finish();
}

fn bench_self_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_self_distance_matrix");
    let m = 50;

    for &n in &[10, 30, 50] {
        let (mat, argvals) = generate_curves(n, m);

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                elastic_self_distance_matrix(
                    black_box(&mat),
                    black_box(&argvals),
                    black_box(0.0),
                )
            });
        });
    }

    group.finish();
}

fn bench_karcher_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("karcher_mean");

    let n = 20;
    let m = 50;
    let (mat, argvals) = generate_curves(n, m);

    group.bench_function("n20_m50", |b| {
        b.iter(|| {
            karcher_mean(
                black_box(&mat),
                black_box(&argvals),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
            )
        });
    });

    // Smaller problem for faster iteration
    let n = 10;
    let (mat_small, argvals_small) = generate_curves(n, m);
    group.bench_function("n10_m50", |b| {
        b.iter(|| {
            karcher_mean(
                black_box(&mat_small),
                black_box(&argvals_small),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_elastic_align_pair,
    bench_self_distance_matrix,
    bench_karcher_mean,
);
criterion_main!(benches);
