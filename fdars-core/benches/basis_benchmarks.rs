//! Benchmarks for basis representation operations
//!
//! Compares performance of:
//! - B-spline basis evaluation (bspline_basis)
//! - Fourier basis evaluation (fourier_basis)
//! - Penalized basis smoothing (smooth_basis)
//! - GCV-optimal smoothing (smooth_basis_gcv)
//! - Basis projection roundtrip (fdata_to_basis_1d / basis_to_fdata_1d)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::basis::{basis_to_fdata_1d, bspline_basis, fdata_to_basis_1d, fourier_basis};
use fdars_core::matrix::FdMatrix;
use fdars_core::smooth_basis::{
    bspline_penalty_matrix, smooth_basis, smooth_basis_gcv, BasisType, FdPar,
};
use std::f64::consts::PI;

/// Generate evaluation points on [0, 1].
fn make_argvals(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Generate synthetic functional data matrix (n curves x m points).
fn generate_basis_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals = make_argvals(m);
    let mut data = vec![0.0; n * m];

    for i in 0..n {
        let c1 = ((i as f64 * 3.7).sin()) * 2.0;
        let c2 = ((i as f64 * 5.1 + 0.3).sin()) * 1.5;
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = c1 * (2.0 * PI * t).sin()
                + c2 * (4.0 * PI * t).cos()
                + 0.3 * ((i as f64 * 11.3 + j as f64 * 0.7).sin());
        }
    }

    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}

fn bench_bspline_basis(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_basis");

    for &nbasis in &[10, 30, 50] {
        for &m in &[100, 500] {
            let argvals = make_argvals(m);
            // nbasis = nknots + order, so nknots = nbasis - order
            let order = 4;
            let nknots = nbasis - order;
            let label = format!("nb{}_m{}", nbasis, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| bspline_basis(black_box(&argvals), black_box(nknots), black_box(order)));
            });
        }
    }

    group.finish();
}

fn bench_fourier_basis(c: &mut Criterion) {
    let mut group = c.benchmark_group("fourier_basis");

    for &nbasis in &[10, 30] {
        for &m in &[100, 500] {
            let argvals = make_argvals(m);
            let label = format!("nb{}_m{}", nbasis, m);

            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| fourier_basis(black_box(&argvals), black_box(nbasis)));
            });
        }
    }

    group.finish();
}

fn bench_smooth_basis(c: &mut Criterion) {
    let mut group = c.benchmark_group("smooth_basis");
    group.sample_size(30);

    let n = 50;
    for &m in &[100, 500] {
        for &nbasis in &[10, 30] {
            let (data, argvals) = generate_basis_data(n, m);
            let order = 4;
            let lfd_order = 2;
            let penalty = bspline_penalty_matrix(&argvals, nbasis, order, lfd_order);

            let fdpar = FdPar {
                basis_type: BasisType::Bspline { order },
                nbasis,
                lambda: 1e-4,
                lfd_order,
                penalty_matrix: penalty,
            };

            let label = format!("n{}_m{}_nb{}", n, m, nbasis);
            group.bench_with_input(BenchmarkId::new("bspline", &label), &label, |b, _| {
                b.iter(|| {
                    smooth_basis(black_box(&data), black_box(&argvals), black_box(&fdpar)).unwrap()
                });
            });
        }
    }

    group.finish();
}

fn bench_smooth_basis_gcv(c: &mut Criterion) {
    let mut group = c.benchmark_group("smooth_basis_gcv");
    group.sample_size(10);

    let n = 50;
    for &m in &[100, 500] {
        let nbasis = 15;
        let (data, argvals) = generate_basis_data(n, m);
        let basis_type = BasisType::Bspline { order: 4 };
        let label = format!("n{}_m{}_nb{}", n, m, nbasis);

        group.bench_with_input(BenchmarkId::new("bspline", &label), &label, |b, _| {
            b.iter(|| {
                smooth_basis_gcv(
                    black_box(&data),
                    black_box(&argvals),
                    black_box(&basis_type),
                    black_box(nbasis),
                    black_box(2),
                    black_box((-6.0, 2.0)),
                    black_box(15),
                )
            });
        });
    }

    group.finish();
}

fn bench_fdata_to_basis_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdata_to_basis_1d");

    let n = 50;
    for &m in &[100, 500] {
        for &nbasis in &[10, 30] {
            let (data, argvals) = generate_basis_data(n, m);
            let label = format!("n{}_m{}_nb{}", n, m, nbasis);

            // basis_type: 0 = bspline, 1 = fourier
            group.bench_with_input(BenchmarkId::new("bspline", &label), &label, |b, _| {
                b.iter(|| {
                    fdata_to_basis_1d(
                        black_box(&data),
                        black_box(&argvals),
                        black_box(nbasis),
                        black_box(0),
                    )
                });
            });
        }
    }

    group.finish();
}

fn bench_basis_to_fdata_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("basis_to_fdata_1d");

    let n = 50;
    for &m in &[100, 500] {
        let nbasis = 15;
        let (data, argvals) = generate_basis_data(n, m);

        // First project to basis
        let proj = fdata_to_basis_1d(&data, &argvals, nbasis, 0).unwrap();
        let coefs = &proj.coefficients;
        let label = format!("n{}_m{}_nb{}", n, m, nbasis);

        group.bench_with_input(BenchmarkId::new("bspline", &label), &label, |b, _| {
            b.iter(|| {
                basis_to_fdata_1d(
                    black_box(coefs),
                    black_box(&argvals),
                    black_box(nbasis),
                    black_box(0),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bspline_basis,
    bench_fourier_basis,
    bench_smooth_basis,
    bench_smooth_basis_gcv,
    bench_fdata_to_basis_1d,
    bench_basis_to_fdata_1d
);
criterion_main!(benches);
