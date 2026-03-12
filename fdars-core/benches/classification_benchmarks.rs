//! Benchmarks for functional classification methods
//!
//! Compares performance of:
//! - FPC + LDA with varying n and ncomp
//! - FPC + QDA with varying n and ncomp
//! - FPC + k-NN with varying n, ncomp, and k
//! - Cross-validation overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::classification::{fclassif_cv, fclassif_knn, fclassif_lda, fclassif_qda};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

/// Generate synthetic functional data with 2 classes.
///
/// Class 0 curves: sin(2πt) + offset_i
/// Class 1 curves: cos(2πt) + offset_i + shift
///
/// Returns (data matrix n×m, labels, argvals).
fn generate_two_class_data(n: usize, m: usize) -> (FdMatrix, Vec<usize>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let n_per_class = n / 2;
    let mut data = vec![0.0; n * m];
    let mut labels = vec![0usize; n];

    for i in 0..n {
        let class = if i < n_per_class { 0 } else { 1 };
        labels[i] = class;
        // Deterministic pseudo-random offset for each curve
        let offset = ((i as f64 * 7.3 + 1.1).sin()) * 0.3;
        for j in 0..m {
            let t = argvals[j];
            let value = if class == 0 {
                (2.0 * PI * t).sin() + offset
            } else {
                (2.0 * PI * t).cos() + offset + 1.0
            };
            data[i + j * n] = value; // column-major
        }
    }

    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, labels, argvals)
}

fn bench_lda(c: &mut Criterion) {
    let mut group = c.benchmark_group("fclassif_lda");
    let m = 50;

    for &n in &[50, 200, 500] {
        for &ncomp in &[3, 5, 10] {
            // ncomp must be < n/2 (each class has n/2 samples)
            if ncomp >= n / 2 {
                continue;
            }
            let (data, y, _argvals) = generate_two_class_data(n, m);
            let label = format!("n{}_nc{}", n, ncomp);
            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    fclassif_lda(black_box(&data), black_box(&y), None, black_box(ncomp))
                });
            });
        }
    }

    group.finish();
}

fn bench_qda(c: &mut Criterion) {
    let mut group = c.benchmark_group("fclassif_qda");
    let m = 50;

    for &n in &[50, 200, 500] {
        for &ncomp in &[3, 5, 10] {
            if ncomp >= n / 2 {
                continue;
            }
            let (data, y, _argvals) = generate_two_class_data(n, m);
            let label = format!("n{}_nc{}", n, ncomp);
            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    fclassif_qda(black_box(&data), black_box(&y), None, black_box(ncomp))
                });
            });
        }
    }

    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("fclassif_knn");
    let m = 50;

    for &n in &[50, 200, 500] {
        for &ncomp in &[3, 5, 10] {
            if ncomp >= n / 2 {
                continue;
            }
            for &k in &[3, 5, 7] {
                let (data, y, _argvals) = generate_two_class_data(n, m);
                let label = format!("n{}_nc{}_k{}", n, ncomp, k);
                group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                    b.iter(|| {
                        fclassif_knn(
                            black_box(&data),
                            black_box(&y),
                            None,
                            black_box(ncomp),
                            black_box(k),
                        )
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fclassif_cv");
    let m = 50;
    let ncomp = 5;
    let seed = 42;

    for &n in &[50, 200] {
        let (data, y, argvals) = generate_two_class_data(n, m);

        for method in &["lda", "qda"] {
            let label = format!("n{}_{}", n, method);
            group.bench_with_input(BenchmarkId::new("params", &label), &label, |b, _| {
                b.iter(|| {
                    fclassif_cv(
                        black_box(&data),
                        black_box(&argvals),
                        black_box(&y),
                        None,
                        black_box(method),
                        black_box(ncomp),
                        5,
                        black_box(seed),
                    )
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_lda, bench_qda, bench_knn, bench_cv);
criterion_main!(benches);
