//! Benchmarks for FdMatrix operations
//!
//! Compares performance of:
//! - Construction (from_column_major)
//! - Row and column access patterns
//! - Efficient row operations (row_to_buf, row_dot, row_l2_sq)
//! - Layout conversion (to_row_major vs iterating with row())
//! - nalgebra roundtrip (to_dmatrix / from_dmatrix)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

/// Generate column-major data with sine-based values.
fn generate_matrix_data(nrows: usize, ncols: usize) -> Vec<f64> {
    let mut data = vec![0.0; nrows * ncols];
    for j in 0..ncols {
        let t = j as f64 / ncols as f64;
        for i in 0..nrows {
            data[i + j * nrows] =
                (2.0 * PI * t).sin() + 0.5 * ((i as f64 * 0.1 + j as f64 * 0.3).sin());
        }
    }
    data
}

/// Matrix sizes to benchmark: (nrows, ncols).
const SIZES: [(usize, usize); 3] = [(100, 50), (1000, 100), (5000, 200)];

fn bench_from_column_major(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_from_column_major");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
            b.iter(|| {
                FdMatrix::from_column_major(
                    black_box(data.clone()),
                    black_box(nrows),
                    black_box(ncols),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_row_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_row");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let mat = FdMatrix::from_column_major(data, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("single", &label), &label, |b, _| {
            b.iter(|| black_box(mat.row(black_box(nrows / 2))));
        });

        group.bench_with_input(BenchmarkId::new("all_rows", &label), &label, |b, _| {
            b.iter(|| {
                for i in 0..nrows {
                    black_box(mat.row(i));
                }
            });
        });
    }

    group.finish();
}

fn bench_column_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_column");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let mat = FdMatrix::from_column_major(data, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("single", &label), &label, |b, _| {
            b.iter(|| black_box(mat.column(black_box(ncols / 2))));
        });

        group.bench_with_input(BenchmarkId::new("all_cols", &label), &label, |b, _| {
            b.iter(|| {
                for j in 0..ncols {
                    black_box(mat.column(j));
                }
            });
        });
    }

    group.finish();
}

fn bench_row_to_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_row_to_buf");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let mat = FdMatrix::from_column_major(data, nrows, ncols).unwrap();
        let mut buf = vec![0.0; ncols];
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("single", &label), &label, |b, _| {
            b.iter(|| {
                mat.row_to_buf(black_box(nrows / 2), black_box(&mut buf));
                black_box(&buf);
            });
        });

        group.bench_with_input(BenchmarkId::new("all_rows", &label), &label, |b, _| {
            b.iter(|| {
                for i in 0..nrows {
                    mat.row_to_buf(i, &mut buf);
                }
                black_box(&buf);
            });
        });
    }

    group.finish();
}

fn bench_row_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_row_dot");

    for &(nrows, ncols) in &SIZES {
        let data_a = generate_matrix_data(nrows, ncols);
        let data_b = generate_matrix_data(nrows, ncols);
        let mat_a = FdMatrix::from_column_major(data_a, nrows, ncols).unwrap();
        let mat_b = FdMatrix::from_column_major(data_b, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("single", &label), &label, |b, _| {
            b.iter(|| black_box(mat_a.row_dot(black_box(0), black_box(&mat_b), black_box(1))));
        });

        // Benchmark all-pairs dot product for first 50 rows
        let n_pairs = nrows.min(50);
        group.bench_with_input(
            BenchmarkId::new(&format!("pairs_{}", n_pairs), &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..n_pairs {
                        for k in (i + 1)..n_pairs {
                            sum += mat_a.row_dot(i, &mat_b, k);
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_row_l2_sq(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_row_l2_sq");

    for &(nrows, ncols) in &SIZES {
        let data_a = generate_matrix_data(nrows, ncols);
        let data_b = generate_matrix_data(nrows, ncols);
        let mat_a = FdMatrix::from_column_major(data_a, nrows, ncols).unwrap();
        let mat_b = FdMatrix::from_column_major(data_b, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("single", &label), &label, |b, _| {
            b.iter(|| black_box(mat_a.row_l2_sq(black_box(0), black_box(&mat_b), black_box(1))));
        });

        let n_pairs = nrows.min(50);
        group.bench_with_input(
            BenchmarkId::new(&format!("pairs_{}", n_pairs), &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..n_pairs {
                        for k in (i + 1)..n_pairs {
                            sum += mat_a.row_l2_sq(i, &mat_b, k);
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_to_row_major(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_to_row_major");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let mat = FdMatrix::from_column_major(data, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        // Dedicated to_row_major method
        group.bench_with_input(BenchmarkId::new("to_row_major", &label), &label, |b, _| {
            b.iter(|| black_box(mat.to_row_major()));
        });

        // Iterating with row() for comparison
        group.bench_with_input(BenchmarkId::new("iterate_row", &label), &label, |b, _| {
            b.iter(|| {
                let mut buf = Vec::with_capacity(nrows * ncols);
                for i in 0..nrows {
                    buf.extend_from_slice(&mat.row(i));
                }
                black_box(buf)
            });
        });
    }

    group.finish();
}

fn bench_dmatrix_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("FdMatrix_dmatrix_roundtrip");

    for &(nrows, ncols) in &SIZES {
        let data = generate_matrix_data(nrows, ncols);
        let mat = FdMatrix::from_column_major(data, nrows, ncols).unwrap();
        let label = format!("{}x{}", nrows, ncols);

        group.bench_with_input(BenchmarkId::new("to_dmatrix", &label), &label, |b, _| {
            b.iter(|| black_box(mat.to_dmatrix()));
        });

        let dmat = mat.to_dmatrix();
        group.bench_with_input(BenchmarkId::new("from_dmatrix", &label), &label, |b, _| {
            b.iter(|| black_box(FdMatrix::from_dmatrix(black_box(&dmat))));
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", &label), &label, |b, _| {
            b.iter(|| {
                let d = mat.to_dmatrix();
                black_box(FdMatrix::from_dmatrix(&d))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_from_column_major,
    bench_row_access,
    bench_column_access,
    bench_row_to_buf,
    bench_row_dot,
    bench_row_l2_sq,
    bench_to_row_major,
    bench_dmatrix_roundtrip
);
criterion_main!(benches);
