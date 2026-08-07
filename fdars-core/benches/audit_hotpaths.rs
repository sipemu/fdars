//! Audit hot-path benchmarks for fdars-core.
//!
//! One representative sentinel per hot-path module at Phase 1 baseline cell.
//! Sizes, caps, and sample_size rationale: see .planning/phases/01-.../01-RESEARCH.md §8.
//! Raw output saved to .planning/research/bench/ per D-06.
//!
//! **4-combo sentinel choice (D-04, A5 resolution):**
//! `fdata_to_pc_1d` was the original candidate for the 4-combo feature-matrix
//! sentinel, but `center_columns` (regression.rs:167-181) uses plain sequential
//! `for` loops and nalgebra SVD is always sequential — so FPCA produces near-
//! identical timings for the `parallel` and non-`parallel` combos and is a poor
//! discriminator.  `karcher_mean` was substituted: its inner N-loop uses
//! `iter_maybe_parallel!` (src/alignment/karcher.rs:185) and genuinely differs
//! across the 4 combos.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::alignment::karcher_mean;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use std::f64::consts::PI;

/// Generate synthetic functional data (n curves, m time points).
///
/// Uses deterministic phase/amplitude variation so no RNG dependency is needed.
/// Column-major layout: element (i, j) at index `i + j * n`.
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

/// Sentinel D-03: FPCA / SVD module baseline.
///
/// Benchmarks `fdata_to_pc_1d` at N=500, M=200 — the representative audit cell
/// for the regression/FPCA module.  This function is not parallel-gated
/// (center_columns is sequential, nalgebra SVD is always sequential), so it
/// serves as the per-module baseline but NOT the 4-combo discriminator.
fn bench_fpca_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    // Tune for audit cell: SVD O(m^3) at m=200 costs ~100 ms/iter
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)))
    });

    group.finish();
}

/// Sentinel D-04: 4-combo feature-matrix discriminator.
///
/// Benchmarks `karcher_mean` at N=100, M=50.  `karcher_mean` uses
/// `iter_maybe_parallel!` in its inner N-loop (karcher.rs:185), so it
/// genuinely exercises different code across the 4 feature combos:
///   - `""` (no-default-features) → sequential iterator
///   - `parallel`                 → rayon parallel iterator
///   - `linalg`                   → sequential (linalg adds faer, not rayon)
///   - `linalg,parallel`          → rayon parallel iterator
fn bench_matrix_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_karcher");
    // N=100, M=50 keeps each of the 4 combo runs fast (<20 s total)
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data),
                black_box(&argvals),
                black_box(10usize),
                black_box(1e-3),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fpca_sentinel, bench_matrix_sentinel);
criterion_main!(benches);
