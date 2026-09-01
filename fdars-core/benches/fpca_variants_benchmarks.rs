//! Phase 51 BENCH-01 — `fpca_variants::fpca_der` coverage (+ cross-covariance `fsvd` cell).
//!
//! `fpca_der` (derivative FPCA) has zero wall-time coverage. `fsvd`/`ssvd` were alloc-profiled in
//! Phase 47 (dhat) but never wall-time benched, so a second cell covers `fsvd` (cross-covariance
//! SVD) on two independent curve sets.
//!
//! Mirrors the criterion structure of `perf_hotpaths.rs`; the deterministic `generate_curves`
//! generator is copied verbatim (bench files are separate compilation units — no shared helper
//! module). A phase-shifted variant builds a second, distinct curve set for `fsvd`.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!   cargo bench -p fdars-core --features linalg,parallel --bench fpca_variants_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::fpca_variants::{fpca_der, fsvd};
use fdars_core::matrix::FdMatrix;

/// Deterministic synthetic curves (column-major, no RNG). Copied verbatim from `perf_hotpaths.rs`.
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * (argvals[j] + phase)).sin();
        }
    }
    (FdMatrix::from_column_major(data, n, m).unwrap(), argvals)
}

/// A phase-/amplitude-shifted second curve set (deterministic, no RNG) for the cross-covariance
/// `fsvd` cell — distinct from `generate_curves` so the cross-covariance is non-trivial.
fn generate_curves_shifted(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let phase = 0.35 + 0.15 * ((i as f64 * 2.3 + 0.9).cos());
        let amp = 1.2 + 0.25 * ((i as f64 * 4.1 + 0.7).cos());
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * (argvals[j] + phase)).cos();
        }
    }
    (FdMatrix::from_column_major(data, n, m).unwrap(), argvals)
}

/// `fpca_der` on 200 curves × 50 argvals, ncomp=5, nderiv=1 (derivative + SVD, medium cost).
fn bench_fpca_der(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpca_der");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (data, argvals) = generate_curves(200, 50);
    group.bench_function("n200_m50_ncomp5_nderiv1", |b| {
        b.iter(|| {
            black_box(
                fpca_der(
                    black_box(&data),
                    black_box(5),
                    black_box(&argvals),
                    black_box(1),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

/// `fsvd` cross-covariance SVD between two independent curve sets (200 × 50 each), ncomp=5.
fn bench_fsvd(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpca_fsvd");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (x, ax) = generate_curves(200, 50);
    let (y, ay) = generate_curves_shifted(200, 50);
    group.bench_function("n200_m50_ncomp5", |b| {
        b.iter(|| {
            black_box(
                fsvd(
                    black_box(&x),
                    black_box(&ax),
                    black_box(&y),
                    black_box(&ay),
                    black_box(5),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_fpca_der, bench_fsvd);
criterion_main!(benches);
