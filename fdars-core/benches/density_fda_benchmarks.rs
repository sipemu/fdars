//! Phase 51 BENCH-01 — `density_fda::lqd_fpca` coverage (+ cheap `wasserstein_barycenter` cell).
//!
//! No prior coverage for the density-FDA module. `lqd_fpca` (log-quantile-density transform + FPCA
//! SVD) is the representative entry; a cheap second cell benches `wasserstein_barycenter`
//! (quantile-average inversion) on the same inputs.
//!
//! Mirrors the criterion structure of `perf_hotpaths.rs`; the deterministic `two_group_densities`
//! generator (strictly-positive density rows) is copied verbatim from `perf_parallelism.rs` (bench
//! files are separate compilation units — no shared helper module).
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!   cargo bench -p fdars-core --features linalg,parallel --bench density_fda_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use fdars_core::density_fda::{lqd_fpca, wasserstein_barycenter};
use fdars_core::matrix::FdMatrix;

/// Deterministic two-group **density** data (rows strictly positive — no RNG). Copied verbatim from
/// `perf_parallelism.rs`. argvals in −3..3; `n = 2 * n_per` strictly-positive Gaussian-shaped rows.
fn two_group_densities(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
    let argvals: Vec<f64> = (0..m)
        .map(|j| -3.0 + 6.0 * j as f64 / (m - 1) as f64)
        .collect();
    let n = 2 * n_per;
    let mut data = vec![0.0; n * m];
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let g = if i < n_per { 0 } else { 1 };
        labels[i] = g;
        let mu = if g == 0 { -0.5 } else { 0.5 } + 0.2 * ((i as f64 * 1.7).sin());
        let sigma = 0.7 + 0.2 * ((i as f64 * 1.3).sin().abs());
        for j in 0..m {
            let z = (argvals[j] - mu) / sigma;
            data[i + j * n] = (-0.5 * z * z).exp() / sigma + 1e-6;
        }
    }
    (
        FdMatrix::from_column_major(data, n, m).unwrap(),
        argvals,
        labels,
    )
}

/// `lqd_fpca` on 100 density rows × 81 argvals, ncomp=3 (LQD transform + FPCA SVD, medium cost).
fn bench_lqd_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_lqd_fpca");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (density_matrix, argvals, _) = two_group_densities(50, 81); // n=100, m=81
    group.bench_function("n100_m81_ncomp3", |b| {
        b.iter(|| {
            black_box(
                lqd_fpca(
                    black_box(&density_matrix),
                    black_box(&argvals),
                    black_box(3),
                    black_box(None),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

/// Cheap second cell: `wasserstein_barycenter` (quantile-average inversion) on the same inputs.
fn bench_wasserstein_barycenter(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_wasserstein_barycenter");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    let (density_matrix, argvals, _) = two_group_densities(50, 81); // n=100, m=81
    group.bench_function("n100_m81", |b| {
        b.iter(|| {
            black_box(
                wasserstein_barycenter(
                    black_box(&density_matrix),
                    black_box(&argvals),
                    black_box(None),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_lqd_fpca, bench_wasserstein_barycenter);
criterion_main!(benches);
