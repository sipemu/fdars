//! PERMANENT thread-scaling benchmarks for Phase 48 (Parallelism-Gap Closure, PERF-03).
//!
//! Each cell measures a newly-parallelized hot path at its PROF-01 measurement size. Run the
//! same cell under a 1-thread and an N-thread rayon pool to expose the parallel speedup:
//!
//!   RAYON_NUM_THREADS=1  TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!     cargo bench -p fdars-core --features linalg,parallel --bench perf_parallelism -- frechet_anova
//!   RAYON_NUM_THREADS=20 TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!     cargo bench -p fdars-core --features linalg,parallel --bench perf_parallelism -- frechet_anova
//!
//! The env-var sweep is the primary thread-scaling signal. If it proves noisy, Task 4 may add an
//! in-bench `rayon::ThreadPoolBuilder` scoped-pool variant to pin thread counts deterministically.
//!
//! These cells are PERMANENT — they become the Phase 51 BENCH-02 regression guards for the
//! parallelism wins landed this phase. Governor state at capture is recorded in
//! `.planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::coclustering::{co_cluster, CoClusterConfig};
use fdars_core::frechet::frechet_anova;
use fdars_core::matrix::FdMatrix;

/// Deterministic two-group **density** data (rows strictly positive — WassersteinDensitySpace).
/// Mirrors the `two_group_densities` helper in `tests/equivalence_phase48.rs` (no RNG in the
/// generator). Group 0: rows 0..n_per; Group 1: rows n_per..2*n_per (contiguous 0..2 labels).
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

/// frechet_anova permutation loop (PROF-01 #4 hotspot, 133 ms). n=24 curves, m=81 argvals,
/// n_perm=999 (ABOVE FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD → parallel branch).
fn bench_frechet_anova(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_parallelism_frechet_anova");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));
    let (data, argvals, labels) = two_group_densities(12, 81);
    group.bench_function("n24_m81_nperm999", |b| {
        b.iter(|| {
            black_box(
                frechet_anova(
                    black_box(&data),
                    black_box(&argvals),
                    black_box(&labels),
                    999,
                    42,
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

/// Deterministic two-latent-row-group curves for co_cluster (no RNG). Row group `i % 2` shifts
/// the sinusoid; mirrors the `co_cluster_data` helper in `tests/equivalence_phase48.rs`.
fn co_cluster_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let grp = i % 2;
        let phase = if grp == 0 { 0.0 } else { 0.35 } + 0.05 * ((i as f64 * 1.7).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 0.9).sin()) + if grp == 0 { 0.0 } else { 0.5 };
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * (argvals[j] + phase)).sin();
        }
    }
    (FdMatrix::from_column_major(data, n, m).unwrap(), argvals)
}

/// co_cluster multi-restart CEM (src/coclustering.rs — PROF-01 candidate). n=200, m=50, n_init=8
/// (ABOVE CO_CLUSTER_INIT_PARALLEL_THRESHOLD → parallel branch).
fn bench_co_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_parallelism_co_cluster");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (data, argvals) = co_cluster_curves(200, 50);
    let mut cfg = CoClusterConfig::default();
    cfg.n_row_blocks = 2;
    cfg.n_col_blocks = 2;
    cfg.ncomp = 3;
    cfg.max_iter = 50;
    cfg.n_init = 8;
    cfg.seed = 42;
    group.bench_function("n200_m50_ninit8", |b| {
        b.iter(|| {
            black_box(co_cluster(black_box(&data), black_box(&argvals), black_box(&cfg)).unwrap())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_frechet_anova, bench_co_cluster);
criterion_main!(benches);
