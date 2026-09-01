//! Criterion benchmarks for the `coclustering` module (Phase 51 BENCH-01).
//!
//! Covers `coclustering::co_cluster_select` — the K×L model-selection sweep over the
//! functional latent-block co-clustering fit. `co_cluster` (single fit) is already
//! benched in `perf_parallelism.rs`; `co_cluster_select` is the distinct sweep entry.
//!
//! The K×L grid is kept TINY (2×2) and per-fit cost bounded (`max_iter=20`) so the
//! cell is not minutes-long.
//!
//! Data is built from a deterministic (non-RNG) generator OUTSIDE `b.iter()`,
//! so timings measure the sweep itself, not data construction.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench coclustering_benchmarks --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::coclustering::{co_cluster_select, CoClusterConfig};
use fdars_core::matrix::FdMatrix;

/// Deterministic two-latent-row-group curves (no RNG). Row group `i % 2` shifts the
/// sinusoid. Copied verbatim from `perf_parallelism.rs` — bench files are separate
/// compilation units with no shared helper module, so each carries its own copy.
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

fn bench_co_cluster_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("coclustering_co_cluster_select");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));

    // n=120 curves, m=40 argvals.
    let (data, argvals) = co_cluster_curves(120, 40);

    // Tiny 2×2 model-selection grid.
    let k_range = [2usize, 3];
    let l_range = [2usize, 3];

    // Bound the grid×restart cost (n_init=3 default, max_iter capped to 20).
    // CoClusterConfig is #[non_exhaustive] → struct-update syntax is unavailable
    // outside the defining crate, so fields are reassigned after default().
    #[allow(clippy::field_reassign_with_default)]
    let config = {
        let mut cfg = CoClusterConfig::default();
        cfg.n_init = 3;
        cfg.max_iter = 20;
        cfg
    };

    group.bench_function("n120_m40_grid2x2", |b| {
        b.iter(|| {
            black_box(
                co_cluster_select(
                    black_box(&data),
                    black_box(&argvals),
                    &k_range,
                    &l_range,
                    black_box(&config),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_co_cluster_select);
criterion_main!(benches);
