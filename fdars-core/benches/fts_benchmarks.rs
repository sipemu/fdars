//! Criterion benchmarks for the `fts` module (Phase 51 BENCH-01).
//!
//! Covers `fts::ftsm` — the functional time-series model (FPCA + AR forecasting
//! of the score processes). `dpca` is already benched in `perf_hotpaths.rs`; `ftsm`
//! is the distinct forecast entry point of the module.
//!
//! Data is built from a deterministic (non-RNG) sinusoid generator OUTSIDE `b.iter()`,
//! so timings measure the model fit itself, not data construction.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench fts_benchmarks --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::fts::ftsm;
use fdars_core::matrix::FdMatrix;

/// Deterministic synthetic curves (column-major). Copied verbatim from
/// `perf_hotpaths.rs` — bench files are separate compilation units with no shared
/// helper module, so each carries its own copy. NO RNG.
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

fn bench_ftsm(c: &mut Criterion) {
    let mut group = c.benchmark_group("fts_ftsm");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    // n=200 curves, m=50 argvals, ncomp=3 — built outside b.iter().
    let (data, argvals) = generate_curves(200, 50);
    group.bench_function("n200_m50_ncomp3", |b| {
        b.iter(|| black_box(ftsm(black_box(&data), 3, black_box(&argvals)).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_ftsm);
criterion_main!(benches);
