//! Phase 51 BENCH-01 — `irreg_fdata::mface_covariance` (multivariate FACE) coverage.
//!
//! Univariate `face_covariance` is already benched in the PERMANENT `perf_hotpaths` bench. This
//! bench covers the distinct multivariate entry `mface_covariance` (block FACE covariance across
//! ≥2 variables). Module path is `irreg_fdata`; the bench filename is `face_benchmarks`.
//!
//! Mirrors the criterion structure and the `IrregFdata::from_lists` sinusoid construction of
//! `perf_hotpaths.rs` (bench files are separate compilation units — no shared helper module). Two
//! variables with differing phase/amplitude are built so the cross-covariance is non-trivial.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!   cargo bench -p fdars-core --features linalg,parallel --bench face_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::irreg_fdata::{mface_covariance, IrregFdata};

/// Build one deterministic `IrregFdata` variable of `n` curves on a regular `m`-point grid.
/// `phase_k`/`amp_k`/`freq_k` vary per variable so the two variables differ. Mirrors the
/// `IrregFdata::from_lists` sinusoid construction in `perf_hotpaths.rs`.
fn make_variable(
    n: usize,
    m: usize,
    phase_k: f64,
    amp_k: f64,
    freq_k: f64,
) -> (IrregFdata, Vec<f64>) {
    let grid: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut argvals_list = Vec::with_capacity(n);
    let mut values_list = Vec::with_capacity(n);
    for i in 0..n {
        let phase = phase_k + 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = amp_k + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        let vals: Vec<f64> = grid
            .iter()
            .map(|&t| amp * (freq_k * PI * (t + phase)).sin())
            .collect();
        argvals_list.push(grid.clone());
        values_list.push(vals);
    }
    (IrregFdata::from_lists(&argvals_list, &values_list), grid)
}

/// `mface_covariance` over 2 variables (n=100 curves, m=30 pts each), bandwidth=0.3
/// (multivariate block FACE covariance — medium/slow cost).
fn bench_mface_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("mface_covariance");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (var0, grid0) = make_variable(100, 30, 0.0, 1.0, 2.0);
    let (var1, grid1) = make_variable(100, 30, 0.4, 1.2, 3.0);
    let variables = vec![var0, var1];
    let grids: Vec<Vec<f64>> = vec![grid0, grid1];
    group.bench_function("vars2_n100_m30", |b| {
        b.iter(|| {
            black_box(
                mface_covariance(black_box(&variables), black_box(&grids), black_box(0.3)).unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_mface_covariance);
criterion_main!(benches);
