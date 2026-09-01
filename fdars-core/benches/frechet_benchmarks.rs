//! Criterion benchmarks for the `frechet` module (Phase 51 BENCH-01).
//!
//! Covers `frechet::frechet_global_reg` — global Fréchet regression with Euclidean
//! predictors (Petersen & Müller 2019). `frechet_anova` is already benched in
//! `perf_parallelism.rs`; `frechet_global_reg` is the distinct regression entry point.
//!
//! The concrete `frechet_global_reg` is used (NOT the generic `_space<S: MetricSpace>`
//! form) to avoid a Sync/object-construction complication in the bench harness.
//!
//! Data is built from a deterministic (non-RNG) density generator OUTSIDE `b.iter()`,
//! so timings measure the regression itself, not data construction.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench frechet_benchmarks --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use fdars_core::frechet::frechet_global_reg;
use fdars_core::matrix::FdMatrix;

/// Deterministic two-group **density** data (rows strictly positive — the response
/// space). Copied verbatim from `perf_parallelism.rs` — bench files are separate
/// compilation units with no shared helper module, so each carries its own copy.
/// The 81-length `argvals` span −3..3 strictly increasing. NO RNG.
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

fn bench_frechet_global_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("frechet_global_reg");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));

    // Responses: 24 density rows × 81 argvals (argvals monotone in −3..3).
    let (responses, argvals, _labels) = two_group_densities(12, 81);
    let n = responses.nrows();

    // Predictors: 1-column Euclidean scalar covariate, one deterministic value per row.
    let pred_vals: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let predictors = FdMatrix::from_column_major(pred_vals, n, 1).unwrap();

    // xout: 5 deterministic query points evenly spaced in the predictor range [0, 1).
    let xout_vals: Vec<f64> = (0..5).map(|k| k as f64 / 5.0).collect();
    let xout = FdMatrix::from_column_major(xout_vals, 5, 1).unwrap();

    group.bench_function("n24_m81_xout5", |b| {
        b.iter(|| {
            black_box(
                frechet_global_reg(
                    black_box(&predictors),
                    black_box(&responses),
                    black_box(&argvals),
                    black_box(&xout),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_frechet_global_reg);
criterion_main!(benches);
