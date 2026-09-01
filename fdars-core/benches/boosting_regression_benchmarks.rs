//! Criterion benchmarks for the `boosting_regression` module (Phase 51 BENCH-01).
//!
//! Covers `boosting_regression::boost_fosr` — component-wise gradient boosting for
//! function-on-scalar regression with penalized B-spline base-learners. No prior
//! coverage existed for this module.
//!
//! Two cells are benched: the representative `BoostingConfig::default()` (mstop=100)
//! and a lighter `mstop=50` variant (RESEARCH A2 — mstop is an executor-tunable knob;
//! the default cell is kept for representativeness).
//!
//! Data is built from a deterministic (non-RNG) sinusoid generator OUTSIDE `b.iter()`,
//! so timings measure the boosting fit itself, not data construction.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench boosting_regression_benchmarks --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::boosting_regression::{boost_fosr, BoostingConfig};
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

/// Deterministic n×2 Euclidean scalar-predictor matrix (2 sinusoid-of-index columns).
fn deterministic_predictors(n: usize) -> FdMatrix {
    let mut vals = vec![0.0; n * 2];
    for i in 0..n {
        // column 0 at index i, column 1 at index i + n (column-major).
        vals[i] = (i as f64 * 0.7).sin();
        vals[i + n] = (i as f64 * 1.3 + 0.5).cos();
    }
    FdMatrix::from_column_major(vals, n, 2).unwrap()
}

fn bench_boost_fosr(c: &mut Criterion) {
    let mut group = c.benchmark_group("boosting_boost_fosr");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));

    // n=100 functional-response curves, m=50 argvals, p=2 scalar predictors.
    let (data, argvals) = generate_curves(100, 50);
    let predictors = deterministic_predictors(100);

    // Representative default (mstop=100).
    let config = BoostingConfig::default();
    group.bench_function("n100_m50_p2_mstop100", |b| {
        b.iter(|| {
            black_box(
                boost_fosr(
                    black_box(&data),
                    black_box(&predictors),
                    black_box(&argvals),
                    black_box(&config),
                )
                .unwrap(),
            )
        })
    });

    // Lighter mstop=50 variant (executor-tunable knob).
    let config50 = BoostingConfig {
        mstop: 50,
        ..BoostingConfig::default()
    };
    group.bench_function("n100_m50_p2_mstop50", |b| {
        b.iter(|| {
            black_box(
                boost_fosr(
                    black_box(&data),
                    black_box(&predictors),
                    black_box(&argvals),
                    black_box(&config50),
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_boost_fosr);
criterion_main!(benches);
