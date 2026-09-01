//! Criterion benchmarks for the `inference` module (Phase 51 BENCH-01).
//!
//! Covers `inference::t_perm_test` — the two-sample functional permutation t-test.
//! Data is built from a deterministic (non-RNG) sinusoid generator OUTSIDE `b.iter()`,
//! so timings measure the permutation test itself, not data construction.
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench inference_benchmarks --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::inference::{t_perm_test, DEFAULT_N_PERM};
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

fn bench_t_perm_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_t_perm_test");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    // Two independent samples sharing the same argvals grid, built outside b.iter().
    let (a, argvals) = generate_curves(30, 50);
    let (b_data, _) = generate_curves(30, 50);
    group.bench_function("na30_nb30_m50_nperm999", |b| {
        b.iter(|| {
            black_box(
                t_perm_test(
                    black_box(&a),
                    black_box(&b_data),
                    black_box(&argvals),
                    black_box(DEFAULT_N_PERM),
                    42,
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_t_perm_test);
criterion_main!(benches);
