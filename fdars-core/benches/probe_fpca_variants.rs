//! THROWAWAY probe bench — Phase 46 (Whole-Crate Profiling & Measurement) tracer.
//!
//! Profiles the `fpca_variants` subsystem hot path `fsvd` (functional SVD /
//! cross-covariance PCA). The gram-matrix eigendecomposition allocation hotspot
//! is at `src/fpca_variants.rs:488` (`DMatrix::from_column_slice(g_dim, g_dim, &gram)`).
//!
//! Copies the `benches/audit_hotpaths.rs` structure verbatim (generate_curves,
//! inputs outside b.iter(), black_box discipline, linalg-gated + empty-stub fallback).
//!
//! REMOVED in Plan 02 — this is not a permanent bench (permanent coverage is Phase 51 / BENCH-01).
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench probe_fpca_variants --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "linalg")]
use fdars_core::fpca_variants::fsvd;
#[cfg(feature = "linalg")]
use fdars_core::matrix::FdMatrix;
#[cfg(feature = "linalg")]
use std::f64::consts::PI;

/// Deterministic synthetic curves, column-major. Kept in sync with
/// `benches/audit_hotpaths.rs:generate_curves` / `tests/alloc_audit_fpca.rs:generate_test_curves`.
#[cfg(feature = "linalg")]
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
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

#[cfg(feature = "linalg")]
fn bench_fsvd(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpca_variants_fsvd");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));
    group.warm_up_time(std::time::Duration::from_secs(3));

    // N×M grid: n50_m50, n200_m50, n50_m200, n1000_m50 (small cell timed fast → all four run).
    let cells = [(50usize, 50usize), (200, 50), (50, 200), (1000, 50)];
    for (n, m) in cells {
        // Second curve set y — a phase-shifted variant so cross-covariance is non-trivial.
        let (x, argvals_x) = generate_curves(n, m);
        let (y, argvals_y) = generate_curves(n, m);
        let ncomp = 5.min(n.min(m) - 1);
        group.bench_function(format!("n{n}_m{m}"), |b| {
            b.iter(|| {
                let r = fsvd(
                    black_box(&x),
                    black_box(&argvals_x),
                    black_box(&y),
                    black_box(&argvals_y),
                    black_box(ncomp),
                );
                black_box(r.unwrap())
            })
        });
    }
    group.finish();
}

// Empty stub so criterion_group! always resolves without the linalg feature.
#[cfg(not(feature = "linalg"))]
fn bench_fsvd(_c: &mut Criterion) {}

criterion_group!(benches, bench_fsvd);
criterion_main!(benches);
