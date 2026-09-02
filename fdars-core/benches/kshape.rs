//! Benchmarks for the SBD / k-Shape clustering pipeline (v0.34.0).
//!
//! Measures the two hot public entry points on a SMALL synthetic curve set:
//! - `sbd_distance_matrix` — the pairwise Shape-Based Distance matrix (the
//!   backend of `sbd_kmedoids`).
//! - `kshape_fd` — full k-Shape clustering (SBD assignment + shape-extraction
//!   centroids) over `n_init` restarts.
//!
//! The dataset is intentionally tiny (n=30, m=64, k=3, n_init=2) so the
//! benchmark runs quickly.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::{kshape_fd, sbd_distance_matrix, KShapeConfig};
use std::f64::consts::PI;

/// Three shifted-sine shape groups with light deterministic noise.
fn synthetic_dataset(n: usize, m: usize) -> FdMatrix {
    let mut flat = vec![0.0f64; n * m];
    for i in 0..n {
        let group = i % 3;
        let freq = 2.0 * (group as f64 + 1.0);
        // Deterministic per-series circular shift + tiny noise (no RNG dep).
        let shift = (i * 7) % m;
        for j in 0..m {
            let jj = (j + shift) % m;
            let base = (freq * PI * jj as f64 / m as f64).sin();
            let hash = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) % 211;
            let noise = 0.02 * (hash as f64 / 211.0 - 0.5);
            flat[i + j * n] = base + noise;
        }
    }
    FdMatrix::from_column_major(flat, n, m).unwrap()
}

fn bench_sbd_pipeline(c: &mut Criterion) {
    let (n, m) = (30usize, 64usize);
    let data = synthetic_dataset(n, m);

    let mut group = c.benchmark_group("kshape");

    group.bench_function("sbd_distance_matrix_n30_m64", |b| {
        b.iter(|| sbd_distance_matrix(black_box(&data)));
    });

    // KShapeConfig is #[non_exhaustive]; build via `new` then set public fields.
    let mut cfg = KShapeConfig::new(3);
    cfg.n_init = 2;
    cfg.seed = 0;
    group.bench_function("kshape_fd_n30_m64_k3", |b| {
        b.iter(|| kshape_fd(black_box(&data), black_box(&cfg)));
    });

    group.finish();
}

criterion_group!(benches, bench_sbd_pipeline);
criterion_main!(benches);
