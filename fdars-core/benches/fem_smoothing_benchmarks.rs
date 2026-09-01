//! Phase 51 BENCH-01 — `fem_smoothing::fem_smooth_gcv` coverage.
//!
//! `fem_smooth` (single-λ) at 576 nodes is already benched in the PERMANENT `perf_hotpaths`
//! bench. This bench covers the distinct GCV-λ-grid entry `fem_smooth_gcv`, which multiplies the
//! O(N³) FEM solve by `n_grid`. To keep wall time bounded we use a SMALLER mesh (k=16 = 256 nodes)
//! than the 576 benched by `fem_smooth`.
//!
//! Mirrors the criterion structure of `perf_hotpaths.rs`; the deterministic `grid_mesh` generator
//! is copied verbatim (bench files are separate compilation units — no shared helper module).
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!   cargo bench -p fdars-core --features linalg,parallel --bench fem_smoothing_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::fem_smoothing::fem_smooth_gcv;

/// Regular k×k triangular mesh over the unit square (copied verbatim from `perf_hotpaths.rs`).
fn grid_mesh(k: usize) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let idx = |r: usize, cc: usize| r * k + cc;
    let mut nodes = Vec::with_capacity(k * k);
    for r in 0..k {
        for cc in 0..k {
            nodes.push([cc as f64 / (k - 1) as f64, r as f64 / (k - 1) as f64]);
        }
    }
    let mut tris = Vec::with_capacity(2 * (k - 1) * (k - 1));
    for r in 0..k - 1 {
        for cc in 0..k - 1 {
            tris.push([idx(r, cc), idx(r, cc + 1), idx(r + 1, cc)]);
            tris.push([idx(r, cc + 1), idx(r + 1, cc + 1), idx(r + 1, cc)]);
        }
    }
    (nodes, tris)
}

/// GCV-selected FEM smoothing over a 5-point log-λ grid on a 256-node mesh (k=16).
/// Smaller than the 576-node `fem_smooth` cell because GCV multiplies the O(N³) solve by n_grid.
fn bench_fem_smooth_gcv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fem_smooth_gcv");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));
    let (nodes, triangles) = grid_mesh(16); // 256 nodes
    let obs_xy: Vec<[f64; 2]> = nodes.clone();
    let y: Vec<f64> = nodes
        .iter()
        .map(|p| (2.0 * PI * p[0]).sin() * (PI * p[1]).cos())
        .collect();
    group.bench_function("nodes256_ngrid5", |b| {
        b.iter(|| {
            black_box(
                fem_smooth_gcv(
                    black_box(&nodes),
                    black_box(&triangles),
                    black_box(&obs_xy),
                    black_box(&y),
                    black_box((-4.0, 0.0)),
                    black_box(5),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_fem_smooth_gcv);
criterion_main!(benches);
