//! PERMANENT before/after performance benchmarks for Phase 47 (Hot-Path & Allocation Performance).
//!
//! These cells guard the PERF-01/PERF-02 wins at the PROF-01 measurement cells and become the
//! Phase 51 BENCH-02 regression guards. Before-numbers (PROF-01, 2026-08-30, governor powersave):
//!   - dpca            n200_m50    : 42 MB / 17,739 alloc blocks (allocation-bound)
//!   - face_covariance n200_m30    : 984 ms
//!   - fem_smooth      576 nodes   : 452 ms
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench --bench perf_hotpaths --features linalg,parallel`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use fdars_core::fem_smoothing::fem_smooth;
use fdars_core::fts::dpca;
use fdars_core::irreg_fdata::{face_covariance, IrregFdata};
use fdars_core::matrix::FdMatrix;

/// Deterministic synthetic curves (column-major). Kept in sync with the Phase 46 probe generators.
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

fn bench_dpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_dpca");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (data, argvals) = generate_curves(200, 50);
    group.bench_function("n200_m50", |b| {
        b.iter(|| black_box(dpca(black_box(&data), black_box(&argvals), 3, None, None).unwrap()))
    });
    group.finish();
}

fn bench_face_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_face_covariance");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (n, m) = (200usize, 30usize);
    let grid: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut argvals_list = Vec::with_capacity(n);
    let mut values_list = Vec::with_capacity(n);
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        let vals: Vec<f64> = grid
            .iter()
            .map(|&t| amp * (2.0 * PI * (t + phase)).sin())
            .collect();
        argvals_list.push(grid.clone());
        values_list.push(vals);
    }
    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
    group.bench_function("n200_m30", |b| {
        b.iter(|| black_box(face_covariance(black_box(&ifd), black_box(&grid), 0.3).unwrap()))
    });
    group.finish();
}

/// Regular k×k triangular mesh over the unit square.
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

fn bench_fem_smooth(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_fem_smooth");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (nodes, triangles) = grid_mesh(24); // 576 nodes / 1058 triangles
    let obs_xy: Vec<[f64; 2]> = nodes.clone();
    let y: Vec<f64> = nodes
        .iter()
        .map(|p| (2.0 * PI * p[0]).sin() * (PI * p[1]).cos())
        .collect();
    group.bench_function("nodes576", |b| {
        b.iter(|| {
            black_box(
                fem_smooth(
                    black_box(&nodes),
                    black_box(&triangles),
                    black_box(&obs_xy),
                    black_box(&y),
                    0.1,
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_dpca, bench_face_covariance, bench_fem_smooth);
criterion_main!(benches);
