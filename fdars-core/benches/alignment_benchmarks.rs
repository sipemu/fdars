//! Benchmarks for elastic alignment methods
//!
//! Compares performance of:
//! - Pairwise elastic alignment with varying curve length m
//! - Self-distance matrix with varying n and m
//! - Karcher mean computation
//! - DP engine options (`WarpOptimConfig` neighbourhood / `dp_scl` / edge cost /
//!   pruning, ALN-DPACC #49; pruning on by default, GH #66)
//! - Closed-curve rotation search (`elastic_align_pair_closed`, ALN-HYGIENE #56)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::alignment::{
    elastic_align_pair, elastic_align_pair_banded, elastic_align_pair_closed,
    elastic_align_pair_with_config, elastic_self_distance_matrix,
    elastic_self_distance_matrix_banded, karcher_mean, karcher_mean_with_band,
    karcher_mean_with_config, DpEdgeCost, WarpOptimConfig,
};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

/// Generate synthetic functional data (n curves, m time points).
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        // Deterministic phase/amplitude variation per curve
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

fn bench_elastic_align_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_align_pair");

    for &m in &[50, 100, 200] {
        let (mat, argvals) = generate_curves(2, m);
        let f1 = mat.row(0);
        let f2 = mat.row(1);

        group.bench_with_input(BenchmarkId::new("m", m), &m, |b, _| {
            b.iter(|| {
                elastic_align_pair(
                    black_box(&f1),
                    black_box(&f2),
                    black_box(&argvals),
                    black_box(0.0),
                )
            });
        });

        // Sakoe–Chiba band at 10% of the domain.
        group.bench_with_input(BenchmarkId::new("m_band0.1", m), &m, |b, _| {
            b.iter(|| {
                elastic_align_pair_banded(
                    black_box(&f1),
                    black_box(&f2),
                    black_box(&argvals),
                    black_box(0.0),
                    black_box(0.1),
                )
            });
        });
    }

    // Also benchmark with lambda > 0
    let (mat, argvals) = generate_curves(2, 100);
    let f1 = mat.row(0);
    let f2 = mat.row(1);
    group.bench_function("m100_lambda0.1", |b| {
        b.iter(|| {
            elastic_align_pair(
                black_box(&f1),
                black_box(&f2),
                black_box(&argvals),
                black_box(0.1),
            )
        });
    });

    group.finish();
}

fn bench_self_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_self_distance_matrix");
    let m = 50;

    for &n in &[10, 30, 50] {
        let (mat, argvals) = generate_curves(n, m);

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                elastic_self_distance_matrix(black_box(&mat), black_box(&argvals), black_box(0.0))
            });
        });

        group.bench_with_input(BenchmarkId::new("n_band0.1", n), &n, |b, _| {
            b.iter(|| {
                elastic_self_distance_matrix_banded(
                    black_box(&mat),
                    black_box(&argvals),
                    black_box(0.0),
                    black_box(0.1),
                )
            });
        });
    }

    group.finish();
}

// Sealed config: external construction is `default()` + field assignment.
#[allow(clippy::field_reassign_with_default)]
fn bench_karcher_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("karcher_mean");

    let n = 20;
    let m = 50;
    let (mat, argvals) = generate_curves(n, m);

    group.bench_function("n20_m50", |b| {
        b.iter(|| {
            karcher_mean(
                black_box(&mat),
                black_box(&argvals),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
            )
        });
    });

    // Smaller problem for faster iteration
    let n = 10;
    let (mat_small, argvals_small) = generate_curves(n, m);
    group.bench_function("n10_m50", |b| {
        b.iter(|| {
            karcher_mean(
                black_box(&mat_small),
                black_box(&argvals_small),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
            )
        });
    });

    // The unpruned reference fill (`dp_prune = false`, GH #66): bit-identical
    // to `n20_m50`, which prunes by default.
    let mut unpruned = WarpOptimConfig::default();
    unpruned.dp_prune = false;
    group.bench_function("n20_m50_unpruned", |b| {
        b.iter(|| {
            karcher_mean_with_config(
                black_box(&mat),
                black_box(&argvals),
                black_box(15),
                black_box(1e-4),
                &unpruned,
            )
            .unwrap()
        });
    });

    group.finish();
}

/// Benchmark the public `karcher_mean_with_band` API at small-to-medium sizes,
/// comparing `None` (exact unbanded) vs `Some(0.1)` (banded) path.
///
/// For large-grid feasibility cells (N=500, M=200) see the existing
/// `bench_p3_karcher_banded` group in `benches/audit_hotpaths.rs`, which already
/// benchmarks `karcher_mean_banded` at those infeasible sizes.
fn bench_karcher_mean_with_band(c: &mut Criterion) {
    let mut group = c.benchmark_group("karcher_mean_with_band");

    let m = 50;
    let n = 20;
    let (mat, argvals) = generate_curves(n, m);

    // Exact unbanded path via the _with_band wrapper (None = same as karcher_mean).
    group.bench_function("n20_m50_none", |b| {
        b.iter(|| {
            karcher_mean_with_band(
                black_box(&mat),
                black_box(&argvals),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
                black_box(None),
            )
        });
    });

    // Banded path via the _with_band wrapper (band_frac = 0.1).
    group.bench_function("n20_m50_band0.1", |b| {
        b.iter(|| {
            karcher_mean_with_band(
                black_box(&mat),
                black_box(&argvals),
                black_box(15),
                black_box(1e-4),
                black_box(0.0),
                black_box(Some(0.1)),
            )
        });
    });

    group.finish();
}

/// DP engine options on one pair (ALN-DPACC #49). `default` is the classic
/// 35-move piecewise-constant engine with exact pruning (on by default since
/// GH #66); the `_unpruned` variants set `dp_prune = false` (the reference
/// fill, bit-identical results). The others trade time for accuracy
/// (`nbhd10_linear` = fdasrsf `DP2`, `scl3` = 3× refined cost).
// Sealed config: external construction is `default()` + field assignment.
#[allow(clippy::field_reassign_with_default)]
fn bench_dp_options(c: &mut Criterion) {
    let mut group = c.benchmark_group("dp_options");
    let variants: [(&str, usize, usize, DpEdgeCost, bool); 8] = [
        ("default", 7, 1, DpEdgeCost::PiecewiseConstant, true),
        (
            "default_unpruned",
            7,
            1,
            DpEdgeCost::PiecewiseConstant,
            false,
        ),
        ("linear7", 7, 1, DpEdgeCost::PiecewiseLinear, true),
        ("linear7_unpruned", 7, 1, DpEdgeCost::PiecewiseLinear, false),
        ("nbhd10_linear", 10, 1, DpEdgeCost::PiecewiseLinear, true),
        (
            "nbhd10_linear_unpruned",
            10,
            1,
            DpEdgeCost::PiecewiseLinear,
            false,
        ),
        ("scl3", 7, 3, DpEdgeCost::PiecewiseConstant, true),
        ("scl3_unpruned", 7, 3, DpEdgeCost::PiecewiseConstant, false),
    ];
    for &m in &[100, 200] {
        let (mat, argvals) = generate_curves(2, m);
        let f1 = mat.row(0);
        let f2 = mat.row(1);
        for &(name, nbhd, scl, cost, prune) in &variants {
            let mut cfg = WarpOptimConfig::default();
            cfg.nbhd_dim = nbhd;
            cfg.dp_scl = scl;
            cfg.dp_cost = cost;
            cfg.dp_prune = prune;
            group.bench_with_input(BenchmarkId::new(name, m), &m, |b, _| {
                b.iter(|| {
                    elastic_align_pair_with_config(
                        black_box(&f1),
                        black_box(&f2),
                        black_box(&argvals),
                        &cfg,
                    )
                    .unwrap()
                });
            });
        }
    }
    group.finish();
}

fn bench_closed_align(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_align_pair_closed");
    for &m in &[50, 100] {
        let (mat, argvals) = generate_curves(2, m);
        let f1 = mat.row(0);
        let f2 = mat.row(1);
        group.bench_with_input(BenchmarkId::new("m", m), &m, |b, _| {
            b.iter(|| {
                elastic_align_pair_closed(
                    black_box(&f1),
                    black_box(&f2),
                    black_box(&argvals),
                    black_box(0.0),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_elastic_align_pair,
    bench_self_distance_matrix,
    bench_karcher_mean,
    bench_karcher_mean_with_band,
    bench_dp_options,
    bench_closed_align,
);
criterion_main!(benches);
