//! Audit hot-path benchmarks for fdars-core.
//!
//! One representative sentinel per hot-path module at Phase 1 baseline cell.
//! Sizes, caps, and sample_size rationale: see .planning/phases/01-.../01-RESEARCH.md §8.
//! Raw output saved to .planning/research/bench/ per D-06.
//!
//! **4-combo sentinel choice (D-04, A5 resolution):**
//! `fdata_to_pc_1d` was the original candidate for the 4-combo feature-matrix
//! sentinel, but `center_columns` (regression.rs:167-181) uses plain sequential
//! `for` loops and nalgebra SVD is always sequential — so FPCA produces near-
//! identical timings for the `parallel` and non-`parallel` combos and is a poor
//! discriminator.  `karcher_mean` was substituted: its inner N-loop uses
//! `iter_maybe_parallel!` (src/alignment/karcher.rs:185) and genuinely differs
//! across the 4 combos.
//!
//! **Workload size caps (D-07):**
//! - Elastic: N=100, M=50 baseline. O(n²·m²) makes N=1000×M=500 ≈ 60s/iter.
//! - CV: N=100, M=50 baseline. Each fold runs FPCA O(m³) + fit + predict × K=5.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::alignment::{
    elastic_cross_distance_matrix, elastic_cross_distance_matrix_banded,
    elastic_self_distance_matrix, elastic_self_distance_matrix_banded, karcher_mean,
    karcher_mean_banded,
};
use fdars_core::classification::fclassif_cv;
use fdars_core::depth::fraiman_muniz_1d;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use fdars_core::smoothing::nadaraya_watson;
use fdars_core::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz};
use std::f64::consts::PI;

/// Generate synthetic functional data (n curves, m time points).
///
/// Uses deterministic phase/amplitude variation so no RNG dependency is needed.
/// Column-major layout: element (i, j) at index `i + j * n`.
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

/// Build alternating binary class labels for n curves (0 / 1).
fn make_class_labels(n: usize) -> Vec<usize> {
    (0..n).map(|i| i % 2).collect()
}

/// Generate noisy sine-curve data for smoothing sentinel.
///
/// Returns (x, y, x_new) where x has n training points and x_new has m prediction points.
fn generate_smoothing_data(n: usize, m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let noise = ((i as f64 * 17.3 + 0.5).sin()) * 0.3;
            (2.0 * PI * xi).sin() + 0.5 * (4.0 * PI * xi).cos() + noise
        })
        .collect();
    let x_new: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    (x, y, x_new)
}

/// Sentinel D-03: FPCA / SVD module baseline.
///
/// Benchmarks `fdata_to_pc_1d` at N=500, M=200 — the representative audit cell
/// for the regression/FPCA module.  This function is not parallel-gated
/// (center_columns is sequential, nalgebra SVD is always sequential), so it
/// serves as the per-module baseline but NOT the 4-combo discriminator.
fn bench_fpca_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    // Tune for audit cell: SVD O(m^3) at m=200 costs ~100 ms/iter
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)))
    });

    group.finish();
}

/// Sentinel D-04: 4-combo feature-matrix discriminator.
///
/// Benchmarks `karcher_mean` at N=100, M=50.  `karcher_mean` uses
/// `iter_maybe_parallel!` in its inner N-loop (karcher.rs:185), so it
/// genuinely exercises different code across the 4 feature combos:
///   - `""` (no-default-features) → sequential iterator
///   - `parallel`                 → rayon parallel iterator
///   - `linalg`                   → sequential (linalg adds faer, not rayon)
///   - `linalg,parallel`          → rayon parallel iterator
fn bench_matrix_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_karcher");
    // N=100, M=50 keeps each of the 4 combo runs fast (<20 s total)
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data),
                black_box(&argvals),
                black_box(10usize),
                black_box(1e-3),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

/// Sentinel: Elastic alignment module baseline.
///
/// Benchmarks `elastic_self_distance_matrix` at N=100, M=50 — CAPPED per D-07.
/// O(n²·m²) DP: N=1000×M=500 ≈ 60s/iter (CONCERNS.md), so N=100, M=50 is used
/// (O(100²×50²) = 25M ops, < 10s/iter).
fn bench_elastic_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_elastic");
    // Capped cell: O(n²·m²) at N=100,M=50 keeps each iter tractable
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter()
    let (data, argvals) = generate_curves(100, 50);
    group.bench_function("n100_m50_capped", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(
                black_box(&data),
                black_box(&argvals),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

/// Sentinel: Depth & distance module baseline.
///
/// Benchmarks `fraiman_muniz_1d` at N=500, M=200 — the representative audit cell.
/// O(n²·m) — tractable at N=500 (existing bench goes to N=2300 at M=200).
fn bench_depth_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_depth");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(15));
    group.warm_up_time(std::time::Duration::from_secs(3));

    // Build input OUTSIDE b.iter()
    // fraiman_muniz_1d takes (&FdMatrix, &FdMatrix, bool) and returns Vec<f64>
    let (data, _argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| black_box(fraiman_muniz_1d(black_box(&data), black_box(&data), true)))
    });

    group.finish();
}

/// Sentinel: CV loops module baseline.
///
/// Benchmarks `fclassif_cv` at N=100, M=50 — CAPPED per D-07.
/// Each fold runs FPCA O(m³) + classifier fit + predict × K=5 folds.
/// N=100, M=50 chosen so 2-run variance check completes within budget.
fn bench_cv_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_cv");
    // Capped cell: N=100,M=50 — each fold runs FPCA+LDA (fast classifier)
    group.sample_size(15);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build inputs OUTSIDE b.iter()
    let (data, argvals) = generate_curves(100, 50);
    let y = make_class_labels(100);
    group.bench_function("n100_m50_capped", |b| {
        b.iter(|| {
            fclassif_cv(
                black_box(&data),
                black_box(argvals.as_slice()),
                black_box(y.as_slice()),
                black_box(None),
                black_box("lda"),
                black_box(5usize),
                black_box(5usize),
                black_box(42u64),
            )
        })
    });

    group.finish();
}

/// Sentinel: Streaming depth module baseline.
///
/// Benchmarks `StreamingFraimanMuniz::depth_batch` at N=500, M=200.
/// Construct-then-query measured as a single unit (matching real incremental usage).
/// O(n·m) build + O(n·m) query — very fast at all sizes.
fn bench_streaming_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_streaming_depth");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(15));
    group.warm_up_time(std::time::Duration::from_secs(3));

    // Build input OUTSIDE b.iter() — only the construct+query goes inside
    let (data, _argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            let state = SortedReferenceState::from_reference(black_box(&data));
            let fm = StreamingFraimanMuniz::new(state, true);
            fm.depth_batch(black_box(&data))
        })
    });

    group.finish();
}

/// Sentinel: Smoothing module baseline.
///
/// Benchmarks `nadaraya_watson` at N=500 training observations, M=200 prediction points.
/// O(n·m) kernel evaluation — the base case for the smoothing module.
fn bench_smooth_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_smooth");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(10));
    group.warm_up_time(std::time::Duration::from_secs(3));

    // Build input OUTSIDE b.iter()
    let (x, y, x_new) = generate_smoothing_data(500, 200);
    let bandwidth = 0.1;
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            nadaraya_watson(
                black_box(&x),
                black_box(&y),
                black_box(&x_new),
                black_box(bandwidth),
                black_box("gaussian"),
            )
            .unwrap()
        })
    });

    group.finish();
}

/// Phase-3 karcher_mean sweep — unbanded full DP at D-06 params, full grid.
///
/// Benchmarks `karcher_mean` at all four Phase-3 grid cells:
/// N∈{100,500} × M∈{50,200}, with the D-06-locked parameters:
/// `max_iter = 20`, `tol = 1e-4`, `lambda = 0.0`.
///
/// **Key fact (D-05 / Anti-Pattern 2):** `karcher_mean()` calls
/// `karcher_mean_impl(.., 0.0)` at `karcher.rs:300`, so `band_frac = 0.0` →
/// full unbanded DP by default (O(max_iter · N · m²)).  Banding is opt-in
/// via `karcher_mean_banded()` — see `bench_p3_karcher_banded` for the twin.
///
/// Timing: N=100 cells use sentinel defaults (20s); N=500×M=200 (borderline
/// workload-matrix cell) uses `measurement_time(60s)` + `sample_size(10)`.
/// All four cells share the same criterion group for artifact clarity.
fn bench_p3_karcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_karcher");

    // --- n100_m50: small cell (~300-640ms/iter, high load variance) ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data100_50),
                black_box(&argvals50),
                black_box(20usize), // D-06 max_iter
                black_box(1e-4),    // D-06 tol
                black_box(0.0),     // D-06 lambda = 0.0
            ))
        })
    });

    // --- n100_m200: ~3-7s/iter depending on system load ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m50: ~1.6-4s/iter ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m200: borderline cell (workload-matrix 60s) ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

/// Phase-3 karcher_mean_banded sweep — band_frac=0.1, full grid.
///
/// Benchmarks `karcher_mean_banded` at all four Phase-3 grid cells:
/// N∈{100,500} × M∈{50,200}, with D-06 params + `band_frac = 0.1` (D-03).
///
/// At M=200: band_radius(0.1, 200) = ceil(0.1×200) = 20 points → ≈10× theoretical
/// DP reduction (m/band = 200/20), ~7× expected after overhead (D-03).
/// At M=50: band_radius(0.1, 50) = 5 points → 10× theoretical still, but smaller
/// absolute DP so overhead dominates more.
///
/// Timing: small cells use sentinel defaults; N=500×M=200 uses
/// `measurement_time(60s)` + `sample_size(10)`.
fn bench_p3_karcher_banded(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_karcher_banded");

    // --- n100_m50: banding reduces cost ~4x vs unbanded ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean_banded(
                black_box(&data100_50),
                black_box(&argvals50),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
                black_box(0.1), // band_frac = 0.1 (D-03)
            ))
        })
    });

    // --- n100_m200: ~1-1.3s/iter with banding ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(karcher_mean_banded(
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m50: ~1.2-1.5s/iter with banding ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean_banded(
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m200: borderline cell ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(karcher_mean_banded(
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    group.finish();
}

/// Phase-3 elastic_self_distance_matrix sweep — unbanded, full grid.
///
/// Benchmarks `elastic_self_distance_matrix` at all four Phase-3 grid cells:
/// N∈{100,500} × M∈{50,200}, with `lambda = 0.0` (no warp penalty, D-06).
/// Square N×N self-distance matrix per D-01 (comparable to cross-distance).
///
/// Phase-1 sentinel recorded N=100×M=50 ≈ 790 ms (comparable cell here).
///
/// Timing by cell (O(n²·m²) DP):
///   - n100_m50:  ~760 ms/iter → sample_size(20), measurement_time(20s)
///   - n100_m200: ~17.6 s/iter → sample_size(10), measurement_time(60s)
///   - n500_m50:  ~27 s/iter  → sample_size(10), measurement_time(60s)
///   - n500_m200: ~very slow  → sample_size(10), measurement_time(60s)
fn bench_p3_elastic_self(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_elastic_self");

    // --- n100_m50: ~760 ms/iter — sentinel defaults ---
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(
                black_box(&data100_50),
                black_box(&argvals50),
                black_box(0.0),
            ))
        })
    });

    // --- n100_m200: ~17.6 s/iter — reduce samples, extend time ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m50: ~27 s/iter — reduce samples, extend time ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m200: borderline cell (workload-matrix 60s cap; very slow) ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix(
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

/// Phase-3 elastic_self_distance_matrix_banded sweep — band_frac=0.1, full grid.
///
/// Benchmarks `elastic_self_distance_matrix_banded` at all four Phase-3 grid cells.
/// Each pairwise alignment is confined to a Sakoe–Chiba corridor of half-width
/// `band_frac = 0.1` (D-03).  Pair with `bench_p3_elastic_self` to quantify the
/// banded-vs-unbanded reduction (~7× expected at M=200 per D-03).
///
/// Timing: n100_m50 uses sentinel defaults; all other cells use
/// `sample_size(10)` + `measurement_time(60s)` due to observed iteration times.
fn bench_p3_elastic_self_banded(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_elastic_self_banded");

    // --- n100_m50: banding reduces cost significantly ---
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix_banded(
                black_box(&data100_50),
                black_box(&argvals50),
                black_box(0.0),
                black_box(0.1), // band_frac = 0.1 (D-03)
            ))
        })
    });

    // --- n100_m200: reduce samples for extended iter time ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix_banded(
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m50 ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix_banded(
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m200: borderline cell ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(elastic_self_distance_matrix_banded(
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    group.finish();
}

/// Phase-3 elastic_cross_distance_matrix sweep — unbanded, square N×N, full grid.
///
/// Benchmarks `elastic_cross_distance_matrix` at all four Phase-3 grid cells:
/// N∈{100,500} × M∈{50,200}, with `lambda = 0.0` (D-06).
///
/// **D-02:** Square N×N cross-distance — data1 = data2 = N curves from the grid.
/// This makes the cost directly comparable to `elastic_self_distance_matrix`
/// (same pairs, same cell sizes). Both datasets are the same `FdMatrix` instance.
///
/// Cross-distance visits all N×N pairs (not just upper-triangular), so it is
/// approximately 2× the cost of self-distance at the same N×M.
///
/// Timing by cell (O(n²·m²) DP, all pairs):
///   - n100_m50:  ~2× self → sample_size(20), measurement_time(30s)
///   - n100_m200: very slow → sample_size(10), measurement_time(60s)
///   - n500_m50:  very slow → sample_size(10), measurement_time(60s)
///   - n500_m200: extreme  → sample_size(10), measurement_time(60s)
fn bench_p3_elastic_cross(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_elastic_cross");

    // --- n100_m50 (square: data1=data2) ---
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(30));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix(
                black_box(&data100_50), // data1
                black_box(&data100_50), // data2 = data1 (square N×N, D-02)
                black_box(&argvals50),
                black_box(0.0),
            ))
        })
    });

    // --- n100_m200: reduce samples due to extended iter time ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix(
                black_box(&data100_200),
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m50: extended time, reduced samples ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix(
                black_box(&data500_50),
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(0.0),
            ))
        })
    });

    // --- n500_m200: borderline cell ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix(
                black_box(&data500_200),
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(0.0),
            ))
        })
    });

    group.finish();
}

/// Phase-3 elastic_cross_distance_matrix_banded sweep — band_frac=0.1, square N×N, full grid.
///
/// Benchmarks `elastic_cross_distance_matrix_banded` at all four Phase-3 grid cells.
/// Square N×N (data1=data2) per D-02; `band_frac = 0.1` per D-03.
///
/// Pairs with `bench_p3_elastic_cross` to quantify the banded-vs-unbanded
/// reduction (~7× expected at M=200, ~10× theoretical when m/band=200/20).
///
/// Timing: n100_m50 uses sentinel defaults; all other cells use
/// `sample_size(10)` + `measurement_time(60s)`.
fn bench_p3_elastic_cross_banded(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_elastic_cross_banded");

    // --- n100_m50 (square: data1=data2) ---
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix_banded(
                black_box(&data100_50),
                black_box(&data100_50), // square N×N (D-02)
                black_box(&argvals50),
                black_box(0.0),
                black_box(0.1), // band_frac = 0.1 (D-03)
            ))
        })
    });

    // --- n100_m200: reduce samples ---
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_200, argvals200) = generate_curves(100, 200);
    group.bench_function("n100_m200", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix_banded(
                black_box(&data100_200),
                black_box(&data100_200),
                black_box(&argvals200),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m50 ---
    let (data500_50, argvals50b) = generate_curves(500, 50);
    group.bench_function("n500_m50", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix_banded(
                black_box(&data500_50),
                black_box(&data500_50),
                black_box(&argvals50b),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    // --- n500_m200: borderline cell ---
    let (data500_200, argvals200b) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| {
            black_box(elastic_cross_distance_matrix_banded(
                black_box(&data500_200),
                black_box(&data500_200),
                black_box(&argvals200b),
                black_box(0.0),
                black_box(0.1),
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fpca_sentinel,
    bench_matrix_sentinel,
    bench_elastic_sentinel,
    bench_depth_sentinel,
    bench_cv_sentinel,
    bench_streaming_sentinel,
    bench_smooth_sentinel,
    bench_p3_karcher,
    bench_p3_karcher_banded,
    bench_p3_elastic_self,
    bench_p3_elastic_self_banded,
    bench_p3_elastic_cross,
    bench_p3_elastic_cross_banded
);
criterion_main!(benches);
