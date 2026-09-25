//! Grid-aligned fdars criterion timings for the external cross-language harness
//! (Phase 116 RPT-01).
//!
//! Purpose: the fdars criterion ids already committed to `benchmarks/results/`
//! (audit_fpca / audit_p4_fpca / fdata_to_pc) do not line up with the harness
//! grid one-to-one — e.g. no fdars `basis_smoothing` bench at all, and `fpca`
//! lacks an `n=300` cell. This bench file defines one `crosslang_<op>` group per
//! harness op, with function ids `n{N}_m{M}` exactly matching the grid the
//! Python/R runners drive, so `benchmarks/collect_rust.py` +
//! `benchmarks/assemble_comparison.py` can pick up a same-shaped Rust row for
//! every cell the report needs (D-09, D-13).
//!
//! Grid: N in {100, 300, 500} x M in {50, 200} (6 cells total, `GRID_CELLS`).
//!
//! Six groups are defined, one per overlapping op (D-09, D-13):
//! `crosslang_fpca`, `crosslang_basis_smoothing`, `crosslang_depth`,
//! `crosslang_functional_boxplot`, `crosslang_fpcr`, `crosslang_elastic` — each
//! with the same six `n{N}_m{M}` ids over `GRID_CELLS`, so every op has a
//! complete per-cell fdars number for the head-to-head and for fitting fdars's
//! own N/M scaling exponents.
//!
//! Data parity: this bench reads the SAME seeded CSV fixtures the Python and R
//! runners read (`benchmarks/generate_data.py`), from
//! `FDARS_CROSSLANG_DATA_DIR` (default `<CARGO_MANIFEST_DIR>/../benchmarks/data`).
//! When `FDARS_CROSSLANG_REQUIRE_CSV=1` is set (as `benchmarks/run_all.py` does
//! for every real harness run), a missing CSV panics with a message naming the
//! file and telling the user to run `benchmarks/generate_data.py` first — silent
//! divergence from the cross-language input would invalidate the whole
//! head-to-head. Outside that mode (e.g. a bare `cargo bench --bench
//! crosslang_grid` with no harness data generated yet), a deterministic
//! `StdRng`-seeded fallback of the same curve/response/pair family is used
//! instead, so the bench file always compiles and runs standalone.
//!
//! Thread note: `benchmarks/run_all.py::run_rust_bench` runs this bench under
//! `RAYON_NUM_THREADS=1` (and the other four thread-pinning env vars) for the
//! single-threaded, fairness-first head-to-head (D-12). `fdata_to_pc` is
//! sequential internally (nalgebra/faer SVD + plain `for`-loop centering), so
//! this bench's timings are parallel-invariant regardless.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::alignment::elastic_align_pair;
use fdars_core::depth::{functional_boxplot, modified_band, DepthMethod};
use fdars_core::dim::Dim;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc;
use fdars_core::scalar_on_function::fregre_lm;
use fdars_core::smooth_basis::{bspline_penalty_matrix, smooth_basis, BasisType, FdPar};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::time::Duration;

/// Locked grid cells, in the fixed order the harness iterates
/// (`benchmarks/run_all.py::GRID_N` x `GRID_M`).
const GRID_CELLS: [(usize, usize); 6] = [
    (100, 50),
    (100, 200),
    (300, 50),
    (300, 200),
    (500, 50),
    (500, 200),
];

/// Directory containing the harness's seeded CSV fixtures.
///
/// Defaults to `<CARGO_MANIFEST_DIR>/../benchmarks/data` (fdars-core's
/// manifest dir is `fdars-core/`, so its parent's `benchmarks/data` is the
/// workspace-root `benchmarks/data/` the Python generator writes to).
fn data_dir() -> std::path::PathBuf {
    match std::env::var("FDARS_CROSSLANG_DATA_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("benchmarks")
            .join("data"),
    }
}

/// Whether a missing harness CSV should panic (real harness runs) or fall back
/// to a deterministic synthetic generator (standalone `cargo bench` runs).
fn require_csv() -> bool {
    std::env::var("FDARS_CROSSLANG_REQUIRE_CSV")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Read a harness curve CSV (`curves_n{n}_m{m}.csv`) into row-major `Vec<Vec<f64>>`.
///
/// Header: `curve_id,t_0..t_{m-1}`, one curve per row. Skips the header line,
/// splits on commas, drops the first column (`curve_id`), parses the rest as
/// f64. Returns `None` if the file does not exist.
fn read_csv_rows(path: &std::path::Path) -> Option<Vec<Vec<f64>>> {
    let contents = std::fs::read_to_string(path).ok()?;
    let mut lines = contents.lines();
    lines.next(); // header
    let rows: Vec<Vec<f64>> = lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split(',')
                .skip(1) // drop curve_id
                .map(|field| {
                    field
                        .trim()
                        .parse::<f64>()
                        .expect("crosslang_grid: failed to parse f64 field in harness CSV")
                })
                .collect()
        })
        .collect();
    Some(rows)
}

/// Deterministic synthetic fallback matching `benchmarks/generate_data.py`'s
/// curve family (amplitude/phase/noise-scale uniform draws, sinusoid + Gaussian
/// noise), seeded with `StdRng::seed_from_u64(42)` for reproducibility when the
/// real harness CSV is unavailable (non-`FDARS_CROSSLANG_REQUIRE_CSV` runs only).
fn generate_fallback_curves(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);

    (0..n)
        .map(|_| {
            let amplitude = rng.gen_range(0.5_f64..2.0_f64);
            let phase = rng.gen_range(0.0_f64..2.0 * std::f64::consts::PI);
            let noise_scale = rng.gen_range(0.0_f64..0.2_f64);
            (0..m)
                .map(|j| {
                    let t = j as f64 / (m - 1) as f64;
                    let noise: f64 = StandardNormal.sample(&mut rng);
                    amplitude * (2.0 * std::f64::consts::PI * t + phase).sin() + noise_scale * noise
                })
                .collect()
        })
        .collect()
}

/// Load the (N, M) curve set for one grid cell as an `FdMatrix` + argvals,
/// preferring the harness's seeded CSV and falling back to the deterministic
/// synthetic generator when the CSV is absent and not required.
///
/// Column-major layout: the value of curve `i` at evaluation point `j` goes to
/// flat index `i + j * n` (per `FdMatrix`'s documented convention).
///
/// # Panics
/// Panics if `FDARS_CROSSLANG_REQUIRE_CSV=1` and the CSV is missing, if the CSV
/// row/column count does not match `(n, m)`, or if `FdMatrix::from_column_major`
/// rejects the assembled buffer (dimension mismatch, which would be a bug in
/// this function).
fn load_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let csv_path = data_dir().join(format!("curves_n{n}_m{m}.csv"));
    let rows = match read_csv_rows(&csv_path) {
        Some(rows) => {
            assert_eq!(
                rows.len(),
                n,
                "crosslang_grid: {} has {} rows, expected n={}",
                csv_path.display(),
                rows.len(),
                n
            );
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    m,
                    "crosslang_grid: {} row {} has {} columns, expected m={}",
                    csv_path.display(),
                    i,
                    row.len(),
                    m
                );
            }
            rows
        }
        None => {
            if require_csv() {
                panic!(
                    "crosslang_grid: required harness CSV not found: {} — run \
                     `benchmarks/generate_data.py --n {} --m {}` first \
                     (FDARS_CROSSLANG_REQUIRE_CSV=1 is set).",
                    csv_path.display(),
                    n,
                    m
                );
            }
            generate_fallback_curves(n, m)
        }
    };

    let mut data = vec![0.0_f64; n * m];
    for (i, row) in rows.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            data[i + j * n] = value;
        }
    }
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let matrix = FdMatrix::from_column_major(data, n, m)
        .expect("crosslang_grid: FdMatrix::from_column_major failed (dimension bug)");
    (matrix, argvals)
}

/// Load the seeded scalar response vector for one grid cell's `n`
/// (`y_n{n}.csv`, header `y`, `n` values), preferring the harness's seeded CSV
/// and falling back to a deterministic `StdRng::seed_from_u64(142)` draw of
/// `StandardNormal`, mirroring `generate_response`'s `seed+100` convention
/// (harness base seed 42 -> 142).
///
/// # Panics
/// Panics if `FDARS_CROSSLANG_REQUIRE_CSV=1` and the CSV is missing, or if the
/// CSV row count does not match `n`.
fn load_response(n: usize) -> Vec<f64> {
    let csv_path = data_dir().join(format!("y_n{n}.csv"));
    match std::fs::read_to_string(&csv_path) {
        Ok(contents) => {
            let values: Vec<f64> = contents
                .lines()
                .skip(1) // header "y"
                .filter(|line| !line.is_empty())
                .map(|line| {
                    line.trim()
                        .parse::<f64>()
                        .expect("crosslang_grid: failed to parse f64 field in y_n{n}.csv")
                })
                .collect();
            assert_eq!(
                values.len(),
                n,
                "crosslang_grid: {} has {} rows, expected n={}",
                csv_path.display(),
                values.len(),
                n
            );
            values
        }
        Err(_) => {
            if require_csv() {
                panic!(
                    "crosslang_grid: required harness CSV not found: {} — run \
                     `benchmarks/generate_data.py --n {} --m <M>` first \
                     (FDARS_CROSSLANG_REQUIRE_CSV=1 is set).",
                    csv_path.display(),
                    n
                );
            }
            let mut rng = StdRng::seed_from_u64(142);
            (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
        }
    }
}

/// Load the 2-curve elastic pairwise fixture for one grid cell's `m`
/// (`elastic_n2_m{m}.csv`), preferring the harness's seeded CSV and falling
/// back to a deterministic `StdRng::seed_from_u64(242)` construction mirroring
/// `generate_elastic_pair`'s `seed+200` convention (harness base seed 42 ->
/// 242): f1 = sin(2*pi*t), f2 = amp2 * sin(2*pi*t + phase_shift) with
/// `phase_shift ~ U(0.2, 0.5)` and `amp2 ~ U(0.8, 1.2)`, argvals uniform on
/// `[0, 1]`.
///
/// Returns `(f1, f2, argvals)`.
///
/// # Panics
/// Panics if `FDARS_CROSSLANG_REQUIRE_CSV=1` and the CSV is missing, or if the
/// CSV does not contain exactly 2 rows of length `m`.
fn load_elastic_pair(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let csv_path = data_dir().join(format!("elastic_n2_m{m}.csv"));
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    match read_csv_rows(&csv_path) {
        Some(rows) => {
            assert_eq!(
                rows.len(),
                2,
                "crosslang_grid: {} has {} rows, expected 2",
                csv_path.display(),
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    m,
                    "crosslang_grid: {} row {} has {} columns, expected m={}",
                    csv_path.display(),
                    i,
                    row.len(),
                    m
                );
            }
            (rows[0].clone(), rows[1].clone(), argvals)
        }
        None => {
            if require_csv() {
                panic!(
                    "crosslang_grid: required harness CSV not found: {} — run \
                     `benchmarks/generate_data.py --n <N> --m {}` first \
                     (FDARS_CROSSLANG_REQUIRE_CSV=1 is set).",
                    csv_path.display(),
                    m
                );
            }
            let mut rng = StdRng::seed_from_u64(242);
            let phase_shift = rng.gen_range(0.2_f64..0.5_f64);
            let amp2 = rng.gen_range(0.8_f64..1.2_f64);
            let f1: Vec<f64> = argvals
                .iter()
                .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
                .collect();
            let f2: Vec<f64> = argvals
                .iter()
                .map(|&t| amp2 * (2.0 * std::f64::consts::PI * t + phase_shift).sin())
                .collect();
            (f1, f2, argvals)
        }
    }
}

/// `crosslang_fpca` group — grid-aligned `fdata_to_pc` timings.
///
/// `n500_m200` is ~16 ms/iter per the Phase 113 `audit_p4_fpca` data, so 30
/// samples in a 10 s measurement window is well inside budget for every cell
/// in the grid (the smaller cells are faster).
fn bench_crosslang_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_fpca");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(FdMatrix, Vec<f64>)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (data, argvals) = cache.get_or_insert_with(|| load_curves(n, m));
            b.iter(|| fdata_to_pc(black_box(data), black_box(5usize), black_box(argvals)))
        });
    }

    group.finish();
}

/// `crosslang_basis_smoothing` group — grid-aligned `smooth_basis` timings.
///
/// Matches the Python/R runners' B-spline basis smoothing parameters: order
/// 4, nbasis 10, domain `[0, 1]`, lambda 0 (pure interpolation, no penalty).
/// The `FdPar` (including its penalty matrix) is built once per cell outside
/// `b.iter`, exactly as `basis_benchmarks.rs::bench_smooth_basis` does; lambda
/// 0 matches the harness's interpolation setting — the penalty matrix is
/// still built (parity with the general `smooth_basis` call shape) but
/// multiplied by zero.
fn bench_crosslang_basis_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_basis_smoothing");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(FdMatrix, Vec<f64>, FdPar)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (data, argvals, fdpar) = cache.get_or_insert_with(|| {
                let (data, argvals) = load_curves(n, m);
                let order = 4;
                let nbasis = 10;
                let lfd_order = 2;
                let penalty_matrix = bspline_penalty_matrix(&argvals, nbasis, order, lfd_order);
                let fdpar = FdPar {
                    basis_type: BasisType::Bspline { order },
                    nbasis,
                    lambda: 0.0,
                    lfd_order,
                    penalty_matrix,
                };
                (data, argvals, fdpar)
            });
            b.iter(|| smooth_basis(black_box(data), black_box(argvals), black_box(fdpar)).unwrap())
        });
    }

    group.finish();
}

/// `crosslang_depth` group — grid-aligned `modified_band` (Modified Band
/// Depth of the sample w.r.t. itself, deterministic) timings, matching the
/// Python/R runners' `ModifiedBandDepth` / `roahd::MBD`.
fn bench_crosslang_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_depth");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(FdMatrix, Vec<f64>)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (data, _argvals) = cache.get_or_insert_with(|| load_curves(n, m));
            b.iter(|| modified_band(black_box(data), black_box(data), Dim::One))
        });
    }

    group.finish();
}

/// `crosslang_functional_boxplot` group — grid-aligned `functional_boxplot`
/// timings (MBD ordering, factor 1.5), matching the Python/R runners'
/// `Boxplot(depth_method=ModifiedBandDepth(), factor=1.5)` /
/// `roahd::fbplot(Depths="MBD")`.
fn bench_crosslang_functional_boxplot(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_functional_boxplot");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(FdMatrix, Vec<f64>)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (data, _argvals) = cache.get_or_insert_with(|| load_curves(n, m));
            b.iter(|| functional_boxplot(black_box(data), DepthMethod::ModifiedBand, 1.5))
        });
    }

    group.finish();
}

/// `crosslang_fpcr` group — grid-aligned `fregre_lm` (5 FPCs, least squares on
/// the seeded response) timings, matching the Python/R runners' FPCA-scores
/// -> OLS approach (`FPCA(n_components=5)` + `numpy.linalg.lstsq` /
/// `fda.usc::fregre.pc`).
///
/// Noise remediation (Phase 116 D-10), round 1: the headline cell
/// `n500_m200` failed the 5% noise gate at 30 samples / 10 s (RSD 11.79%,
/// CI-width 8.11%), so the group is raised to 60 samples / 20 s.
fn bench_crosslang_fpcr(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_fpcr");
    group.sample_size(60);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(FdMatrix, Vec<f64>, Vec<f64>)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (data, _argvals, y) = cache.get_or_insert_with(|| {
                let (data, argvals) = load_curves(n, m);
                let y = load_response(n);
                (data, argvals, y)
            });
            b.iter(|| fregre_lm(black_box(data), black_box(y), None, black_box(5)))
        });
    }

    group.finish();
}

/// `crosslang_elastic` group — grid-aligned pairwise Fisher-Rao alignment
/// timings for the 2-curve fixture (`elastic_n2_m{m}.csv`), lambda 0,
/// matching the Python/R runners' `FisherRaoElasticRegistration(penalty=0.0)`
/// / `fdasrvf::time_warping`.
///
/// `elastic_align_pair` at m=200 is ~52 ms/iter per the Phase 113
/// `audit_p3_elastic_cross` data, so 30 linear samples need about 25s;
/// `measurement_time` is raised to 30s (`warm_up_time` 5s) to leave headroom.
///
/// All six grid cells are benched even though the pairwise computation
/// depends only on `m` (the two-curve fixture is `n`-independent by
/// construction) so the report can show the fdars N-scaling exponent is
/// approximately 0 for this op (D-13).
fn bench_crosslang_elastic(c: &mut Criterion) {
    let mut group = c.benchmark_group("crosslang_elastic");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(5));

    for &(n, m) in GRID_CELLS.iter() {
        let mut cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
        group.bench_function(format!("n{n}_m{m}"), |b| {
            let (f1, f2, argvals) = cache.get_or_insert_with(|| load_elastic_pair(m));
            b.iter(|| {
                elastic_align_pair(
                    black_box(f1),
                    black_box(f2),
                    black_box(argvals),
                    black_box(0.0),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_crosslang_fpca,
    bench_crosslang_basis_smoothing,
    bench_crosslang_depth,
    bench_crosslang_functional_boxplot,
    bench_crosslang_fpcr,
    bench_crosslang_elastic,
);
criterion_main!(benches);
