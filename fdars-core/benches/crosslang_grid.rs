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
//! every cell the report needs (D-09, D-13). This tracer plan (116-01) ships
//! only the `fpca` group; 116-02 adds the other five ops on this same scaffold.
//!
//! Grid: N in {100, 300, 500} x M in {50, 200} (6 cells total, `GRID_CELLS`).
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
//! `StdRng::seed_from_u64(42)` fallback of the same curve family is used
//! instead, so the bench file always compiles and runs standalone.
//!
//! Thread note: `benchmarks/run_all.py::run_rust_bench` runs this bench under
//! `RAYON_NUM_THREADS=1` (and the other four thread-pinning env vars) for the
//! single-threaded, fairness-first head-to-head (D-12). `fdata_to_pc` is
//! sequential internally (nalgebra/faer SVD + plain `for`-loop centering), so
//! this bench's timings are parallel-invariant regardless.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc;
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

criterion_group!(benches, bench_crosslang_fpca);
criterion_main!(benches);
