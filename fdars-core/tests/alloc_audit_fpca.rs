//! dhat allocation-profiling integration tests for Phase 4 FPCA/SVD audit.
//!
//! Three measurement cells: `fdata_to_pc_1d` (N=500,M=200), `vert_fpca` (N=100,M=50),
//! and `joint_fpca` (N=100,M=50 with balance_c=Some(1.0) to bypass the golden-section
//! optimizer).  All cells are gated under `#[cfg(feature = "dhat-heap")]`.
//!
//! **Why integration test (not inline `#[cfg(test)]`)?**
//! dhat requires a *separate process* to set the global allocator.  Placing it inside
//! a library's `#[cfg(test)]` module would contaminate the allocator for every other
//! test in that compilation unit and produce incorrect counts.  Integration tests
//! under `fdars-core/tests/` each link into their own test binary — exactly the
//! separate-process guarantee dhat requires.
//!
//! **Run with:**
//! ```bash
//! TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
//!     cargo test -p fdars-core --features dhat-heap,linalg \
//!     -- count_fpca_allocations_n500_m200 --nocapture
//! ```
//!
//! **Feature gate:**  Every dhat symbol is under `#[cfg(feature = "dhat-heap")]`.
//! Without the feature, this file compiles to an empty test binary.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
use fdars_core::alignment::karcher_mean;
#[cfg(feature = "dhat-heap")]
use fdars_core::elastic_fpca::{joint_fpca, vert_fpca};
#[cfg(feature = "dhat-heap")]
use fdars_core::matrix::FdMatrix;
#[cfg(feature = "dhat-heap")]
use fdars_core::regression::fdata_to_pc_1d;
#[cfg(feature = "dhat-heap")]
use std::f64::consts::PI;

/// Generate deterministic functional data — column-major layout.
///
/// Replicates `generate_curves` from `audit_hotpaths.rs` lines 38-52 verbatim.
/// `data[i + j * n]` is column-major: observation `i` at evaluation point `j`.
///
/// NOTE: kept in sync with `benches/audit_hotpaths.rs:generate_curves`. Any change
/// to the amplitude/phase formula must be mirrored there to keep allocation baselines
/// and criterion numbers commensurable.
#[cfg(feature = "dhat-heap")]
fn generate_test_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        // Deterministic phase/amplitude variation per curve (no RNG dependency)
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

/// Count heap allocations for `fdata_to_pc_1d` at N=500, M=200.
///
/// Expected allocation profile (RESEARCH.md §4C):
/// - `regression.rs:167`  `center_columns`  → FdMatrix::zeros(n, m)     — 800 KB
/// - `regression.rs:291`  `centered.clone()` → FdMatrix (n×m copy)       — 800 KB
/// - `regression.rs:298`  `weighted.to_dmatrix()` → DMatrix<f64> copy   — 800 KB
///
/// Three O(n·m) allocations → ~2.4 MB total, ~1.6 MB peak (two live at peak).
/// Hard assertions are NOT made — this is a baseline, not a regression gate.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_fpca_allocations_n500_m200() {
    // Build test data OUTSIDE the profiler — setup, not the measurement target.
    let (data, argvals) = generate_test_curves(500, 200);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdata_to_pc_1d(&data, 5, &argvals);
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes: {}", stats.total_bytes);
    println!("Peak heap bytes: {}", stats.max_bytes);
    // Record — do not hard-assert a specific count (baseline, not regression gate).
}

/// Count heap allocations for `vert_fpca` at N=100, M=50.
///
/// `vert_fpca` uses `to_dmatrix()` at `elastic_fpca.rs:214` — the primary
/// elastic-FPCA SVD copy site.  KarcherMeanResult is built outside the profiler
/// scope (setup, not the measurement target).
///
/// Hard assertions are NOT made — this is a baseline.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_vert_fpca_allocations_n100_m50() {
    let (data, argvals) = generate_test_curves(100, 50);
    // Build KarcherMeanResult OUTSIDE the profiler — karcher_mean is setup, not target.
    let karcher = karcher_mean(&data, &argvals, 10, 1e-3, 0.0);

    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = vert_fpca(&karcher, &argvals, 5);
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes: {}", stats.total_bytes);
    println!("Peak heap bytes: {}", stats.max_bytes);
    // Record — do not hard-assert a specific count (baseline, not regression gate).
}

/// Count heap allocations for `joint_fpca` at N=100, M=50 (optimizer bypassed).
///
/// `joint_fpca` uses `to_dmatrix()` at `elastic_fpca.rs:317`.  `balance_c =
/// Some(1.0)` bypasses `optimize_balance_c_raw` (Pitfall B, RESEARCH.md §4B) so
/// this cell isolates the main SVD-copy path from the golden-section loop.
///
/// Hard assertions are NOT made — this is a baseline.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_joint_fpca_allocations_n100_m50() {
    let (data, argvals) = generate_test_curves(100, 50);
    // Build KarcherMeanResult OUTSIDE the profiler — karcher_mean is setup, not target.
    let karcher = karcher_mean(&data, &argvals, 10, 1e-3, 0.0);

    let _profiler = dhat::Profiler::builder().testing().build();
    // balance_c = Some(1.0) to bypass optimize_balance_c_raw (Pitfall B)
    let _ = joint_fpca(&karcher, &argvals, 5, Some(1.0));
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes: {}", stats.total_bytes);
    println!("Peak heap bytes: {}", stats.max_bytes);
    // Record — do not hard-assert a specific count (baseline, not regression gate).
}
