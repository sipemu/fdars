//! THROWAWAY dhat allocation probes — Phase 46 (Whole-Crate Profiling & Measurement).
//!
//! Baseline allocation measurements for the reuse-first v0.19–v0.29 subsystems'
//! hot paths. Separate integration-test binary so dhat's `#[global_allocator]`
//! occupies its own process (mirrors `tests/alloc_audit_fpca.rs`).
//!
//! REMOVED in Plan 02 Task 3 — not a permanent test (measurements live in
//! `PROF-01-hotpath-targets.md`).
//!
//! Run: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core \
//!        --features dhat-heap,linalg -- count_ --nocapture`

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
use fdars_core::matrix::FdMatrix;
#[cfg(feature = "dhat-heap")]
use std::f64::consts::PI;

/// Deterministic synthetic curves, column-major. Mirrors
/// `benches/audit_hotpaths.rs:generate_curves` verbatim.
#[cfg(feature = "dhat-heap")]
fn generate_test_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
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

/// Allocation baseline for `fpca_variants::fsvd` at N=200, M=50.
///
/// Hotspot: `src/fpca_variants.rs:488` — `DMatrix::from_column_slice(g_dim, g_dim, &gram)`
/// (gram-matrix eigendecomposition copy). Baseline only — NOT a hard-asserted gate.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_fsvd_allocations_n200_m50() {
    let (x, argvals_x) = generate_test_curves(200, 50);
    let (y, argvals_y) = generate_test_curves(200, 50);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdars_core::fpca_variants::fsvd(&x, &argvals_x, &y, &argvals_y, 5);
    let stats = dhat::HeapStats::get();
    println!("[fsvd n200_m50] total_blocks: {}", stats.total_blocks);
    println!("[fsvd n200_m50] total_bytes: {}", stats.total_bytes);
    println!("[fsvd n200_m50] max_bytes: {}", stats.max_bytes);
}
