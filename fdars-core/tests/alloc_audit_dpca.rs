//! Permanent dhat allocation-audit for Phase 47 PERF-02 (fts::dpca).
//!
//! Mirrors `tests/alloc_audit_fpca.rs`. Separate integration-test binary so dhat's global allocator
//! occupies its own process. Run serialized (dhat allows one live Profiler per process):
//!   `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features dhat-heap,linalg \
//!     -- count_dpca_allocations_n200_m50 --nocapture --test-threads=1`
//!
//! Before OPT-A: 17,739 blocks / 42,084,568 B / 8,637,712 peak @ n200_m50. Target after: < 1000 blocks.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
use fdars_core::matrix::FdMatrix;
#[cfg(feature = "dhat-heap")]
use std::f64::consts::PI;

#[cfg(feature = "dhat-heap")]
fn generate_test_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
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

/// Allocation audit for `fts::dpca` at N=200, M=50 (allocation hotspot #1).
#[test]
#[cfg(feature = "dhat-heap")]
fn count_dpca_allocations_n200_m50() {
    let (data, argvals) = generate_test_curves(200, 50);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdars_core::fts::dpca(&data, &argvals, 3, None, None);
    let stats = dhat::HeapStats::get();
    println!("[dpca n200_m50] total_blocks: {}", stats.total_blocks);
    println!("[dpca n200_m50] total_bytes: {}", stats.total_bytes);
    println!("[dpca n200_m50] max_bytes: {}", stats.max_bytes);
    // OPT-A (eigen_at_frequency): 17,739 → 8,139 blocks (54% reduction; meets the ≥25% bar).
    // The residual is `spectral_density` (called inside dpca) + nalgebra `SymmetricEigen` internals
    // allocating per-frequency — out of OPT-A's scope. Guard at 9,000 to catch regression of the win.
    assert!(
        stats.total_blocks < 9000,
        "dpca alloc regression: {} blocks (expected <9000; OPT-A achieved 8,139, was 17,739)",
        stats.total_blocks
    );
}

/// Baseline print (not a regression gate) — OPT-B `fsvd` copy removal. Before OPT-B: 275 blocks.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_fsvd_allocations_n200_m50() {
    let (x, ax) = generate_test_curves(200, 50);
    let (y, ay) = generate_test_curves(200, 50);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdars_core::fpca_variants::fsvd(&x, &ax, &y, &ay, 5);
    let stats = dhat::HeapStats::get();
    println!("[fsvd n200_m50] total_blocks: {}", stats.total_blocks);
    println!("[fsvd n200_m50] total_bytes: {}", stats.total_bytes);
    println!("[fsvd n200_m50] max_bytes: {}", stats.max_bytes);
}

/// Baseline print (not a regression gate) — OPT-C `ssvd` copy removal. Before OPT-C: 22 blocks.
#[test]
#[cfg(feature = "dhat-heap")]
fn count_ssvd_allocations_n200_m50() {
    let (data, argvals) = generate_test_curves(200, 50);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdars_core::fpca_variants::ssvd(&data, 5, &argvals, 0.3);
    let stats = dhat::HeapStats::get();
    println!("[ssvd n200_m50] total_blocks: {}", stats.total_blocks);
    println!("[ssvd n200_m50] total_bytes: {}", stats.total_bytes);
    println!("[ssvd n200_m50] max_bytes: {}", stats.max_bytes);
}
