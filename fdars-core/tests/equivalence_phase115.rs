//! Permanent golden-equivalence tests for Phase 115 (SVD/FPCA & Covariance Optimization).
//!
//! OPT-02 refactors `cov_irreg` in `src/irreg_fdata/smoothing.rs` to exploit the
//! symmetry of the covariance surface when `s_grid == t_grid`: only the upper
//! triangle is computed and the lower half is filled by a bit-copy (`sym_cov_irreg`).
//! The `face_covariance` call site in `src/irreg_fdata/face.rs` (line 148) is
//! updated from `cov_irreg(ifd, grid, grid, bandwidth)` to
//! `sym_cov_irreg(ifd, grid, bandwidth)`.
//!
//! This fixture locks the pre-optimization `face_covariance` output for a fixed
//! `n=10, m=11` sample so that Plan 02's edit can be verified to change nothing
//! numerically:
//!
//!   - `golden_face_covariance_n10_m11`: asserts selected cells (upper-triangle,
//!     mirrored partners, diagonal) against literal `f64` constants at tolerance
//!     `1e-10`, and asserts the full symmetry invariant within `1e-10`.
//!
//! Reference values are captured from the CURRENT (pre-OPT-02) code in Plan 01
//! (Wave 1, 2026-09-23), before any `fdars-core/src` change. The refactor must
//! reproduce every constant within `1e-10` (OPT-02 is a pure upper-triangle
//! bit-copy — the upper half is identical; the mirrored half is a bit-exact copy,
//! so `1e-10` is the documented hard gate).

// Golden reference constants are pasted at full printed precision (`{:.17e}`); the
// extra digits beyond f64's exact representation are intentional (round-trip
// fidelity), so silence the lint.
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use fdars_core::irreg_fdata::{face_covariance, IrregFdata};

/// Absolute-closeness assertion: `|a - b| <= tol`.
fn assert_abs_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "abs-close failed: a={a:.17e} b={b:.17e} abs_err={:.3e} tol={tol:.1e}",
        (a - b).abs()
    );
}

/// Deterministic synthetic irregular functional data matching the bench/probe generator.
///
/// Grid: `j / (m-1)` for `j` in `0..m`.
/// Per-curve: `phase = 0.2 * sin(i*3.7 + 0.5)`, `amp = 1.0 + 0.3 * sin(i*5.1 + 0.3)`,
/// values: `amp * sin(2π * (t + phase))`.
///
/// This is the identical generator used in `benches/perf_hotpaths.rs:bench_face_covariance`
/// (and the `perf_face_covariance/n200_m30` criterion cell), so that benchmark
/// measurements and golden constants are commensurable.
fn make_test_ifd(n: usize, m: usize) -> (IrregFdata, Vec<f64>) {
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
    (IrregFdata::from_lists(&argvals_list, &values_list), grid)
}

// ─── OPT-02: face_covariance golden constants (n=10, m=11) ────────────────────────────────────────
// Reference constants captured from the PRE-OPT-02 code (Plan 01, Wave 1, 2026-09-23).
// Tolerance 1e-10: OPT-02 is a pure upper-triangle bit-copy — upper half arithmetic
// is unchanged; lower half becomes a bit-exact mirror of upper half.
//
// Cells covered:
//   diagonal:        (0,0), (5,5), (10,10)
//   upper-triangle:  (0,5), (2,7)
//   mirrored:        (5,0), (7,2)
#[test]
fn golden_face_covariance_n10_m11() {
    let (ifd, grid) = make_test_ifd(10, 11);
    let cov = face_covariance(&ifd, &grid, 0.3).unwrap();

    // Structural invariant: shape must be (m, m) = (11, 11).
    assert_eq!(cov.shape(), (11, 11));

    // ── Diagonal cells ─────────────────────────────────────────────────────
    const COV_0_0: f64 = 2.01522424817674774e-2;
    const COV_5_5: f64 = 2.26413976377977649e-3;
    const COV_10_10: f64 = 1.47740963472988169e-2;

    assert_abs_close(cov[(0, 0)], COV_0_0, 1e-10);
    assert_abs_close(cov[(5, 5)], COV_5_5, 1e-10);
    assert_abs_close(cov[(10, 10)], COV_10_10, 1e-10);

    // ── Upper-triangle cell (0, 5) and its mirror (5, 0) ───────────────────
    // Pre-opt: both are computed independently by cov_irreg and may differ by ~1e-17.
    // Post-opt: (5,0) is a bit-copy of (0,5) — exactly equal.
    // The constant below is from the upper-triangle cell (0,5); use assert_abs_close
    // for the mirrored partner so the fixture is green on BOTH pre- and post-opt code.
    const COV_0_5: f64 = -3.31275058129442681e-3;
    const COV_5_0: f64 = -3.31275058129442768e-3;

    assert_abs_close(cov[(0, 5)], COV_0_5, 1e-10);
    assert_abs_close(cov[(5, 0)], COV_5_0, 1e-10);

    // ── Upper-triangle cell (2, 7) and its mirror (7, 2) ───────────────────
    const COV_2_7: f64 = -5.92895218221224912e-3;
    const COV_7_2: f64 = -5.92895218221224998e-3;

    assert_abs_close(cov[(2, 7)], COV_2_7, 1e-10);
    assert_abs_close(cov[(7, 2)], COV_7_2, 1e-10);

    // ── Symmetry structural invariant ──────────────────────────────────────
    // On un-optimized code: cov_irreg computes both halves independently;
    // symmetry holds within ~1e-17 (floating-point commutativity).
    // On post-opt code: lower half is a bit-copy of upper; exact equality.
    // Use assert_abs_close at 1e-10 so the fixture stays green on both.
    let m = 11usize;
    for i in 0..m {
        for j in 0..i {
            assert_abs_close(cov[(i, j)], cov[(j, i)], 1e-10);
        }
    }
}
