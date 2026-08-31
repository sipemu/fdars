//! Permanent golden-equivalence tests for Phase 47 (Hot-Path & Allocation Performance).
//!
//! Each optimization (OPT-A..OPT-F) must reproduce the CURRENT (pre-optimization) numeric output
//! within a documented tolerance: **exact** for counting/integer paths, **relative ≤ 1e-10** for
//! float / SVD / eigen paths (47-CONTEXT.md). Reference values are captured from the current code
//! and hardcoded as `const`s; the refactor must keep them within `assert_rel_close`.
//!
//! Per-OPT tests are added by their owning plans/tasks; this module owns the shared helpers.

// Golden reference constants are pasted at full printed precision (`{:.17e}`); the extra digits
// beyond f64's exact representation are intentional (round-trip fidelity), so silence the lint.
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use fdars_core::fts::dpca;
use fdars_core::matrix::FdMatrix;

/// Relative-closeness assertion: `|a-b| <= tol * max(|a|, 1e-12)`.
fn assert_rel_close(a: f64, b: f64, tol: f64) {
    let scale = a.abs().max(1e-12);
    assert!(
        (a - b).abs() <= tol * scale,
        "rel-close failed: a={a:.17e} b={b:.17e} rel_err={:.3e} tol={tol:.1e}",
        (a - b).abs() / scale
    );
}

/// Deterministic synthetic curves (column-major), matching the bench/probe generators.
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

#[test]
fn helpers_smoke() {
    assert_rel_close(1.0, 1.0 + 1e-13, 1e-10);
    let (data, argvals) = generate_curves(50, 10);
    assert_eq!(data.shape(), (50, 10));
    assert_eq!(argvals.len(), 10);
}

// ─── OPT-A: fts::dpca allocation refactor (eigen_at_frequency) ────────────────────────────────
// Reference constants captured from the PRE-OPT-A code (Task 2). Tolerance 1e-12: OPT-A is a pure
// copy-elimination refactor — eigendecomposition inputs/order/sign-alignment are unchanged.
#[test]
fn golden_dpca_n50_m10() {
    let (data, argvals) = generate_curves(50, 10);
    let r = dpca(&data, &argvals, 2, None, None).unwrap();

    // Structural invariants.
    assert_eq!(r.ncomp, 2);
    assert_eq!(r.filters.len(), 2);
    assert_eq!(r.n_freqs, 50);
    assert_eq!(r.eigenvalues.len(), 2);

    let nf = r.n_freqs;
    // Reference checkpoints captured from the PRE-OPT-A code (2026-08-31). Tolerance 1e-12: OPT-A is
    // a pure copy-elimination refactor — eigendecomposition order/sign-alignment are unchanged.
    const EIG0_0: f64 = 5.13206226443637392e-2;
    const EIG1_0: f64 = 3.70777740086235749e-2;
    const EIG0_MID: f64 = 6.81620794100073035e-1;
    const FILT0_0: f64 = 8.46588981863219425e-4;
    const FILT1_0: f64 = 4.87470752471493040e-4;
    const SCORES_0: f64 = 5.09449869041621484e-1;
    const SCORES_LAST: f64 = -3.73917270422230330e-1;

    assert_rel_close(r.eigenvalues[0][0], EIG0_0, 1e-12);
    assert_rel_close(r.eigenvalues[1][0], EIG1_0, 1e-12);
    assert_rel_close(r.eigenvalues[0][nf / 2], EIG0_MID, 1e-12);
    assert_rel_close(r.filters[0].as_slice()[0], FILT0_0, 1e-12);
    assert_rel_close(r.filters[1].as_slice()[0], FILT1_0, 1e-12);
    assert_rel_close(r.scores.as_slice()[0], SCORES_0, 1e-12);
    assert_rel_close(*r.scores.as_slice().last().unwrap(), SCORES_LAST, 1e-12);
}
