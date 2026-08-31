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

use fdars_core::fpca_variants::{fsvd, ssvd};
use fdars_core::fts::{dpca, functional_acf};
use fdars_core::irreg_fdata::{face_covariance, IrregFdata};
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

// ─── OPT-C: ssvd sandwich-eigen copy removal (fpca_variants.rs) ────────────────────────────────
#[test]
fn golden_ssvd_n30_m12() {
    let (data, argvals) = generate_curves(30, 12);
    let r = ssvd(&data, 3, &argvals, 0.1).unwrap();
    // Reference captured from pre-OPT-C code. rel 1e-12: from_fn is byte-identical to
    // from_column_slice(&c_scaled) (same column-major arithmetic).
    assert_rel_close(r.singular_values[0], 2.24787453825566086e0, 1e-12);
    assert_rel_close(r.singular_values[1], 1.01807594146085556e0, 1e-12);
    assert_rel_close(r.singular_values[2], 2.32292395913555848e-8, 1e-10);
    assert_rel_close(r.rotation.as_slice()[0], 1.50783709265766985e0, 1e-12);
    assert_rel_close(r.scores.as_slice()[0], 4.22050180009691023e-1, 1e-12);
}

// ─── OPT-B: fsvd gram copy removal (fpca_variants.rs) ─────────────────────────────────────────
#[test]
fn golden_fsvd_n20_p15_q10() {
    let (x, ax) = generate_curves(20, 15);
    let (y, ay) = generate_curves(20, 10);
    let r = fsvd(&x, &ax, &y, &ay, 3).unwrap();
    // Reference captured from pre-OPT-B code. rel 1e-12: from_fn is byte-identical to
    // from_column_slice(&gram) (same column-major arithmetic).
    assert_rel_close(r.singular_values[0], 2.50821617715771994e-1, 1e-12);
    assert_rel_close(r.singular_values[1], 6.02876754087493597e-2, 1e-12);
    assert_rel_close(r.left_functions.as_slice()[0], 1.41398040862575569e0, 1e-12);
    assert_rel_close(
        r.right_functions.as_slice()[0],
        1.42469272612456255e0,
        1e-12,
    );
}

// ─── OPT-D: functional_acf c0_scaled from_fn + sqrt_w precompute (fts/acf.rs) ──────────────────
#[test]
fn golden_functional_acf_n40_m12() {
    let (data, argvals) = generate_curves(40, 12);
    let r = functional_acf(&data, &argvals, Some(5), 50, 0.95, 7).unwrap();
    // Reference captured from pre-OPT-D code. acf/pacf are the deterministic covariance path
    // (rel 1e-12); upper_band is the seed-fixed Monte-Carlo path (also exact here, rel 1e-10).
    assert_rel_close(r.acf[0], 7.10234356424632618e-1, 1e-12);
    assert_rel_close(r.acf[1], 3.62777392633280904e-1, 1e-12);
    assert_rel_close(r.pacf[0], 7.10234356424632618e-1, 1e-12);
    assert_rel_close(r.pacf[1], -2.85845108686378857e-1, 1e-12);
    assert_rel_close(r.upper_band[0], 2.50742939451547353e-1, 1e-10);
}

// ─── OPT-E: face_covariance kernel-weight-table precompute (irreg_fdata/smoothing.rs) ──────────
#[test]
fn golden_face_covariance_n40() {
    // Deterministic irregular data: 40 curves, 3-5 obs points each, no RNG.
    let mut argvals_list: Vec<Vec<f64>> = Vec::with_capacity(40);
    let mut values_list: Vec<Vec<f64>> = Vec::with_capacity(40);
    for i in 0..40usize {
        let npts = 3 + (i % 3); // 3,4,5
        let mut av = Vec::with_capacity(npts);
        let mut vv = Vec::with_capacity(npts);
        for k in 0..npts {
            let t = (k as f64 + 0.3 * (i as f64 * 0.7).sin().abs()) / (npts as f64);
            let t = t.clamp(0.0, 1.0);
            av.push(t);
            vv.push((2.0 * PI * (t + 0.1 * i as f64)).sin() + 0.05 * i as f64);
        }
        argvals_list.push(av);
        values_list.push(vv);
    }
    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
    let grid: Vec<f64> = (0..10).map(|j| j as f64 / 9.0).collect();
    let cov = face_covariance(&ifd, &grid, 0.15).unwrap();
    let s = cov.as_slice();
    let (ns, nt) = cov.shape();
    // Reference captured from pre-OPT-E code. rel 1e-12: the w_s/w_t factoring is exact
    // (Gaussian kernel separable in s,t) and preserves the per-cell i→j1→j2 summation order.
    assert_eq!(s.len(), 100);
    assert_rel_close(s[0], 5.02285392885318593e-1, 1e-12);
    assert_rel_close(s[ns - 1], 1.05308144194836459e-1, 1e-12);
    assert_rel_close(s[ns * (nt - 1)], 1.05308144194836389e-1, 1e-12);
    assert_rel_close(s[ns * nt - 1], 6.66851657132077946e-1, 1e-12);
    assert_rel_close(s[ns * nt / 2], 1.62141589034421940e-1, 1e-12);
}
