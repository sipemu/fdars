//! Permanent golden-equivalence tests for Phase 114 (Elastic Alignment Band-Iteration Optimization).
//!
//! OPT-01 refactors the inner loop of `dp_fill_grid` in `src/alignment/mod.rs` to eliminate
//! redundant bound re-checks inside the banded corridor. Both the banded and unbanded arms of
//! `dp_fill_grid` are restructured; this fixture locks the pre-optimization output for both
//! so that Plan 02's edit can be verified to change nothing numerically.
//!
//! Reference values are captured from the CURRENT (pre-optimization) code in Plan 01 (Wave 1),
//! before any `fdars-core/src` change. The refactor must reproduce every constant within
//! `assert_rel_close` at tolerance 1e-10 (OPT-01 is a pure loop-structure refactor — no
//! arithmetic changes, so tighter is achievable but 1e-10 is the documented hard gate).
//!
//! Two tests:
//!   - `golden_elastic_self_distance_unbanded_n20_m50`: exercises the UNBANDED dp_fill_grid arm.
//!   - `golden_elastic_self_distance_banded_n20_m50`:  exercises the BANDED dp_fill_grid arm
//!       (band_frac=0.1); THIS is the arm Plan 02 restructures.
//!
//! Both tests use `generate_curves(20, 50)` — the same deterministic no-RNG generator as the
//! criterion harness in `benches/audit_hotpaths.rs`.

// Golden reference constants are pasted at full printed precision (`{:.17e}`); the extra digits
// beyond f64's exact representation are intentional (round-trip fidelity), so silence the lint.
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use fdars_core::alignment::{elastic_self_distance_matrix, elastic_self_distance_matrix_banded};
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
///
/// Uses the identical amplitude/phase formula as `benches/audit_hotpaths.rs:generate_curves`
/// so that benchmark measurements and golden constants are commensurable.
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

// ─── OPT-01: Unbanded dp_fill_grid arm ────────────────────────────────────────────────────────────
// Reference constants captured from the PRE-OPT-01 code (Plan 01, 2026-09-23). Tolerance 1e-10:
// OPT-01 is a pure loop-structure refactor — DP arithmetic and accumulation order are unchanged.
#[test]
fn golden_elastic_self_distance_unbanded_n20_m50() {
    let (data, argvals) = generate_curves(20, 50);
    let dist = elastic_self_distance_matrix(&data, &argvals, 0.0);

    // Structural invariants.
    assert_eq!(dist.shape(), (20, 20));

    // Zero diagonal.
    assert_eq!(dist[(0, 0)], 0.0);
    assert_eq!(dist[(10, 10)], 0.0);
    assert_eq!(dist[(19, 19)], 0.0);

    // Reference checkpoints (selected cells covering diverse curve pairs).
    // Unbanded path: all dp_fill_grid cells visited without corridor restriction.
    const DIST_0_1: f64 = 1.17458127979673499e0;
    const DIST_0_19: f64 = 4.85253341940299066e-1;
    const DIST_9_10: f64 = 8.69320275292602251e-1;
    const DIST_19_18: f64 = 1.56594861489696280e0;
    const DIST_5_15: f64 = 6.69029915915371198e-1;

    assert_rel_close(dist[(0, 1)], DIST_0_1, 1e-10);
    assert_rel_close(dist[(0, 19)], DIST_0_19, 1e-10);
    assert_rel_close(dist[(9, 10)], DIST_9_10, 1e-10);
    assert_rel_close(dist[(19, 18)], DIST_19_18, 1e-10);
    assert_rel_close(dist[(5, 15)], DIST_5_15, 1e-10);

    // Symmetry: dist[i,j] == dist[j,i].
    assert_rel_close(dist[(1, 0)], DIST_0_1, 1e-10);
    assert_rel_close(dist[(19, 0)], DIST_0_19, 1e-10);
    assert_rel_close(dist[(10, 9)], DIST_9_10, 1e-10);
    assert_rel_close(dist[(18, 19)], DIST_19_18, 1e-10);
    assert_rel_close(dist[(15, 5)], DIST_5_15, 1e-10);
}

// ─── OPT-01: Banded dp_fill_grid arm (the Plan 02 edit target) ────────────────────────────────────
// Reference constants captured from the PRE-OPT-01 code (Plan 01, 2026-09-23). Tolerance 1e-10:
// OPT-01 is a pure loop-structure refactor — DP arithmetic and accumulation order are unchanged.
//
// band_frac=0.1 → band_radius = ceil(0.1 * 50) = 5. The corridor restricts each row's column
// scan to [j-band, j+band], reducing work O(m²) → O(m·band). After the Plan 02 refactor, these
// values MUST match to within 1e-10 — a different number means arithmetic was accidentally changed.
#[test]
fn golden_elastic_self_distance_banded_n20_m50() {
    let (data, argvals) = generate_curves(20, 50);
    let dist = elastic_self_distance_matrix_banded(&data, &argvals, 0.0, 0.1);

    // Structural invariants.
    assert_eq!(dist.shape(), (20, 20));

    // Zero diagonal.
    assert_eq!(dist[(0, 0)], 0.0);
    assert_eq!(dist[(10, 10)], 0.0);
    assert_eq!(dist[(19, 19)], 0.0);

    // Reference checkpoints (banded path, band_frac=0.1).
    // These differ from the unbanded constants above: the corridor cuts off far-off-diagonal
    // DP paths, producing a band-approximation of the exact elastic distance.
    const DIST_0_1: f64 = 2.15003440465124251e0;
    const DIST_0_19: f64 = 4.85253341940299066e-1;
    const DIST_9_10: f64 = 1.35275846614977113e0;
    const DIST_19_18: f64 = 2.90036187866837514e0;
    const DIST_5_15: f64 = 8.10038567393554620e-1;

    assert_rel_close(dist[(0, 1)], DIST_0_1, 1e-10);
    assert_rel_close(dist[(0, 19)], DIST_0_19, 1e-10);
    assert_rel_close(dist[(9, 10)], DIST_9_10, 1e-10);
    assert_rel_close(dist[(19, 18)], DIST_19_18, 1e-10);
    assert_rel_close(dist[(5, 15)], DIST_5_15, 1e-10);

    // Symmetry: dist[i,j] == dist[j,i].
    assert_rel_close(dist[(1, 0)], DIST_0_1, 1e-10);
    assert_rel_close(dist[(19, 0)], DIST_0_19, 1e-10);
    assert_rel_close(dist[(10, 9)], DIST_9_10, 1e-10);
    assert_rel_close(dist[(18, 19)], DIST_19_18, 1e-10);
    assert_rel_close(dist[(15, 5)], DIST_5_15, 1e-10);
}
