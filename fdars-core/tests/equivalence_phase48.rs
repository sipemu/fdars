//! Permanent golden-equivalence tests for Phase 48 (Parallelism-Gap Closure).
//!
//! BEHAVIOR-PRESERVING parallelism phase: each golden reference is the CURRENT (pre-parallel) seeded
//! output of the target fn, captured as a `const`. The parallelized code must reproduce it
//! **BIT-IDENTICALLY** (`assert_eq!`, NOT tolerance — p-values are exact rationals `(n_ge+1)/(n_perm+1)`
//! and co_cluster reductions are deterministic), and the test must pass under BOTH
//! `--features linalg,parallel` AND `--no-default-features --features linalg`. Determinism holds because
//! each iteration reseeds `StdRng::seed_from_u64(seed + k)`, so output is independent of thread count.

#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use fdars_core::frechet::frechet_anova;
use fdars_core::matrix::FdMatrix;

/// Deterministic two-group **density** data (rows strictly positive — WassersteinDensitySpace).
/// Group 0: rows 0..n_per; Group 1: rows n_per..2*n_per (contiguous 0..2 labels).
fn two_group_densities(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
    let argvals: Vec<f64> = (0..m)
        .map(|j| -3.0 + 6.0 * j as f64 / (m - 1) as f64)
        .collect();
    let n = 2 * n_per;
    let mut data = vec![0.0; n * m];
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let g = if i < n_per { 0 } else { 1 };
        labels[i] = g;
        // Group-shifted Gaussian density (strictly positive).
        let mu = if g == 0 { -0.5 } else { 0.5 } + 0.2 * ((i as f64 * 1.7).sin());
        let sigma = 0.7 + 0.2 * ((i as f64 * 1.3).sin().abs());
        for j in 0..m {
            let z = (argvals[j] - mu) / sigma;
            data[i + j * n] = (-0.5 * z * z).exp() / sigma + 1e-6;
        }
    }
    (
        FdMatrix::from_column_major(data, n, m).unwrap(),
        argvals,
        labels,
    )
}

#[test]
fn helpers_smoke() {
    let (d, a, l) = two_group_densities(6, 20);
    assert_eq!(d.shape(), (12, 20));
    assert_eq!(a.len(), 20);
    assert_eq!(l, vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    let _ = PI;
}

// ─── frechet_anova parallelization (src/frechet/anova.rs) ──────────────────────────────────────
// Reference captured from PRE-parallel sequential code; must remain bit-identical after the
// iter_maybe_parallel! swap AND under both feature configs.
#[test]
fn golden_frechet_anova_parallel() {
    // n_perm=999 is ABOVE the payback threshold → parallel branch.
    let (data, argvals, labels) = two_group_densities(12, 20);
    let r = frechet_anova(&data, &argvals, &labels, 999, 42).unwrap();
    // Captured from pre-parallel sequential frechet_anova; BIT-IDENTICAL after the iter_maybe_parallel!
    // swap and under both feature configs (each perm reseeds seed+perm → order-independent).
    assert_eq!(r.statistic, 1.17320834419366224e3);
    assert_eq!(r.p_value_permutation, 1.00000000000000002e-3);
    assert_eq!(r.p_value_asymptotic, 4.05441402554881990e-257);
}

#[test]
fn golden_frechet_anova_below_threshold() {
    // small n_perm → BELOW threshold → sequential branch.
    let (data, argvals, labels) = two_group_densities(12, 20);
    let r = frechet_anova(&data, &argvals, &labels, 50, 42).unwrap();
    // Below-threshold sequential branch — also bit-identical.
    assert_eq!(r.statistic, 1.17320834419366224e3);
    assert_eq!(r.p_value_permutation, 1.96078431372549017e-2);
}
