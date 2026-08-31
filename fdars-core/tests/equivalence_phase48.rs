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

use fdars_core::coclustering::{co_cluster, CoClusterConfig};
use fdars_core::frechet::frechet_anova;
use fdars_core::matrix::FdMatrix;

/// Deterministic two-latent-row-group data for co_cluster goldens (no RNG). Row group `i % 2`
/// shifts the sinusoid phase/amplitude; columns carry a smooth argument-point structure.
fn co_cluster_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let grp = i % 2;
        let phase = if grp == 0 { 0.0 } else { 0.35 } + 0.05 * ((i as f64 * 1.7).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 0.9).sin()) + if grp == 0 { 0.0 } else { 0.5 };
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * (argvals[j] + phase)).sin();
        }
    }
    (FdMatrix::from_column_major(data, n, m).unwrap(), argvals)
}

fn co_cluster_config(n_init: usize) -> CoClusterConfig {
    // CoClusterConfig is #[non_exhaustive] — build from Default and set the fields we pin.
    let mut cfg = CoClusterConfig::default();
    cfg.n_row_blocks = 2;
    cfg.n_col_blocks = 2;
    cfg.ncomp = 3;
    cfg.max_iter = 50;
    cfg.tol = 1e-6;
    cfg.n_init = n_init;
    cfg.seed = 42;
    cfg
}

// ─── co_cluster n_init parallelization (src/coclustering.rs) ───────────────────────────────────
// References captured from PRE-parallel sequential co_cluster; must remain bit-identical after the
// parallel n_init map + SEQUENTIAL strict-`>` reduce (lowest-init-index tie-break), under both
// feature configs. Row labels alternate by latent group; col labels are the best-fit partition.
#[test]
fn golden_co_cluster_parallel() {
    // n_init=4 is ABOVE CO_CLUSTER_INIT_PARALLEL_THRESHOLD → parallel branch.
    let (data, argvals) = co_cluster_data(60, 40);
    let r = co_cluster(&data, &argvals, &co_cluster_config(4)).unwrap();
    assert_eq!(r.log_likelihood, 4.34232518676733207e2);
    let row_ref: Vec<usize> = (0..60).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    assert_eq!(r.row_labels, row_ref);
    let col_ref = vec![
        1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    assert_eq!(r.col_labels, col_ref);
}

#[test]
fn golden_co_cluster_below_threshold() {
    // n_init=2 is BELOW the threshold → sequential branch.
    let (data, argvals) = co_cluster_data(60, 40);
    let r = co_cluster(&data, &argvals, &co_cluster_config(2)).unwrap();
    assert_eq!(r.log_likelihood, 4.15588733019948734e2);
    let row_ref: Vec<usize> = (0..60).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    assert_eq!(r.row_labels, row_ref);
    let col_ref = vec![
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(r.col_labels, col_ref);
}

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
