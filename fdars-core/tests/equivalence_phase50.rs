//! Permanent golden-equivalence tests for Phase 50 (Additive API-Surface Consolidation).
//!
//! ADDITIVE / BEHAVIOR-PRESERVING phase: new seeded entry points are introduced alongside the
//! existing functions, and the existing functions become `#[deprecated]` shims that delegate to the
//! new seeded form with the LEGACY fixed seed. Each golden reference is the CURRENT (pre-change) f64
//! output captured as a `const` from the code that shipped before the seeded API existed. The new
//! seeded function called with the legacy seed, AND the deprecated shim, must reproduce every value
//! **BIT-IDENTICALLY** — `assert_eq!`, NOT tolerance — because the shim performs the exact same
//! arithmetic on the exact same RNG stream. The suite must pass under BOTH
//! `--features linalg,parallel` AND `--no-default-features --features linalg`.
//!
//! ── Plan 50-02: `fanova` / `fanova_seeded` ──────────────────────────────────────────────────────
//! CRITICAL (RESEARCH pitfall #3): `fanova` does NOT use `StdRng` — it uses a hand-rolled LCG seeded
//! with a hardcoded 42 (multiplier 6_364_136_223_846_793_005, increment 1, bit-extract
//! `(rng_state >> 33) as usize % (i+1)`). `fanova_seeded` keeps that EXACT LCG and changes ONLY the
//! seed source. `p_value` is the linchpin: it is seed-DEPENDENT (driven by the permutation stream),
//! so a bit-identical `p_value` from `fanova_seeded(…, 42)` proves the LCG stream is preserved.
//! `global_statistic` is seed-INDEPENDENT (deterministic observed statistic).
//!
//! This file is the shared Wave-2 golden harness for Phase-50 plans; plan 50-03 appends its own
//! goldens (the depth/fdata `_2d` shims) to it.

#![allow(clippy::excessive_precision)]

use fdars_core::matrix::FdMatrix;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// FANOVA seedable-permutation goldens (API-01/API-02, plan 50-02). Captured from the CURRENT
// (pre-change) `fanova(&data, &groups, n_perm)` with its hardcoded-42 LCG on the deterministic
// fixture below (n_perm=199, matching the existing integration tests). `fanova_seeded(…, 42)` and
// the `#[deprecated]` `fanova` shim must both reproduce these bits EXACTLY under BOTH feature configs.
// ════════════════════════════════════════════════════════════════════════════════════════════════

const FANOVA_N_PERM: usize = 199;
// global_statistic — seed-INDEPENDENT integrated observed F.
const FANOVA_GLOBAL_STATISTIC: f64 = 42.36027574578335;
// p_value — seed-DEPENDENT (LCG permutation stream). The linchpin proving the LCG is preserved.
const FANOVA_P_VALUE_SEED42: f64 = 0.11;
// p_value with a DIFFERENT seed (7): must differ from the seed-42 value (proves the seed threads
// into the LCG) while the seed-independent global_statistic stays identical.
const FANOVA_P_VALUE_SEED7: f64 = 0.105;

/// Deterministic two-group functional fixture (n=6 curves × m=5 points) with a clear group effect,
/// driving `fanova`. Fixed literals — no RNG, no time dependence — so the goldens are reproducible.
fn fanova_fixture() -> (FdMatrix, Vec<usize>) {
    // 6 observations, 5 grid points. Group 0 = rows 0..3 (low), group 1 = rows 3..6 (high).
    // Column-major flatten: element (i, j) at index i + j * n.
    let n = 6usize;
    let m = 5usize;
    let mut cm = vec![0.0f64; n * m];
    for j in 0..m {
        let t = j as f64 / (m - 1) as f64;
        for i in 0..n {
            let g = if i < 3 { 0.0 } else { 1.5 };
            // deterministic per-(i,j) value with a group-level shift
            let v = (t * (i as f64 + 1.0)).sin() + g + 0.1 * (i as f64) * t;
            cm[i + j * n] = v;
        }
    }
    let data = FdMatrix::from_column_major(cm, n, m).unwrap();
    let groups = vec![0usize, 0, 0, 1, 1, 1];
    (data, groups)
}

/// The deprecated `fanova` shim MUST reproduce the CURRENT (pre-change) output bit-identically — it
/// delegates to `fanova_seeded(…, 42)` with the legacy fixed seed. `#[allow(deprecated)]` because
/// this test deliberately exercises the deprecated path to pin its output.
#[allow(deprecated)]
#[test]
fn fanova_shim_seed42_bit_identical() {
    use fdars_core::function_on_scalar::fanova;
    let (data, groups) = fanova_fixture();
    let r = fanova(&data, &groups, FANOVA_N_PERM).unwrap();
    assert_eq!(r.global_statistic, FANOVA_GLOBAL_STATISTIC);
    assert_eq!(r.p_value, FANOVA_P_VALUE_SEED42);
    assert_eq!(r.n_perm, FANOVA_N_PERM);
}

/// `fanova_seeded(…, 42)` — the new seeded entry point with the legacy seed — must reproduce the
/// SAME bits as the deprecated shim (the LCG stream is preserved verbatim). Not deprecated, so no
/// `#[allow(deprecated)]` needed.
#[test]
fn fanova_seeded_seed42_bit_identical() {
    use fdars_core::function_on_scalar::fanova_seeded;
    let (data, groups) = fanova_fixture();
    let r = fanova_seeded(&data, &groups, FANOVA_N_PERM, 42).unwrap();
    assert_eq!(r.global_statistic, FANOVA_GLOBAL_STATISTIC);
    assert_eq!(r.p_value, FANOVA_P_VALUE_SEED42);
    assert_eq!(r.n_perm, FANOVA_N_PERM);
}

/// A DIFFERENT seed (7) must change the p_value (proving the seed threads into the LCG) while
/// leaving the seed-independent global_statistic identical. This is the sanity check that
/// `fanova_seeded` is not silently ignoring `seed`.
#[test]
fn fanova_seeded_different_seed_changes_pvalue_not_statistic() {
    use fdars_core::function_on_scalar::fanova_seeded;
    let (data, groups) = fanova_fixture();
    let r7 = fanova_seeded(&data, &groups, FANOVA_N_PERM, 7).unwrap();
    // global_statistic is seed-independent — identical across seeds.
    assert_eq!(r7.global_statistic, FANOVA_GLOBAL_STATISTIC);
    // p_value is seed-dependent — differs from seed 42, matches the captured seed-7 golden.
    assert_eq!(r7.p_value, FANOVA_P_VALUE_SEED7);
    assert_ne!(r7.p_value, FANOVA_P_VALUE_SEED42);
}
