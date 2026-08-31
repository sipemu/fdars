//! The single authoritative permutation-test scaffold (CONS-02, Plan A).
//!
//! Consolidates the per-permutation `reseed → shuffle → recompute → count →
//! (1+n_ge)/(1+n_perm)` pattern into ONE `pub(crate)` helper,
//! [`permutation_pvalue`]. The scaffold is **feature-gated parallel** (threshold
//! owned by the caller) and **bit-identical** across thread counts and feature
//! configs because every permutation reseeds its own RNG via
//! [`crate::helpers::seed_for_thread`] — so the parallel `.sum()` reduction is
//! order-independent and equals the sequential one.
//!
//! # Scope
//!
//! This module is deliberately named `permutation_test` to avoid colliding with
//! the existing [`crate::inference::permutation`] submodule. It is the CONS-02
//! consolidation of Plan A: only `frechet_anova`'s primary loop migrates onto it,
//! because that is the single site whose loop already matches the contract
//! (per-permutation reseeded `StdRng`, threshold-gated parallel,
//! `(1+n_ge)/(1+n_perm)`). The advancing-single-RNG sites (`t_perm_test`,
//! `f_perm_test`, `explain/importance`, `famm`) and the fixed-42 LCG site
//! (`function_on_scalar::fanova`) are documented-and-excluded — migrating them
//! WOULD change their p-values.
//!
//! # Draw-application contract
//!
//! The helper draws the permutation as a shuffled `Vec<usize>` of `0..n` and
//! hands the closure `&perm_idx` (a permutation *of positions*). The caller
//! applies that permutation itself — typically by GATHERING its own per-position
//! data through `perm_idx`. Because the helper applies exactly ONE
//! `SliceRandom::shuffle` (Fisher–Yates) to a length-`n` slice under the same
//! per-permutation seed a hand-rolled loop would use, `perm_idx` is the same
//! position-permutation that loop's in-place `shuffle` produced — so gathering
//! through it reproduces the old shuffled vector bit-for-bit.

use crate::helpers::seed_for_thread;
use crate::iter_maybe_parallel;
use rand::seq::SliceRandom;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute a seeded permutation p-value `(1 + n_ge) / (1 + n_perm)`.
///
/// For each `perm` in `0..n_perm`, reseeds an `StdRng` at `seed + perm` (via
/// [`seed_for_thread`]), shuffles a fresh `Vec<usize>` of `0..n` in place, and
/// passes the resulting position-permutation `&perm_idx` to `stat`. Counts how
/// many permutation statistics are `>= observed`, then returns
/// `(n_ge + 1) / (n_perm + 1)` as an `f64`.
///
/// Dispatch is parallel via `iter_maybe_parallel!` when
/// `n_perm >= threshold` (the caller owns the payback point), else sequential.
/// The reduction is a plain `.sum()`, which is order-independent here because
/// each permutation reseeds — so the parallel and sequential branches, and the
/// `parallel`-on / `parallel`-off feature configs, are all BIT-IDENTICAL.
///
/// A degenerate permutation (e.g. a compute error the caller wants to skip
/// conservatively) is expressed by having `stat` return [`f64::NEG_INFINITY`],
/// so the `>= observed` comparison yields `false` (counts 0).
pub(crate) fn permutation_pvalue<F>(
    observed: f64,
    n: usize,
    n_perm: usize,
    seed: u64,
    threshold: usize,
    stat: F,
) -> f64
where
    F: Fn(&[usize]) -> f64 + Sync,
{
    let count_ge = |perm: usize| -> usize {
        let mut rng = seed_for_thread(seed, perm);
        let mut perm_idx: Vec<usize> = (0..n).collect();
        perm_idx.shuffle(&mut rng);
        usize::from(stat(&perm_idx) >= observed)
    };
    let n_ge: usize = if n_perm >= threshold {
        iter_maybe_parallel!(0..n_perm).map(count_ge).sum()
    } else {
        (0..n_perm).map(count_ge).sum()
    };
    (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel (>= threshold) and sequential (< threshold) branches yield the
    /// SAME count for the same seed, because each permutation reseeds.
    #[test]
    fn parallel_and_sequential_branches_agree() {
        // A deterministic statistic that depends only on perm_idx contents.
        let stat = |idx: &[usize]| -> f64 { idx.iter().map(|&x| x as f64).take(3).sum() };
        let n = 30usize;
        let n_perm = 400usize;
        // threshold=200 → parallel; threshold=usize::MAX → sequential. Same seed.
        let observed = 20.0;
        let p_par = permutation_pvalue(observed, n, n_perm, 42, 200, stat);
        let p_seq = permutation_pvalue(observed, n, n_perm, 42, usize::MAX, stat);
        assert_eq!(p_par, p_seq);
    }

    /// A statistic returning NEG_INFINITY never counts (degenerate-perm skip).
    #[test]
    fn neg_infinity_stat_counts_zero() {
        let stat = |_idx: &[usize]| -> f64 { f64::NEG_INFINITY };
        let p = permutation_pvalue(0.0, 10, 99, 1, 200, stat);
        // n_ge = 0 → (0+1)/(99+1).
        assert_eq!(p, 1.0 / 100.0);
    }

    /// The returned p-value is exactly the rational `(1+n_ge)/(1+n_perm)`.
    #[test]
    fn pvalue_is_exact_rational() {
        // stat >= observed for every perm → n_ge = n_perm.
        let stat = |_idx: &[usize]| -> f64 { 1.0 };
        let p = permutation_pvalue(0.0, 5, 49, 7, 200, stat);
        assert_eq!(p, (49.0 + 1.0) / (49.0 + 1.0));
    }
}
