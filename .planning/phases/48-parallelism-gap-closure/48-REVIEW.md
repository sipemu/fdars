---
phase: 48-parallelism-gap-closure
reviewed: 2026-08-31T12:49:01Z
depth: deep
files_reviewed: 5
files_reviewed_list:
  - fdars-core/src/frechet/anova.rs
  - fdars-core/src/coclustering.rs
  - fdars-core/tests/equivalence_phase48.rs
  - fdars-core/benches/perf_parallelism.rs
  - fdars-core/Cargo.toml
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: clean
---

# Phase 48: Code Review Report

**Reviewed:** 2026-08-31T12:49:01Z
**Depth:** deep
**Files Reviewed:** 5
**Status:** clean

## Summary

Phase 48 parallelizes two hot paths — the `frechet_anova` permutation loop and the
`co_cluster` multi-restart CEM loop — behind payback thresholds, plus adds a permanent
thread-scaling benchmark and bit-identical golden tests. This is a declared
BEHAVIOR-PRESERVING milestone, so the review focused on parallel/sequential result
equivalence, RNG determinism, reduction order-independence, `collect::<Result>` error
semantics, and Send/Sync soundness.

I traced every correctness-critical property and found no bug, no concurrency hazard, and no
security issue. The two parallelizations are genuinely order-independent, and the code changes
preserve output bit-for-bit. Only two minor INFO-level observations, neither a defect.

Detailed verification of the load-bearing claims:

**1. frechet_anova permutation count — order-independent (VERIFIED).** Each permutation reseeds
`StdRng::seed_from_u64(seed.wrapping_add(perm))`, so every `count_ge(perm)` is a pure function of
`perm` with no cross-iteration state. The reduction is an integer `.sum()`, which is
associative/commutative and identical under `Iterator::sum` (sequential branch / feature off) and
`ParallelIterator::sum` (parallel branch). Bit-identical across thread counts confirmed by
construction. The `if let Ok { if tn_perm >= tn_obs }` → `match { Ok(..) if tn_perm >= tn_obs => 1, _ => 0 }`
rewrite is exactly equivalent: both the `Err` arm and the `Ok`-with-`tn_perm < tn_obs` arm yield 0,
matching the original "skip degenerate permutation conservatively" semantics.

**2. co_cluster reduction — tie-break preserved (VERIFIED).** Original: `best=None`, first result
always taken, replaced only on strict `>`; winner = lowest index among max log_likelihood. New:
`results` is collected in index order, then `into_iter().reduce(|acc, r| if r.ll > acc.ll { r } else { acc })`
folds left-to-right, keeping `acc` (earlier index) unless strictly greater — same winner. NaN and
`NEG_INFINITY` (empty `per_iter_ll`, e.g. `max_iter=0`) log-likelihoods both compare false under `>`,
so the earliest index is kept in both old and new code. Equivalent under all value classes.

**3. collect::<Result<Vec,_>> ordering (VERIFIED).** `Range<usize>` is an `IndexedParallelIterator`
in rayon 1.11; `FromParallelIterator` for `Result<Vec<T>, E>` preserves index order and returns an
`Err`. The comment claims it returns "the first Err in iteration order" — rayon does not strictly
guarantee lowest-index Err selection under scheduling, BUT this is moot here: `run_init`'s only
fallible call is `kmeans_fd`, whose only error paths are input validation (empty matrix, `k==0`,
`k>n`, `argvals` length) that depend solely on `data`/`argvals`/`k_blocks` — all identical across
every `init`. If one init errors, all inits produce the *same* error value, so error-selection
order is unobservable. No behavior change. (See IN-01 for the wording nuance.)

**4. Send/Sync soundness (VERIFIED).** `co_cluster::run_init` captures only shared refs
(`&FdMatrix`, `&[f64]`, `&DMatrix`, `&CoClusterConfig`) and `Copy` scalars — all `Sync`;
`CoClusterResult` (Vecs + f64 + block params) is `Send`. `frechet_anova::count_ge` captures shared
refs / `Copy` and returns `usize`. No interior mutability, no shared mutable state. Sound.

**5. Feature-gating (VERIFIED).** `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;`
is required by the `.map()/.collect()/.sum()` chain on the parallel branch and is cfg'd out when
`parallel` is off, so no unused-import warning in either configuration. `iter_maybe_parallel!`
supplies `IntoParallelIterator`. Below-threshold and feature-off paths use `std::iter` exclusively.

**6. Threshold dispatch (VERIFIED).** In `frechet_anova`, `n_perm==0` is defaulted to 999 (line 170)
*before* the threshold check (line 189), so a `0` request correctly takes the parallel branch — and
the golden test at `n_perm=999` exercises it. Below-threshold (`n_perm=50`) golden exercises the
sequential path. Both `co_cluster` branches (`n_init=4` parallel, `n_init=2` sequential) are
golden-covered. Test scaffolding correctly builds the `#[non_exhaustive]` `CoClusterConfig` via
`Default` + field assignment.

## Info

### IN-01: `collect::<Result>` comment overstates rayon's Err-ordering guarantee

**File:** `fdars-core/src/coclustering.rs:974-976`
**Issue:** The comment states `collect::<Result<Vec,_>>()` "yields the first Err in iteration order —
the same error the sequential `?` propagated." Rayon's parallel `Result` collection returns *an*
`Err` but does not contractually guarantee it is the lowest-index one under arbitrary scheduling.
This is harmless in practice here because `run_init`'s only fallible operation (`kmeans_fd`) fails
only on init-independent input validation, so all inits yield the identical error value — but the
comment's stated rationale ("first Err in iteration order") is stronger than what rayon promises and
could mislead a future maintainer who adds an init-dependent fallible call inside `run_init`.
**Fix:** Tighten the comment to reflect the actual invariant, e.g.: "collect yields an Err if any
init fails; here `kmeans_fd`'s errors are init-independent input-validation failures, so every init
produces the identical error value and Err-selection order is unobservable. If an init-dependent
fallible call is ever added here, re-evaluate error-selection determinism."

### IN-02: Threshold constants are `pub(crate)` but referenced only within their own module

**File:** `fdars-core/src/coclustering.rs:55`, `fdars-core/src/frechet/anova.rs:25`
**Issue:** `CO_CLUSTER_INIT_PARALLEL_THRESHOLD` and `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD` are
declared `pub(crate)` but each is used only at its own dispatch site in the same file (tests/benches
reference them in comments only). The broader visibility is unnecessary.
**Fix:** Demote to a plain module-private `const` unless a future cross-module consumer (e.g. a test
importing the constant to derive the branch boundary) is anticipated. Purely a scope-hygiene nit; no
functional impact and no dead-code warning (each is used within the crate).

---

_Reviewed: 2026-08-31T12:49:01Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
