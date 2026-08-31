---
phase: 49-code-consolidation-dedup
reviewed: 2026-08-31T00:00:00Z
depth: deep
files_reviewed: 22
files_reviewed_list:
  - fdars-core/src/distributions.rs
  - fdars-core/src/inference/dist.rs
  - fdars-core/src/inference/permutation.rs
  - fdars-core/src/spm/chi_squared.rs
  - fdars-core/src/spm/bootstrap.rs
  - fdars-core/src/spm/mod.rs
  - fdars-core/src/regression.rs
  - fdars-core/src/pace_fpca.rs
  - fdars-core/src/helpers.rs
  - fdars-core/src/permutation_test.rs
  - fdars-core/src/frechet/anova.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/alignment/generative.rs
  - fdars-core/src/elastic_explain.rs
  - fdars-core/src/explain/importance.rs
  - fdars-core/src/famm.rs
  - fdars-core/src/function_on_scalar.rs
  - fdars-core/src/outliers.rs
  - fdars-core/src/scalar_on_function/bootstrap.rs
  - fdars-core/src/tolerance/equivalence.rs
  - fdars-core/src/tolerance/fpca.rs
  - fdars-core/tests/equivalence_phase49.rs
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 49: Code Review Report

**Reviewed:** 2026-08-31
**Depth:** deep (cross-file, behavior-preservation focus)
**Files Reviewed:** 22
**Status:** issues_found (2 WARNING, 2 INFO — no BLOCKERs)

## Summary

Phase 49 is a behavior-preserving consolidation: four hand-rolled patterns (χ²/gamma
special functions, SVD sign-decision, per-thread RNG seeding, permutation-test scaffold)
were pulled into shared `pub(crate)` homes and call sites migrated. I reviewed every
migrated site against its pre-refactor source, verified the tail-policy split, empirically
re-derived the permutation gather-contract, and audited the newly reachable public surface.

The consolidation is **sound and behavior-preserving**. Every claim in the module docs
checks out:

- **χ²/gamma tail split correctly preserved.** `chi2_sf` keeps its OWN Q-direct continued
  fraction (`tiny = 1e-300`, inline `1e-15`, no underflow guard) and does NOT route through
  `1 − P`; the CDF family keeps the P-direct path (`tiny = 1e-30`, `eps = 1e-14`, `−700`
  guard). No constant changed between old kernels and new. The `ln_gamma` leading coefficient
  differs textually between the two old copies (`...809_93` vs `...809_9`) but parses to the
  **bit-identical** f64 `0x3feffffffffff950` (verified). The far-tail linchpin
  (χ²-SF x=70.59,k=1 = 4.397e-17 vs the `1−P` cliff's 0.0) is locked by `assert_eq!` goldens.
- **Permutation gather-contract is bit-identical.** Empirically re-ran 1000 permutations
  with an uneven-group label vector: `group_labels.to_vec().shuffle()` (old) vs
  `(0..n).shuffle()` then `gather group_labels[perm_idx[i]]` (new) → **0 mismatches**.
  Fisher–Yates depends only on length + RNG draws, so the position-permutation is identical
  and the gather reproduces the old shuffled vector exactly. No off-by-one, no double-shuffle.
- **Determinism preserved.** `seed_for_thread` body is exactly
  `StdRng::seed_from_u64(seed.wrapping_add(k as u64))`; all 14 migrated sites pass the correct
  `(seed, k)` with no offset change. `rng_stream_seed_for_thread_bit_identical` golden confirms.
- **SVD sign-decision core** is a verbatim extraction; both the two-matrix (`fix_svd_signs`)
  and single-matrix (`pace_fpca`) sites gate from it with matching row bounds. Goldens lock both.
- **No genuine new public API.** `distributions` and `permutation_test` are `pub(crate)`;
  `spm::chi_squared` widened `pub(super)→pub(crate)` but stays crate-internal (unreachable
  externally). The only externally-reachable additions are the `#[doc(hidden)]`
  `__equivalence_test_support` forwarders (see WR-01).

The two WARNINGs below are risk call-outs, not correctness defects: neither changes any
observable result for realistic inputs.

## Warnings

### WR-01: New externally-reachable (doc-hidden) public API surface — `__equivalence_test_support`

**File:** `fdars-core/src/lib.rs:145-232`
**Issue:** The consolidation adds a genuinely new `pub mod __equivalence_test_support` tree
(`current`, `distributions`, `helpers` submodules with 13 `pub fn` forwarders). It is
`#[doc(hidden)]`, but `#[doc(hidden)]` restricts documentation visibility ONLY — it does NOT
restrict language visibility. These `pub fn`s ARE callable by any external crate
(`fdars_core::__equivalence_test_support::distributions::chi2_sf(...)`, etc.). This was the
mechanism the golden harness needed to reach `pub(crate)` items from the external
`tests/equivalence_phase49.rs` integration crate.

For the "additive-only / no public signature change" milestone rule this is *additive* (nothing
removed or altered) and the leading `__` + `#[doc(hidden)]` signals non-stability, so it is an
acceptable escape hatch. But it is a real, permanent widening of the externally-reachable
surface and should be tracked: external code CAN bind to it, and nothing yet prevents it from
outliving the Phase-49 test that justified it.

**Fix:** Acceptable as-is per the "doc(hidden) is acceptable but call it out" rule. To reduce
long-term surface, consider one of:
- Gate the module behind a dedicated internal test feature so it is absent from normal builds:
  ```rust
  #[doc(hidden)]
  #[cfg(any(test, feature = "__internal_equivalence_tests"))]
  pub mod __equivalence_test_support { /* ... */ }
  ```
  (and add `required-features` / a dev-only feature for `tests/equivalence_phase49.rs`).
- At minimum, add a tracking note so this module is removed when the Phase-49 goldens are
  retired or folded into an in-crate `#[cfg(test)]` location.

### WR-02: `wrapping_add` unifies two pre-refactor seed forms — debug-overflow behavior change at extreme seeds

**File:** `fdars-core/src/helpers.rs:29-32` (helper); call sites in `tolerance/equivalence.rs`,
`tolerance/fpca.rs`, `alignment/generative.rs`
**Issue:** The 14 migrated sites previously used TWO forms: `seed + b as u64` (e.g.
`tolerance/equivalence.rs`, `tolerance/fpca.rs`, `alignment/generative.rs`) and
`seed.wrapping_add(b as u64)` (e.g. `outliers.rs`, `scalar_on_function/bootstrap.rs`,
`elastic_explain.rs`). `seed_for_thread` uses `wrapping_add` for all. For every non-overflowing
input this is bit-identical (verified). The only divergence is for the former `seed + b` sites
when `seed + k` would overflow `u64`: the old code would **panic in debug builds** (overflow
check) and wrap in release; the new code always wraps silently. This is a behavior change for
`seed` values within `k` of `u64::MAX` in debug builds only.

**Fix:** No action required — this is practically unreachable (seeds are small in all call
sites and tests) and the doc comment already documents the intent. Noted for completeness so it
is a conscious decision rather than an accident. If strict bit-for-bit debug parity with the old
`seed + b` sites were ever required, those specific sites would need the panicking form — but
that is undesirable and the `wrapping_add` unification is the correct choice.

## Info

### IN-01: Permutation closure bound is `Sync` (not `Send`) — correct, but worth documenting

**File:** `fdars-core/src/permutation_test.rs:52-67`
**Issue:** `permutation_pvalue`'s `F: Fn(&[usize]) -> f64 + Sync` bound omits `Send`. This is
correct for rayon's `map` (the closure is shared by reference across worker threads, so `Sync`
suffices; `Send` is not required because the closure itself is not moved per-thread). The
`frechet_anova` closure captures only plain data (`WassersteinDensitySpace`, `Vec<Vec<f64>>`,
`&[usize]`, `usize`, `f64`) with no interior mutability, so it is `Sync`. Compiles clean under
both feature configs (session reports clippy/fmt green, 2586 tests both configs).
**Fix:** None needed. Optionally add a one-line note on the bound explaining why `Send` is
intentionally absent, to pre-empt a future "should this be `+ Send`?" question.

### IN-02: Documented-and-excluded permutation/seed sites are correctly justified

**File:** `fdars-core/src/inference/permutation.rs:173,238`; `fdars-core/src/explain/importance.rs`
(4 sites); `fdars-core/src/famm.rs:874`; `fdars-core/src/function_on_scalar.rs:835`;
`fdars-core/src/frechet/anova.rs:270` (second loop)
**Issue:** Six permutation sites and the frechet generic-`MetricSpace` second loop were left
un-migrated with explanatory comments. Each exclusion is legitimate and behavior-preserving:
- `t_perm_test` / `f_perm_test` / `explain::importance` (×4) / `famm` use a SINGLE advancing
  `StdRng` — a per-perm reseed WOULD change their p-values, so migrating would be
  behavior-changing. Correctly deferred.
- `function_on_scalar::fanova` uses a hardcoded-42 LCG with no `seed` param — outside the
  scaffold's `StdRng` contract. Correctly excluded.
- `frechet_anova_space` (generic) is sequential with no Phase-48 golden backstop; migrating
  would introduce parallelism with no bit-identity guard. Correctly left inline.
**Fix:** None. These are correct scoping decisions; flagged only to confirm they were reviewed
and are not oversights. The advancing-RNG constraint is the real reason each cannot be folded in
without a re-baselined golden.

---

_Reviewed: 2026-08-31_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
