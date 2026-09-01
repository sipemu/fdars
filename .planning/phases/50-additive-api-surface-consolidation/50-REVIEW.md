---
phase: 50-additive-api-surface-consolidation
reviewed: 2026-09-01T00:00:00Z
depth: deep
files_reviewed: 19
files_reviewed_list:
  - fdars-core/examples/21_function_on_scalar/main.rs
  - fdars-core/src/boosting_regression/mod.rs
  - fdars-core/src/depth/fraiman_muniz.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/depth/modal.rs
  - fdars-core/src/depth/random_projection.rs
  - fdars-core/src/depth/random_tukey.rs
  - fdars-core/src/depth/tests.rs
  - fdars-core/src/dim.rs
  - fdars-core/src/fdata.rs
  - fdars-core/src/function_on_scalar.rs
  - fdars-core/src/inference/anova.rs
  - fdars-core/src/inference/permutation.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/src/regression.rs
  - fdars-core/tests/equivalence_phase50.rs
  - fdars-core/tests/validate_against_r.rs
  - fdars-core/tests/validate_new_modules.rs
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: clean
---

# Phase 50: Code Review Report

**Reviewed:** 2026-09-01
**Depth:** deep
**Files Reviewed:** 19
**Status:** clean

## Summary

Phase 50 (Additive API-Surface Consolidation) was reviewed adversarially with the
primary hypothesis that an "additive-only" refactor had quietly broken a public
signature, mis-forwarded a dispatcher, or subtly altered the `fanova` RNG stream.
None of those materialized. The diff is disciplined and the additive-only rule is
**fully upheld** — every hunk is a pure addition (new fn, new `impl Default`, new
`Dim` enum, new dispatchers, `#[deprecated]` attributes, `#[allow(deprecated)]`
guards, docs). No public function, struct, field, or enum variant was renamed,
removed, or had its signature changed.

Key verifications performed:

1. **Additive-only (BLOCKER class): PASS.** `git diff` of `boosting_regression/mod.rs`
   shows zero removed/changed public items. `fanova`'s signature
   `fn fanova(&FdMatrix, &[usize], usize) -> Result<FanovaResult, FdarError>` is
   byte-identical pre/post — it gained only `#[deprecated]`. All old re-export names
   (`fraiman_muniz_2d`, `modal_2d`, `random_projection_2d`, `random_tukey_2d`,
   `mean_2d`, `fanova`) remain re-exported from `lib.rs`/`depth/mod.rs`/`prelude.rs`.
   Confirmed NO `_1d` variant was deprecated, and `functional_spatial_2d`, `deriv_*`,
   `geometric_median_*`, `_nd` families were NOT touched. Exactly 6 source-level
   `#[deprecated]` items were introduced (the 5 `_2d` shims + `fanova`), matching the plan.

2. **fanova bit-identity: PASS.** Diffed the old inline `fanova` body against the new
   `fanova_seeded` body line-by-line: identical except `let mut rng_state: u64 = seed;`
   (was `= 42`), and the deprecated `fanova` shim delegates with literal `42`. LCG
   multiplier `6_364_136_223_846_793_005`, increment `1`, and extraction
   `(rng_state >> 33) as usize % (i + 1)` are preserved verbatim; Fisher-Yates loop
   ordering, `n_ge` accumulation, and the `(n_ge+1)/(n_perm+1)` p-value are unchanged.
   The golden suite (`equivalence_phase50.rs`) pins this with `assert_eq!` (not tolerance)
   on both `global_statistic` and the seed-dependent `p_value`, plus a seed-7
   divergence check — a sound design.

3. **Dispatcher correctness: PASS.** All 5 dispatchers are single-arm
   `Dim::One | Dim::Two => name_1d(...)`. Argument order/types forwarded to each `_1d`
   primitive match exactly (verified against the `_1d` signatures): `modal(obj,ori,h)`,
   `fraiman_muniz(obj,ori,scale)`, `random_projection(obj,ori,nproj)`,
   `random_tukey(obj,ori,nproj)`, `mean(data)`. The RNG dispatchers are correctly
   verified structurally (`len == n_obs` and every value ∈ [0,1]) rather than with
   `assert_eq!`, which is the right call given `thread_rng()` non-determinism.

4. **Deprecation hygiene: PASS.** All 20 `#[allow(deprecated)]` sites are outer
   attributes scoped to a specific re-export statement, a specific `use`, or a specific
   test fn — there are NO blanket `#![allow(deprecated)]` inner attributes anywhere.
   `test_functional_spatial_2d_delegates` correctly has NO allow (that fn is not
   deprecated), demonstrating the guards are minimal and not over-broad. Both feature
   configs (`linalg,parallel` and `--no-default-features --features linalg`) build clean;
   the changed example (`function_on_scalar`) also builds warning-free after its
   migration to `fanova_seeded`.

5. **Default values: PASS.** All three `impl Default` bodies match their struct's
   field-level doc-comments and doc-block prose (BoostingConfig mstop:100/nu:0.1/
   nbasis:10/order:4/lfd_order:2/lambda:1.0/ncomp_x:3/seed:0; BayesianConfig ncomp:4/
   tau2:100.0/IG(0.001,0.001)/n_iter:400/burn_in:200/thin:1/seed:0; StabilityConfig
   n_resamples:100/pi_thr:0.9/seed:0). A dedicated inline test asserts these exact
   values. No obviously-wrong default. (The existing test-helper constructors use
   smaller fast-test values — mstop:30, nbasis:8 — but those are test-speed fixtures,
   not the intended defaults; the Default impls correctly follow the documented
   FDboost/Meinshausen-Bühlmann conventions instead.)

Only two low-severity Info observations follow; neither blocks shipping.

## Info

### IN-01: `since = "0.30.0"` is a forward-looking version while crate is at 0.29.0

**File:** `fdars-core/src/function_on_scalar.rs:913`, `fdars-core/src/fdata.rs:203`, `fdars-core/src/depth/{modal,fraiman_muniz,random_projection,random_tukey}.rs`
**Issue:** All six `#[deprecated(since = "0.30.0", …)]` attributes cite `0.30.0`, but
`fdars-core/Cargo.toml` currently declares `version = "0.29.0"`. This is intentional
and correct (the deprecations take effect in the next release), but it creates a soft
coupling: if this phase ships in a release that is NOT cut as `0.30.0` (e.g. a `0.29.1`
patch or a jump to `0.31.0`), the `since` string will be inaccurate and could mislead
downstream users reading the deprecation notice.
**Fix:** No code change required now. At release time, verify the milestone version
bump lands on `0.30.0`; if the version target changes, sweep the six `since` strings to
match. Consider adding this to the milestone release checklist.

### IN-02: `BayesianConfig::default` doc claims it "mirrors `bayesian::tests::default_config`" but the seed differs

**File:** `fdars-core/src/boosting_regression/mod.rs:113-115`
**Issue:** The doc comment says the defaults mirror `bayesian::tests::default_config`.
That is true for every field EXCEPT `seed` — the test constructor uses
`seed: 20260824` whereas `Default` uses `seed: 0`. The `seed: 0` choice is correct and
consistent with the sibling `Default` impls, but the "mirrors" wording slightly
overstates the match and could confuse a reader who diffs the two.
**Fix:** Tighten the doc to "mirrors `bayesian::tests::default_config` (except `seed`,
which defaults to 0)" for precision. Cosmetic only.

---

_Reviewed: 2026-09-01_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
