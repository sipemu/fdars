---
phase: 20-two-sample-tests-inference-module
plan: 01
subsystem: inference
tags: [functional-inference, permutation-test, hotelling-t2, scb, degras, fpca, two-sample]

# Dependency graph
requires:
  - phase: 16-r-ecosystem-capability-inventory
    provides: "R-parity gap map identifying Inference (Area 5) as 0/22 present, INF-01 prioritized"
provides:
  - "fdars-core/src/inference/ module — first standalone functional-inference surface"
  - "t_perm_test, f_perm_test (permutation two-sample mean tests; fda::tperm.fd / Fperm.fd)"
  - "two_sample_mean_test (Hotelling T² on a shared FPC basis; fda.usc-style mean equality)"
  - "mean_scb, scb_two_sample_test (Degras SCB for the mean / mean difference; SCBmeanfd)"
  - "TestResult { statistic, p_value, n_perm } shared result struct (crate-root re-exported)"
  - "pub(crate) integrated_f_statistic helper factored out of fanova (shared integrated-F core)"
affects: [phase-21-flm-inference, INF-02, INF-03, inference]

# Actuals
actuals:
  tokens: 10863
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []  # no new crate dependencies — statrs deliberately avoided (self-contained chi-square SF)
  patterns:
    - "Reuse-first inference surface: lift existing permutation/Hotelling/Degras machinery, factor shared cores rather than reimplement"
    - "Self-contained chi-square survival function (regularized upper incomplete gamma + Lanczos ln_gamma) to avoid a new dependency"

key-files:
  created:
    - fdars-core/src/inference/mod.rs
    - fdars-core/src/inference/permutation.rs
    - fdars-core/src/inference/hotelling.rs
    - fdars-core/src/inference/scb.rs
  modified:
    - fdars-core/src/function_on_scalar.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "f_perm_test merged forward from planned Task 2 into Task 1's commit so the module compiles clean under -D warnings (no dead public fn)"
  - "two_sample_mean_test converts Hotelling T² to a p-value via a self-contained chi-square(ncomp) upper tail (no statrs dependency added — package-legitimacy gate avoided)"
  - "scb_two_sample_test builds the SCB on a paired difference matrix d[i]=a[i]-b[i] (min(n_a,n_b) rows) and rejects when the simultaneous band excludes zero; p_value is a conservative 0/1 band-decision encoding"
  - "TestResult carries n_perm (0 for the non-permutation Hotelling/SCB paths); no reject bool added"

patterns-established:
  - "Inference two-sample fns share a validate_two_samples entry-check (equal non-zero columns, argvals width, >= 2 rows per sample)"
  - "Deterministic StdRng::seed_from_u64(seed) permutation null with Fisher–Yates relabelling; p=(n_ge+1)/(n_perm+1)"

requirements-completed: [INF-01]

coverage:
  - id: D1
    description: "inference/ module exists and t_perm_test, f_perm_test, two_sample_mean_test, mean_scb, scb_two_sample_test, TestResult are reachable at the crate root"
    requirement: INF-01
    verification:
      - kind: unit
        ref: "fdars-core inline tests inference::* (17 tests) + crate-root reachability compile check"
        status: pass
  - id: D2
    description: "t_perm_test / f_perm_test: p≈0 for clearly-separated samples, large p under the null, deterministic under a fixed seed"
    requirement: INF-01
    verification:
      - kind: unit
        ref: "inference::permutation::tests (t_perm_separated_small_p, t_perm_null_large_p, t_perm_deterministic, f_perm_* incl. f_perm_agrees_with_fanova_decision)"
        status: pass
  - id: D3
    description: "two_sample_mean_test rejects on differing group means and fails to reject when means coincide"
    requirement: INF-01
    verification:
      - kind: unit
        ref: "inference::hotelling::tests (mean_test_differ_rejects, mean_test_coincide_fails_to_reject, chi_square_sf_sane)"
        status: pass
  - id: D4
    description: "mean_scb covers the true mean at ~requested coverage; scb_two_sample_test flags a real mean difference"
    requirement: INF-01
    verification:
      - kind: unit
        ref: "inference::scb::tests (mean_scb_covers_true_mean, scb_two_sample_detects_difference, scb_two_sample_no_difference)"
        status: pass

status: complete
metrics:
  duration: "~1 session"
  completed: 2026-08-16
---

# Phase 20 Plan 01: Two-Sample Functional Tests & `inference/` Module Summary

Delivered fdars' first standalone functional-inference surface — a new `fdars-core/src/inference/` module with two-sample permutation (`t_perm_test`, `f_perm_test`), FPC-basis Hotelling-T² (`two_sample_mean_test`), and Degras SCB (`mean_scb`, `scb_two_sample_test`) tests plus a shared `TestResult` struct, all crate-root re-exported — closing R-parity Inference (Area 5) requirement INF-01 by reusing existing machinery rather than reimplementing it.

## Accomplishments

- **`inference/` module scaffold + `TestResult`** — `mod.rs` with module-level docs naming the R baselines (`tperm.fd`/`Fperm.fd`, `fda.usc`, `SCBmeanfd`); `TestResult { statistic, p_value, n_perm }` (Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`). Wired into `lib.rs` (`pub mod inference;` alphabetical) with a crate-root re-export block.
- **`t_perm_test`** — integrated L2-of-difference two-sample permutation test; sample means, Simpson-weighted integration of the squared mean difference, pooled Fisher–Yates permutation null seeded by `StdRng::seed_from_u64(seed)`, `p=(n_ge+1)/(n_perm+1)`.
- **`f_perm_test`** — integrated-F permutation test (k=2 case of functional ANOVA). Factored the reusable integrated-F core out of `fanova` into `pub(crate) fn integrated_f_statistic` in `function_on_scalar.rs`; both `fanova` and `f_perm_test` call it. `fanova`'s public signature and numerical output are unchanged (its 4 inline tests stay green).
- **`two_sample_mean_test`** — projects both samples onto a shared FPC basis (`fdata_to_pc_1d` on pooled data, `FpcaResult::project` per sample), Hotelling-T² (`spm::stats::hotelling_t2`) on the score-mean difference scaled by `sqrt(n_a·n_b/(n_a+n_b))`, eigenvalues from pooled singular values via `sv²/(n_pooled−1)` (mfpca convention). T² → p-value via a **self-contained chi-square(ncomp) survival function** (regularized upper incomplete gamma via series/continued-fraction + Lanczos `ln_gamma`), so **no new crate dependency** was introduced.
- **`mean_scb`** — thin inference-facing wrapper over `tolerance::scb_mean_degras` (band math not reimplemented).
- **`scb_two_sample_test`** — builds a simultaneous band around the mean difference from a paired difference matrix `d[i]=a[i]−b[i]`, reusing the Degras multiplier bootstrap; rejects when the band excludes zero at any grid point.

## Tests

- **17 inline tests added** (9 permutation + 4 hotelling + 4 scb): separated→p<0.05, null→p>0.1, determinism under fixed seed, `f_perm` agreement with `fanova`'s decision, Hotelling differ→reject / coincide→fail-to-reject, chi-square SF vs known χ² quantiles, `mean_scb` covers the true mean at every grid point, SCB two-sample difference detection, and invalid-input `Err` for all fns.
- **Full suite: 2027 lib tests + all integration/doc tests pass** under `--features linalg,parallel` (0 failures).
- **`cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean** (CI parity — lints test/bench code too).
- Crate-root reachability confirmed: all 6 symbols + `DEFAULT_N_PERM` resolve as `fdars_core::<name>`.

## Deviations from Plan

### Auto-fixed / scope adjustments

**1. [Rule 3 - Blocking] `f_perm_test` merged forward from Task 2 into Task 1's commit**
- **Found during:** Task 1 (module scaffold).
- **Issue:** Implementing only `t_perm_test` while `permutation.rs` also contained a fully-written `f_perm_test` (or leaving `f_perm_test` unexported) would trip `-D dead_code`/broken-doc-link gates; the pre-commit doc gate also rejects intra-doc links to not-yet-defined items.
- **Fix:** Implemented and re-exported `t_perm_test` + `f_perm_test` together in the Task 1 commit (both reuse the shared `integrated_f_statistic` core). Not-yet-defined module-doc links (`two_sample_mean_test`, `mean_scb`, `scb_two_sample_test`) were kept as plain backticks in Task 1 and promoted to doc-links as each item landed in Tasks 2/3.
- **Commit:** 5ef8c8fc

**2. [Rule 3 - Blocking] Self-contained chi-square survival function instead of `statrs`**
- **Found during:** Task 2 (Hotelling p-value).
- **Issue:** The plan suggested `statrs::distribution::ChiSquared`, but `statrs` is not a dependency and there is no in-crate chi-square CDF. Adding a package is excluded from auto-fix (package-legitimacy gate).
- **Fix:** Implemented a self-contained `chi_square_sf` (regularized upper incomplete gamma via series + continued fraction, Lanczos `ln_gamma`), validated in a unit test against known χ²(1)/χ²(2) 5% quantiles (< 1e-3 error). No dependency added.
- **Commit:** 52f8a526

**3. [Rule 1 - Bug] Test noise generator was not zero-mean**
- **Found during:** Task 3 (`mean_scb` coverage test failed 0/40).
- **Issue:** The deterministic test noise `((state>>33)/2^31) - 1.0` spans `[-1, 0)` (always negative, mean ≈ −0.5), biasing the empirical sample mean below the true mean. Harmless for permutation/Hotelling tests (a constant offset cancels in differences and relabelling — those tests passed), but it broke SCB mean-coverage.
- **Fix:** Corrected the scb-module test helper to genuinely zero-mean noise `2u−1` over `[-1, 1)`; used a gentle near-linear mean with a small bandwidth so smoothing bias is negligible and coverage is exact at every grid point.
- **Commit:** da2ea6ee

## Existing public signatures — unchanged (additive/non-breaking)

- `pub fn fanova(data, groups, n_perm) -> Result<FanovaResult, FdarError>` — body refactored to call the new `pub(crate) integrated_f_statistic`; signature and output unchanged (4 inline tests green).
- `pub fn hotelling_t2(scores, eigenvalues) -> Result<Vec<f64>, FdarError>` — unchanged.
- `pub fn scb_mean_degras(...) -> Result<ToleranceBand, FdarError>` — unchanged.
- No existing `pub use` line was altered; only additive re-exports were introduced.

## Commits

- `5ef8c8fc` feat(20-01): inference module scaffold + t_perm_test/f_perm_test permutation tests
- `52f8a526` feat(20-01): two_sample_mean_test (Hotelling T2 on shared FPC basis)
- `da2ea6ee` feat(20-01): mean_scb + scb_two_sample_test (Degras SCB reuse)

## Known Stubs

None — every delivered function is fully wired to real reused machinery; no placeholder/mock data paths.

## Self-Check: PASSED

- All 4 inference source files present on disk.
- All 3 task commits (5ef8c8fc, 52f8a526, da2ea6ee) exist in git history.
- SUMMARY.md present.
