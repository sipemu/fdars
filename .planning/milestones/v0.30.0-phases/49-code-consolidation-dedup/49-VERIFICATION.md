---
phase: 49-code-consolidation-dedup
verified: 2026-08-31T00:00:00Z
status: passed
score: 5/5 must-haves verified
gap_closure: >
  The lone gap (thread-offset RNG sites not migrated, breaking the ROADMAP "every call site
  migrated" goal) was CLOSED in two passes. Pass 1 (commit c2444ee5, plan 49-05) migrated
  spm/arl.rs:177/248/301, alignment/shape_ci.rs:121, boosting_regression/stability.rs:137, and
  explain_generic/importance.rs:66. Pass 2 (commit 6e2ecaf1) caught two the first pass missed —
  explain_generic/shap.rs:78 (seed.wrapping_add(i), migrated) and frechet/anova.rs:276
  (seed.wrapping_add(perm) in the document-and-skipped second loop, RNG construction migrated). A
  grep for residual seed_from_u64(seed_field + loop_var) thread-offset patterns is now empty. The
  only remaining plain-seed sequential RNGs (explain/shap.rs:192, explain_generic/importance.rs:151)
  carry in-code rationale comments — different contract (advancing RNG), correctly NOT migrated.
  Both feature configs green (2586 lib tests each), clippy --all-targets clean, all goldens
  bit-identical. Every same-contract call site is now migrated → SC#1/SC#2 satisfied.
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "SC#1/SC#2 (ROADMAP): the per-thread seeded-RNG determinism contract is consolidated with EVERY call site migrated"
    status: partial
    reason: >
      helpers::seed_for_thread(seed,k) exists (pub(crate), body exactly
      StdRng::seed_from_u64(seed.wrapping_add(k as u64))) and 14 sites migrated
      bit-identically — but FIVE genuine same-contract thread-offset RNG sites
      remain hand-rolled. They match the exact StdRng::seed_from_u64(seed_field +
      loop_var as u64) pattern inside iter_maybe_parallel! loops that seed_for_thread
      was created to eliminate. They were excluded on plan-scoping grounds ("not in
      plan 49-03's files_modified"), NOT on any contract difference, and carry NO
      in-code Phase-49 rationale comment. The ROADMAP goal is verbatim "every call
      site migrated" / SC#2 "call sites migrated"; the drift-risk the phase set out
      to kill therefore survives at these 5 sites. No later milestone phase (50 API
      surface, 51 benchmarks) covers them, so this is not a deferral.
    artifacts:
      - path: "fdars-core/src/spm/arl.rs"
        issue: "lines 177, 248, 301 — StdRng::seed_from_u64(config.seed + rep as u64) in iter_maybe_parallel! loop; not migrated to seed_for_thread, no rationale comment"
      - path: "fdars-core/src/alignment/shape_ci.rs"
        issue: "line 121 — StdRng::seed_from_u64(config.seed + b as u64) bootstrap loop; not migrated, no rationale comment"
      - path: "fdars-core/src/boosting_regression/stability.rs"
        issue: "line 137 — StdRng::seed_from_u64(stab_config.seed.wrapping_add(b as u64)) resample loop; not migrated, no rationale comment"
    missing:
      - "Migrate spm/arl.rs:177,248,301 to crate::helpers::seed_for_thread(config.seed, rep) — bit-identical (wrapping_add == + for non-overflow)"
      - "Migrate alignment/shape_ci.rs:121 to seed_for_thread(config.seed, b)"
      - "Migrate boosting_regression/stability.rs:137 to seed_for_thread(stab_config.seed, b)"
      - "OR: add an in-code Phase-49 rationale comment at each site AND a VERIFICATION.md override documenting why these same-contract sites are intentionally left un-migrated (as was done correctly for the 49-04 permutation exclusions)"
      - "Decide explicitly on explain_generic/importance.rs:66 + shap.rs:78 (per-component seed.wrapping_add reseeds): either migrate or add the in-code rationale that 49-04's narrative claims — they currently carry NO in-code Phase-49 comment"
deferred: []
---

# Phase 49: Code Consolidation / Dedup — Verification Report

**Phase Goal:** Duplicated numerical and statistical-test machinery scattered across the v0.19–v0.29 modules is factored into shared `pub(crate)` helpers with every call site migrated — reducing surface area and drift risk while leaving all observable behavior unchanged.
**Verified:** 2026-08-31
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CONS-01 χ²/gamma consolidated into shared pub(crate) primitives; SF keeps Q-direct tail, CDF keeps P-direct tail; call sites migrated bit-identically | ✓ VERIFIED | `distributions.rs` has pub(crate) `ln_gamma`/`reg_gamma_p`/`reg_gamma_q`/`chi2_cdf`/`chi2_quantile`/`chi2_sf` + private SF-tuned CF/series. inference/dist.rs delegates to `chi2_sf`; spm/chi_squared.rs delegates to `distributions::*`; spm/bootstrap.rs:377 uses `reg_gamma_p(0.5, x*x)`. Local gamma kernels removed. gamma goldens (SF+CDF, current+new, incl. far tail) pass. |
| 2 | CONS-01 SVD sign convention lives in ONE pub(crate) decision core; fix_svd_signs + pace_fpca both gate flips from it; signs bit-identical | ✓ VERIFIED | `regression.rs:186 dominant_sign_negative()` (max-abs-index + `<0.0`). `fix_svd_signs` (line 212) and `pace_fpca.rs:223` both call it. Goldens `svd_sign_fpca_two_matrix` + `svd_sign_pace_eigenfunctions_single_matrix` pass. |
| 3 | CONS-02 seeded-RNG determinism contract consolidated into seed_for_thread with EVERY call site migrated; bit-identical; exclusions documented | ✗ FAILED (partial) | Helper exists (`helpers.rs:28`, exact body). 14 sites migrated, rng_stream golden passes. BUT 5 genuine same-contract thread-offset sites (spm/arl.rs x3, shape_ci.rs x1, boosting stability x1) remain hand-rolled, excluded on plan-scoping grounds with no in-code rationale. See Gaps. |
| 4 | CONS-02 permutation scaffold has ONE authoritative pub(crate) permutation_pvalue; frechet_anova primary loop migrated bit-identically; incompatible sites documented-excluded | ✓ VERIFIED | `permutation_test.rs:57 permutation_pvalue<F>` (per-perm reseed via seed_for_thread, threshold-gated parallel, (1+n_ge)/(1+n_perm)). frechet/anova.rs:181 calls it with gather contract. Phase-48 frechet goldens + phase49 frechet golden pass. 5 advancing-RNG/LCG/multi-stat sites + 2nd frechet loop each carry a Phase-49 CONS-02 rationale comment (genuine contract differences: advancing single RNG changes p-values; LCG has no seed param). |
| 5 | SC#3/SC#4: suite green, no public signature change, no numeric output altered, no new dependency | ✓ VERIFIED | All migrated fns retain identical `pub(crate)` signatures. Cargo.toml/Cargo.lock dependency sections unchanged across the phase-49 commit range (only a `[[bench]]` entry, no deps). Suite reported green both configs (2586 lib each); goldens assert_eq! bit-identical. |

**Score:** 4/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/distributions.rs` | shared χ²/gamma primitives + 2 tail wrappers | ✓ VERIFIED | 6 pub(crate) fns + private SF CF/series; wired into 3 consumers |
| `fdars-core/src/regression.rs::dominant_sign_negative` | pub(crate) SVD sign decision core | ✓ VERIFIED | Used by fix_svd_signs + pace_fpca |
| `fdars-core/src/helpers.rs::seed_for_thread` | pub(crate) determinism helper | ✓ VERIFIED (helper) / ⚠️ under-consumed | Exact body; 14 sites wired, 5 same-contract sites still hand-rolled |
| `fdars-core/src/permutation_test.rs::permutation_pvalue` | pub(crate) permutation scaffold | ✓ VERIFIED | Wired into frechet_anova primary loop |
| `fdars-core/tests/equivalence_phase49.rs` | golden harness for all 4 targets | ✓ VERIFIED | 8 goldens (gamma current+new SF+CDF, 2× SVD sign, rng_stream, frechet) — all pass |

### Key Link Verification

| From | To | Via | Status |
|------|-----|-----|--------|
| spm/chi_squared.rs, inference/dist.rs, spm/bootstrap.rs | distributions::* | delegating calls | ✓ WIRED |
| fix_svd_signs + pace_fpca | regression::dominant_sign_negative | gated flip | ✓ WIRED |
| 14 thread-offset sites | helpers::seed_for_thread | inline fully-qualified call | ✓ WIRED |
| 5 residual thread-offset sites | helpers::seed_for_thread | (none — still hand-rolled) | ✗ NOT_WIRED |
| frechet_anova primary loop | permutation_test::permutation_pvalue | closure + gather contract | ✓ WIRED |

### Behavioral Spot-Checks / Probe Execution

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-49 goldens bit-identical | `cargo test --test equivalence_phase49` (linalg,parallel) | 8 passed, 0 failed | ✓ PASS |
| Phase-48 frechet backstop | `cargo test --test equivalence_phase48 frechet` | 2 passed, 0 failed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| CONS-01 | 49-01, 49-02 | ✓ SATISFIED | χ²/gamma + SVD sign-fix consolidated, migrated, bit-identical |
| CONS-02 | 49-03, 49-04 | ⚠️ PARTIAL | permutation target fully satisfied; seeded-RNG target leaves 5 same-contract sites un-migrated → "every call site migrated" not met |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| distributions.rs, permutation_test.rs, regression.rs, helpers.rs | (none) | — | No TODO/FIXME/stub/placeholder in new/modified helper files |

### Notable (Info)

- `lib.rs::__equivalence_test_support` is a NEW `pub` (but `#[doc(hidden)]`, `__`-prefixed) module compiled into the library (not `#[cfg(test)]`-gated). It only forwards to existing pub(crate) internals and widens no existing signature, so SC#3 ("no public signature changed") holds. It is the documented mechanism for the external equivalence test crate to reach `pub(crate)` helpers. Not a blocker; noted for awareness.

### Gaps Summary

CONS-01 (χ²/gamma + SVD sign-fix) and the CONS-02 permutation scaffold are fully and correctly
consolidated: shared pub(crate) helpers exist, duplicated kernels are removed, every migrated site
is backed by an assert_eq! bit-identical golden, and the 49-04 permutation exclusions are genuine
contract differences documented with in-code rationale.

The one gap is in the CONS-02 seeded-RNG target. `seed_for_thread` is correct and 14 sites migrated
bit-identically, but FIVE genuine same-contract thread-offset sites remain hand-rolled — spm/arl.rs
(3), alignment/shape_ci.rs (1), boosting_regression/stability.rs (1). They match the exact
`StdRng::seed_from_u64(seed_field + loop_var)` pattern in `iter_maybe_parallel!` loops that the helper
was built to eliminate, and were excluded only because they fell outside plan 49-03's `files_modified`
list — a scoping artifact, not a contract difference — and carry no in-code rationale comment. The
ROADMAP goal is verbatim "every call site migrated," so the drift-risk this phase set out to kill
survives at these sites. No later phase covers them, so this is a real gap, not a deferral.

**Resolution is cheap and low-risk:** migrate the 3 clearly-identical sites to `seed_for_thread`
(bit-identical, wrapping_add == + for non-overflowing seeds), OR add an in-code rationale comment at
each plus a VERIFICATION override formally accepting them as intentional residuals (mirroring the
correct 49-04 treatment). Also make an explicit decision on the 2 explain_generic per-component
reseeds (importance.rs:66, shap.rs:78), which currently carry no in-code Phase-49 comment despite
49-04's narrative claiming they are owned/deferred.

---

_Verified: 2026-08-31_
_Verifier: Claude (gsd-verifier)_
