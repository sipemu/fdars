---
phase: 25-functional-glm-exponential-family
verified: 2026-08-17T16:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 25: Functional GLM (Exponential Family) Verification Report

**Phase Goal:** Users can fit a functional GLM for a scalar response over functional predictors across the four mainstream exponential-family families, generalizing the existing logistic path without breaking it.
**Verified:** 2026-08-17T16:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `functional_glm(data, y, family, …)` returns `Result<FunctionalGlmResult, FdarError>`; crate-root re-exported; `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` each with canonical link + variance (SC1) | VERIFIED | `glm.rs:509`, `mod.rs:344-353`, `lib.rs:249,252,254` — all four variants present with `inv_link`, `link_deriv`, `irls_weight`, `deviance`, `log_likelihood`; crate-root re-export confirmed in `lib.rs:249,252,254` |
| 2 | `functional_glm(..., GlmFamily::Binomial)` reproduces `functional_logistic` coefficients/fitted values within 1e-6 tolerance; `functional_logistic` signature retained unchanged (SC2) | VERIFIED | `test_binomial_parity_with_logistic` passes (runs both at tol=1e-12, asserts per-element agreement < 1e-6); `logistic.rs` has zero commits in this phase — `git log --all -- fdars-core/src/scalar_on_function/logistic.rs` shows no phase-25 entry |
| 3 | Poisson (log link) and Gamma (inverse link) recover known generative signal within tolerance; fitted_values all finite and positive (SC3) | VERIFIED | `test_poisson_recovery` passes (Pearson corr > 0.9 against true mu from log(mu)=1.0+1.5*s); `test_gamma_recovery` passes (Pearson corr > 0.9, all fitted_values > 0 and finite); confirmed by 11/11 tests green |
| 4 | IRLS runs over `fdata_to_pc_1d` FPC scores; out-of-domain responses (negative/non-integer Poisson, non-positive Gamma, Binomial not in {0,1}, NaN/Inf) and dimension mismatch return `FdarError`, never panic (SC4) | VERIFIED | Guards at `glm.rs:167-208` (non-finite check before per-family guards; Binomial, Poisson, Gamma guards); dimension check at `glm.rs:521-551`; 7 guard tests pass: `test_binomial_out_of_range_guard`, `test_poisson_negative_guard`, `test_poisson_noninteger_guard`, `test_gamma_nonpositive_guard`, `test_dimension_mismatch_guard`, `test_nonfinite_response_guard` (NaN+Inf), `test_predict_dimension_guard` |
| 5 | Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green; no new crate dependency (SC5) | VERIFIED | Full suite: 2081 lib tests + 305 integration/doc tests — 0 failed; clippy exits 0; `fdars-core/Cargo.toml` has no `statrs` entry — Poisson log(y!) uses Lanczos `ln_gamma` implemented inline (`glm.rs:216-241`) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/scalar_on_function/glm.rs` | functional_glm, predict_functional_glm, GlmFamily methods, IRLS step/loop, inline tests | VERIFIED | 997 lines; substantive implementation with full IRLS, domain guards, result assembly, 11 inline tests |
| `GlmFamily` enum + `FunctionalGlmResult` struct | Defined in mod.rs (per SUMMARY decision), with `#[non_exhaustive]`, conditional serde | VERIFIED | `mod.rs:342-395`; `#[non_exhaustive]`, `#[derive(Debug,Clone,Copy,PartialEq)]`, `#[cfg_attr(feature="serde",...)]` |
| Barrel `pub use` in `scalar_on_function/mod.rs` | `mod glm;` + `pub use glm::{functional_glm, predict_functional_glm}` + GlmFamily/FunctionalGlmResult in module | VERIFIED | `mod.rs:27` (`mod glm;`), `mod.rs:40-41` (pub use), types defined at `mod.rs:344,362` |
| Crate-root re-export in `src/lib.rs` | `functional_glm, predict_functional_glm, GlmFamily, FunctionalGlmResult` appended to existing block | VERIFIED | `lib.rs:249,252,254` — all four symbols in the `pub use scalar_on_function::{...}` block |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `glm.rs:irls_step_glm` | `GlmFamily::link_deriv` | working response z = η + (y-μ)·g′(μ) — stored separately from irls_weight | WIRED | `glm.rs:265-267`: explicit comment "MUST use link_deriv, NOT 1/weight"; `family.link_deriv(mu[i])` called in z computation |
| `GlmFamily::Gamma` | `irls_weight` | w = μ² (not 1/μ²) — Gamma bug fix | WIRED | `glm.rs:95`: `GlmFamily::Gamma => mu.max(1e-10).powi(2)` — positive μ² confirmed; code comment cross-references derivation |
| `GlmFamily::Gamma` | `init_beta` | β₀ = 1/mean(y) prevents η=0/μ=∞ on first step | WIRED | `glm.rs:343-348`: `if let GlmFamily::Gamma = family { beta[0] = 1.0/mean_y.max(1e-10); }` |
| `glm.rs:functional_glm` | `fdata_to_pc_1d` | IRLS runs over FPC scores from FPCA | WIRED | `glm.rs:558`: `let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;` |
| `glm.rs:validate_response` | Domain guards | Non-finite check fires before per-family guards | WIRED | `glm.rs:173-177`: NaN/Inf check first; per-family guards at `glm.rs:179-207` |
| `Binomial IRLS` | `logistic.rs irls_step` | Identical w = μ(1-μ), z = η + (y-μ)/w | WIRED | `glm.rs:92-93` (irls_weight), `glm.rs:74` (link_deriv = 1/(μ(1-μ))) — produces identical z to logistic.rs |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `functional_glm` | `fitted_values` | `build_glm_result:371` — `family.inv_link(eta)` where `eta` comes from `irls_loop_glm` over real FPCA scores | Yes | FLOWING |
| `functional_glm` | `beta_t` | `recover_beta_t(&beta[1..=ncomp], &fpca.rotation, m)` — real FPC rotation from `fdata_to_pc_1d` | Yes | FLOWING |
| `predict_functional_glm` | return value | `glm.rs:633-652` — projects new data through stored `fit.fpca`, computes `eta`, applies `fit.family.inv_link(eta)` | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 11 GLM unit tests pass | `cargo test --lib glm -- --test-threads=1` | 11 passed; 0 failed | PASS |
| Gaussian smoke test | included in above | `test_gaussian_smoke: ok` | PASS |
| Binomial parity with logistic | included in above | `test_binomial_parity_with_logistic: ok` | PASS |
| Poisson recovery (corr > 0.9) | included in above | `test_poisson_recovery: ok` | PASS |
| Gamma recovery (corr > 0.9) | included in above | `test_gamma_recovery: ok` | PASS |
| All domain guards | included in above | 5 guard tests + NaN/Inf + predict guard: all ok | PASS |
| Full suite | `cargo test -p fdars-core --features linalg,parallel` | 2081 lib + 305 integration/doc — 0 failed | PASS |
| Clippy | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | exit 0 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| REG-02 | 25-01-PLAN.md | Functional GLM for scalar response with GlmFamily{Binomial,Poisson,Gamma,Gaussian}, IRLS over FPC scores, functional_logistic retained | SATISFIED | `functional_glm` implemented; all 5 SCs verified; `REQUIREMENTS.md:16` shows REG-02 marked complete with commit `cb839d52` |

No orphaned requirements: REQUIREMENTS.md maps REG-02 exclusively to Phase 25, and Phase 25 claims only REG-02. Coverage is 1/1 (complete).

### Anti-Patterns Found

No anti-patterns found in `glm.rs`. Scanned for: TBD/FIXME/XXX (0), TODO/HACK/PLACEHOLDER (0), return null/empty stubs (0). Comments in the file are substantive documentation (mathematical derivations, safety invariants, convention divergence notes).

### Human Verification Required

None — all truths are verifiable programmatically and all behavioral tests passed.

### Gaps Summary

No gaps. All five success criteria are verified against the actual codebase:

1. **SC1 (API + re-export):** `functional_glm`, `predict_functional_glm`, `GlmFamily`, `FunctionalGlmResult` exist at the stated locations and are re-exported at the crate root.
2. **SC2 (Binomial parity + logistic unchanged):** `test_binomial_parity_with_logistic` asserts < 1e-6 agreement at convergence; `logistic.rs` has zero phase-25 commits.
3. **SC3 (Poisson + Gamma recovery):** Both recovery tests pass with Pearson corr > 0.9 and all fitted_values finite and positive.
4. **SC4 (domain guards + dimension checks):** Seven guard tests cover all specified error paths — Binomial out-of-range, negative Poisson, non-integer Poisson, non-positive Gamma, NaN/Inf (all families), and dimension mismatch — all return `FdarError`, no panics.
5. **SC5 (full suite + clippy + no new dep):** 2081 lib tests pass, clippy exits 0, no `statrs` or other new entry in `fdars-core/Cargo.toml` — Poisson `log(y!)` uses an inline Lanczos `ln_gamma`.

Notable code-quality point: the critical Gamma IRLS weight bug (`1/μ²` vs `μ²`) was caught by the code review and fixed before this verification. The current code at `glm.rs:95` correctly uses `mu.max(1e-10).powi(2)` (positive μ²), and the Gamma `link_deriv` correctly returns `-1/μ²` (negative), satisfying the sign-safety invariant that the working response `z = η + (y-μ)·g′(μ)` must use the link derivative directly.

---

_Verified: 2026-08-17T16:00:00Z_
_Verifier: Claude (gsd-verifier)_
