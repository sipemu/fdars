---
phase: 76-differentiable-elastic-distance-fpca-scores
verified: 2026-09-06T00:00:00Z
status: passed
score: 11/11 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 76: Differentiable Elastic Distance & FPCA Scores Verification Report

**Phase Goal:** The two scoped FDA operations — elastic distance and FPCA score projection — are generic over `Scalar`, so at `Dual` they yield exact forward-mode gradients w.r.t. a curve's input values, while at `f64` they reproduce the existing numerics.
**Verified:** 2026-09-06
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | soft-DTW `Dual` gradient matches hand-written oracle ≤1e-9 (SC #1) | ✓ VERIFIED | `dual_gradient_vs_oracle` PASS (soft_dtw.rs:519). See SC #1 nuance below — validated against a **corrected** endpoint-preserving oracle, itself independently confirmed by central FD to 3.4e-10 in 76-REVIEW.md. Non-circular. |
| 2 | soft-DTW `Dual` gradient matches central FD ≤1e-6 (SC #3) | ✓ VERIFIED | `dual_gradient_vs_fd` PASS across gamma∈{0.5,1.0,2.0}, all 5 coords (soft_dtw.rs:558). True `(f(x+h)−f(x−h))/2h` via `::<f64>` — not the Dual re-derived. |
| 3 | soft-DTW `::<f64>` reproduces `soft_dtw_distance` bit-identically (SC #2) | ✓ VERIFIED | `f64_parity` PASS with `assert_eq!` bit-identity across gamma∈{0.1,1.0,10.0} + 20-pt series (soft_dtw.rs:536). `soft_dtw_distance` delegates to `soft_dtw_distance_inner::<f64>`. |
| 4 | amplitude-at-warp `Dual` gradient w.r.t. curve2 matches central FD ≤1e-5 (SC #4) | ✓ VERIFIED | `amplitude_gradient_vs_fd` PASS, all 20 pts, grid [0.1,0.9] avoids SRSF sqrt singularities (differentiable.rs:200). |
| 5 | amplitude-at-warp `::<f64>` reproduces hand-composed interp→SRSF→wL2 reference ≤1e-10 (SC #7) | ✓ VERIFIED | `amplitude_f64_parity` PASS (differentiable.rs:174). |
| 6 | warp-SEARCHED `elastic_distance` NOT genericized (discrete DP argmin) — documented deferred | ✓ VERIFIED | Module doc comment differentiable.rs:9-15 states the rationale; in-scope-correct per research scope call. `elastic_distance` signature unchanged (pairwise.rs:103). |
| 7 | FPCA `project_scores_generic::<Dual>` gradient equals closed form `rotation[(j,k)]·weights[j]` ≤1e-12 (SC #5 strong) | ✓ VERIFIED | `fpca_score_gradient_dual` PASS, all j∈0..40, k∈0..3, spanning full-rank curves (regression.rs:1750). |
| 8 | FPCA `Dual` gradient matches central FD ≤1e-6 (SC #5 FD) | ✓ VERIFIED | `fpca_score_gradient_vs_fd` PASS (regression.rs:1810). |
| 9 | FPCA `::<f64>` reproduces `FpcaResult::project` ≤1e-12 (SC #6) | ✓ VERIFIED | `fpca_score_generic_f64_parity` PASS (regression.rs:1787). Tolerance (not `assert_eq!`) correct — left- vs right-assoc multiply ~1 ULP. |
| 10 | No existing public f64 signature changed; `FpcaResult::project` untouched | ✓ VERIFIED | `soft_dtw_distance`, `elastic_distance`, `amplitude_distance`, `FpcaResult::project` signatures all present + unchanged (grep confirmed). Additive methods only. |
| 11 | No new crate dependency; MSRV 1.81 preserved (SC #5 non-breaking) | ✓ VERIFIED | `git diff` on Cargo.toml empty; `grep num-traits` = 0; only `S::from_f64` + existing Scalar ops used. |

**Score:** 11/11 truths verified (0 present, behavior-unverified)

All gradient truths are behavior-dependent (derivative/state computations); each is backed by a passing behavioral test exercising the actual gradient value — so they qualify as VERIFIED on behavioral evidence, not presence alone.

### SC #1 Pre-existing-oracle nuance (recorded per instruction, NOT a gap)

ROADMAP SC #1 requires the elastic-distance forward-mode gradient to match BOTH central FD AND "the existing hand-written `soft_dtw` gradient". The shipped `soft_dtw_accumulate_gradient` / `soft_dtw_backward` oracle has a **CONFIRMED PRE-EXISTING bug**: the reverse loop overwrites the `E[n][m]=1.0` endpoint seed with `a+b+c=0`, zeroing the whole backward pass → returns an all-zeros gradient for any input.

- **Pre-existing + out of additive scope:** 76-REVIEW.md independently reproduced the all-zeros output and traced the identical code back to commit `6bd5c4ce` (far before this phase). Phase 76 did not touch `soft_dtw_backward` (additive-only constraint). A one-line fix cascades into `soft_dtw_barycenter` numerics — out of scope to re-validate here.
- **SC #1 honestly met against a correct reference:** the phase validates the `Dual` gradient against a **corrected** in-test oracle (`corrected_oracle_gradient`, endpoint preserved, soft_dtw.rs:467) — a hand-written analytic forward/backward DP recursion, algorithmically distinct from forward-mode operator overloading, and itself independently confirmed by central FD to 3.4e-10 (76-REVIEW.md). The check is genuinely non-circular ("not zeros compared to zeros").
- **Disposition:** the differentiability GOAL (a correct, verified elastic-distance gradient) is achieved. The buggy shipped oracle is a separate pre-existing defect logged to backlog (SUMMARY + 76-REVIEW.md IN-01). Treating SC #1 as MET on the corrected-oracle + FD basis; NOT failing the phase for a pre-existing bug outside its additive scope.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `fdars-core/src/metric/soft_dtw.rs` | softmin3_generic, soft_dtw_distance_inner, soft_dtw_distance_generic + delegation | ✓ VERIFIED | All three present (lines 56,101,157); `soft_dtw_distance` delegates (line 136). Real DP recurrence, not stub. |
| `fdars-core/src/alignment/differentiable.rs` | new module: generic_linear_interp, generic_srsf_central_diff, generic_l2_srsf_distance, amplitude_distance_at_warp_generic | ✓ VERIFIED | File exists, substantive composition (line 125); module doc documents deferred warp-search. |
| `fdars-core/src/regression.rs` | project_scores_generic free fn + FpcaResult::project_generic wrapper | ✓ VERIFIED | Free fn (line 232), method wrapper (line 139); real linear projection. |
| `fdars-core/src/lib.rs` | project_scores_generic added to pub use | ✓ VERIFIED | Present (line 582). |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `soft_dtw_distance` | `soft_dtw_distance_inner::<f64>` | delegation body | ✓ WIRED | soft_dtw.rs:136; f64-parity `assert_eq!` gate green. |
| `amplitude_distance_at_warp_generic` | interp→SRSF→L2 helpers | composition over S:Scalar | ✓ WIRED | differentiable.rs:133-137. |
| `alignment/mod.rs` | `amplitude_distance_at_warp_generic` | `pub use differentiable::...` | ✓ WIRED | mod.rs:19,64. |
| `FpcaResult::project_generic` | `project_scores_generic` | wrapper delegation | ✓ WIRED | regression.rs:139-140. |
| `lib.rs` crate root | `project_scores_generic` | `pub use regression::{...}` | ✓ WIRED | lib.rs:582. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| New soft-DTW gradient/parity tests | `cargo test ... soft_dtw` | 9 passed, 0 failed (incl. dual_gradient_vs_oracle, f64_parity, dual_gradient_vs_fd) | ✓ PASS |
| Amplitude gradient/parity tests | `cargo test ... differentiable` | amplitude_gradient_vs_fd, amplitude_f64_parity — 5 passed, 0 failed | ✓ PASS |
| FPCA gradient/parity tests | `cargo test ... fpca` | fpca_score_gradient_dual, fpca_score_generic_f64_parity, fpca_score_gradient_vs_fd — 0 failed | ✓ PASS |
| Whole crate suite | `cargo test -p fdars-core --features linalg,parallel` | 2853 lib + all integration/doc, 3673 total, **0 failed** | ✓ PASS |
| Lint | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | exit 0, clean | ✓ PASS |
| Format | `cargo fmt --check` | exit 0, no drift | ✓ PASS |
| No new dependency | `git diff Cargo.toml` + `grep num-traits` | empty diff, count 0 | ✓ PASS |
| Additive signatures | grep `soft_dtw_distance`/`elastic_distance`/`amplitude_distance`/`FpcaResult::project` | all present, unchanged | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| DIF-02 | 76-01 | Differentiable soft-DTW + fixed-warp amplitude distance | ✓ SATISFIED | Truths 1-6; all targeted tests green. |
| DIF-03 | 76-02 | Differentiable FPCA score projection | ✓ SATISFIED | Truths 7-9; all targeted tests green. |

### Anti-Patterns Found

None. No TBD/FIXME/XXX debt markers introduced. The two LOW findings in 76-REVIEW.md (SRSF/L2 `sqrt` NaN-tangent at derivative-zero / distance-zero) are documented, accepted domain singularities covered by the threat model (T-76-01) and module docs — not defects introduced by this work.

### Human Verification Required

None. All success criteria are numerically testable and verified by passing behavioral tests.

### Gaps Summary

No gaps. All 5 ROADMAP success criteria and all 11 plan must-haves are satisfied with codebase + passing-test evidence. Full suite (0 failed), clippy `--all-targets` clean, fmt clean, no new dependency, and every existing public f64 signature unchanged. The single SC #1 subtlety (buggy shipped soft-DTW oracle) is a confirmed PRE-EXISTING defect outside the phase's additive scope; SC #1 is honestly met via a corrected, independently-FD-confirmed reference oracle, and the shipped bug is logged to backlog. The deferred warp-searched `elastic_distance` is in-scope-correct (non-differentiable discrete DP argmin).

---

_Verified: 2026-09-06_
_Verifier: Claude (gsd-verifier)_
