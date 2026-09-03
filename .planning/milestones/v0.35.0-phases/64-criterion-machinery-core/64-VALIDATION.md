---
phase: "64"
slug: "criterion-machinery-core"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-02"
---

# Phase 64 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[test]`, `#[cfg(test)] mod tests`) |
| **Config file** | none — Cargo's `cargo test` runner |
| **Quick run command** | `cargo test -p fdars-core --features linalg optimal_design` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | full suite ~minutes (2657 lib tests); quick module run ~seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg optimal_design`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`
- **Max feedback latency:** quick module run — seconds

---

## Per-Task Verification Map

| Task ID | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|-------------|-----------------|-----------|-------------------|-------------|--------|
| build_sigma_design | FOD-01/02 | Σ_d shape \|S\|×\|S\| + σ²I; ridge-retry never panics | unit | `cargo test -p fdars-core optimal_design::tests::test_ridge_retry` | ❌ W0 | ⬜ pending |
| trajectory ∅ | FOD-01 | `MSE(∅) = Σλ_k`, grid-invariant | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_empty_set` | ❌ W0 | ⬜ pending |
| trajectory reduce | FOD-01 | MSE decreases on adding a point | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_reduces_on_point` | ❌ W0 | ⬜ pending |
| trajectory monotone | FOD-01 | monotone non-increasing (sign gate) | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_trajectory` | ❌ W0 | ⬜ pending |
| trajectory grid-inv | FOD-01 | MSE(∅) unchanged m=21/51/101 | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_grid_invariance` | ❌ W0 | ⬜ pending |
| score A ∅ | FOD-02 | `A(∅) = Σλ_k` | unit | `cargo test -p fdars-core optimal_design::tests::test_score_a_empty_set` | ❌ W0 | ⬜ pending |
| score D ∅ | FOD-02 | `D(∅) = Σ log λ_k` (negative) | unit | `cargo test -p fdars-core optimal_design::tests::test_score_d_empty_set` | ❌ W0 | ⬜ pending |
| score prior | FOD-02 | `Cov(ξ|∅) = diag(λ)` | unit | `cargo test -p fdars-core optimal_design::tests::test_score_prior_recovery` | ❌ W0 | ⬜ pending |
| A monotone | FOD-02 | A-opt monotone non-increasing | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_a_opt` | ❌ W0 | ⬜ pending |
| D monotone | FOD-02 | D-opt monotone non-increasing | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_d_opt` | ❌ W0 | ⬜ pending |
| enum dispatch | FOD-03 | 3 DesignCriterion variants route correctly | unit | `cargo test -p fdars-core optimal_design::tests::test_enum_dispatch` | ❌ W0 | ⬜ pending |
| validate index | FOD-03 | out-of-range index → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_index_range` | ❌ W0 | ⬜ pending |
| validate sigma2 | FOD-03 | sigma2 ≤ 0 → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_sigma2` | ❌ W0 | ⬜ pending |
| validate ncomp | FOD-03 | ncomp == 0 → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_ncomp` | ❌ W0 | ⬜ pending |
| lib re-export | FOD-03 | additive re-export compiles, nothing broken | build | `cargo build -p fdars-core` | ❌ W0 | ⬜ pending |
| serde derives | FOD-03 | serde-gated derives compile | build | `cargo build -p fdars-core --features serde` | ❌ W0 | ⬜ pending |
| suite unbroken | FOD-01/02/03 | 2657+ existing tests still pass | integration | `cargo test -p fdars-core --features linalg,parallel` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/optimal_design.rs` — new file; all `#[cfg(test)] mod tests` gates above
- [ ] `lib.rs` additive lines: `pub mod optimal_design;` + re-export `DesignCriterion`, `OptimalityKind`, `design_criterion`

*No existing test infrastructure covers `optimal_design` — all module tests are new in Wave 0.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated known-answer verification. |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < module-test seconds
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
