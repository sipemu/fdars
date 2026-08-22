---
phase: 42
slug: object-data-fr-chet-regression
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-22
---

# Phase 42 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — `cargo test` on the existing workspace |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel frechet` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 s (full); frechet-scoped quick run ~10 s |

---

## Sampling Rate

- **After every task commit:** module-scoped `cargo test ... frechet`.
- **After every plan wave:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + full suite.
- **Before `/gsd-verify-work`:** full suite + clippy green; **existing density Fréchet tests must still pass** (regression after weight/Tₙ helper extraction).
- **Max feedback latency:** 120 s.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|--------|
| 42-01-xx | 01 | 1 | FRE-02-01 | SPD Frobenius/power-α/log-Cholesky: identical-object mean recovers object; power α=1 ≡ Frobenius; log-Cholesky mean(I,4I)=2I | unit | `cargo test -p fdars-core --features linalg,parallel frechet::spaces` | ⬜ pending |
| 42-01-xx | 01 | 1 | FRE-02-02 | correlation mean has unit diagonal; identical recovers object | unit | same | ⬜ pending |
| 42-02-xx | 02 | 1 | FRE-02-03 | spherical antipodal distance = π; Karcher midpoint of two near vectors; unit-norm output | unit | `cargo test -p fdars-core --features linalg,parallel frechet::spaces` | ⬜ pending |
| 42-02-xx | 02 | 1 | FRE-02-04 | network Laplacian mean of identical graphs recovers graph | unit | same | ⬜ pending |
| 42-02-xx | 02 | 1 | FRE-02-05 | point-process intensity L2 + weighted-average mean | unit | same | ⬜ pending |
| 42-03-xx | 03 | 2 | FRE-02-06 | generic global+local regression with SPD backend; constant response ⇒ constant prediction | unit | `cargo test -p fdars-core --features linalg,parallel frechet::regression` | ⬜ pending |
| 42-03-xx | 03 | 2 | FRE-02-07 | generic ANOVA over a non-density space; homogeneous sample non-significant + seed-reproducible; existing density tests still pass | unit | `cargo test -p fdars-core --features linalg,parallel frechet` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Inline `#[cfg(test)] mod tests` in the new backend module(s) and alongside the generic regression/ANOVA additions.
- The existing `frechet::{regression,anova,space,mean}` tests act as the regression suite for the helper extraction — no new framework.

*Existing Rust test infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Divergences from R `frechet` 0.3.0 (Frobenius correlation geometry, log-Cholesky convention, extrinsic Karcher init, Frobenius-on-Laplacian, L2 point-process) documented in rustdoc | FRE-02-01..05 | Prose/rustdoc claim | `cargo doc --no-deps` renders; reviewer confirms divergence notes present |

*All numeric behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Existing density Fréchet tests pass post-refactor (regression gate)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
