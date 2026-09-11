---
phase: "95"
slug: "generic-scalar-hot-path-signatures"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-11"
---

# Phase 95 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) + doctests |
| **Config file** | none — existing `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel helpers::tests utility::tests warping::tests` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 s full suite |

---

## Sampling Rate

- **After every task commit:** the kernel's module tests (`cargo test -p fdars-core --features linalg,parallel <module>`).
- **After every plan wave:** full `cargo test -p fdars-core --features linalg,parallel`.
- **Before `/gsd-verify-work`:** the full non-breaking compile-gate must be green (see below).
- **Max feedback latency:** ~180 s.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 95-0x-W0 | — | 0 | GEN-01 | — | N/A | unit (RED) | `cargo test -p fdars-core --features linalg,parallel` (new parity+FD tests fail pre-impl) | ❌ W0 | ⬜ pending |
| 95-xx | — | 1 | GEN-01 | — | N/A | unit | per-kernel f64-parity + Dual/Var FD spot-check | ❌ W0 | ⬜ pending |
| 95-gate | — | last | GEN-01 | — | N/A | build | non-breaking compile-gate (examples + serde + wasm + clippy + doctests) | ❌ W0 | ⬜ pending |

*Task IDs indicative; the planner sets the authoritative map.*

---

## Wave 0 Requirements

- Existing Rust harness covers everything — no framework install. Wave 0 = author the RED tests (per RESEARCH.md, ~ten new tests across `helpers.rs`, `utility.rs`, `warping.rs`): f64-parity per kernel + a forward-mode `Dual` and reverse-mode `Var` finite-difference spot-check per kernel. They fail (or don't compile) before the in-place generalization lands, pass after.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* The non-breaking guarantee (GEN-01) is proven by the compile-gate, not manual inspection:
- **f64 parity:** each generalized kernel at `T = f64` reproduces the pre-change result bit-identically (or within 1e-12).
- **Autodiff flow:** each kernel accepts `Dual` and `Var`; a light central-FD spot-check (tol 1e-6) confirms correct gradients.
- **Non-breaking compile-gate (the GEN-01 deliverable):**
  - `cargo build -p fdars-core --features linalg,parallel`
  - `cargo build -p fdars-core --examples --features linalg,parallel` (all 28 examples)
  - `cargo build -p fdars-core --features serde`
  - `cargo build -p fdars-core --target wasm32-unknown-unknown` (WASM surface)
  - `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
  - `cargo test -p fdars-core --features linalg,parallel` + `--doc`
  - `git diff --stat` shows ONLY the kernel files changed (helpers.rs / utility.rs / warping.rs + their tests) — no call-site churn.
  - R bindings: source-level grep of the external `fdars-r` package confirms it calls the kernels with `f64` data only (cannot build the external package here).

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
