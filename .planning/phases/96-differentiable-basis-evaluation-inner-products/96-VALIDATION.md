---
phase: "96"
slug: "differentiable-basis-evaluation-inner-products"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-11"
---

# Phase 96 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`, `basis/tests.rs`) + doctests |
| **Config file** | none — existing `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel basis::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 s full suite |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel basis::` (basis module tests).
- **After every plan wave:** full `cargo test -p fdars-core --features linalg,parallel`.
- **Before `/gsd-verify-work`:** full suite + clippy `--all-targets` + `--features serde` build + 28 examples green.
- **Max feedback latency:** ~180 s.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 96-0x-W0 | — | 0 | DOP-01 | — | N/A | unit (RED) | new parity + Dual/Var FD tests fail pre-impl | ❌ W0 | ⬜ pending |
| 96-bspline | — | 1 | DOP-01 | — | N/A | unit | `cargo test ... basis::` (bspline parity + FD) | ❌ W0 | ⬜ pending |
| 96-fourier | — | 2 | DOP-01 | — | N/A | unit | `cargo test ... basis::` (fourier parity + FD) | ❌ W0 | ⬜ pending |
| 96-gate | — | last | DOP-01 | — | N/A | build | non-breaking gate (examples + serde + wasm + clippy + doctests) | ❌ W0 | ⬜ pending |

*Task IDs indicative; the planner sets the authoritative map.*

---

## Wave 0 Requirements

- Existing Rust harness covers everything — no framework install. Wave 0 = author the RED tests: per-basis f64-parity (bit-identical / partition-of-unity 1e-10) and a combined basis-eval → inner-product objective differentiated w.r.t. `t`, FD-checked at `Dual` and `Var(vjp)`. They fail (or don't compile) before the generalization lands.
- Preserve the existing basis regression tests (basis/tests.rs): B-spline dimensions / partition-of-unity(1e-10) / non-negativity / boundary; Fourier dimensions / DC-constant / sin-cos-range / period / single-point edge case.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*
- **f64 parity:** `bspline_basis_from_knots::<f64>` (and the generic Fourier core at `f64`) reproduce current numerics bit-identically (`assert_eq!` on the recurrence body; partition-of-unity within existing 1e-10). The bit-parity rests on: `f64::from_f64` is identity, and the recurrence preserves op order `((t_val - knots[j]) / d1) * b[j]` (division-then-multiply, NOT reciprocal-multiply).
- **Autodiff FD (DOP-01 criterion #3):** a combined objective — evaluate basis at `t`, inner-product against a fixed curve, sum to scalar — differentiated w.r.t. `t`; central-FD (h=1e-6, tol 1e-6) matches gradients at BOTH `Dual` and `Var` (via `vjp`).
- **Non-breaking gate:** `cargo build --examples` (28), `--features serde`, `--target wasm32-unknown-unknown`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test` + `--doc`, and `git diff --stat` shows the change confined to `basis/bspline.rs`, `basis/fourier.rs` (+ their tests) and any in-crate caller of the Fourier core delegating to it. **Public signatures of `bspline_basis`, `fourier_basis`, `fourier_basis_with_period` MUST be unchanged** (Fourier generic work goes into a NEW additive core the f64 wrappers delegate to).

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
