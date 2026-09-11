---
phase: "97"
slug: "differentiable-regression-prediction-smoothing-penalties"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-11"
---

# Phase 97 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + doctests |
| **Config file** | none — existing `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel scalar_on_function:: smooth_basis::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 s full suite |

---

## Sampling Rate

- **After every task commit:** the touched module's tests (`scalar_on_function::` or `smooth_basis::`).
- **After every plan wave:** full `cargo test -p fdars-core --features linalg,parallel`.
- **Before `/gsd-verify-work`:** full suite + clippy `--all-targets` + 28 examples + serde + wasm green.
- **Max feedback latency:** ~180 s.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 97-0x-W0 | — | 0 | DOP-02/03 | unit (RED) | new parity + Dual/Var FD tests fail pre-impl | ⬜ pending |
| 97-pred | — | 1 | DOP-02 | unit | `cargo test ... scalar_on_function::` (predict_curve_generic parity + FD) | ⬜ pending |
| 97-pen | — | 2 | DOP-03 | unit | `cargo test ... smooth_basis::` (penalty_value_generic parity + FD) | ⬜ pending |
| 97-gate | — | last | DOP-02/03 | build | non-breaking gate | ⬜ pending |

---

## Wave 0 Requirements

- Existing Rust harness covers everything. Wave 0 = author the RED tests: prediction f64 reference + Dual/Var FD (grad w.r.t. curve); penalty f64 reference + Dual/Var FD (grad w.r.t. coef). Preserve `scalar_on_function/tests.rs:117` and `smooth_basis.rs:1318+` regression tests.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*
- **DOP-02 f64 parity:** `predict_curve_generic::<f64>(curve, fit)` matches `predict_fregre_lm` for the corresponding curve within ~1e-12 (NOT bit-identical — the two paths differ in f64 accumulation order by ~1e-14; the standard tolerance for this phase's parity assertion is 1e-9, comfortably above the ULP divergence and below the existing test's 1e-6). `predict_fregre_lm` itself is UNCHANGED, so `test_predict_fregre_lm_on_training_data` stays green trivially.
- **DOP-03 f64 parity:** `penalty_value_generic::<f64>(coef, R, lambda)` is bit-identical to a direct inline `λ·cᵀRc` reference (R exactly symmetric by construction), tol 1e-12 / `assert_eq!`.
- **Autodiff FD:** prediction gradient w.r.t. curve and penalty gradient w.r.t. coef match central FD (h=1e-6) at BOTH `Dual` and `Var(vjp)`, tolerance `1e-6*(1.0 + fd.abs())` (absolute-floor form from Phase 96).
- **Non-breaking gate:** full suite + 28 examples + serde + wasm + clippy `--all-targets` + doctests; `git diff --stat` confined to `scalar_on_function/fregre_lm.rs`, `smooth_basis.rs` (+ their tests) and any additive re-exports; no new dependency; `predict_fregre_lm` + penalty-matrix constructor signatures unchanged.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
