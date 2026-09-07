---
phase: "77"
slug: "gradient-api-composition-demo-integration"
status: validated
nyquist_compliant: true
wave_0_complete: false
created: "2026-09-06"
---

# Phase 77 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + doctests (`cargo test --doc`) |
| **Config file** | none |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel autodiff grad` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` (+ `cargo test -p fdars-core --features linalg,parallel --doc`) |
| **Estimated runtime** | ~60–180 seconds |

---

## Sampling Rate

- **After every task commit:** quick command for the touched area.
- **After every plan wave:** full suite + `--doc`.
- **Before `/gsd-verify-work`:** full suite + doctests green + clippy `--all-targets` clean + fmt clean.
- **Max feedback latency:** ~180 seconds.

---

## Per-Task Verification Map

| Task | Req | Test Type | Automated Command | Status |
|------|-----|-----------|-------------------|--------|
| `grad`/`jacobian` multi-input gradient API | DIF-04 | unit | `cargo test ... grad` | ⬜ pending |
| Composition demo + FD check | DIF-04 | unit | `cargo test ... compos` | ⬜ pending |
| Crate-root + prelude re-exports | DIF-04 | compile | `cargo test ...` (use-path test) | ⬜ pending |
| Module doctest | DIF-04 | doctest | `cargo test ... --doc autodiff` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red*

### Validation architecture
1. **`grad` correctness (SC #1):** `grad(f, x)` returns `(value, gradient)` where gradient length == x length; on a known closed-form multi-input objective (e.g. `Σ xᵢ² `→ grad `2xᵢ`), gradient matches analytically ≤1e-12.
2. **Composition demo (SC #2):** compose ≥2 Phase-76 differentiable ops into one scalar objective; the `grad` gradient matches central finite differences ≤1e-6 on SPANNING full-rank curves (random combos of sin(πt), cos(2πt), sin(3πt); avoid SRSF zero-derivative points).
3. **Re-exports (SC #3):** a test (or the doctest) references `fdars_core::{Scalar, Dual, diff, grad, soft_dtw_distance_generic, ...}` via crate root AND `fdars_core::prelude::*` — compiling proves reachability.
4. **Doctest (SC #4):** `cargo test -p fdars-core --features linalg,parallel --doc` green (the new module doctest runs and asserts).

---

## Wave 0 Requirements

- [ ] Inline tests for `grad` + composition FD check; doctest in the module doc.
- [ ] No framework install.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification (including the doctest).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Doctest included and runs under `cargo test --doc`
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
