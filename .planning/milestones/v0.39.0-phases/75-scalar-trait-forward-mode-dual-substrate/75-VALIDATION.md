---
phase: "75"
slug: "scalar-trait-forward-mode-dual-substrate"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-06"
---

# Phase 75 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) |
| **Config file** | none — inline tests in `src/autodiff.rs` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel autodiff` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 seconds (full suite; incremental after warm build) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel autodiff`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite must be green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean + `cargo fmt --check` clean
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 75-01-* | 01 | 1 | DIF-01 | — | N/A (numeric library, no security surface) | unit | `cargo test -p fdars-core --features linalg,parallel autodiff` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Validation tiers (from 75-RESEARCH.md § Validation Architecture):
1. **Known-answer tests** — composed elementary functions with closed-form derivatives; `Dual` result matches analytical derivative to ≤1e-10 (the DIF-01 gate).
2. **Central finite-difference cross-check** — dual gradient vs `(f(x+h) − f(x−h)) / 2h` to ≤1e-6 (catches sign errors / missing chain-rule terms).
3. **f64 parity** — the `Scalar` impl for `f64` reproduces the current transcendental numerics bit-for-bit (`assert_eq!`, zero tolerance).

---

## Wave 0 Requirements

- [ ] `src/autodiff.rs` `#[cfg(test)] mod tests` — known-answer + finite-difference + f64-parity tests for DIF-01
- [ ] No framework install needed — Rust built-in harness

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
