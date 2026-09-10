---
phase: "94"
slug: "reverse-mode-autodiff-core-vjp-tape"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-10"
---

# Phase 94 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) + doctests |
| **Config file** | none — existing `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel autodiff` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–180 seconds (full suite ~2857 tests) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel autodiff::reverse` (the reverse-mode module tests)
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 94-01-01 | 01 | 1 | RAD-01 | — | N/A (pure numerics) | unit | `cargo test -p fdars-core --features linalg,parallel autodiff::reverse` | ❌ W0 | ⬜ pending |
| 94-01-02 | 01 | 1 | RAD-01 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel autodiff::reverse` | ❌ W0 | ⬜ pending |
| 94-02-01 | 02 | 2 | RAD-02 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel autodiff::reverse::vjp` | ❌ W0 | ⬜ pending |
| 94-03-01 | 03 | 3 | RAD-03 | — | N/A | integration | `cargo test -p fdars-core --features linalg,parallel autodiff` | ❌ W0 | ⬜ pending |

*Task IDs are indicative; the planner sets the authoritative map. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements — the Rust built-in test harness and doctest machinery are already in place (see `autodiff.rs:629–1143` for the tiered test pattern to mirror). No new framework install; no new test config. Reverse-mode tests live inline in `autodiff/reverse.rs` under `#[cfg(test)] mod tests`.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Reverse-mode gradients are validated by:
- Tier 1 known-answer unit derivatives (tol 1e-10) for every `Var` op.
- Guard tests for singular points (sqrt/ln/powf at 0) — NaN/Inf adjoint parity with forward-mode.
- Reverse-vs-`Dual` agreement (bit-close, 1e-10/1e-12).
- Reverse-vs-central-FD cross-checks (tol 1e-6) on `soft_dtw_distance_generic` and `project_scores_generic`.
- Composed-objective `vjp` test (`soft_dtw + λ·Σscores²`), FD-checked per component.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
