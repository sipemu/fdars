---
phase: "101"
slug: "in-crate-l-bfgs-barycenter-optimizer"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 101 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]`) |
| **Config file** | none — inline `mod tests` in `fdars-core/src/metric/soft_dtw.rs` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel soft_dtw` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel soft_dtw`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 101-01-01 | 01 | 0 | SDTW-01 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New unit tests in `fdars-core/src/metric/soft_dtw.rs` `mod tests` covering SDTW-01:
  - cost monotonic non-increase across L-BFGS iterations,
  - convergence to a bounded barycenter within the data's convex hull (no divergence),
  - `max_iter` / `tol` stopping honored,
  - f64 determinism (same input → identical barycenter),
  - L-BFGS cost ≤ prior MM-step cost on a curved objective.
- [ ] wasm32 compile guard: `cargo check -p fdars-core --target wasm32-unknown-unknown` (no `linalg`/`argmin`).

*Existing regression guard: `test_soft_dtw_barycenter_vs_tslearn` (validate_against_r.rs) continues to pass.*

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
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
