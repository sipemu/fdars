---
phase: "103"
slug: "softdtwbarycenterconfig-api-legacy-wrapper"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 103 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + doctests |
| **Config file** | none — inline `mod tests` in `fdars-core/src/metric/soft_dtw.rs` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel soft_dtw` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Doctest command** | `cargo test -p fdars-core --doc soft_dtw` |
| **Estimated runtime** | ~60–120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel soft_dtw`
- **After every plan wave:** `cargo test -p fdars-core --features linalg,parallel` + doctests
- **Before `/gsd-verify-work`:** Full suite + doctests green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 103-01-01 | 01 | 0 | SDTW-03 | — | N/A | unit+doc | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Tests covering SDTW-03:
  - `SoftDtwBarycenterConfig::default()` yields the documented defaults; `#[non_exhaustive]`,
  - `soft_dtw_barycenter_with_config` runs and returns a valid barycenter,
  - legacy `soft_dtw_barycenter(data,gamma,max_iter,tol)` == config path with `n_restarts=1` (bit-identical parity),
  - new items re-exported at crate root AND prelude (compile-time use in a test/doctest),
  - a running doctest on the new entry point (`cargo test --doc`).
- [ ] Non-breaking guard: full `cargo test` + all 28 examples + doctests compile unchanged.

*Existing regression guard: `test_soft_dtw_barycenter_vs_tslearn` continues to pass.*

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
