---
phase: 10
slug: capability-gaps-spline-interpolation-functional-summary-statistics
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-10
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — inline `mod tests` per source file |
| **Quick run command** | `cargo test -p fdars-core --features linalg helpers::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg <module>::`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg`
- **Before `/gsd-verify-work`:** Full suite must be green + `cargo clippy -p fdars-core --features linalg` clean
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | FEAT-01 | — | N/A | unit | `cargo test -p fdars-core --features linalg helpers::` | ✅ | ⬜ pending |
| 10-02-01 | 02 | 1 | FEAT-02 | — | N/A | unit | `cargo test -p fdars-core --features linalg fdata::` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements (inline `#[cfg(test)]` modules already present in `helpers.rs` and `fdata.rs`; no framework install needed).*

---

## Manual-Only Verifications

*All phase behaviors have automated verification (numerical correctness verified by inline unit tests against hand-computed references).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
