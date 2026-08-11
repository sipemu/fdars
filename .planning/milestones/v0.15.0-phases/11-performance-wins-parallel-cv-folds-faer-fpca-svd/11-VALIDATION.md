---
phase: 11
slug: performance-wins-parallel-cv-folds-faer-fpca-svd
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-11
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[test]` / `#[cfg(test)] mod tests`) |
| **Config file** | none — existing `fdars-core` test suite |
| **Quick run command** | `cargo test -p fdars-core --features linalg <test_name>` |
| **Full suite command** | `cargo test -p fdars-core --features linalg && cargo test -p fdars-core --features parallel && cargo clippy -p fdars-core --features linalg` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg <touched_module>`
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | PERF-01 | — | N/A (no new external input; parallelism is deterministic per-fold) | unit/integration | `cargo test -p fdars-core --features parallel` | ✅ | ⬜ pending |
| 11-02-01 | 02 | 1 | PERF-02 | — | N/A (numerical backend swap; no attack surface) | unit | `cargo test -p fdars-core --features linalg` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* Both changes are internal to `fdars-core`; the built-in Rust test harness and the existing criterion benchmark harness require no new setup.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Sequential-vs-parallel equivalence (PERF-01) and faer-vs-nalgebra `FpcaResult` equivalence (PERF-02) are both expressible as deterministic `#[test]` assertions.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
