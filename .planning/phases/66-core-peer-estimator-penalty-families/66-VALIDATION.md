---
phase: "66"
slug: "core-peer-estimator-penalty-families"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-04"
---

# Phase 66 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) |
| **Config file** | none — existing crate test infrastructure |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel peer::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~5 s (module tests) / minutes (full suite) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel peer::`
- **After every plan wave:** Run `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + module tests
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds (module tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 66-01-01 | 01 | 1 | PER-01 | — | β(t) recovered within tolerance on synthetic known-β data | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ❌ W0 | ⬜ pending |
| 66-01-02 | 01 | 1 | PER-02 | — | all three penalty families (Ridge, Difference{2}, Decree) fit without error | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ❌ W0 | ⬜ pending |
| 66-01-03 | 01 | 1 | PER-02 | — | decree-Q partition boundary yields β(t) distinct from plain-roughness fit | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ❌ W0 | ⬜ pending |
| 66-01-04 | 01 | 1 | PER-01/02 | — | wrong-dim Q → descriptive FdarError, never panic; no NaN β(t) | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/peer.rs` inline `#[cfg(test)] mod tests` — synthetic known-β(t) fixture (design curves, chosen coefficient function, response = ∫X·β + noise)
- [ ] Reuse `crate::test_helpers::uniform_grid` for argvals

*Existing Rust test infrastructure covers the framework; no new deps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (known-answer numeric tests).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
