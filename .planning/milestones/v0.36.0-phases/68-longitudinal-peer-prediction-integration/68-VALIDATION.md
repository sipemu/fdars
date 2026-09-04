---
phase: "68"
slug: "longitudinal-peer-prediction-integration"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-04"
---

# Phase 68 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + doctests (`#[cfg(test)] mod tests` + `//!` doctest) |
| **Config file** | none — existing crate test infrastructure |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel peer::` |
| **Doctest command** | `cargo test -p fdars-core --doc --features linalg,parallel peer` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~5 s (module) / ~6 s (doc) / minutes (full) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel peer::`
- **After the doctest/export task:** also `cargo test -p fdars-core --doc --features linalg,parallel peer`
- **After every plan wave:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + module + doc tests
- **Before `/gsd-verify-work`:** Full suite + doctests green
- **Max feedback latency:** ~10 s (module tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 68-01-01 | 01 | 1 | PER-04 | — | `lpeer()` fits subject random effects; variance components σ²≥0 | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ (peer.rs) | ⬜ pending |
| 68-01-02 | 01 | 1 | PER-04 | — | lpeer recovers β(t) + fitted σ²_subject tracks injected between-subject variance on synthetic longitudinal data | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ | ⬜ pending |
| 68-01-03 | 01 | 1 | PER-05 | — | `predict` self-consistent (re-passed training curves == training fitted values); finite on new curves; validates ncols | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ | ⬜ pending |
| 68-01-04 | 01 | 2 | PER-05 | — | full surface reachable from crate root + prelude; running module doctest passes under `cargo test --doc` | integration+doc | `cargo test -p fdars-core --doc --features linalg,parallel peer` | ✅ (lib.rs/prelude.rs) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Synthetic **longitudinal** fixture in `peer.rs` tests: repeated per-subject curves with a KNOWN injected between-subject variance + known β(t), so `lpeer` recovery + variance tracking are assertable.
- [ ] Reuse the spanning pseudo-random design pattern (full-rank) from `make_fixture`.

*Existing Rust test infrastructure covers the framework; no new deps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated known-answer verification (including the doctest gate).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
