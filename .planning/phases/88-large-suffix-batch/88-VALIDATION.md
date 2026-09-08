---
phase: "88"
slug: "large-suffix-batch"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-08"
---

# Phase 88 — Validation Strategy

> API-shape-only mass rename/consolidation. Validation is compile-time + existing-suite non-regression + a code-review byte-identical-body gate. No new behavioral tests. The `_1d` bodies are untouched (only visibility flips to `pub(crate)`, or a plain-rename of the symbol) so the existing 2857-test suite is the numeric non-regression proof; all 28 examples + doctests + tests + benches are the compile-time proof.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + doctests |
| **Quick run** | `cargo build -p fdars-core --features linalg,parallel` |
| **Full suite** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~120–300 s |

## Sampling Rate

- **After every task:** `cargo build -p fdars-core --features linalg,parallel`
- **After all tasks:** full gate set (fmt, clippy `--all-targets`, test, serde build, `--examples`, `--benches`, doctests)
- **Max feedback latency:** ~300 s

## Per-Task Verification Map

| Task | Requirement | Command | Status |
|------|-------------|---------|--------|
| 88-01-* (per group) | NAME-04 | `cargo build -p fdars-core --features linalg,parallel` | ⬜ pending |
| 88-01-final | NAME-04 | full gate set incl. `--examples --benches`, doctests, clippy `--all-targets` | ⬜ pending |

## Wave 0 Requirements

*Existing infrastructure covers all requirements.* The full suite + doctests + 28 examples + benches already exist and provide the compile-time + non-regression coverage. `tests/equivalence_phase50.rs` bit-identity goldens must be migrated to the dispatcher (per research Pitfall 2) and continue to hold.

## Manual-Only Verifications

*All phase behaviors have automated verification.* Byte-identical-body equivalence confirmed by the code-review gate.

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify
- [ ] No 3 consecutive tasks without automated verify
- [ ] `nyquist_compliant: true` set

**Approval:** pending
