---
phase: "87"
slug: "targeted-renames"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-08"
---

# Phase 87 — Validation Strategy

> Per-phase validation contract. API-shape-only rename/consolidation — validation is compile-time + existing-suite non-regression + a code-review byte-identical-body gate. No new behavioral tests are authored.

The `_impl` bodies are moved **verbatim** from the existing `_1d`/`_2d` functions, so the existing 2857-test suite is the numeric non-regression proof; the 28 examples + doctests + `tests/` are the compile-time proof the consolidated signatures are complete and self-consistent. A code-review gate confirms each `_impl` body is byte-identical to its pre-consolidation source.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + doctests |
| **Config file** | `fdars-core/Cargo.toml` (features: `default=["parallel"]`, `linalg`, `serde`) |
| **Quick run command** | `cargo build -p fdars-core --features linalg,parallel` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~120–300 seconds |

---

## Sampling Rate

- **After every task commit:** `cargo build -p fdars-core --features linalg,parallel` (compile proof)
- **After every plan wave:** `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** full gate set green (fmt, clippy `--all-targets`, test, `--features serde` build, `--examples`, `--benches`, doctests)
- **Max feedback latency:** ~300 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 87-01-A | 01 | 1 | NAME-05 | compile | `cargo build -p fdars-core --features linalg,parallel` | ⬜ pending |
| 87-01-B | 01 | 1 | NAME-01/02/03 | compile | `cargo build -p fdars-core --features linalg,parallel` | ⬜ pending |
| 87-01-C | 01 | 2 | all | gate | full gate set incl. `--examples`, `--benches`, `--doc`, clippy `--all-targets` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* No new test files — the existing suite + doctests + examples + `tests/validate_against_r.rs` + benches provide the compile-time + non-regression coverage.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Byte-identical-body equivalence is confirmed by the code-review gate (`gsd-code-review`), not manual inspection.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (none — existing infra suffices)
- [ ] No watch-mode flags
- [ ] Feedback latency < 300s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
