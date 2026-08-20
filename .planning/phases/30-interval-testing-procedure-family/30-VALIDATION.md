---
phase: 30
slug: interval-testing-procedure-family
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-20
---

# Phase 30 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[cfg(test)]` (inline module tests) |
| **Config file** | none — Cargo built-in test harness |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib inference::itp` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (`inference::itp` filter)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> The closure-adjustment (`pval_correct`) helper carries a dedicated hand-computed small-case unit
> test (RESEARCH Open Question 1) before it is trusted for larger fixtures.

| Task ID | Plan | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|-------------|-----------|-------------------|-------------|--------|
| 30-01-01 | 01 | INF-03 | unit | `cargo test -p fdars-core --features linalg,parallel --lib inference::itp::tests::pval_correct` | ❌ W0 | ⬜ pending |
| 30-01-02 | 01 | INF-03 | unit | `cargo test -p fdars-core --features linalg,parallel --lib inference::itp::tests::one_population` | ❌ W0 | ⬜ pending |
| 30-02-01 | 02 | INF-03 | unit | `cargo test -p fdars-core --features linalg,parallel --lib inference::itp::tests::two_population` | ❌ W0 | ⬜ pending |
| 30-02-02 | 02 | INF-03 | unit | `cargo test -p fdars-core --features linalg,parallel --lib inference::itp::tests::flm` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Plan/task breakdown is provisional — the planner sets authoritative task IDs.*

---

## Wave 0 Requirements

- [ ] New `src/inference/itp.rs` with the three entry points + `ItpResult` + private `pval_correct` helper + inline `#[cfg(test)] mod tests`
- [ ] A hand-computed small-case (~4 component) test locking `pval_correct` before larger fixtures
- [ ] No framework install — Rust built-in test harness already present

*Existing infrastructure (Cargo test harness + `inference` tests) covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification — adjusted p-values are asserted against expected
significance on synthetic localized-difference / null fixtures with seeded determinism.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
