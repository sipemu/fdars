---
phase: 29
slug: outlier-detector-suite
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-19
---

# Phase 29 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[cfg(test)]` (inline module tests) |
| **Config file** | none — Cargo built-in test harness |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib outliers::` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (`outliers::` filter)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> One detector family per task; each carries inline `#[cfg(test)]` tests asserting the expected
> magnitude/shape outlier index sets on synthetic fixtures plus error paths.

| Task ID | Plan | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|-------------|-----------|-------------------|-------------|--------|
| 29-01-01 | 01 | OUT-01 | unit | `cargo test -p fdars-core --features linalg,parallel --lib outliers::tvdmss` | ❌ W0 | ⬜ pending |
| 29-01-02 | 01 | OUT-01 | unit | `cargo test -p fdars-core --features linalg,parallel --lib outliers::muod` | ❌ W0 | ⬜ pending |
| 29-02-01 | 02 | OUT-01 | unit | `cargo test -p fdars-core --features linalg,parallel --lib outliers::seq_transform` | ❌ W0 | ⬜ pending |
| 29-02-02 | 02 | OUT-01 | unit | `cargo test -p fdars-core --features linalg,parallel --lib outliers::depthgram` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Plan/task breakdown is provisional — the planner sets authoritative task IDs.*

---

## Wave 0 Requirements

- [ ] New detectors + result structs added inline in `src/outliers.rs` with `#[cfg(test)] mod tests`
- [ ] Private `iqr_fence` helper (Q1/Q3 via `quantile_sorted`, fence = Q ± factor·IQR)
- [ ] No framework install — Rust built-in test harness already present

*Existing infrastructure (Cargo test harness + `outliers` tests) covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification — numeric detector outputs asserted against expected outlier index sets and error paths inline.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
