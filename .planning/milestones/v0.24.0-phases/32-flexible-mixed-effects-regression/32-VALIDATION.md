---
phase: 32
slug: flexible-mixed-effects-regression
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-20
---

# Phase 32 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel famm fof` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90 seconds full suite; ~5s for famm/fof-only |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel famm fof`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite (incl. doctests) + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 32-01-01 | 01 | 1 | REG-05 | — / — | denseFLMM recovers fixed effects + variance components; invalid input → FdarError | unit | `cargo test -p fdars-core --features linalg,parallel famm::tests::dense_flmm` | ❌ W0 | ⬜ pending |
| 32-01-02 | 01 | 1 | REG-05 | — / — | multiFAMM multivariate recovery | unit | `cargo test -p fdars-core --features linalg,parallel famm::tests::multi_famm` | ❌ W0 | ⬜ pending |
| 32-01-03 | 01 | 1 | REG-05 | — / — | fastFMM massively-univariate inference recovers fixed effect | unit | `cargo test -p fdars-core --features linalg,parallel famm::tests::fast_fmm` | ❌ W0 | ⬜ pending |
| 32-02-01 | 02 | 2 | REG-05 | — / — | flexible-RE function-on-function recovers RE structure; base FoF untouched | unit | `cargo test -p fdars-core --features linalg,parallel fof::tests::fof_re` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are indicative — the planner sets the authoritative task breakdown.*

---

## Wave 0 Requirements

- [ ] Extend `fdars-core/src/famm.rs` with new estimators + inline `#[cfg(test)]` tests
- [ ] Extend `fdars-core/src/fof_regression.rs` with the flexible-RE variant + inline tests
- [ ] Promote 6 private famm.rs helpers to `pub(crate)` (non-breaking) for sibling-module reuse
- [ ] No new test framework needed — Rust built-in harness.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (synthetic mixed-model recovery + variance-component + fitted-curve + invalid-input tests).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
