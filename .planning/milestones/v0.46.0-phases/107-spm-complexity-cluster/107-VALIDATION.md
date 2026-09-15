---
phase: "107"
slug: "spm-complexity-cluster"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-14"
---

# Phase 107 — Validation Strategy

> Per-phase validation contract. Behavior-preserving refactor of `src/spm/`:
> the signal is the existing suite staying green (numeric/behavior identical)
> AND pmat complexity dropping below threshold for each decomposed SPM function.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + pmat 3.18.0 for complexity |
| **Config file** | none — existing suite (2942 lib + 215 doctests + 28 examples) |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --lib spm` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Complexity check** | `pmat analyze complexity` (filter to `src/spm/`; confirm each target below cyc 30 / cog 25) |
| **Estimated runtime** | full suite ~120–300s; `--lib spm` subset ~10–40s |

---

## Sampling Rate

- **After every task commit:** touched-module test subset + `pmat analyze complexity` on the touched file (confirm the decomposed function dropped below threshold).
- **After every plan wave:** full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Before `/gsd-verify-work`:** full suite green + `cargo fmt --check` clean + every Phase-107 SPM target below pmat error thresholds.
- **Max feedback latency:** ~300 seconds.

---

## Per-Task Verification Map

No new behavior added → no new tests required; the existing SPM tests are the
non-regression + numeric-identity oracle. Each flagged SPM function gets:
(1) its module tests stay green, (2) pmat confirms the drop.

| Target group | Requirement | Test Type | Automated Command | Status |
|--------------|-------------|-----------|-------------------|--------|
| amewma + iterative (tracer) | QUAL-02 | non-regression + complexity | `--lib spm` + `pmat analyze complexity` | ⬜ pending |
| mfpca + select_ncomp + contrib | QUAL-02 | non-regression + complexity | `--lib spm` + `pmat analyze complexity` | ⬜ pending |
| ewma + mewma + stats + phase | QUAL-02 | non-regression + complexity | `--lib spm` + `pmat analyze complexity` | ⬜ pending |
| partial + frcc + profile | QUAL-02 | non-regression + complexity | `--lib spm` + `pmat analyze complexity` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements — refactor phase. The SPM
tests already exercise every flagged function; they are the behavior-preservation
oracle. No new test files required.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Behavior-preservation via the
existing automated suite; complexity reduction via `pmat analyze complexity`.
Numeric hazard to watch (documented in RESEARCH.md): the AMEWMA sequential
recurrence (`q_prev`/`lambda_prev`/`sum_lambda_sq_prod` carried i→i+1) and the
`mfpca` `scales` in-place mutation — both must stay bit-identical, proven by the
existing SPM tests.

---

## Validation Sign-Off

- [ ] Every flagged SPM function's module tests stay green (numeric/behavior identical)
- [ ] `pmat analyze complexity` shows each Phase-107 target below cyc 30 / cog 25, all new helpers under threshold
- [ ] Full suite green, clippy `--all-targets` clean, `cargo fmt --check` clean
- [ ] No public API / signature / visibility change (git diff shows only added private helpers)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
