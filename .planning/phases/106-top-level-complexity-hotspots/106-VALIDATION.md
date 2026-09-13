---
phase: "106"
slug: "top-level-complexity-hotspots"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 106 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> This is a **behavior-preserving refactor** phase: the validation signal is that
> the existing suite stays green (numeric/behavior identical) AND pmat complexity
> drops below threshold for each decomposed function.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[test]`/`#[cfg(test)]`) + criterion; pmat 3.18.0 for complexity |
| **Config file** | none — existing suite (2942 lib tests + 215 doctests + 28 examples) |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --lib <module>` (per touched module) |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Complexity check** | `pmat analyze complexity` (per-function; confirm each target ≤ threshold) |
| **Estimated runtime** | full suite ~120–300s; per-module quick run ~10–40s |

---

## Sampling Rate

- **After every task commit:** Run the quick per-module test subset + `pmat analyze complexity` on the touched file to confirm the decomposed function dropped below threshold.
- **After every plan wave:** Run the full suite `cargo test -p fdars-core --features linalg,parallel` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Before `/gsd-verify-work`:** Full suite green + `cargo fmt --check` clean + every Phase-106 target function below pmat cognitive+cyclomatic thresholds.
- **Max feedback latency:** ~300 seconds (full suite).

---

## Per-Task Verification Map

Each target function gets: (1) its module's existing tests stay green (behavior/numeric-identical), (2) pmat confirms complexity dropped. No new behavior is added, so no new tests are required — the **existing** tests are the non-regression oracle.

| Target Function | Requirement | Test Type | Automated Command | Status |
|-----------------|-------------|-----------|-------------------|--------|
| `funfem_cluster` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `elastic_net_cd` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `wnet_cv_lambda` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `wnet` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `boost_fofr` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `pace_fpca` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `bayesian_fosr` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |
| `kcfc_cluster` | QUAL-01 | non-regression + complexity | module tests + `pmat analyze complexity` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements — this is a refactor phase.
The 2942 lib tests + 215 doctests already exercise every target function; they are
the behavior-preservation oracle. No new test files required.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Behavior-preservation is proven
by the existing automated suite; complexity reduction is proven by `pmat analyze
complexity`. Nothing is manual-only.

---

## Validation Sign-Off

- [ ] Every target function's module tests stay green (numeric/behavior identical)
- [ ] `pmat analyze complexity` shows each target below cognitive AND cyclomatic thresholds
- [ ] Full suite green, clippy `--all-targets` clean, `cargo fmt --check` clean
- [ ] No public API / signature / visibility change (git diff shows only private helpers added)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
