---
phase: 27
slug: elastic-multinomial-regression
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-19
---

# Phase 27 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib elastic_multinomial` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~30 seconds (full lib suite) |

## Sampling Rate

- **After every task commit:** module-scoped tests
- **After every plan wave:** full suite
- **Before `/gsd-verify-work`:** full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 27-01-* | 01 | 1 | REG-03 | unit | `cargo test -p fdars-core --features linalg,parallel --lib elastic_multinomial` | ⬜ pending |

**Behaviors requiring automated coverage (SC → test):**
- SC1 — `elastic_multinomial(...)` returns `Result<ElasticMultinomialResult, FdarError>` (K≥2), crate-root re-exported (shape/smoke).
- SC2 — `predict_elastic_multinomial` returns labels for new curves.
- SC3 — well-separated K-class templates recovered within accuracy threshold; K=2 agrees with binary `elastic_logistic`.
- SC4 — reuse SRVF/warping; invalid inputs (K<2, non-contiguous labels, count mismatch, empty) → `FdarError`, no panic.
- SC5 — binary `elastic_logistic` signature unchanged; full suite + clippy `--all-targets` green; no new dependency.

## Wave 0 Requirements

- [ ] `elastic_regression/logistic.rs` inline `#[cfg(test)] mod tests` — new tests (shape, K-class recovery, K=2↔binary agreement, error guards).

## Manual-Only Verifications

*All phase behaviors have automated verification.*

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
