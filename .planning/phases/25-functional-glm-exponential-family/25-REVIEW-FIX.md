---
phase: 25-functional-glm-exponential-family
fixed_at: 2026-08-17T00:00:00Z
findings_in_scope: 8
fixed: 8
skipped: 0
iteration: 1
status: all_fixed
---

# Phase 25 — Code Review Fix Report

All 8 findings from `25-REVIEW.md` (4 critical, 3 warning, 1 info) were fixed.
CR-01 was fixed by the initial (stalled) fixer run and recovered via fast-forward
merge of its committed work; the remaining 7 were applied directly after the fixer
subagent stalled on the slow per-commit hook.

## Fixed

| ID | Severity | Fix | Commit |
|----|----------|-----|--------|
| CR-01 | BLOCKER | Gamma IRLS weight corrected `1/μ²` → `μ²` (Fisher `w = (dμ/dη)²/V(μ) = μ²`); `test_gamma_recovery` strengthened to assert accuracy | `0b0a8f16` |
| CR-02 | BLOCKER | (a) reject non-finite responses so `+Inf` no longer passes Poisson `floor()`/saturates `as u64`; (b) replaced O(y) `Σ ln(k)` log-factorial with self-contained Lanczos `ln_gamma(y+1)` — O(1), overflow-free, **no new dependency** | `07a6ac8c` |
| CR-03 | BLOCKER | `predict_functional_glm` now returns `Result`; validates `new_data.ncols() == training grid length` → `InvalidDimension` (no OOB panic / silent truncation) | `07a6ac8c` |
| CR-04 | BLOCKER | `functional_glm` validates `scalar_covariates` row count `== n` → `InvalidDimension` | `07a6ac8c` |
| WR-01 | WARNING | Dispersion φ estimated (Pearson χ²/dof) for Gaussian & Gamma standard errors; φ=1 for Binomial/Poisson | `07a6ac8c` |
| WR-02 | WARNING | Binomial-parity test made deterministic: both `functional_logistic` and `functional_glm` run to full convergence (tol 1e-12, max_iter 100) so the deviance-vs-coefficient stopping-criterion difference cannot cause flakiness | `07a6ac8c` |
| WR-03 | WARNING | `predict_functional_glm` validates `new_scalar` row & column counts (and rejects `None` when the model has scalar covariates) → `InvalidDimension` | `07a6ac8c` |
| IN-01 | INFO | Non-finite (NaN/±Inf) responses rejected for all families in `validate_response` → `InvalidParameter` | `07a6ac8c` |

## Regression tests added

- `test_nonfinite_response_guard` — NaN (Gamma) and `+Inf` (Poisson) → `InvalidParameter` (covers CR-02a, IN-01).
- `test_predict_dimension_guard` — mismatched predict grid length → `InvalidDimension` (covers CR-03).
- `test_gamma_recovery` strengthened to a Pearson-correlation accuracy assertion (covers CR-01).

## Verification

- `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (exit 0), full suite (2081 lib tests) and doctests — **all green** (pre-commit hook: "✅ All checks passed").
- `functional_logistic` / `logistic.rs` untouched; Binomial parity retained (< 1e-6).
- No new crate dependency (Lanczos `ln_gamma` is vendored inline).

## Notes

- The API change to `predict_functional_glm` (and the delegating `FunctionalGlmResult::predict`) returning `Result` is non-breaking against released APIs: both symbols are new in this (unreleased) phase.
- The first fixer subagent stalled (stream watchdog, 600s) mid-run on the slow per-commit hook; its one committed fix (CR-01) was recovered and the remainder completed inline. The stale `gsd-reviewfix/25-*` worktree/branch were cleaned up.
