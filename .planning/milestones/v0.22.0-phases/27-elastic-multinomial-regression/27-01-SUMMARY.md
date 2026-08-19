---
phase: 27-elastic-multinomial-regression
plan: "01"
subsystem: elastic-regression
status: complete
completed: "2026-08-19"
duration_minutes: 20

tags:
  - elastic-regression
  - multinomial
  - one-vs-rest
  - classification
  - srsf

dependency_graph:
  requires:
    - fdars-core/src/elastic_regression/logistic.rs (existing binary elastic_logistic)
  provides:
    - fdars_core::elastic_multinomial
    - fdars_core::predict_elastic_multinomial
    - fdars_core::ElasticMultinomialResult
  affects:
    - fdars-core/src/elastic_regression/mod.rs (barrel re-export)
    - fdars-core/src/lib.rs (crate-root re-export)

tech_stack:
  added: []
  patterns:
    - One-vs-rest (OvR) multinomial via K binary elastic_logistic calls
    - Row-normalised sigmoid outputs as class posteriors (zero-row guard)
    - #[non_exhaustive] result struct with conditional serde

key_files:
  created: []
  modified:
    - fdars-core/src/elastic_regression/logistic.rs
    - fdars-core/src/elastic_regression/mod.rs
    - fdars-core/src/lib.rs

decisions:
  - "OvR approach: reuse binary elastic_logistic K times unchanged (maximal reuse, no new dep)"
  - "Row-normalise OvR sigmoid scores to class posteriors; zero-row guard assigns uniform 1/K"
  - "Labels must be contiguous 0..K usize; validated at entry; FdarError on violation"
  - "K=2 OvR predicted labels agree with binary elastic_logistic on separable data (tested)"

estimate:
  tokens: 62000
  raw_tokens: 34000
  tasks: 4
  confidence: high

actuals:
  tokens: 18000
  tasks: 4
  commits: 2
---

# Phase 27 Plan 01: Elastic Multinomial Regression Summary

One-liner: Elastic multinomial logistic regression (K-class OvR) over SRSF space using K binary `elastic_logistic` fits with row-normalised class posteriors, crate-root exported.

## What Was Built

Added `elastic_multinomial`, `predict_elastic_multinomial`, and `ElasticMultinomialResult` to
`fdars-core/src/elastic_regression/logistic.rs` — extending the elastic-regression family with
multi-class classification. Implementation uses one-vs-rest (OvR): fits K binary
`elastic_logistic` models (class k → +1, rest → −1) via the existing SRSF/warping/IRLS machinery
unchanged, then row-normalises the K sigmoid columns to produce class posterior probabilities.

### New public symbols

- `ElasticMultinomialResult` — `#[non_exhaustive]`, `#[derive(Debug, Clone, PartialEq)]`,
  conditional serde. Fields: `n_classes`, `classes`, `class_models` (K binary fits),
  `train_probabilities` (n×K, row-sum=1), `predicted_classes`, `train_accuracy`.
- `elastic_multinomial(data, y: &[usize], argvals, ncomp_beta, lambda, max_iter, tol)`
  — `#[must_use]`. Entry-point guards: n==0, y.len()!=n, K<2, non-contiguous labels → FdarError.
- `predict_elastic_multinomial(fit, new_data, argvals) -> Vec<usize>` — reuses
  `predict_elastic_logistic` per class, row-normalises, argmax → label.

### Re-exports

- `elastic_regression/mod.rs` barrel extended with 3 new symbols (additive).
- `fdars-core/src/lib.rs` crate root extended with 3 new symbols (additive).

### Tests (7 inline, all pass)

| Test | Purpose |
|---|---|
| `elastic_multinomial_shape_smoke` | n=6, K=3: struct shapes, row-sums==1, accuracy in [0,1] |
| `elastic_multinomial_recovers_separated_classes` | K=3 bump templates: train_accuracy >= 0.8, predict on held-out templates |
| `elastic_multinomial_k2_agrees_with_binary` | K=2 separable data: multinomial predicted == binary elastic_logistic predicted |
| `elastic_multinomial_rejects_count_mismatch` | y.len() != n → Err |
| `elastic_multinomial_rejects_single_class` | K<2 → Err |
| `elastic_multinomial_rejects_noncontiguous_labels` | labels {0,2} → Err |
| `elastic_multinomial_rejects_empty` | n==0 → Err |

## Verification Results

- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — CLEAN
- `cargo test -p fdars-core --features linalg,parallel --lib elastic_multinomial` — 7/7 PASS
- `cargo test -p fdars-core --features linalg,parallel --lib elastic` — 128/128 PASS
- Binary `elastic_logistic` / `predict_elastic_logistic` / `ElasticLogisticResult` — byte-for-byte unchanged
- No new dependency introduced

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy: redundant pattern matching in reject tests**
- **Found during:** Task 4 clippy gate
- **Issue:** `assert!(matches!(result, Err(_)), ...)` triggers clippy `redundant_pattern_matching`
- **Fix:** Replaced 4 occurrences with `assert!(result.is_err(), ...)`
- **Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
- **Commit:** f3a36802

**2. [Rule 3 - Blocking] rustfmt reformatted logistic call and if-else in elastic_multinomial**
- **Found during:** First commit pre-commit hook (fmt check)
- **Issue:** Long call to `elastic_logistic(...)` and inline `let scale = if ... { } else { }` exceeded line width 100
- **Fix:** `cargo fmt -p fdars-core` applied before re-staging
- **Files modified:** `fdars-core/src/elastic_regression/logistic.rs`

## Threat Mitigations Applied

| Threat | Status |
|---|---|
| T-27-01 DoS: unbounded work on malformed input | Mitigated — entry-point guards (n==0, K<2, non-contiguous) return FdarError before any binary fit |
| T-27-02 Numeric integrity of probabilities | Mitigated — explicit zero-row-sum guard assigns uniform 1/K; rows always sum to 1 |
| T-27-03 Supply chain | Accepted — no new dependency |

## Threat Flags

None — no new network endpoints, auth paths, file access, or trust-boundary schema changes introduced.

## Known Stubs

None.

## Self-Check

- [x] `fdars-core/src/elastic_regression/logistic.rs` — modified, present
- [x] `fdars-core/src/elastic_regression/mod.rs` — modified, present
- [x] `fdars-core/src/lib.rs` — modified, present
- [x] commit 72cf61d7 — feat(27-01): add ElasticMultinomialResult and elastic_multinomial OvR fitter
- [x] commit f3a36802 — feat(27-01): barrel + crate-root re-export of elastic multinomial symbols

## Self-Check: PASSED
