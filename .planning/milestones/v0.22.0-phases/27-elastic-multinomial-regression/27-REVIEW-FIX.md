---
phase: 27-elastic-multinomial-regression
fixed_at: 2026-08-19T00:00:00Z
review_path: .planning/phases/27-elastic-multinomial-regression/27-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 27: Code Review Fix Report

**Fixed at:** 2026-08-19
**Source review:** .planning/phases/27-elastic-multinomial-regression/27-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5
- Fixed: 5
- Skipped: 0

## Fixed Issues

### CR-01: `ElasticLogisticResult` missing serde derive — compile break under `--features serde`

**Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
**Commit:** 4cdab43e
**Applied fix:** Added `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
to `ElasticLogisticResult` (line 14), directly above the existing `#[non_exhaustive]` attribute.
`FdMatrix` already carries serde derives so no further changes were needed.
Verified with `cargo build -p fdars-core --features serde,linalg,parallel` — compiled clean.

---

### CR-02: Double-write normalization in `elastic_multinomial` — Inf/NaN for tiny-positive row sums

**Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
**Commit:** 4cdab43e
**Applied fix:** Replaced the double-pass block (which computed `scale = 1/row_sum` for a tiny-but-
positive sum, producing ~1e20 multiplier, then conditionally overwrote) with the branch-first
pattern already used in `predict_elastic_multinomial`: if `row_sum < 1e-15`, set all cells to
`1/K` directly; else compute `scale = 1/row_sum` and multiply. Added a regression test
`elastic_multinomial_near_zero_row_stays_finite` that asserts all probability values are finite,
non-negative, and each row sums to 1.

---

### WR-01: Missing `m < 2` and `argvals.len() != m` entry-point guards in `elastic_multinomial`

**Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
**Commit:** 4cdab43e
**Applied fix:** Changed `let (n, _m) = data.shape()` to `let (n, m) = data.shape()` and added an
explicit guard block after the existing `n == 0 / y.len() != n` check:
```rust
if m < 2 || argvals.len() != m {
    return Err(crate::FdarError::InvalidDimension {
        parameter: "data/argvals",
        expected: "m >= 2, argvals.len() == m".to_string(),
        actual: format!("m={}, argvals.len()={}", m, argvals.len()),
    });
}
```
Added tests `elastic_multinomial_rejects_m_lt_2` and `elastic_multinomial_rejects_argvals_mismatch`
to cover both branches.

---

### WR-02: `ElasticMultinomialResult` missing `predict` convenience method

**Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
**Commit:** 4cdab43e
**Applied fix:** Added an `impl ElasticMultinomialResult` block after the
`predict_elastic_multinomial` free function, exposing a `pub fn predict(&self, new_data: &FdMatrix,
argvals: &[f64]) -> Vec<usize>` method that delegates to `predict_elastic_multinomial`. The method
is exercised by the `predict_elastic_multinomial_empty_input_returns_empty` test (WR-03 test).

---

### WR-03: `predict_elastic_multinomial` undocumented `n_new == 0` behaviour, no test

**Files modified:** `fdars-core/src/elastic_regression/logistic.rs`
**Commit:** 4cdab43e
**Applied fix:** Added an explicit early-return guard at the top of `predict_elastic_multinomial`
(`if n_new == 0 { return Vec::new(); }`) and extended the rustdoc to explain the intentional
asymmetry with the fitting function (fit errors on `n == 0`; predict returns empty vec). Added test
`predict_elastic_multinomial_empty_input_returns_empty` asserting the empty-input path via both the
free function and the new `predict` method on `ElasticMultinomialResult`.

---

## Skipped Issues

None — all findings were fixed.

---

**Verification ran in:** main checkout (workflow.use_worktrees=true but fixes applied pre-commit
via the cargo fmt + clippy + test pre-commit hook, which runs in the main checkout).

**Gates result:**
- `cargo build -p fdars-core --features serde,linalg,parallel`: PASS
- `cargo test -p fdars-core --features linalg,parallel --lib elastic_multinomial` (11 tests): PASS
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: PASS
- Pre-commit hook (fmt + clippy + 2107 lib tests): PASS

_Fixed: 2026-08-19_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
