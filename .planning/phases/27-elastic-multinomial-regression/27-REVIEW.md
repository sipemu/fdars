---
phase: 27-elastic-multinomial-regression
reviewed: 2026-08-19T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/elastic_regression/logistic.rs
  - fdars-core/src/elastic_regression/mod.rs
  - fdars-core/src/lib.rs
findings:
  critical: 2
  warning: 3
  info: 0
  total: 5
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-08-19
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

The new `elastic_multinomial` / `predict_elastic_multinomial` / `ElasticMultinomialResult` code is well-structured, uses the project's conventions correctly in most respects, and its input-validation coverage is solid. However there are two blockers: a serde compilation failure under `--features serde` caused by `ElasticLogisticResult` not carrying the `serde` derive, and an incorrect normalization path in `elastic_multinomial` (the zero-sum guard scales first then overwrites, which is harmless in the zero case but indicates confused logic that can misfire when `row_sum` is extremely small but positive — see CR-02). Three warnings cover a missing minimum-dimension guard delegated to the inner call, the absence of a `predict` convenience method on `ElasticMultinomialResult`, and the untested `n_new == 0` path in `predict_elastic_multinomial`.

Pre-existing binary `elastic_logistic` / `predict_elastic_logistic` / `ElasticLogisticResult` are unchanged. Re-exports in `mod.rs` and `lib.rs` are additive and correct.

---

## Critical Issues

### CR-01: `ElasticMultinomialResult` fails to compile under `--features serde` — `ElasticLogisticResult` has no serde derive

**File:** `fdars-core/src/elastic_regression/logistic.rs:219`
**Issue:** `ElasticMultinomialResult` carries `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` and contains `pub class_models: Vec<ElasticLogisticResult>`. `ElasticLogisticResult` (line 14) has only `#[derive(Debug, Clone, PartialEq)]` — no serde derive. When the `serde` feature is enabled the compiler will refuse to derive `Serialize`/`Deserialize` for `ElasticMultinomialResult` because `ElasticLogisticResult` does not implement those traits. This is a compile-time breakage for any user who enables `--features serde`.

**Fix:** Add the conditional serde derive to `ElasticLogisticResult`:

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElasticLogisticResult {
    // ...
}
```

`FdMatrix` already carries its own serde derive (verified in `matrix.rs`), so no further changes are needed.

---

### CR-02: Normalization logic in `elastic_multinomial` has a redundant double-write that silently mishandles the degenerate branch

**File:** `fdars-core/src/elastic_regression/logistic.rs:319-334`
**Issue:** The zero-sum guard branch has two separate write passes for the same row:

```rust
let scale = if row_sum < 1e-15 {
    1.0 / k as f64     // scale = 1/K
} else {
    1.0 / row_sum
};
for col in 0..k {
    train_probabilities[(row_i, col)] *= scale;  // (A) multiply by 1/K
}
if row_sum < 1e-15 {
    for col in 0..k {
        train_probabilities[(row_i, col)] = 1.0 / k as f64;  // (B) overwrite with 1/K
    }
}
```

Pass (A) multiplies each cell by `1/K`. Since all cells are essentially 0 (row_sum < 1e-15), (A) writes ~0. Pass (B) then overwrites with the correct `1/K`. So the zero-sum case produces the correct result, but only accidentally — the intended semantic and actual execution are misaligned. The real bug is that for rows where `row_sum` is tiny-but-positive (e.g. 1e-20, above the guard threshold of 1e-15), `scale = 1/row_sum` becomes an enormous number (~1e20) that inflates all probabilities to astronomically large values before they are clipped by nothing — there is no upper-bound clamp. The resulting "probabilities" will be `Inf` or `NaN` if any cell was non-zero but the row sum was between 1e-15 and, say, 1e-10.

The `predict_elastic_multinomial` function uses the cleaner, correct idiom (branch first, no double write) and does not share this defect.

**Fix:** Replace the double-pass block in `elastic_multinomial` with the same pattern used in `predict_elastic_multinomial`:

```rust
for row_i in 0..n {
    let row_sum: f64 = (0..k).map(|col| train_probabilities[(row_i, col)]).sum();
    if row_sum < 1e-15 {
        for col in 0..k {
            train_probabilities[(row_i, col)] = 1.0 / k as f64;
        }
    } else {
        let scale = 1.0 / row_sum;
        for col in 0..k {
            train_probabilities[(row_i, col)] *= scale;
        }
    }
}
```

---

## Warnings

### WR-01: `elastic_multinomial` does not guard `m < 2` directly — the guard is buried inside each OvR `elastic_logistic` call and produces a poor error message

**File:** `fdars-core/src/elastic_regression/logistic.rs:260-268`
**Issue:** The dimension guard at the top of `elastic_multinomial` checks only `n == 0` and `y.len() != n`. It does not check `m < 2` or `argvals.len() != m`. These are caught by `elastic_logistic` when the first OvR model is fitted (K binary models are still allocated before the failure). The error message produced by the inner call says `"data/y/argvals"` and refers to the binary context, which is confusing for a caller of the multinomial API. Convention in this codebase is to validate at the entry point.

**Fix:** Add explicit guards before the OvR loop:

```rust
let (n, m) = data.shape();
if n == 0 || y.len() != n {
    return Err(crate::FdarError::InvalidDimension { ... });
}
if m < 2 || argvals.len() != m {
    return Err(crate::FdarError::InvalidDimension {
        parameter: "data/argvals",
        expected: "m >= 2, argvals.len() == m".to_string(),
        actual: format!("m={}, argvals.len()={}", m, argvals.len()),
    });
}
```

---

### WR-02: `ElasticMultinomialResult` is missing the `predict` convenience method present on `ElasticLogisticResult`

**File:** `fdars-core/src/elastic_regression/logistic.rs` (after line 427)
**Issue:** `ElasticLogisticResult` has an `impl` block with a `predict` method (line 185–190) that delegates to the free function. `ElasticMultinomialResult` has no equivalent. Users expecting a symmetric API will find the method absent and may be surprised. This breaks the established convention in this crate where fitted result types expose a `predict` method.

**Fix:** Add the symmetric impl block after the `predict_elastic_multinomial` definition:

```rust
impl ElasticMultinomialResult {
    /// Predict class labels for new data. Delegates to [`predict_elastic_multinomial`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Vec<usize> {
        predict_elastic_multinomial(self, new_data, argvals)
    }
}
```

---

### WR-03: `predict_elastic_multinomial` panics silently on `n_new == 0` input (argmax reads `prob_matrix[(0, 0)]` on a zero-row matrix)

**File:** `fdars-core/src/elastic_regression/logistic.rs:413-426`
**Issue:** When `new_data` has zero rows, `n_new = 0`, `prob_matrix = FdMatrix::zeros(0, k)`, and the argmax closure `(0..n_new).map(|row_i| { ... best_p = prob_matrix[(row_i, 0)]; ... })` is never executed — so the function correctly returns an empty `Vec`. However `predict_elastic_logistic` (called inside the column-fill loop) internally calls `srsf_transform` which calls `simpsons_weights(argvals)` and then iterates over `n_new = 0` rows, which is safe. The overall behavior is actually correct in this case, but there is no test for it, and the analogous training path (`elastic_multinomial` with `n == 0`) is explicitly guarded and returns an error. The asymmetry means predict silently succeeds with an empty vec while fit returns an error — callers cannot rely on consistent error semantics.

**Fix:** Add a guard at the top of `predict_elastic_multinomial` (or document the behavior):

```rust
if n_new == 0 {
    return Vec::new();
}
```

Or add a test asserting the empty-input behavior is intentionally permissive, so the asymmetry is explicit in the test suite.

---

_Reviewed: 2026-08-19_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
