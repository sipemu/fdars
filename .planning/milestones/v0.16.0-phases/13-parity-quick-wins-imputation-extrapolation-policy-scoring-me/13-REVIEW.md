---
phase: 13-parity-quick-wins-imputation-extrapolation-policy-scoring-me
reviewed: 2026-08-11T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/helpers.rs
  - fdars-core/src/scoring.rs
  - fdars-core/src/lib.rs
findings:
  critical: 2
  warning: 1
  info: 1
  total: 4
status: needs-attention
---

# Phase 13: Code Review Report

**Reviewed:** 2026-08-11
**Depth:** standard
**Files Reviewed:** 3 (helpers.rs FEAT-03/04 additions, scoring.rs new module, lib.rs re-exports)
**Status:** needs-attention

## Summary

Phase 13 adds three additive features to `fdars-core`: `ExtrapolationPolicy` + `fdata_interpolate_with_policy` (FEAT-04), `ImputationMethod` + `impute_missing_values` (FEAT-03), and five functional scoring metrics in a new `scoring.rs` module (FEAT-05). The overall structure is sound — conventions are followed, signatures are non-breaking, and the test suite covers the primary happy paths and expected error paths.

Two correctness bugs were found:

1. **Periodic wrap with a degenerate (zero-length) domain produces silent NaN output.** When `argvals` has identical first and last values (`domain_len == 0.0`), any OOB query with `ExtrapolationPolicy::Periodic` computes `x % 0.0 = NaN` (IEEE 754), which propagates silently through `linear_interp` returning garbage. No validation blocks this path.

2. **`functional_explained_variance` near-zero guard has a logic error.** When `ss_tot < NUMERICAL_EPS` (constant true curve), the guard checks `ss_res < NUMERICAL_EPS` to determine "perfect fit". But if `ss_tot = 5e-11` and `ss_res = 8e-11`, both values are below `NUMERICAL_EPS` yet `ss_res > ss_tot`, meaning the prediction is worse than the baseline — the function incorrectly returns `1.0` instead of `0.0`.

One warning-level issue and one info item round out the findings.

---

## Critical Issues

### CR-01: Periodic extrapolation silently produces NaN when domain_len = 0

**File:** `fdars-core/src/helpers.rs:808`

**Issue:** `fdata_interpolate_with_policy` computes `((t - t_min) % domain_len + domain_len) % domain_len` where `domain_len = t_max - t_min`. If `argvals` has repeated identical values (e.g. `[5.0, 5.0, 5.0]`), then `domain_len == 0.0`. In Rust `f64`, `x % 0.0 = NaN` (IEEE 754 — no panic). The `NaN` propagates to `wrapped`, which is then passed to `linear_interp`. Inside `linear_interp`, all comparisons with `NaN` return `false`, so the clamp/search path falls through `binary_search_by` using `partial_cmp(...).unwrap_or(Equal)` and returns an indeterminate value from the array. The caller receives `Ok(...)` containing garbage values rather than an error.

This cannot happen with the `Boundary`, `Exception`, or `Fill` variants because they do not divide by `domain_len`. The existing validation only checks `m < 2`, not that `argvals` is strictly increasing.

**Trigger:** Any caller passing argvals with `argvals[0] == argvals[m-1]` (degenerate domain) together with `ExtrapolationPolicy::Periodic` and at least one OOB query point.

**Fix:** Add a domain_len guard in `fdata_interpolate_with_policy`, either at entry for all policies or specifically before the `Periodic` branch:

```rust
// Option A: guard only the Periodic branch (minimal change)
ExtrapolationPolicy::Periodic => {
    if domain_len <= 0.0 {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "argvals",
            message: "Periodic extrapolation requires a positive domain length \
                      (argvals[0] < argvals[m-1])".to_string(),
        });
    }
    let wrapped = t_min + ((t - t_min) % domain_len + domain_len) % domain_len;
    // ...
}

// Option B: validate at function entry (preferred — catches degenerate argvals early)
if domain_len <= 0.0 && matches!(policy, ExtrapolationPolicy::Periodic) {
    return Err(crate::FdarError::InvalidParameter {
        parameter: "argvals",
        message: "Periodic extrapolation requires argvals[0] < argvals[m-1]".to_string(),
    });
}
```

---

### CR-02: functional_explained_variance returns 1.0 when ss_res > ss_tot (both < NUMERICAL_EPS)

**File:** `fdars-core/src/scoring.rs:258-265`

**Issue:** The near-zero guard for `SS_tot`:

```rust
let ev_i = if ss_tot < NUMERICAL_EPS {
    if ss_res < NUMERICAL_EPS {
        1.0   // <-- BUG: returned even when ss_res > ss_tot
    } else {
        0.0
    }
} else {
    1.0 - ss_res / ss_tot
};
```

When `ss_tot = 5e-11` and `ss_res = 8e-11`, both values are below `NUMERICAL_EPS = 1e-10`. The code returns `1.0` (perfect fit), but since `ss_res > ss_tot` the prediction actually has MORE variation than the true curve — the correct result is `<= 0.0`.

**Concrete scenario:** `y_true` is a constant curve (e.g., all values `5.0`), `y_pred` is that constant plus a tiny-amplitude oscillation (e.g., `5.0 + 1e-6 * sin(t)`). Then `ss_tot ≈ 0` (constant baseline), `ss_res ≈ (1e-6)^2 * domain_len ≈ 1e-12` — both below `NUMERICAL_EPS`. The function returns `1.0` (claiming perfect prediction) even though the prediction oscillates around the true constant.

**Fix:** The inner guard should compare `ss_res` against `ss_tot` relatively, not against the absolute `NUMERICAL_EPS`:

```rust
let ev_i = if ss_tot < NUMERICAL_EPS {
    // Constant true curve: "perfect fit" only when residual variance also vanishes
    // relative to ss_tot. Compare ss_res <= ss_tot (with small tolerance) rather
    // than both vs an absolute threshold.
    if ss_res <= ss_tot * (1.0 + 1e-6) {
        1.0
    } else {
        0.0
    }
} else {
    1.0 - ss_res / ss_tot
};
```

Alternatively, use `ss_res < NUMERICAL_EPS * NUMERICAL_EPS` (squaring the epsilon since SS values are quadratic in the signal) or replace the double-absolute check with a single relative-residual test before the `ss_tot` branch.

---

## Warnings

### WR-01: impute_missing_values returns wrong error variant for m=0 input

**File:** `fdars-core/src/helpers.rs:878-883`

**Issue:** If `data` has `n > 0` rows and `m = 0` columns (and `argvals` is empty, so the dimension check passes at line 868), the function enters the per-curve loop. Each curve's row is empty (`data.row(i)` returns `vec![]`), so `valid_count = 0`. The code then returns `Err(InvalidParameter { parameter: "data", message: "curve 0 contains only NaN values" })`. This error message is factually wrong — the curve has no NaN values; it has no values at all. The caller receives a misleading error that a curve is "all NaN" when the real problem is a degenerate zero-column matrix.

While `FdMatrix` with `m = 0` is unusual, it is constructible and the validation should handle it distinctly from the all-NaN case.

**Fix:** Add a degenerate-column guard before the per-curve loop:

```rust
if m == 0 {
    return Err(crate::FdarError::InvalidDimension {
        parameter: "data",
        expected: "m >= 1".to_string(),
        actual: "m=0".to_string(),
    });
}
```

---

## Info

### IN-01: Redundant #[cfg(test)] attribute on use statement inside test module

**File:** `fdars-core/src/scoring.rs:277`

**Issue:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]          // <-- redundant: already inside #[cfg(test)] mod
    use crate::test_helpers::uniform_grid;
```

The outer `#[cfg(test)]` on the `mod tests` block already gates the entire module, including all `use` statements inside it. The inner `#[cfg(test)]` on the `use` is harmless but signals unfamiliarity with the pattern. The same pattern is not present in other test modules in the codebase.

**Fix:** Remove the inner `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
```

---

## Findings Not Raised

The following areas were explicitly checked and found correct:

- **Periodic modulo math:** The guarded-modulo recipe `((t - t_min) % L + L) % L` correctly handles negative remainders in Rust's truncated-division semantics for all non-degenerate `L > 0`. The `t = t_max` exact boundary is caught by the `in_range` check (inclusive), so the modulo is never applied to boundary points in the normal case.
- **Simpson's weights for scoring:** `simpsons_weights(argvals)` is called once per metric and its sum equals the domain length. The per-curve integral accumulator (`total`) is divided by `n` once at the end — this is mathematically equivalent to averaging per-curve integrals.
- **MAE/MSE formulas:** Correct. The double loop sums `|error| * w_j` across all `(i, j)`, then divides by `n`.
- **MAPE pre-scan:** The full `(n, m)` scan before integration is correct fail-fast behavior. MAPE uses `y_true[(i,j)].abs()` (not `y_true[(i,j)]`), correctly guarding negative denominators too.
- **MSLE domain guard:** The threshold `= -1.0 + NUMERICAL_EPS` correctly guards `ln_1p(x)` from producing NaN: values exactly at `-1.0` are rejected (since `-1.0 <= -1.0 + 1e-10` is true), and values in `(-1, -1 + 2*eps)` produce finite (very negative) but valid floats.
- **Linear imputation neighbor search:** `valid_idxs.iter().rev().find(k < j)` finds the nearest left neighbor; `valid_idxs.iter().find(k > j)` finds the nearest right neighbor. Both are correct because `valid_idxs` is built from `(0..row.len()).filter(...)` in ascending order.
- **Column-major write-back in imputation:** `out_data[i + j * n] = imputed[j]` correctly maps curve `i`, eval point `j` into column-major layout.
- **Non-breaking signatures:** `fdata_interpolate`, `spline_interpolate`, `linear_interp`, and `cubic_hermite_interp` are unchanged. All new items are additive.
- **Crate-root re-exports:** All six new public items (`ExtrapolationPolicy`, `ImputationMethod`, `fdata_interpolate_with_policy`, `impute_missing_values`, and all five scoring functions) are correctly re-exported from `lib.rs`.
- **explained_variance formula:** The centered-residual definition (`SS_res = ∫(residual - mean_res)^2`) correctly implements explained variance (distinct from R^2, matching scikit-fda's `explained_variance_score`). The `mean_true` computation via `integral / domain_len` is the correct functional mean.

---

_Reviewed: 2026-08-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
