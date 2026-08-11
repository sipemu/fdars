---
phase: 10-capability-gaps-spline-interpolation-functional-summary-stat
reviewed: 2026-08-10T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/helpers.rs
  - fdars-core/src/fdata.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-08-10
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 10 added `spline_interpolate` (B-spline fit-then-evaluate) in `helpers.rs` and five functional
descriptive-statistics functions (`functional_variance`, `functional_std`, `functional_covariance`,
`depth_based_median`, `trim_mean`) in `fdata.rs`, with crate-root re-exports in `lib.rs`.

The five statistics functions are numerically correct: Bessel correction is applied, the covariance
overflow guard (`checked_mul`) is present, `trim_mean` correctly prevents division by zero for
`alpha < 1`, and NaN comparators use the established `unwrap_or(Equal)` pattern. No critical
correctness or security issues were found.

`spline_interpolate` has three quality gaps. First, the knot-count formula
`nknots = m.saturating_sub(order).max(2)` produces one extra basis function (nbasis = m + 1 > m)
when `order == m − 1`, making the normal-equation system underdetermined; the pseudoinverse then
returns a minimum-norm approximation rather than an interpolant, silently violating the function's
documentation contract. Second, the out-of-domain guard for query points uses `argvals[0]` and
`argvals[m-1]` as bounds but does not reject `NaN` query values (IEEE comparisons with `NaN` are
always `false`). Third, the documented precondition that `argvals` must be sorted is not validated
at runtime, while internally `bspline_basis` computes `t_min`/`t_max` from the actual min/max of
`argvals`, creating a latent inconsistency for callers who pass unsorted grids.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `spline_interpolate` — underdetermined system when `order == m − 1`

**File:** `fdars-core/src/helpers.rs:460`

**Issue:** The knot count `nknots = m.saturating_sub(order).max(2)` clamps to 2 when
`m − order == 1` (i.e. `order == m − 1`), yielding `nbasis = nknots + order = m + 1`.
This produces more unknowns than equations in the B × coef = y system (B is m × (m+1)).
The SVD pseudoinverse returns a minimum-norm least-squares solution instead of an exact
interpolant. The function is documented, named, and tested as an "interpolant" — callers
evaluating at the original `argvals` after fitting will observe nonzero residuals.

The case arises in practice whenever a caller passes a high-order spline close to the
data count (e.g. `m = 5, order = 4` for cubic-in-disguise experiments).

**Fix:** Add an explicit guard that rejects or caps `order` to avoid the underdetermined
branch, or document the minimum-norm fallback clearly in the error conditions:

```rust
// After the order == 0 || order >= m check, add:
let nknots = m.saturating_sub(order).max(2);
let nbasis = nknots + order;
if nbasis > m {
    return Err(crate::FdarError::InvalidParameter {
        parameter: "order",
        message: format!(
            "order={order} with m={m} produces an underdetermined spline system \
             (nbasis={nbasis} > m); use order <= m-2 for exact interpolation"
        ),
    });
}
```

Alternatively, cap `nknots` so `nbasis` never exceeds `m`:

```rust
let nknots = m.saturating_sub(order).max(2);
// nbasis = nknots + order; cap so we stay exactly determined
let nknots = nknots.min(m.saturating_sub(order)); // recompute to guarantee nbasis == m
```

---

### WR-02: `spline_interpolate` — `NaN` query points bypass the out-of-domain guard

**File:** `fdars-core/src/helpers.rs:447-455`

**Issue:** The bounds check `if q < t_min || q > t_max` silently passes for `NaN` query
points because IEEE 754 comparisons with `NaN` always evaluate to `false`. A `NaN`
query point is then forwarded to the B-spline evaluator, which produces `NaN` columns in
the output `FdMatrix` without returning an error — despite the documented contract that
all query points must lie in `[argvals[0], argvals[m-1]]`.

**Fix:** Add an explicit `is_nan` check alongside the bounds test:

```rust
for &q in query_points {
    if q.is_nan() || q < t_min || q > t_max {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "query_points",
            message: format!(
                "all query points must be finite and lie in [{t_min}, {t_max}]; \
                 found {q} which is outside the interpolation domain"
            ),
        });
    }
}
```

---

### WR-03: `spline_interpolate` — documented `argvals` sort precondition is unvalidated; bounds-check uses endpoints instead of actual min/max

**File:** `fdars-core/src/helpers.rs:404,445-446`

**Issue:** The doc comment states `argvals` "must be sorted" but there is no runtime
check. Internally, `bspline_basis` computes the knot domain from `argvals.iter().fold(…,
f64::min/max)` — the true minimum and maximum — whereas the query-point bounds check uses
`argvals[0]` and `argvals[m-1]`. For an unsorted `argvals` slice these differ, creating
two failure modes:

1. A query point that lies within the true domain `[min, max]` is rejected as
   out-of-domain if it falls outside `[argvals[0], argvals[m-1]]`.
2. A query point that lies outside the true domain is silently accepted and extrapolated
   against an incorrect knot vector.

**Fix:** Either validate `argvals` is sorted before use, or align the bounds check with
the actual domain used by `bspline_basis`:

```rust
// Compute actual domain bounds (consistent with bspline_basis internals)
let t_min = argvals.iter().copied().fold(f64::INFINITY, f64::min);
let t_max = argvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
// Then validate monotonicity:
if argvals.windows(2).any(|w| w[1] < w[0]) {
    return Err(crate::FdarError::InvalidParameter {
        parameter: "argvals",
        message: "argvals must be sorted in non-decreasing order".to_string(),
    });
}
```

---

## Info

### IN-01: Misleading error message in `depth_based_median` for the `m == 0` case

**File:** `fdars-core/src/fdata.rs:443-446`

**Issue:** The `ComputationFailed` error detail string reads:

> "depth vector is empty (should not occur with n >= 1)"

This message is inaccurate when `m == 0` (zero evaluation points): a non-empty `data`
matrix with zero columns can yield an empty depth vector even with `n >= 1`, because
`fraiman_muniz_1d` may return an empty result for degenerate inputs. The phrase "should
not occur with n >= 1" is false in that edge case and will confuse callers.

**Fix:**

```rust
.ok_or_else(|| FdarError::ComputationFailed {
    operation: "depth_based_median",
    detail: "depth vector is empty; data must have n >= 1 rows and m >= 1 columns"
        .to_string(),
})
```

---

### IN-02: `spline_interpolate` missing `#[must_use]` annotation (project convention)

**File:** `fdars-core/src/helpers.rs:416`

**Issue:** Per the project convention documented in `CLAUDE.md`, expensive computation
functions are annotated with `#[must_use]`. `fdata_interpolate` immediately above carries
`#[must_use]` (line 365), but `spline_interpolate` does not. The compiler's built-in
`#[must_use]` on `Result` provides partial protection, but the project-wide convention
should be followed for consistency.

**Fix:**

```rust
#[must_use]
pub fn spline_interpolate(
```

---

_Reviewed: 2026-08-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
