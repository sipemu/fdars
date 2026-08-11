# Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics - Research

**Researched:** 2026-08-11
**Domain:** Rust FDA codebase — interpolation, NaN imputation, functional scoring metrics
**Confidence:** HIGH (all claims verified by reading source files this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Non-breaking, additive.** Do NOT change existing public signatures (e.g. `fdata_interpolate`,
  `spline_interpolate`). Where a new option must reach an existing function, use the established
  Phase-12 pattern: a new `*_with_policy` wrapper or an `Option`-typed addition — never a breaking
  positional param. New functions are fine.
- **Every new public function returns `Result<_, FdarError>`** with dimension/parameter validation
  (never panic on input) — this OVERRIDES the audit backlog's sketch of bare-`f64` scoring returns.
- **FEAT-04 enum name & variants:** `ExtrapolationPolicy` with `Boundary` (clamp to nearest edge),
  `Exception` (return `FdarError` on out-of-range — name matches scikit-fda
  `ExceptionExtrapolation` and REQUIREMENTS, NOT the backlog's `Error`), `Fill(f64)` (constant
  fill), `Periodic` (wrap modulo domain). Derive `Debug, Clone, PartialEq` (+ conditional serde
  per crate convention).

### Claude's Discretion
- **FEAT-03:** `impute_missing_values` returning `Result<FdMatrix, FdarError>` (or an in-place
  `&mut` variant — planner picks the cleaner Result-based signature). Provide at least two
  strategies via an `ImputationMethod` enum: `Linear` (reuse `helpers::linear_interp` between
  nearest non-NaN neighbors) and `Mean`/`Constant(f64)`. Validate + error on all-NaN curves or
  unsupported input. Location: `helpers.rs` (reuse `irreg_fdata/` interp infra if it fits).
- **FEAT-04:** thread the policy through the interpolation/evaluation path non-breakingly (wrapper
  or new fn). Reuse the existing spline/linear machinery from v0.15.0.
- **FEAT-05:** new `fdars-core/src/scoring.rs` module; `functional_mae`, `functional_mse`,
  `functional_mape`, `functional_msle`, `functional_explained_variance` over
  `(y_true, y_pred, argvals)`, integrated over `argvals`, each `Result`-returning with shape
  validation; re-export at crate root. Reuse existing `r_squared`/integration helpers where
  sensible.

### Deferred Ideas (OUT OF SCOPE)
- Additional imputation strategies beyond mean/linear (spline-based, KNN) — future.
- Extrapolation policies beyond the four named variants — future.
- Additional metrics beyond the five named — future.
</user_constraints>

---

## Summary

Phase 13 adds three independent, additive capability gaps to `fdars-core`. All work is
non-breaking: no existing public signature changes. The implementation sites are fully
identified. Two features (FEAT-03 imputation and FEAT-04 extrapolation policy) both touch
`helpers.rs` and must be serialized in the plan. FEAT-05 (scoring.rs) is a new file and fully
independent.

**FEAT-03 (imputation):** No NaN imputation exists today. The `impute_missing_values` function
will live in `helpers.rs`, using `helpers::linear_interp` directly for the `Linear` strategy.
The `irreg_fdata` infra (CSR layout, `pub(super) fn linear_interp`) is a different abstraction
optimized for sparse data and is NOT reusable as-is; use `helpers::linear_interp` instead.

**FEAT-04 (extrapolation policy):** `fdata_interpolate` silently clamps at boundaries
(`helpers.rs:172–191` and `:513–525`). `spline_interpolate` errors for out-of-range query points
(`helpers.rs:447–455`). The cleanest non-breaking extension is a new function
`fdata_interpolate_with_policy` that applies the policy at each query point and delegates
interpolation to the existing private helpers. Policy is applied AFTER the interpolation step
(post-call dispatch), not inside the low-level interpolators.

**FEAT-05 (scoring):** No `scoring.rs` exists. Five metrics are added, each integrating an
error function over `argvals` using `helpers::simpsons_weights`. The integration pattern is
identical to `utility::integrate_simpson` (weight dot-product). `r_squared`/`r_squared_adj` in
`helpers.rs` accept `(y_true, residuals)` — the new `functional_explained_variance` takes
`(y_true, y_pred, argvals)` and computes SS_res/SS_tot via Simpson weights.

**Primary recommendation:** Implement FEAT-03 + FEAT-04 sequentially in `helpers.rs` (one plan
wave), then FEAT-05 independently in a new `scoring.rs` (can be a parallel wave or separate
plan).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| NaN imputation | Shared utility (`helpers.rs`) | — | Operates on `FdMatrix` like other matrix-level helpers |
| Extrapolation policy enum | Shared utility (`helpers.rs`) | — | Collocated with `InterpolationMethod` enum and `fdata_interpolate` |
| Extrapolation wrapper fn | Shared utility (`helpers.rs`) | — | Wraps existing private interpolation helpers |
| Functional scoring metrics | New module (`scoring.rs`) | — | Independent; analogous to `cv.rs` metric functions but integrated over argvals |

---

## Standard Stack

### Core (no new dependencies)

All three features use only the existing crate-internal machinery. No new crates are required.

| Tool | Version | Purpose | Source |
|------|---------|---------|--------|
| `helpers::simpsons_weights` | crate-internal | Integration weights for scoring metrics | `[VERIFIED: fdars-core/src/helpers.rs:57-86]` |
| `helpers::linear_interp` | crate-internal | Linear interpolation for imputation and policy | `[VERIFIED: fdars-core/src/helpers.rs:172-191]` |
| `helpers::fdata_interpolate` | crate-internal | Existing interpolation dispatcher | `[VERIFIED: fdars-core/src/helpers.rs:366-391]` |
| `helpers::cubic_hermite_interp` | crate-internal (private) | CubicHermite branch | `[VERIFIED: fdars-core/src/helpers.rs:513-570]` |
| `matrix::FdMatrix` | crate-internal | Data container; column-major layout | `[VERIFIED: fdars-core/src/matrix.rs:1-44]` |
| `utility::integrate_simpson` | crate-internal | Integration pattern reference | `[VERIFIED: fdars-core/src/utility.rs:15-26]` |

### Package Legitimacy Audit

> Not applicable — no external crates are added in this phase.

---

## Interpolation Path Detail (FEAT-04)

### Current Signatures

**`linear_interp`** — `[VERIFIED: fdars-core/src/helpers.rs:172-191]`
```rust
pub fn linear_interp(x: &[f64], y: &[f64], t: f64) -> f64
```
Current out-of-range behavior: silent boundary clamp.
- `t <= x[0]` → returns `y[0]` (line 173–175)
- `t >= x[last]` → returns `y[last]` (line 177–179)

**`cubic_hermite_interp`** — `[VERIFIED: fdars-core/src/helpers.rs:513-525]` (private `fn`)
```rust
fn cubic_hermite_interp(x: &[f64], y: &[f64], t: f64) -> f64
```
Current out-of-range behavior: identical silent clamp at lines 520–525.

**`fdata_interpolate`** — `[VERIFIED: fdars-core/src/helpers.rs:365-391]`
```rust
#[must_use]
pub fn fdata_interpolate(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    new_argvals: &[f64],
    method: InterpolationMethod,
) -> crate::matrix::FdMatrix
```
No validation; no `Result`. Returns silently clamped values for any out-of-range `new_argvals`.
Doc comment says "within original domain" but does not enforce it. [VERIFIED: fdars-core/src/helpers.rs:352-391]

**`spline_interpolate`** — `[VERIFIED: fdars-core/src/helpers.rs:416-508]`
```rust
pub fn spline_interpolate(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    query_points: &[f64],
    order: usize,
) -> Result<crate::matrix::FdMatrix, crate::FdarError>
```
Out-of-range behavior: returns `Err(FdarError::InvalidParameter { parameter: "query_points", ... })`
for any query point outside `[argvals[0], argvals[m-1]]` (lines 446-455).
This means `spline_interpolate` already behaves like `Exception` policy — it errors on OOB.

### Existing Call Sites

`fdata_interpolate` and `spline_interpolate` are re-exported from `lib.rs:172-177` but are NOT
called anywhere inside `src/` other than `helpers.rs` itself (verified by grep: only `lib.rs`
re-export found). No examples or benchmarks call them. [VERIFIED: grep output — zero internal
call sites besides lib.rs re-export]

This means a new `fdata_interpolate_with_policy` function cannot break any existing internal
callers. The original `fdata_interpolate` stays untouched.

### Recommended Non-Breaking Design

**New function** (Phase-12 `*_with_band` pattern):

```rust
/// Extrapolation policy controlling behavior when a query point falls
/// outside the domain of `argvals`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtrapolationPolicy {
    /// Clamp to the nearest boundary value.
    Boundary,
    /// Return an error for any out-of-range query point.
    Exception,
    /// Fill with a constant value.
    Fill(f64),
    /// Wrap query points modulo the domain length (periodic extension).
    Periodic,
}

#[must_use]
pub fn fdata_interpolate_with_policy(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    new_argvals: &[f64],
    method: InterpolationMethod,
    policy: ExtrapolationPolicy,
) -> Result<crate::matrix::FdMatrix, crate::FdarError>
```

**Where the policy is applied:** Inside `fdata_interpolate_with_policy`, before calling the
private interpolation logic, each query point `t` in `new_argvals` is tested against
`[argvals[0], argvals[m-1]]`. Out-of-range points are handled by the policy:

- `Boundary`: clamp `t` to `[t_min, t_max]` then pass to existing `linear_interp` /
  `cubic_hermite_interp`.
- `Exception`: return `Err(FdarError::InvalidParameter { parameter: "new_argvals", message: ... })`
  immediately.
- `Fill(v)`: write `v` directly into the result cell, skip interpolation.
- `Periodic`: transform `t` → `t_min + ((t - t_min) % domain_len + domain_len) % domain_len`,
  then interpolate normally. `domain_len = argvals[m-1] - argvals[0]`.

The policy is NOT threaded into `linear_interp` or `cubic_hermite_interp` — those remain
unchanged private helpers. The wrapper owns the dispatch.

**Why not `Option<ExtrapolationPolicy>` on existing functions:** The existing
`fdata_interpolate` returns bare `FdMatrix` (no `Result`). Adding an `Option` param would either
require a breaking signature change or an awkward `unwrap_or` pattern inside the function that
silently ignores errors. The new-function approach (Phase-12 pattern) is cleaner and matches
precedent. [VERIFIED: fdars-core/src/alignment/karcher.rs:343-359 — `karcher_mean_with_band`
delegates to `karcher_mean_impl` with `band_frac: Option<f64>`]

**Existing tests for `fdata_interpolate` / `spline_interpolate`:** located in
`helpers.rs:907-1112` (inline `#[cfg(test)] mod tests`). None of these tests pass out-of-range
query points to `fdata_interpolate` (they only test in-range values), so adding the new function
cannot affect them.

---

## Imputation Detail (FEAT-03)

### Existing NaN Handling

No imputation exists in the codebase. NaN values propagate silently through most operations.
The only NaN-specific code found: `helpers::sort_nan_safe` (line 10-12), `utility.rs:312-315`
(guards a log computation), and `cv.rs:876` (NaN ordering comment). No `impute_missing_values`,
no `ImputationMethod` enum, no `is_nan()` checks in `FdMatrix` methods.
[VERIFIED: grep for `impute\|ImputationMethod\|missing_values` found zero matches in src/]

### What Can Be Reused

**`helpers::linear_interp`** (line 172) — directly reusable for the `Linear` imputation
strategy. The imputation logic finds the nearest non-NaN neighbors on each side of a gap and
calls `linear_interp(valid_argvals, valid_values, t)` for each NaN position.

**`irreg_fdata/mod.rs:193-213` — `pub(super) fn linear_interp`** — this is a different
function (`pub(super)`, CSR-layout specific). It cannot be reused from `helpers.rs`. Ignore for
imputation.
[VERIFIED: fdars-core/src/irreg_fdata/mod.rs:192-213]

### Column-Major Access Pattern for Imputation

FdMatrix uses column-major layout: `data[(i, j)]` is at `data.data[i + j * nrows]`.
- Row `i` (curve `i`): access via `data.row(i)` → `Vec<f64>` (O(ncols) allocation) or
  `data.row_to_buf(i, &mut buf)` (zero-alloc).
- Column `j` (all observations at eval point `j`): `data.column(j)` → contiguous `&[f64]`.

For imputation, the natural unit is a **row** (one curve). Allocating a `Vec<f64>` per row via
`data.row(i)` is acceptable for a function that already allocates a result `FdMatrix`.
[VERIFIED: fdars-core/src/matrix.rs:146-173]

### Recommended Signature

```rust
/// Imputation strategy for in-grid NaN values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImputationMethod {
    /// Linear interpolation between nearest non-NaN neighbors.
    Linear,
    /// Replace NaN with the curve's mean of non-NaN values.
    Mean,
    /// Replace NaN with a constant value.
    Constant(f64),
}

/// Impute NaN values in a regular functional data matrix.
///
/// Returns a new `FdMatrix` with NaN values replaced according to `method`.
/// Leading/trailing NaN (at boundary — no neighbor on one side) are filled by
/// the nearest valid value (boundary extension) for `Linear`.
///
/// # Errors
/// - `FdarError::InvalidParameter` if any curve consists entirely of NaN values.
/// - `FdarError::InvalidDimension` if `argvals.len() != data.ncols()`.
#[must_use]
pub fn impute_missing_values(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    method: ImputationMethod,
) -> Result<crate::matrix::FdMatrix, crate::FdarError>
```

**Why `Result<FdMatrix, FdarError>` not in-place `&mut`:** Matches crate convention for all
public transformations (`fdata_interpolate` → `FdMatrix`, `spline_interpolate` →
`Result<FdMatrix, …>`). In-place `&mut` forces the caller to have a mutable binding and is less
composable. The `#[must_use]` annotation is mandatory per crate convention (74+ expensive
functions so marked).

**All-NaN detection:** For each curve `i`, collect `valid_count = row.iter().filter(|v|
v.is_finite()).count()`. If `valid_count == 0`, return
`Err(FdarError::InvalidParameter { parameter: "data", message: "curve {i} contains only NaN values" })`.

**Linear imputation algorithm per curve:**
1. Gather the row as `Vec<f64>` via `data.row(i)`.
2. Find indices where `!y[j].is_nan()` → `valid_indices: Vec<usize>`.
3. For each NaN position `j`:
   - Find left neighbor `(j_l, y_l)` = last valid index ≤ j.
   - Find right neighbor `(j_r, y_r)` = first valid index ≥ j.
   - If both exist: call `linear_interp(&[argvals[j_l], argvals[j_r]], &[y_l, y_r], argvals[j])`.
   - If only left (leading gap on left): `y_l` (boundary fill).
   - If only right (leading gap on right): `y_r` (boundary fill).
4. Write back to result matrix column-major.

---

## Scoring Metrics Detail (FEAT-05)

### Existing Scoring Infrastructure

**`helpers::r_squared(y_true: &[f64], residuals: &[f64]) -> f64`** — `[VERIFIED: fdars-core/src/helpers.rs:301-313]`
Takes `residuals`, not `y_pred`. Accepts plain slices. Returns `f64`, not `Result`. Not directly
reusable for functional metrics (no argvals, different interface).

**`helpers::r_squared_adj(y_true: &[f64], residuals: &[f64], p: usize) -> f64`** — `[VERIFIED: fdars-core/src/helpers.rs:317-323]`

**`cv::metric_mae(y_true: &[f64], y_pred: &[f64]) -> f64`** — `[VERIFIED: fdars-core/src/cv.rs:126-132]`
Pointwise MAE (no integration, no argvals). Not directly reusable but is the reference formula.

**`cv::metric_r_squared(y_true: &[f64], y_pred: &[f64]) -> f64`** — `[VERIFIED: fdars-core/src/cv.rs:135-148]`
Pointwise R². Takes `y_pred` (not residuals). Reference formula only.

**`utility::integrate_simpson(values: &[f64], argvals: &[f64]) -> f64`** — `[VERIFIED: fdars-core/src/utility.rs:15-26]`
The canonical integration primitive. Called as: `let weights = simpsons_weights(argvals); dot(values, weights)`.
All five scoring functions should use `simpsons_weights` the same way.

### No scoring.rs Exists

`ls fdars-core/src/scoring.rs` → not found. [VERIFIED: bash output]

### Recommended Module Structure

New file: `fdars-core/src/scoring.rs`

```rust
//! Functional scoring metrics — MAE, MSE, MAPE, MSLE, explained variance.
//!
//! All metrics integrate the pointwise error function over `argvals` using
//! Simpson's rule, producing a single scalar score per metric.

use crate::helpers::simpsons_weights;
use crate::FdarError;
use crate::matrix::FdMatrix;

/// Functional Mean Absolute Error integrated over argvals.
///
/// `functional_mae = ∫ |y_true(t) - y_pred(t)| dt` approximated by Simpson's rule.
///
/// # Errors
/// - `InvalidDimension` if shapes of `y_true`, `y_pred`, or `argvals` are inconsistent.
pub fn functional_mae(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

/// Functional Mean Squared Error integrated over argvals.
pub fn functional_mse(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

/// Functional Mean Absolute Percentage Error.
///
/// MAPE = ∫ |y_true - y_pred| / |y_true| dt, integrated by Simpson's rule.
///
/// # Errors
/// - `InvalidParameter` if any value of `y_true` is zero or near-zero
///   (|y_true| < NUMERICAL_EPS), which would cause division by zero.
pub fn functional_mape(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

/// Functional Mean Squared Logarithmic Error.
///
/// MSLE = ∫ (log(1 + y_true) - log(1 + y_pred))^2 dt.
///
/// # Errors
/// - `InvalidParameter` if any value of `y_true` or `y_pred` is < -1
///   (making log(1+x) undefined for negative values below -1).
pub fn functional_msle(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

/// Functional Explained Variance Score.
///
/// EV = 1 - Var(y_true - y_pred) / Var(y_true), where variance is computed
/// as the integrated squared deviation from the integrated mean.
///
/// Returns 1.0 for perfect prediction; 0.0 if prediction equals mean.
/// Can be negative for worse-than-mean predictions.
///
/// # Errors
/// - `InvalidDimension` if shapes are inconsistent.
pub fn functional_explained_variance(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>
```

**Shape contract for all five functions:**
- `y_true.shape() == y_pred.shape()` → else `InvalidDimension { parameter: "y_pred", ... }`
- `argvals.len() == y_true.ncols()` → else `InvalidDimension { parameter: "argvals", ... }`
- `y_true.nrows() >= 1` and `argvals.len() >= 2` → else `InvalidDimension` (degenerate input)

**Integration pattern (identical for all five):**
```rust
let weights = simpsons_weights(argvals);   // Vec<f64>, length m
let m = argvals.len();
let n = y_true.nrows();
let mut total = 0.0;
for i in 0..n {
    for j in 0..m {
        let err = pointwise_error(y_true[(i,j)], y_pred[(i,j)]);  // metric-specific
        total += err * weights[j];
    }
}
total / n as f64  // average over curves
```

### lib.rs Wiring

Two changes to `fdars-core/src/lib.rs`:

1. Add `pub mod scoring;` in the "Shared utility modules" block (around line 91, adjacent to
   `pub mod helpers;`).
2. Add a new `pub use scoring::{...}` block with all five function names.

Prelude (`prelude.rs`) does NOT need to import scoring functions — they are standalone utilities,
not commonly composed types. (Precedent: `integrate_simpson`, `metric_mae`, etc. are not in
prelude either.)

---

## FdarError Variants (for Validation)

`[VERIFIED: fdars-core/src/error.rs:1-51]`

```rust
pub enum FdarError {
    InvalidDimension {
        parameter: &'static str,
        expected: String,
        actual: String,
    },
    InvalidParameter {
        parameter: &'static str,
        message: String,
    },
    ComputationFailed {
        operation: &'static str,
        detail: String,
    },
    InvalidEnumValue { enum_name: &'static str, value: i32 },
}
```

Mapping for Phase 13:
- Shape mismatch (`y_true` vs `y_pred`, `argvals` vs `ncols`) → `InvalidDimension`
- Zero denominator (MAPE), domain violation (MSLE log), out-of-range query (Exception policy),
  all-NaN curve → `InvalidParameter`
- SVD/numerical failure → `ComputationFailed` (not expected here)
- `InvalidEnumValue` — not needed (no integer-to-enum conversion in these features)

---

## Test and Prelude Conventions

### Test Location

All tests are **inline** in the same file as the implementation, in a `#[cfg(test)] mod tests`
block. [VERIFIED: fdars-core/src/helpers.rs:707-1113 — inline test block]

No separate integration test files are required for these features, but one could be added to
`fdars-core/tests/` for cross-module coverage if desired.

Shared helpers for tests: `crate::test_helpers::uniform_grid(n)` — available only under
`#[cfg(test)]`. [VERIFIED: fdars-core/src/test_helpers.rs:1-8]

### Re-export Block Location in lib.rs

The re-export block for `helpers` is at `lib.rs:172-177`:
```rust
pub use helpers::{
    aic, bandwidth_candidates_from_dists, bic, cumulative_trapz, extract_curves, fdata_interpolate,
    gaussian_kernel, gradient, gradient_nonuniform, gradient_uniform, l2_distance, linear_interp,
    quantile_sorted, r_squared, r_squared_adj, simpsons_weights, simpsons_weights_2d,
    spline_interpolate, trapz, InterpolationMethod, DEFAULT_CONVERGENCE_TOL, NUMERICAL_EPS,
};
```
[VERIFIED: fdars-core/src/lib.rs:172-177]

Phase 13 additions to this block:
- `fdata_interpolate_with_policy` (new function)
- `ExtrapolationPolicy` (new enum)
- `impute_missing_values` (new function)
- `ImputationMethod` (new enum)

New separate block for scoring (at the end of lib.rs, before line 437):
```rust
pub mod scoring;
pub use scoring::{
    functional_explained_variance, functional_mae, functional_mape, functional_mse,
    functional_msle,
};
```

### Serde Convention

`InterpolationMethod` in `helpers.rs` does NOT have conditional serde (`[VERIFIED: helpers.rs:342-350]`).

The CONTEXT.md locked decision for `ExtrapolationPolicy` is `Derive Debug, Clone, PartialEq (+
conditional serde per crate convention)`. The crate convention (from `FdMatrix`, `FpcaResult`,
etc.) is:
```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
```
Apply this to `ExtrapolationPolicy` and `ImputationMethod`.

Note: `InterpolationMethod` in helpers.rs currently lacks the serde derive — this is a
pre-existing omission, not a convention to follow for new enums. New enums in Phase 13 SHOULD
include serde per the locked decision.

---

## Architecture Patterns

### Recommended Project Structure Changes

```
fdars-core/src/
├── helpers.rs          # FEAT-03 + FEAT-04: add ImputationMethod, ExtrapolationPolicy,
│                       #   impute_missing_values, fdata_interpolate_with_policy
│                       #   (SERIALIZED — both touch this file)
├── scoring.rs          # FEAT-05: new module — 5 functional metric fns
│                       #   (INDEPENDENT — can proceed in parallel with helpers.rs work)
└── lib.rs              # update mod list + re-export blocks for all three features
```

### Pattern: `*_with_policy` Wrapper (Phase-12 pattern)

```rust
// From alignment/karcher.rs:343-359 — reference pattern [VERIFIED]
pub fn karcher_mean_with_band(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
    band_frac: Option<f64>,   // ← new parameter
) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, band_frac.unwrap_or(0.0))
}
```

Apply identically for `fdata_interpolate_with_policy`:
- Original `fdata_interpolate` stays untouched (no `Result`, no policy param)
- New `fdata_interpolate_with_policy` adds `policy: ExtrapolationPolicy`, returns `Result<FdMatrix, FdarError>`
- Internally calls existing private `linear_interp` / `cubic_hermite_interp` for in-range points

### Pattern: Simpson's Integration in Scoring

Reference from `utility::integrate_simpson` `[VERIFIED: fdars-core/src/utility.rs:15-26]`:
```rust
pub fn integrate_simpson(values: &[f64], argvals: &[f64]) -> f64 {
    if values.len() != argvals.len() || values.is_empty() {
        return 0.0;
    }
    let weights = simpsons_weights(argvals);
    values.iter().zip(weights.iter()).map(|(&v, &w)| v * w).sum()
}
```
The scoring functions should NOT call `integrate_simpson` directly (it takes `&[f64]`, not
`FdMatrix` pairs). Instead, they call `simpsons_weights(argvals)` once and apply the weights
across the error values in a single pass.

### Anti-Patterns to Avoid

- **Modifying `linear_interp` or `cubic_hermite_interp` to accept a policy:** These are
  low-level scalar functions called per point. Adding policy there forces every caller to pass
  a policy, breaking signatures.
- **Returning `f64::NAN` from scoring functions on validation failure:** Crate convention is
  `Result<_, FdarError>` — all public functions return `Err` on bad input.
- **Using `irreg_fdata`'s `pub(super) linear_interp` for imputation:** It is `pub(super)` and
  not accessible from `helpers.rs`. Use `helpers::linear_interp` instead.
- **Calling `fdata_interpolate` inside `fdata_interpolate_with_policy`:** The original fn
  returns bare `FdMatrix` (no Result) and always clamps. The wrapper must duplicate the
  per-point dispatch to apply policy before interpolation.

---

## Common Pitfalls

### Pitfall 1: MAPE Division by Zero
**What goes wrong:** `y_true` contains values near zero → `|y_true - y_pred| / |y_true|` → inf
or NaN.
**Why it happens:** MAPE is undefined when the true value is zero.
**How to avoid:** Before computing the integral, scan all `y_true[(i,j)]` values. If
`y_true[(i,j)].abs() < NUMERICAL_EPS` (≈ 1e-10, `[VERIFIED: helpers.rs:4]`), return
`Err(FdarError::InvalidParameter { parameter: "y_true", message: "MAPE is undefined when y_true contains values near zero" })`.
**Warning signs:** Users get `Inf` or `NaN` from scoring — test with a zero-containing curve.

### Pitfall 2: MSLE Domain Violation
**What goes wrong:** `log(1 + y)` requires `y > -1`. If `y_true` or `y_pred` < -1, the log
is NaN or errors.
**Why it happens:** MSLE is designed for non-negative targets (counts, prices). Applying it to
arbitrary functional data violates the domain.
**How to avoid:** Validate that `y_true[(i,j)] > -1.0 - NUMERICAL_EPS` and `y_pred[(i,j)] > -1.0
- NUMERICAL_EPS` for all `(i,j)`. Return `InvalidParameter` if violated.
**Warning signs:** `f64::NAN` propagation silently producing zero integrals.

### Pitfall 3: Periodic Wrap Arithmetic
**What goes wrong:** `t % domain_len` in Rust gives the remainder, not the modulo for negative
values. If `t < t_min`, `(t - t_min)` is negative and `% domain_len` returns a negative number.
**Why it happens:** Rust `%` is remainder-after-truncation, not mathematical modulo.
**How to avoid:** Use `((t - t_min) % domain_len + domain_len) % domain_len` to ensure a
non-negative result. Then the query point is `t_min + wrapped`.
**Warning signs:** Queries significantly below `t_min` produce boundary values instead of
wrapping correctly.

### Pitfall 4: NaN Propagation in Imputation
**What goes wrong:** If NaN detection uses `==` comparison instead of `is_nan()`, NaN values are
not found (NaN != NaN in IEEE 754).
**How to avoid:** Use `y[j].is_nan()` (not `y[j] == f64::NAN`) throughout imputation code.
**Warning signs:** Test that a synthetic NaN at position j=2 is actually replaced.

### Pitfall 5: helpers.rs File-Overlap Wave Collision
**What goes wrong:** FEAT-03 and FEAT-04 both write to `helpers.rs`. If planned as parallel
worktree tasks, the second write rebases onto a stale base and loses the first set of changes.
**How to avoid:** The plan MUST serialize FEAT-03 and FEAT-04 into the same plan or sequential
waves. FEAT-05 (`scoring.rs`) is a new file and is fully independent.
**Warning signs:** Git merge conflicts on `helpers.rs` mid-phase.

### Pitfall 6: `#[must_use]` on `fdata_interpolate_with_policy`
**What goes wrong:** Omitting `#[must_use]` on an expensive computation violates crate
convention for functions that return computed results (74+ functions annotated).
`fdata_interpolate` itself is already `#[must_use]` at line 365.
**How to avoid:** Add `#[must_use]` to `fdata_interpolate_with_policy` and
`impute_missing_values`.

### Pitfall 7: `ExtrapolationPolicy` Is `#[non_exhaustive]`?
**What goes wrong:** The CONTEXT.md locked decisions do not mention `#[non_exhaustive]`.
`InterpolationMethod` in helpers.rs at line 344 has `#[non_exhaustive]`. The locked decision
says derive `Debug, Clone, PartialEq` only. If `#[non_exhaustive]` is added, callers cannot
match all variants exhaustively without a wildcard arm — inconvenient for an enum intended for
exhaustive pattern matching.
**Recommendation:** Do NOT add `#[non_exhaustive]` to `ExtrapolationPolicy` or
`ImputationMethod`. The CONTEXT.md does not request it, and exhaustive matching is desirable for
these small, closed enums. The `Deferred Ideas` section notes no additional variants are planned.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Integration weights | Custom weight formula | `helpers::simpsons_weights` | Already handles uniform/non-uniform grids, edge cases |
| Linear interpolation for imputation | New per-segment interpolator | `helpers::linear_interp` | Binary-search based, correct boundary behavior |
| Periodic wrap | Custom modulo | `((t - t_min) % domain_len + domain_len) % domain_len` | Standard IEEE modulo recipe |
| Test grid generation | Inline linspace | `crate::test_helpers::uniform_grid(n)` | Used throughout existing tests |

---

## Code Examples

### ExtrapolationPolicy Dispatch Skeleton

```rust
// Source: derived from helpers.rs:366-391 structure [VERIFIED] + karcher.rs:343-359 pattern [VERIFIED]
pub fn fdata_interpolate_with_policy(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    new_argvals: &[f64],
    method: InterpolationMethod,
    policy: ExtrapolationPolicy,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();
    let m_new = new_argvals.len();
    // ... dimension checks ...
    let t_min = argvals[0];
    let t_max = argvals[m - 1];
    let domain_len = t_max - t_min;

    let mut result = crate::matrix::FdMatrix::zeros(n, m_new);
    for i in 0..n {
        let y: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        for (j, &t) in new_argvals.iter().enumerate() {
            let in_range = t >= t_min && t <= t_max;
            result[(i, j)] = if in_range {
                match method {
                    InterpolationMethod::Linear => linear_interp(argvals, &y, t),
                    InterpolationMethod::CubicHermite => cubic_hermite_interp(argvals, &y, t),
                }
            } else {
                match &policy {
                    ExtrapolationPolicy::Boundary => match method {
                        InterpolationMethod::Linear => linear_interp(argvals, &y, t.clamp(t_min, t_max)),
                        InterpolationMethod::CubicHermite => cubic_hermite_interp(argvals, &y, t.clamp(t_min, t_max)),
                    },
                    ExtrapolationPolicy::Exception => {
                        return Err(crate::FdarError::InvalidParameter {
                            parameter: "new_argvals",
                            message: format!("query {t} is outside domain [{t_min}, {t_max}]"),
                        });
                    }
                    ExtrapolationPolicy::Fill(v) => *v,
                    ExtrapolationPolicy::Periodic => {
                        let wrapped = t_min + ((t - t_min) % domain_len + domain_len) % domain_len;
                        match method {
                            InterpolationMethod::Linear => linear_interp(argvals, &y, wrapped),
                            InterpolationMethod::CubicHermite => cubic_hermite_interp(argvals, &y, wrapped),
                        }
                    }
                }
            };
        }
    }
    Ok(result)
}
```

Note: `cubic_hermite_interp` is currently `fn` (private). It must remain callable from within
`helpers.rs` itself (it is in the same file), so no visibility change is needed.

### Imputation Skeleton

```rust
// Source: derived from helpers.rs:172-191 [VERIFIED] and matrix.rs:146-173 [VERIFIED]
pub fn impute_missing_values(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    method: ImputationMethod,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    let mut out_data = vec![0.0_f64; n * m]; // column-major output
    for i in 0..n {
        let row: Vec<f64> = data.row(i);
        // Check all-NaN
        let valid_count = row.iter().filter(|v| !v.is_nan()).count();
        if valid_count == 0 {
            return Err(crate::FdarError::InvalidParameter {
                parameter: "data",
                message: format!("curve {i} contains only NaN values"),
            });
        }
        let imputed = impute_row(&row, argvals, &method);
        for j in 0..m {
            out_data[i + j * n] = imputed[j]; // column-major write
        }
    }
    crate::matrix::FdMatrix::from_column_major(out_data, n, m)
        .map_err(|e| e) // from_column_major already returns FdarError
}

fn impute_row(row: &[f64], argvals: &[f64], method: &ImputationMethod) -> Vec<f64> {
    let mut result = row.to_vec();
    match method {
        ImputationMethod::Mean => {
            let mean = row.iter().filter(|v| !v.is_nan()).sum::<f64>()
                / row.iter().filter(|v| !v.is_nan()).count() as f64;
            for v in &mut result {
                if v.is_nan() { *v = mean; }
            }
        }
        ImputationMethod::Constant(c) => {
            for v in &mut result {
                if v.is_nan() { *v = *c; }
            }
        }
        ImputationMethod::Linear => {
            // find valid indices, interpolate between neighbors
            let valid_idxs: Vec<usize> = (0..row.len()).filter(|&j| !row[j].is_nan()).collect();
            for j in 0..row.len() {
                if result[j].is_nan() {
                    let left = valid_idxs.iter().rev().find(|&&k| k < j);
                    let right = valid_idxs.iter().find(|&&k| k > j);
                    result[j] = match (left, right) {
                        (Some(&l), Some(&r)) => linear_interp(
                            &[argvals[l], argvals[r]],
                            &[row[l], row[r]],
                            argvals[j],
                        ),
                        (Some(&l), None) => row[l],   // boundary fill
                        (None, Some(&r)) => row[r],   // boundary fill
                        (None, None) => unreachable!(), // all-NaN already rejected
                    };
                }
            }
        }
    }
    result
}
```

### Functional MAE Skeleton (representative of all five metrics)

```rust
// Source: integration pattern from utility.rs:15-26 [VERIFIED], formula reference cv.rs:126-132 [VERIFIED]
pub fn functional_mae(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = y_true.shape();
    if y_pred.shape() != (n, m) {
        return Err(FdarError::InvalidDimension {
            parameter: "y_pred",
            expected: format!("({n}, {m})"),
            actual: format!("{:?}", y_pred.shape()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n == 0 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 1 and m >= 2".to_string(),
            actual: format!("n={n}, m={m}"),
        });
    }
    let weights = simpsons_weights(argvals);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            total += (y_true[(i, j)] - y_pred[(i, j)]).abs() * weights[j];
        }
    }
    Ok(total / n as f64)
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No NaN imputation | `impute_missing_values` (FEAT-03) | Phase 13 | Closes scikit-fda PREP-03 parity gap |
| Silent clamp on OOB interpolation | `ExtrapolationPolicy` via new wrapper | Phase 13 | Closes scikit-fda REPR-03 parity gap |
| No functional scoring | 5 integrated metrics in `scoring.rs` | Phase 13 | Closes scikit-fda MISC-04 parity gap |

**Deprecated/outdated for this phase:** Nothing deprecated. All existing functions remain
unchanged.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + criterion 0.5 for benchmarks |
| Config file | none (inline `#[cfg(test)]` blocks) |
| Quick run command | `cargo test -p fdars-core --features linalg 2>&1 \| tail -5` |
| Full suite command | `cargo test -p fdars-core --features linalg` |

### Phase Requirements → Test Map

| Req | Behavior | Test Type | File | Automated Command |
|-----|----------|-----------|------|-------------------|
| FEAT-03-a | `impute_missing_values` reproduces linear values on synthetic gap | unit | `helpers.rs` inline | `cargo test -p fdars-core test_impute_linear` |
| FEAT-03-b | `impute_missing_values` mean strategy replaces with curve mean | unit | `helpers.rs` inline | `cargo test -p fdars-core test_impute_mean` |
| FEAT-03-c | All-NaN curve returns `InvalidParameter` | unit | `helpers.rs` inline | `cargo test -p fdars-core test_impute_all_nan` |
| FEAT-04-a | `Boundary` policy clamps out-of-range queries | unit | `helpers.rs` inline | `cargo test -p fdars-core test_extrapolation_boundary` |
| FEAT-04-b | `Exception` policy returns `InvalidParameter` on OOB | unit | `helpers.rs` inline | `cargo test -p fdars-core test_extrapolation_exception` |
| FEAT-04-c | `Fill(v)` policy inserts constant | unit | `helpers.rs` inline | `cargo test -p fdars-core test_extrapolation_fill` |
| FEAT-04-d | `Periodic` policy wraps correctly | unit | `helpers.rs` inline | `cargo test -p fdars-core test_extrapolation_periodic` |
| FEAT-05-a | `functional_mae` matches hand-computed value for constant error | unit | `scoring.rs` inline | `cargo test -p fdars-core test_functional_mae` |
| FEAT-05-b | `functional_mse` matches hand-computed value | unit | `scoring.rs` inline | `cargo test -p fdars-core test_functional_mse` |
| FEAT-05-c | `functional_mape` errors on zero y_true | unit | `scoring.rs` inline | `cargo test -p fdars-core test_functional_mape_zero` |
| FEAT-05-d | `functional_msle` errors on y < -1 | unit | `scoring.rs` inline | `cargo test -p fdars-core test_functional_msle_domain` |
| FEAT-05-e | `functional_explained_variance` = 1.0 for perfect prediction | unit | `scoring.rs` inline | `cargo test -p fdars-core test_explained_variance_perfect` |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg -q 2>&1 | tail -5`
- **Per wave merge:** `cargo test -p fdars-core --features linalg`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/scoring.rs` — does not exist, must be created in Wave 0 / task 1
- [ ] No new test files needed (all inline), but scoring module must be declared in `lib.rs`

---

## Security Domain

> `security_enforcement: true`, ASVS level 1 from config.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a — pure computation library |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | `FdarError::InvalidDimension` / `InvalidParameter` at function entry; no silent truncation |
| V6 Cryptography | no | n/a |

### Known Threat Patterns for Rust numeric library

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in index arithmetic | Tampering | Rust's bounds checks + `debug_assert!`; column-major index `i + j * n` uses `usize` — safe for realistic dimensions |
| NaN/Inf propagation producing misleading results | Information Disclosure | Explicit `is_nan()` checks in imputation; `NUMERICAL_EPS` guard in MAPE |
| Division by zero in MAPE | Tampering | Pre-scan `y_true` for near-zero values before integration |
| log of non-positive in MSLE | Tampering | Pre-validate `y_true > -1` and `y_pred > -1` before computation |

---

## Open Questions

1. **`Constant(f64)` vs `Mean` naming in ImputationMethod**
   - What we know: CONTEXT.md says "at least two strategies: `Linear` and `Mean`/`Constant(f64)`" — the slash suggests both are acceptable or that one variant is `Constant(f64)` and the name `Mean` is an alternative name for another.
   - What's unclear: Whether the enum should have `Mean` and `Constant(f64)` as separate variants, or just `Mean` with the constant embedded.
   - Recommendation: Provide both as separate variants: `Mean` (curve-level mean of non-NaN values) and `Constant(f64)` (user-supplied constant). This mirrors scikit-fda's approach and is more composable.

2. **`functional_explained_variance` multi-curve aggregation**
   - What we know: EV = 1 - Var(residual)/Var(y_true), but "variance" for a set of curves can be computed two ways: (a) per-curve then averaged, or (b) global across all curves.
   - What's unclear: Which aggregation scikit-fda uses.
   - Recommendation: Compute per-curve (curve-level SS_res and SS_tot integrated over argvals), then average. This is consistent with how `functional_mae`/`functional_mse` average over curves.

3. **Leading/trailing NaN boundary behavior in `Linear` imputation**
   - What we know: At the left edge of a curve with leading NaN, there is no left neighbor, so boundary fill (nearest valid value) is used.
   - Recommendation: Document this explicitly in the function doc comment. Silently filling boundary NaN with the nearest valid neighbor is the scikit-fda default behavior.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cubic_hermite_interp` remains private `fn` (not `pub`) — calling it from within `fdata_interpolate_with_policy` in the same file is valid | Interpolation path | If moved to a submodule, visibility would need adjustment |
| A2 | No external callers of `fdata_interpolate` or `spline_interpolate` exist outside the repo (crates.io users) — non-breaking means no signature change to these functions | Interpolation path | Published API users relying on the exact signature; mitigated by adding only new functions |
| A3 | `functional_explained_variance` aggregates per-curve (not global) | Scoring metrics | Different aggregation gives different numerical results |

---

## Sources

### Primary (HIGH confidence — file contents read this session)

- `[VERIFIED: fdars-core/src/helpers.rs:1-1113]` — complete helpers module: `linear_interp` (172-191), `fdata_interpolate` (365-391), `spline_interpolate` (416-508), `cubic_hermite_interp` (513-570), `simpsons_weights` (57-86), `r_squared` (301-313), `InterpolationMethod` enum (342-350), full inline test suite (707-1113)
- `[VERIFIED: fdars-core/src/error.rs:1-51]` — `FdarError` enum with all four variants and their fields
- `[VERIFIED: fdars-core/src/lib.rs:64-438]` — module declarations (64-107), `pub use helpers` re-export (172-177), full re-export block structure
- `[VERIFIED: fdars-core/src/matrix.rs:1-214]` — `FdMatrix` struct, `from_column_major` (50-63), `column`/`column_mut` (127-140), `row` (146-150), `row_to_buf` (157-173), column-major layout documentation
- `[VERIFIED: fdars-core/src/utility.rs:15-46]` — `integrate_simpson` (15-26) and `inner_product` (34-46) — integration pattern
- `[VERIFIED: fdars-core/src/cv.rs:116-148]` — `metric_rmse`, `metric_mae`, `metric_r_squared` — reference scalar metric formulas
- `[VERIFIED: fdars-core/src/irreg_fdata/mod.rs:192-213]` — `pub(super) fn linear_interp` — confirmed not reusable from helpers.rs
- `[VERIFIED: fdars-core/src/prelude.rs:1-76]` — prelude re-exports, confirmed no metric fns in prelude
- `[VERIFIED: fdars-core/src/test_helpers.rs:1-8]` — `uniform_grid` test helper
- `[VERIFIED: fdars-core/src/alignment/karcher.rs:343-359]` — `karcher_mean_with_band` as Phase-12 `*_with_band` pattern reference
- `[VERIFIED: grep output]` — zero internal call sites for `fdata_interpolate` / `spline_interpolate` in `src/` (only lib.rs re-export); no `scoring.rs` file; no `ImputationMethod` / `ExtrapolationPolicy` / `impute_missing_values` definitions anywhere in `src/`

### Secondary (MEDIUM confidence)

- CONTEXT.md (13-CONTEXT.md) — locked decisions, feature requirements, deferred scope

---

## Metadata

**Confidence breakdown:**
- Interpolation path (FEAT-04): HIGH — all function signatures, behaviors, and call sites verified by reading source
- Imputation (FEAT-03): HIGH — confirmed no existing imputation, verified reusable helpers
- Scoring (FEAT-05): HIGH — verified no scoring.rs, confirmed integration pattern, confirmed lib.rs wiring location
- FdarError variants: HIGH — read error.rs in full
- Test/prelude conventions: HIGH — read test_helpers.rs, prelude.rs, and inline test blocks in helpers.rs

**Research date:** 2026-08-11
**Valid until:** 2026-09-11 (stable Rust codebase; re-verify if helpers.rs is touched by another phase before planning completes)
