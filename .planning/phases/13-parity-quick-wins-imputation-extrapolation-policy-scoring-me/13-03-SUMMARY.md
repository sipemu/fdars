---
phase: 13-parity-quick-wins-imputation-extrapolation-policy-scoring-me
plan: 13-03
subsystem: helpers / scoring
tags: [bug-fix, gap-closure, extrapolation, scoring, imputation]
status: complete
reviewed: 2026-08-12T00:00:00Z
commits:
  - efa190ad
  - ef8f6999
duration_minutes: 25
tasks_completed: 2
files_modified:
  - fdars-core/src/helpers.rs
  - fdars-core/src/scoring.rs
  - fdars-core/src/lib.rs
---

# Phase 13 Plan 03: Review Bug-Fix Pass + FEAT-04 Gap Closure Summary

**One-liner:** Fix four code-review findings (CR-01 Periodic NaN, CR-02 EV logic error, WR-01 misleading m=0 error, IN-01 redundant cfg-test) and add `spline_interpolate_with_policy` to close the VERIFICATION gap against ROADMAP SC#2.

---

## Commit A — `fix(13)`: efa190ad

Addresses all four findings from `13-REVIEW.md`.

### CR-01 (critical) — Periodic extrapolation silently produces NaN on zero-length domain

**File:** `fdars-core/src/helpers.rs` — `fdata_interpolate_with_policy`

**Root cause:** When `argvals[0] == argvals[m-1]` (degenerate domain), `domain_len == 0.0`. The Periodic branch computed `x % 0.0` which is `NaN` in IEEE 754, not a panic. The NaN silently propagated through `linear_interp` / `cubic_hermite_interp` and the function returned `Ok(...)` containing garbage values.

**Fix:** Added a pre-loop guard:
```rust
if domain_len <= 0.0 && matches!(policy, ExtrapolationPolicy::Periodic) {
    return Err(FdarError::InvalidParameter { parameter: "argvals", .. });
}
```

**Test added:** `test_extrapolation_periodic_zero_length_domain_errors` — verifies `argvals = [5.0, 5.0, 5.0]` + Periodic + OOB query returns `InvalidParameter("argvals")`.

---

### CR-02 (critical) — `functional_explained_variance` returns 1.0 when ss_res > ss_tot

**File:** `fdars-core/src/scoring.rs` — `functional_explained_variance`

**Root cause:** The inner guard `if ss_res < NUMERICAL_EPS` used an absolute threshold. When both `ss_tot` and `ss_res` fell below `NUMERICAL_EPS` but `ss_res > ss_tot`, the function returned `1.0` (perfect fit). Concretely: a constant `y_true` + tiny oscillating `y_pred` would report EV = 1.0 instead of <= 0.0.

**Fix:** Replaced the absolute inner check with a relative one:
```rust
// Before:
if ss_res < NUMERICAL_EPS { 1.0 } else { 0.0 }
// After:
if ss_res <= ss_tot * (1.0 + 1e-6) { 1.0 } else { 0.0 }
```

**Test added:** `test_explained_variance_constant_true_perturbed_pred` — constant `y_true = 5.0`, `y_pred = 5.0 + 1e-6 * sin(2πt)` over 100 points; asserts `ev <= 0.0`.

---

### WR-01 (warning) — `impute_missing_values` misleading error for m=0 matrix

**File:** `fdars-core/src/helpers.rs` — `impute_missing_values`

**Root cause:** Without an explicit m=0 guard, the per-curve loop ran and reported `"curve 0 contains only NaN values"` — factually incorrect because a zero-column curve has no values at all.

**Fix:** Added a degenerate-column guard before the loop:
```rust
if m == 0 {
    return Err(FdarError::InvalidDimension { parameter: "data", expected: "m >= 1", actual: "m=0" });
}
```

**Test added:** `test_impute_zero_columns_errors` — `FdMatrix::zeros(2, 0)` with empty argvals returns `InvalidDimension("data")`.

---

### IN-01 (info) — Redundant `#[cfg(test)]` on use inside test module

**File:** `fdars-core/src/scoring.rs` — `mod tests`

**Fix:** Removed the inner `#[cfg(test)]` attribute on the `use crate::test_helpers::uniform_grid;` line, which was already inside a `#[cfg(test)] mod tests` block. Purely cosmetic; no behavior change.

---

## Commit B — `feat(13)`: ef8f6999

### FEAT-04 Gap Closure — `spline_interpolate_with_policy`

**Verification gap closed:** `13-VERIFICATION.md` score 7/8 — truth #8 FAILED.

ROADMAP SC#2 requires `ExtrapolationPolicy` to thread through **both** the linear/cubic path and `spline_interpolate`. The original phase implementation (`fdata_interpolate_with_policy`) covered only the linear + cubic-Hermite path. `spline_interpolate` retained its original signature and always errored on OOB (Exception-equivalent only).

**New public function:** `spline_interpolate_with_policy(data, argvals, query_points, order, policy: ExtrapolationPolicy) -> Result<FdMatrix, FdarError>`

- Located in `fdars-core/src/helpers.rs`, immediately after `spline_interpolate`
- Non-breaking: existing `spline_interpolate` signature is unchanged
- Reuses the same SVD pseudoinverse / B-spline basis machinery as `spline_interpolate`
- Policy semantics match `fdata_interpolate_with_policy` exactly:
  - `Boundary`: clamp OOB query to `t_min`/`t_max` before spline evaluation
  - `Exception`: return `InvalidParameter("query_points")` on first OOB query
  - `Fill(v)`: set OOB output cells to constant `v`; in-range cells use full spline interpolation
  - `Periodic`: wrap OOB query via `((q - t_min) % domain_len + domain_len) % domain_len`; zero-length-domain guard identical to CR-01 fix
- Re-exported at crate root in `lib.rs` next to `spline_interpolate`

**Tests added (6):**
- `test_spline_with_policy_in_range_matches_spline` — in-range output identical to `spline_interpolate`
- `test_spline_with_policy_boundary` — OOB clamped to nearest boundary value
- `test_spline_with_policy_exception` — OOB returns error; in-range succeeds
- `test_spline_with_policy_fill` — OOB cells get fill value; in-range uses spline
- `test_spline_with_policy_periodic` — OOB queries wrapped modulo domain
- `test_spline_with_policy_periodic_zero_length_domain_errors` — degenerate domain + Periodic returns error

---

## Test Suite After Both Commits

| Suite | Before | After | Delta |
|-------|--------|-------|-------|
| Unit tests (fdars-core lib) | 1984 | 1993 | +9 |
| Integration tests | 532 | 532 | 0 |
| Doc tests | 138 | 138 | 0 |
| **Total** | **2654** | **2663** | **+9** |

All 2663 tests pass. CI-parity clippy (`--all-targets --all-features -D warnings`) clean.

---

## Deviations from Plan

None — all four review findings were closed as specified. The `spline_interpolate_with_policy` implementation follows the VERIFICATION gap description exactly. The existing `spline_interpolate` signature is unchanged.

---

## Self-Check

- [x] `fdars-core/src/helpers.rs` modified (CR-01, WR-01, FEAT-04)
- [x] `fdars-core/src/scoring.rs` modified (CR-02, IN-01)
- [x] `fdars-core/src/lib.rs` modified (`spline_interpolate_with_policy` re-exported)
- [x] Commit A: `efa190ad`
- [x] Commit B: `ef8f6999`
- [x] All 1993 unit tests pass
- [x] Clippy clean
- [x] VERIFICATION gap (truth #8) closed
