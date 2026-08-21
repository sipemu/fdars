---
phase: 34-functional-serial-dependence-tooling
reviewed: 2026-08-21T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/fts/acf.rs
  - fdars-core/src/fts/mod.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 1
  info: 1
  total: 3
status: findings
---

# Phase 34: Code Review Report

**Reviewed:** 2026-08-21
**Depth:** standard
**Files Reviewed:** 3
**Status:** findings

## Summary

Reviewed the three files added or modified during Phase 34 (FTS-02: functional serial-dependence tooling). The implementation is well-structured, correct for the core numerical algorithms (fACF HS-norm, Bartlett HAC, Durbin-Levinson fPACF, KPSS stationarity statistic), and follows project conventions throughout — Result-returning, #[non_exhaustive] structs, #[must_use] annotations, divergence notes in rustdoc, column-major indexing, and the standard seeded-RNG permutation pattern.

One blocker was found: `functional_acf` (and `functional_pacf`) does not validate the `n_sim` parameter and passes `0` directly into `mc_band_threshold`, which panics with a usize underflow on the `n_sim - 1` subtraction (and a subsequent out-of-bounds access on the empty `realizations` Vec). Every other public entry point validates its analogous count parameter (`n_perm == 0` is checked in `stationarity_test`), making this an inconsistency.

One warning was found: the `ci` confidence-level parameter is accepted without range validation. While it cannot cause a panic, values outside (0.0, 1.0) silently produce semantically meaningless results.

The remaining finding is informational: there is no test that exercises the `n_sim == 0` error path (which would only be detected once the guard is added).

## Critical Issues

### CR-01: `n_sim = 0` panics instead of returning `FdarError::InvalidParameter`

**File:** `fdars-core/src/fts/acf.rs:166`

**Issue:** `functional_acf` (and `functional_pacf` which delegates to it) accepts `n_sim: usize` without validating that it is at least 1. When `n_sim == 0` is passed and the truncated eigenvalue list is non-empty, `mc_band_threshold` is called. Inside that function, the MC loop body never executes, leaving `realizations` as an empty `Vec`. The subsequent line:

```rust
let idx = ((ci * n_sim as f64) as usize).min(n_sim - 1);
```

performs `0usize - 1`, which panics in debug mode (overflow check) and wraps to `usize::MAX` in release mode, immediately followed by `realizations[idx]` which panics with an out-of-bounds access on the empty Vec. Either way the function panics rather than returning `FdarError::InvalidParameter`.

The analogous parameter `n_perm` in `stationarity_test` *is* correctly validated (`if n_perm == 0 { return Err(...) }`), making this a clear inconsistency.

**Fix:** Add the guard immediately after the `validate_fts_input` call in `functional_acf` (before the eigendecomposition path):

```rust
if n_sim == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "n_sim",
        message: "must be >= 1".to_string(),
    });
}
```

Also update the `# Errors` section of `functional_acf`'s doc comment to document this variant:

```text
/// * [`FdarError::InvalidParameter`] — `max_lag == 0` (must be ≥ 1)
///   **or `n_sim == 0`** (must be ≥ 1).
```

## Warnings

### WR-01: `ci` confidence level not validated; out-of-range values silently produce wrong bands

**File:** `fdars-core/src/fts/acf.rs:149,166`

**Issue:** The `ci` parameter (confidence level for the white-noise band) is passed through to `mc_band_threshold` without any validation. The quantile index is computed as:

```rust
let idx = ((ci * n_sim as f64) as usize).min(n_sim - 1);
```

- `ci < 0.0`: casting a negative f64 to usize yields 0 in Rust (saturating cast), silently returning the *minimum* realization as the band threshold. The result is numerically valid but statistically meaningless.
- `ci > 1.0`: the expression `ci * n_sim` exceeds `n_sim`; `.min(n_sim - 1)` clamps it to the maximum, silently returning the *maximum* as the band threshold.
- Neither case panics, but neither signals the caller of the error.

The project convention is to return `FdarError::InvalidParameter` for parameter-range violations.

**Fix:** Add a range check in `functional_acf` alongside the `n_sim` guard (or inside `mc_band_threshold`):

```rust
if !(ci > 0.0 && ci < 1.0) {
    return Err(FdarError::InvalidParameter {
        parameter: "ci",
        message: "must be in the open interval (0.0, 1.0)".to_string(),
    });
}
```

Update the `# Errors` rustdoc to include this variant.

## Info

### IN-01: No test covers the `n_sim = 0` error path

**File:** `fdars-core/src/fts/acf.rs:861-941` (the `error_handling` consolidated test)

**Issue:** The `error_handling` test and all other tests pass positive `n_sim` values. After fixing CR-01 by adding a guard, the new `InvalidParameter` variant will go untested. By project convention (`cargo clippy --all-targets` and coverage target 70%), every new error path should have a test.

**Fix:** Add a case to the `error_handling` test (or a new focused test) for `n_sim == 0`:

```rust
// functional_acf: n_sim == 0 must return InvalidParameter
let (good, argvals) = make_whitenoise_curves(20, 10, 99);
assert!(
    matches!(
        functional_acf(&good, &argvals, None, 0, 0.95, 1),
        Err(FdarError::InvalidParameter { parameter: "n_sim", .. })
    ),
    "functional_acf: n_sim == 0 must return InvalidParameter"
);
```

---

_Reviewed: 2026-08-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
