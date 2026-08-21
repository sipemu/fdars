---
phase: 34-functional-serial-dependence-tooling
fixed_at: 2026-08-21T00:00:00Z
review_path: .planning/phases/34-functional-serial-dependence-tooling/34-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 34: Code Review Fix Report

**Fixed at:** 2026-08-21
**Source review:** .planning/phases/34-functional-serial-dependence-tooling/34-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### CR-01: `n_sim = 0` panics instead of returning `FdarError::InvalidParameter`

**Files modified:** `fdars-core/src/fts/acf.rs`
**Commit:** 86598bd6
**Applied fix:** Added an early guard immediately after `validate_fts_input` in `functional_acf`:
```rust
if n_sim == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "n_sim",
        message: "must be >= 1".to_string(),
    });
}
```
Also updated the `# Errors` rustdoc to document the new variant. `functional_pacf` inherits the guard automatically via its delegation to `functional_acf`.

### WR-01: `ci` confidence level not validated

**Files modified:** `fdars-core/src/fts/acf.rs`
**Commit:** 86598bd6
**Applied fix:** Added a range guard alongside the `n_sim` guard:
```rust
if !(ci > 0.0 && ci < 1.0) {
    return Err(FdarError::InvalidParameter {
        parameter: "ci",
        message: "must be in the open interval (0.0, 1.0)".to_string(),
    });
}
```
The `!(ci > 0.0 && ci < 1.0)` form correctly rejects `NaN` (since `NaN > 0.0` is false), `0.0`, negative values, `1.0`, and values above `1.0`. Updated `# Errors` rustdoc to document this variant.

### IN-01: No test covers the `n_sim = 0` error path

**Files modified:** `fdars-core/src/fts/acf.rs`
**Commit:** 86598bd6
**Applied fix:** Added a new `invalid_parameter_guards` test at the end of the `tests` module covering:
- `functional_acf` with `n_sim == 0` → `InvalidParameter { parameter: "n_sim" }`
- `functional_pacf` with `n_sim == 0` → `InvalidParameter { parameter: "n_sim" }`
- `functional_acf` with `ci = 1.5` → `InvalidParameter { parameter: "ci" }`
- `functional_acf` with `ci = 0.0` → `InvalidParameter { parameter: "ci" }`
- `functional_acf` with `ci = -0.1` → `InvalidParameter { parameter: "ci" }`
- `functional_pacf` with `ci = 1.0` → `InvalidParameter { parameter: "ci" }`

## Verification

**Test run (main checkout):** `cargo test -p fdars-core --features linalg,parallel fts::`
- 25 tests passed (24 pre-existing + 1 new `invalid_parameter_guards`), 0 failed.

**Clippy (main checkout):** `cargo clippy -p fdars-core --features linalg,parallel -- -D warnings`
- No warnings.

---

_Fixed: 2026-08-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
