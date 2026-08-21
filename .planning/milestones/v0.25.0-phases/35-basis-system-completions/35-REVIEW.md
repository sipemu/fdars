---
phase: 35-basis-system-completions
reviewed: 2026-08-21T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - fdars-core/src/basis/basis_system.rs
  - fdars-core/src/basis/monomial.rs
  - fdars-core/src/basis/exponential.rs
  - fdars-core/src/basis/power.rs
  - fdars-core/src/basis/polygonal.rs
  - fdars-core/src/multi_fdata.rs
  - fdars-core/src/pda.rs
  - fdars-core/src/basis/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/smooth_basis.rs
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 35: Code Review Report

**Reviewed:** 2026-08-21
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found (2 warnings, 3 info — no critical/blockers)

## Summary

Phase 35 adds four basis factories (`monomial_basis`, `exponential_basis`, `power_basis`,
`polygonal_basis`) returning a new `BasisSystem` struct, a `MultiFunData` multi-domain
container, an `Lfd` linear differential operator, and `principal_differential_analysis` (PDA).

**Overall assessment:** The implementation is numerically correct for its stated domain.
Column-major indexing is consistent with the `bspline_basis` convention (`basis[ti + j*n]`).
The analytic falling-factorial Gram formula is mathematically sound, and the unreachable
code paths (negative-power integrals, logarithmic degenerate case) are genuinely unreachable
for the current fixed `lfd_order=2` with non-negative integer exponents. The PDA SVD
pseudoinverse is correctly formed (`V · Σ⁻¹ · Uᵀ · y`). The hat-function partition-of-unity
invariant holds at all interior and boundary knots, including the degenerate shared-knot case,
via the deliberate half-open interval convention. The `smooth_basis.rs` change is confirmed
as a pure visibility promotion of two private helpers to `pub(crate)` with no logic change.

Two warnings are filed: a silent domain clamp in the power-basis numeric penalty, and an
undocumented and unguarded zero-order `Lfd` (empty `coefs`). Three info items cover a
quadrature-density coupling, a missing minimum `n_pts` guard in PDA/Lfd, and the
conservative-but-silent zero return in the `gram_entry` logarithmic guard.

---

## Critical Issues

None.

---

## Warnings

### WR-01: `power_penalty_numeric` silently shifts the quadrature domain for argvals with t_min in (0, 1e-10)

**File:** `fdars-core/src/basis/power.rs:208`

**Issue:** `power_penalty_numeric` clamps `t_min` to `1e-10` via `t_min.max(1e-10)`, but
the **evaluation matrix** in `power_basis` uses the original `argvals[0]` as the left
boundary. If a user supplies a strictly-positive `argvals[0]` smaller than `1e-10`
(e.g. `1e-11`), the penalty is integrated over `[1e-10, t_max]` while the basis is
evaluated over `[1e-11, t_max]`. The two matrices therefore describe different domains,
producing a penalty that is slightly too small on the interval `[t_min, 1e-10)`.

This only triggers when `requires_positive == true` (at least one non-integer or negative
exponent) and `argvals[0] < 1e-10`. It does not produce NaN or Inf, but the returned
`BasisSystem` has an inconsistent penalty domain. There is no warning to the caller.

**Fix:** Remove the `t_min_safe` clamping and use the caller-validated `t_min` directly.
The domain check before this function already guarantees all `argvals > 0`, so `t_min > 0`
is assured. If a numerical guard is truly needed for the specific case of `t.powf(e)` at
very small `t` with large negative exponent, document the guard in the docstring.

```rust
// Before (line 208–213):
let t_min_safe = t_min.max(1e-10);
// ...
let quad_t: Vec<f64> = (0..n_quad)
    .map(|i| t_min_safe + (t_max - t_min_safe) * i as f64 / (n_quad - 1) as f64)
    .collect();
// ...
let h = (t_max - t_min_safe) / (n_quad - 1) as f64;

// After:
// t_min is already guaranteed > 0 by the requires_positive domain check in power_basis.
let quad_t: Vec<f64> = (0..n_quad)
    .map(|i| t_min + (t_max - t_min) * i as f64 / (n_quad - 1) as f64)
    .collect();
let h = (t_max - t_min) / (n_quad - 1) as f64;
```

---

### WR-02: `Lfd::apply` silently accepts empty `coefs` (order 0) as an identity operator

**File:** `fdars-core/src/pda.rs:101`

**Issue:** When `self.coefs` is empty (`m = 0`), `Lfd::apply` computes `D⁰x = x` and
returns a copy of `data` with no derivatives applied. This is technically the identity
operator `Lx = x`, but it is neither documented nor guarded. A user who accidentally
creates `Lfd { coefs: vec![] }` receives silently copied data rather than a
descriptive error.

The coef-length validation loop (lines 113–120) is a no-op when `coefs` is empty, so
no `InvalidDimension` is raised.

**Fix:** Add an entry-point guard in `Lfd::apply` or in a constructor:

```rust
// At the top of Lfd::apply, after computing m:
if m == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "coefs",
        message: "Lfd requires at least one weight function (coefs.len() >= 1); \
                  use coefs = vec![vec![0.0]] for the pure-derivative operator".to_string(),
    });
}
```

Alternatively, document the zero-order identity behaviour explicitly in the `Lfd` struct
docstring and the `apply` docstring if the identity semantics are intentional.

---

## Info

### IN-01: Logarithmic guard in `gram_entry` returns 0 silently for improper integrals

**File:** `fdars-core/src/basis/monomial.rs:144–146` and `fdars-core/src/basis/power.rs:168–171`

**Issue:** When `|p| < 1e-15` and `a <= 0`, the function returns `0.0` with a comment
"improper integral on [0, b]; set to 0 conservatively". This is mathematically wrong
(the correct value is `+∞`), but the case is currently unreachable for the fixed
`lfd_order = 2` with non-negative integer exponents: any `(ei, ej)` pair with non-zero
falling factorials satisfies `ei >= 2` and `ej >= 2`, giving `p = ei+ej-3 >= 1 > 0`,
so `|p| < 1e-15` cannot occur.

If `lfd_order` is made user-configurable in a future version (e.g., `lfd_order = 1`),
the condition becomes reachable (e.g., `ei = ej = 0, lfd_order = 0` gives `p = 1`;
`ei = 0, ej = 1, lfd_order = 1` gives `p = 0`), and the silent `0.0` will produce a
wrong penalty with no warning.

**Fix:** Replace the silent `0.0` with a `ComputationFailed` propagation or at minimum
a `debug_assert!(false, "improper integral reached")`:

```rust
if p.abs() < 1e-15 {
    if a <= 0.0 {
        // Integral ∫₀ᵇ t⁻¹ dt is improper — this path is unreachable for the
        // current lfd_order=2 with non-negative integer exponents. If lfd_order
        // is ever made user-configurable, this must return Err, not 0.
        debug_assert!(false, "gram_entry: improper integral t^(-1) encountered; \
                               penalty result would be wrong");
        return 0.0;
    }
    ci * cj * (b.ln() - a.ln())
}
```

---

### IN-02: `polygonal_penalty_numeric` quadrature density is derived from `argvals.len()` not from knot count

**File:** `fdars-core/src/basis/polygonal.rs:192`

**Issue:** The fine quadrature grid has `n_quad = (argvals.len() - 1) * 10 + 1` points,
but the integration domain is the knot span `[knots[0], knots[-1]]`. When `argvals` is
coarse relative to the knot sequence (e.g., 2-point `argvals` with 20 knots), the 11-point
grid covers 19 piecewise-linear intervals with less than 1 point per interval, producing
inaccurate penalty entries.

**Concrete case:** `polygonal_basis(&[0.0, 1.0], &knots_of_length_50)` uses only 11
quadrature points for 49 intervals.

This does not crash and the penalty is still symmetric and PSD, but it may be numerically
inaccurate for smoothing applications.

**Fix:** Drive `n_quad` from the knot sequence length rather than from `argvals`:

```rust
// Replace line 192:
let n_quad = (argvals.len() - 1) * n_sub + 1;
// With:
let n_quad = (knots.len() - 1) * n_sub + 1;
```

This ensures at least 10 quadrature sub-points per piecewise-linear interval regardless
of how many evaluation points the user supplies.

---

### IN-03: `principal_differential_analysis` and `Lfd::apply` have no minimum `n_pts` guard

**File:** `fdars-core/src/pda.rs:248` (`principal_differential_analysis`) and line 104 (`Lfd::apply`)

**Issue:** Neither function checks `n_pts >= 2` before calling `crate::helpers::gradient`.
When `n_pts == 1`, `gradient` correctly returns `vec![0.0]` (all derivatives are zero),
so there is no panic. However, `principal_differential_analysis` silently returns all-zero
coefficient functions when given a single-point grid, which is numerically meaningless.
A user expecting PDA to detect or report this degenerate case receives no error.

This is consistent with the behaviour of `Lfd::apply` (returns a copy of the curve when
`n_pts == 1`), but both functions would be clearer with an explicit guard.

**Fix:** Add a check at the entry points:

```rust
// In principal_differential_analysis, after the argvals length check:
if n_pts < 2 {
    return Err(FdarError::InvalidDimension {
        parameter: "argvals",
        expected: ">= 2 (required for finite-difference derivatives)".to_string(),
        actual: n_pts.to_string(),
    });
}
```

The same guard would be reasonable in `Lfd::apply`. Note that the existing basis
factories already validate `argvals.len() >= 2`, so adding this check to PDA/Lfd
makes the API surface consistent.

---

_Reviewed: 2026-08-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
