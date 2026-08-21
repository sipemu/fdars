---
phase: 35-basis-system-completions
fixed_at: 2026-08-21T00:00:00Z
review_path: .planning/phases/35-basis-system-completions/35-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 35: Code Review Fix Report

**Fixed at:** 2026-08-21
**Source review:** `.planning/phases/35-basis-system-completions/35-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (2 Warning + 3 Info)
- Fixed: 5
- Skipped: 0

## Fixed Issues

### WR-01: `power_penalty_numeric` silently shifts the quadrature domain

**Files modified:** `fdars-core/src/basis/power.rs`
**Commit:** `6a2cdc81`
**Applied fix:** Removed `let t_min_safe = t_min.max(1e-10)` and replaced all three
uses of `t_min_safe` with `t_min` directly. Added a comment confirming `t_min > 0` is
guaranteed by the `requires_positive` domain check upstream. Added regression test
`power_penalty_domain_matches_eval_for_tiny_t_min` that supplies `argvals[0] = 1e-11`
and verifies the resulting penalty is finite and symmetric (which it was NOT before the
fix due to the domain mismatch).

---

### WR-02: `Lfd::apply` silently accepts empty `coefs` as identity

**Files modified:** `fdars-core/src/pda.rs`
**Commit:** `6a2cdc81`
**Applied fix:** Added an early-return guard immediately after computing `m = self.coefs.len()`:
when `m == 0` returns `FdarError::InvalidParameter { parameter: "coefs", message: "..." }`.
Updated the `# Errors` rustdoc section to document the new error. Added test
`lfd_empty_coefs_returns_err`.

---

### IN-01: Logarithmic guard in `gram_entry` returns 0 silently for improper integrals

**Files modified:** `fdars-core/src/basis/monomial.rs`, `fdars-core/src/basis/power.rs`
**Commit:** `6a2cdc81`
**Applied fix:** Added `debug_assert!(false, "gram_entry: improper integral t^(-1) encountered; ...")` in both copies of the `a <= 0` branch inside the `p.abs() < 1e-15` guard. The runtime `return 0.0` is preserved (the branch is currently unreachable for `lfd_order=2` with non-negative integer exponents). The `debug_assert!` will fire in debug builds if the branch ever becomes reachable after a future change. No tests added — the branch is verified unreachable by the existing geometry, and adding a test would require bypassing the caller's parameter validation.

---

### IN-02: `polygonal_penalty_numeric` quadrature density from `argvals.len()` not knot count

**Files modified:** `fdars-core/src/basis/polygonal.rs`
**Commit:** `6a2cdc81`
**Applied fix:** Changed `let n_quad = (argvals.len() - 1) * n_sub + 1` to
`let n_quad = (knots.len() - 1) * n_sub + 1`. Added a comment explaining the
rationale. This ensures at least 10 quadrature sub-points per piecewise-linear
interval regardless of how coarse the evaluation grid is relative to the knot sequence.
Existing polygonal tests still pass; no new test added (the fix is purely numeric
accuracy, not a new error path).

---

### IN-03: No `n_pts >= 2` guard in `principal_differential_analysis` and `Lfd::apply`

**Files modified:** `fdars-core/src/pda.rs`
**Commit:** `6a2cdc81`
**Applied fix:** Added `if n_pts < 2 { return Err(FdarError::InvalidDimension { ... }) }`
in both `Lfd::apply` (after the empty-coefs guard) and `principal_differential_analysis`
(after the `argvals.len() != n_pts` check). Updated `# Errors` rustdoc on both functions.
Added tests `lfd_single_point_grid_returns_err` and `pda_single_point_grid_returns_err`.

## Skipped Issues

None — all 5 findings were successfully fixed.

---

## Verification

Tests and static analysis were run in the main checkout (workflow.use_worktrees=false):

```
cargo test -p fdars-core --features linalg,parallel
  → 2363 lib/integration tests: 0 failed
  → 163 doc tests: 0 failed (4 ignored, pre-existing)
cargo clippy --all-targets --features linalg,parallel -- -D warnings
  → clean (0 warnings)
```

---

_Fixed: 2026-08-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
