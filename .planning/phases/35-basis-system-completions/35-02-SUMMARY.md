---
phase: 35-basis-system-completions
plan: "02"
subsystem: basis
status: complete
tags: [basis, exponential, power, polygonal, BasisSystem, numeric-gram, hat-functions, penalty-matrix]
depends_on: [35-01]
provides:
  - exponential_basis factory (exp(rate*t) eval; numeric 2nd-order Gram penalty)
  - power_basis factory (t^exponent eval; analytic Gram for integer exponents, numeric otherwise; domain guard)
  - polygonal_basis factory (piecewise-linear hat functions; 1st-order numeric Gram penalty)
affects:
  - fdars-core/src/basis/exponential.rs
  - fdars-core/src/basis/power.rs
  - fdars-core/src/basis/polygonal.rs
  - fdars-core/src/basis/mod.rs
  - fdars-core/src/lib.rs
tech_stack:
  added: []
  patterns:
    - "Numeric Gram via pub(crate) smooth_basis helpers (promoted in 35-01) for exponential + polygonal"
    - "Analytic falling-factorial Gram for integer-exponent power_basis (shared helper from monomial)"
    - "Half-open interval [knots[j-1], knots[j]) convention for hat-function partition-of-unity"
    - "Domain guard: FdarError::InvalidParameter when non-positive argvals used with non-integer/negative exponents"
key_files:
  created:
    - fdars-core/src/basis/exponential.rs
    - fdars-core/src/basis/power.rs
    - fdars-core/src/basis/polygonal.rs
  modified:
    - fdars-core/src/basis/mod.rs
    - fdars-core/src/lib.rs
decisions:
  - "Numeric Gram for exponential (avoids r_i+r_j=0 special case in analytic formula; consistent with bspline/polygonal pattern)"
  - "Analytic Gram for integer-exponent power_basis, numeric Gram otherwise (reuses gram_entry helper from monomial.rs)"
  - "Half-open interval [knots[j-1], knots[j]) on left ramp for polygonal hat function (prevents double-counting at shared knot; closed right ramp owns the peak)"
  - "lfd_order=1 for polygonal (D^2 of piecewise-linear is 0 a.e.; documented in rustdoc)"
  - "10 sub-points per original interval for numeric Gram grid (same density as bspline_penalty_matrix)"
metrics:
  duration_minutes: 18
  completed: "2026-08-21"
  tasks_completed: 3
  commits: 3
estimate:
  tokens: 62000
  tasks: 3
actuals:
  tokens: 12000
  tasks: 3
  commits: 3
---

# Phase 35 Plan 02: exponential_basis + power_basis + polygonal_basis Summary

**One-liner:** Three remaining basis factories (exponential, power, polygonal) completing the four-factory REP-01 family — each returning `BasisSystem` with eval matrix and penalty, all crate-root re-exported, clippy gate clean.

## What Was Built

### Task 1: exponential_basis + power_basis factories

**`fdars-core/src/basis/exponential.rs`** — new factory:
- `exponential_basis(argvals, rates) -> Result<BasisSystem, FdarError>`
- `eval_matrix[i + j*n] = exp(rates[j] * t_i)` — `rates[j]=0` → identically 1.0 (constant)
- Numeric 2nd-order Gram penalty via `pub(crate) differentiate_basis_columns` +
  `integrate_symmetric_penalty` (promoted in 35-01), fine grid at 10 sub-points per interval
- Errors: `InvalidDimension` for `argvals.len() < 2`; `InvalidParameter` for empty `rates`
- 9 inline unit tests: at-zero eval (all 1), rate=0 constant, closed-form [1, exp(-1)],
  shape invariants, penalty symmetry, PSD diagonal, derives

**`fdars-core/src/basis/power.rs`** — new factory:
- `power_basis(argvals, exponents) -> Result<BasisSystem, FdarError>`
- `eval_matrix[i + j*n] = t.powf(exponents[j])`
- **Domain guard (T-35-02 mitigation):** if any exponent is non-integer or negative, all
  `argvals` must be `> 0.0`; otherwise `FdarError::InvalidParameter` (prevents NaN/Inf)
- Penalty: analytic falling-factorial Gram when all exponents are non-negative integers
  (exact, same formula as `monomial_basis`); numeric Gram otherwise
- 12 inline unit tests: error paths, integer-exponent eval matches monomial exactly,
  non-integer spot check, domain guard (zero/negative argvals), shape invariants,
  penalty symmetry/PSD (both analytic and numeric paths), low-exponent zeros, derives

**Reference values verified:**
- `t=[0,1,2], rates=[0,-1]` → col0 `[1,1]`, col1 `[1, exp(-1)]` (within 1e-12)
- `t=[0,1,2], exponents=[0,1,2]` → matches `monomial_basis` eval matrix exactly
- `power_basis([−1,1], [−1])` → `Err(InvalidParameter)` (negative argval + negative exp)
- `power_basis([0,1], [0.5])` → `Err(InvalidParameter)` (zero argval + fractional exp)

### Task 2: polygonal_basis factory (hat functions, 1st-order penalty)

**`fdars-core/src/basis/polygonal.rs`** — new factory:
- `polygonal_basis(argvals, knots) -> Result<BasisSystem, FdarError>` — `nbasis = knots.len()`
- Hat function evaluation via half-open interval convention:
  - Left ramp: `(t - knots[j-1]) / (knots[j] - knots[j-1])` on `[knots[j-1], knots[j])`
  - Right ramp: `(knots[j+1] - t) / (knots[j+1] - knots[j])` on `[knots[j], knots[j+1]]`
  - At `t == knots[j]` only the right ramp is active → `B_j(knots[j]) = 1.0` exactly
  - Final boundary knot uses closed interval on the left ramp
- **Partition-of-unity:** `Σ_j B_j(t) == 1.0` verified over 21-point grid within 1e-12
- 1st-order numeric Gram penalty (`lfd_order = 1`); 2nd derivative is 0 a.e. for
  piecewise-linear basis — documented in rustdoc
- **Knot validation (T-35-03 mitigation):** non-monotone or duplicate knots →
  `FdarError::InvalidParameter` (zero-width intervals produce division-by-zero)
- 12 inline unit tests: error paths, hat peaks, midpoint values, interior knot eval,
  partition-of-unity, shape invariants, penalty symmetry/PSD, derives

**Reference values verified (knots=[0,0.5,1]):**
- `B_j(knots[j]) = 1.0` for all j
- `t=0.25`: B₀=0.5, B₁=0.5, B₂=0.0
- `t=0.5`: B₀=0.0, B₁=1.0, B₂=0.0
- `Σ_j B_j(t) = 1.0` at all 21 test points

### Task 3: Registration in basis/mod.rs and crate-root lib.rs

**`fdars-core/src/basis/mod.rs`** — additive extensions:
- `pub mod exponential;`, `pub mod power;`, `pub mod polygonal;` added
- `pub use exponential::exponential_basis;`, `pub use power::power_basis;`,
  `pub use polygonal::polygonal_basis;` added in alphabetical order
- All existing exports preserved

**`fdars-core/src/lib.rs`** — crate-root re-export:
- `exponential_basis`, `power_basis`, `polygonal_basis` added to `pub use basis::{...}`
- All four factories (monomial + three new) callable as `fdars_core::{factory}`

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test basis::exponential` | 9/9 passed |
| `cargo test basis::power` | 12/12 passed |
| `cargo test basis::polygonal` | 12/12 passed |
| `cargo test basis::` | 196/196 passed (163 from 35-01 + 33 new) |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | clean |
| Existing bspline/fourier/constant/monomial untouched | confirmed (git diff additive only) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Double-counting at shared knot boundaries in polygonal hat function**
- **Found during:** Task 2, test run (3 tests failed: `poly_hat_peaks_at_knot`,
  `poly_eval_at_interior_knot`, `poly_partition_of_unity`)
- **Issue:** Original implementation used `t >= knots[j-1] && t <= knots[j]` for the left
  ramp (closed on both ends). At `t == knots[j]` both left ramp (= 1.0) and right ramp
  (= 1.0) activated, yielding `B_j(knots[j]) = 2.0` instead of 1.0 and violating
  partition-of-unity (sum = 2 at shared knot).
- **Fix:** Changed left ramp to half-open `[knots[j-1], knots[j])` using strict `t < knots[j]`
  (with a special case for the final boundary knot which uses closed `[knots[j-1], knots[j]]`).
  Right ramp owns the peak point `t = knots[j]` → evaluates to 1.0 exactly.
- **Files modified:** `fdars-core/src/basis/polygonal.rs`
- **Commit:** included in `5f8e533e`

## Known Stubs

None. All three factories evaluate to exact closed-form values on reference points; penalties are computed analytically (power/integer) or via numeric quadrature (exponential/polygonal). No placeholder data.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. All additions are pure numerical computation.

**T-35-02 mitigated:** `power_basis` rejects `argvals` containing `≤ 0.0` when any exponent is non-integer or negative — no NaN/Inf silently poisons eval/penalty matrices.

**T-35-03 mitigated:** `polygonal_basis` rejects non-monotone/duplicate knots at entry via `FdarError::InvalidParameter` — zero-width interval division-by-zero cannot occur.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| `exponential.rs` | FOUND |
| `power.rs` | FOUND |
| `polygonal.rs` | FOUND |
| `35-02-SUMMARY.md` | FOUND |
| `578c52fb` (task 1 — exp+power) | FOUND |
| `5f8e533e` (task 2 — polygonal) | FOUND |
| `4ef51e47` (task 3 — registration+clippy) | FOUND |
