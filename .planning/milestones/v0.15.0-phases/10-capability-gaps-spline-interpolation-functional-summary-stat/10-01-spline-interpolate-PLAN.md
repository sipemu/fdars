---
phase: 10-capability-gaps-spline-interpolation-functional-summary-statistics
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/helpers.rs
  - fdars-core/src/lib.rs
autonomous: true
requirements: [FEAT-01]
estimate:
  tokens: 55000
  raw_tokens: 30000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "spline_interpolate reproduces the input exactly (within 1e-10) when query_points == argvals (FEAT-01, Success Criterion 1)"
    - "spline_interpolate reproduces known cubic-spline values within 1e-10 at off-grid query points (FEAT-01, Success Criterion 1)"
    - "spline_interpolate returns FdarError::InvalidParameter when order == 0 or order >= m, never panicking (Success Criterion 3)"
    - "spline_interpolate returns FdarError::InvalidParameter when any query point falls outside [argvals[0], argvals[m-1]], never panicking (Success Criterion 3)"
    - "spline_interpolate returns FdarError::InvalidDimension when argvals.len() != data.ncols() or query_points is empty, never panicking (Success Criterion 3)"
    - "the existing linear/cubic-Hermite fdata_interpolate path remains present and unchanged (additive, not a removal — Success Criterion 4)"
    - "cargo test -p fdars-core --features linalg and cargo clippy -p fdars-core --features linalg pass with the new function covered (Success Criterion 4)"
  artifacts:
    - "fdars-core/src/helpers.rs — pub fn spline_interpolate"
    - "fdars-core/src/lib.rs — spline_interpolate added to the helpers re-export block"
  key_links:
    - "spline_interpolate -> basis::bspline::construct_bspline_knots / bspline_basis / bspline_basis_from_knots (fit-then-evaluate on the SAME knot vector)"
    - "spline_interpolate -> nalgebra SVD for the per-curve B^T B coefficient solve"
    - "lib.rs re-export -> crate-root visibility of spline_interpolate (mirrors fdata_interpolate)"
---

<objective>
Add a new public `spline_interpolate` function to `helpers.rs` that fits an order-k B-spline per curve over the existing `basis/` B-spline system and evaluates it at arbitrary off-grid query points, returning a new `FdMatrix`. Re-export it at the crate root following the existing `fdata_interpolate` pattern.

Purpose: Close capability gap FEAT-01 (audit REPR-02) — callers can currently interpolate only linearly / via cubic Hermite; this adds true order-k spline interpolation with O(h^{2k}) off-grid accuracy for smooth curves.

Output: `pub fn spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` in `helpers.rs`, re-exported in `lib.rs`, with inline `#[cfg(test)]` tests covering exact-reproduction, off-grid accuracy, and all input-validation error paths.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-capability-gaps-spline-interpolation-functional-summary-stat/10-RESEARCH.md

@fdars-core/src/helpers.rs
@fdars-core/src/basis/bspline.rs
@fdars-core/src/basis/pspline.rs
@fdars-core/src/matrix.rs
@fdars-core/src/error.rs
</context>

## Artifacts this phase produces (plan 10-01)

New public symbols introduced by this plan:

| Symbol | File | Kind |
|--------|------|------|
| `spline_interpolate` | `fdars-core/src/helpers.rs` | `pub fn` |
| `spline_interpolate` re-export | `fdars-core/src/lib.rs` | added to the `pub use helpers::{...}` block (lib.rs:170-175) |

Nothing is removed. The existing `fdata_interpolate`, `linear_interp`, `InterpolationMethod` (lib.rs:171-174) remain.

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end spline_interpolate — one curve fit + evaluate, exact-reproduction test green</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - `fdars-core/src/helpers.rs` (the file being modified — study existing `linear_interp`, `fdata_interpolate`, and the inline `#[cfg(test)] mod tests` block at helpers.rs:590-844 to match placement and test style)
    - `fdars-core/src/basis/pspline.rs:66-187` — the reference fit-then-evaluate pattern (`pspline_fit_1d` builds `DMatrix::from_column_slice(m, actual_nbasis, &basis)` at pspline.rs:86-87; `pspline_evaluate` evaluates on a new grid at pspline.rs:163-187). Reuse this structure MINUS the smoothing penalty.
    - `fdars-core/src/basis/bspline.rs:4-125` — `construct_bspline_knots(t_min, t_max, nknots, order)` (bspline.rs:4-17), `bspline_basis(t, nknots, order)` (bspline.rs:100-125), `bspline_basis_from_knots(t, knots, order)` (bspline.rs:62-83). Note the basis layout: `basis[ti + k*n]` = basis function k at point t[ti].
    - `fdars-core/src/matrix.rs:50-130` — `FdMatrix::from_column_major`, `FdMatrix::zeros`, `FdMatrix::column`, `shape()`, indexed access `[(i,j)]`.
    - `fdars-core/src/test_helpers.rs:1-8` — `uniform_grid(n)` signature.
  </read_first>
  <behavior>
    - Test `spline_interpolate_reproduces_argvals`: given a single cubic curve y = t^3 sampled on `uniform_grid(20)`, calling `spline_interpolate(&data, &t, &t, 4)` (query_points == argvals) reproduces the input at every point within 1e-10.
  </behavior>
  <action>
    Implement the thinnest complete path: `pub fn spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>`, marked `#[must_use]` (per project convention on expensive computations).

    Wire ONE happy path end-to-end following the `pspline_evaluate` reference (basis/pspline.rs:163-187) with NO smoothing penalty (this is interpolation, not smoothing — do NOT call `pspline_fit_1d`):
    1. Read `(n, m) = data.shape()`.
    2. Compute `nknots = m.saturating_sub(order).max(2)`; `t_min = argvals[0]`, `t_max = argvals[m-1]`.
    3. Build the knot vector once: `basis::bspline::construct_bspline_knots(t_min, t_max, nknots, order)`.
    4. Build the fit basis on argvals: `basis::bspline::bspline_basis(argvals, nknots, order)`; derive `nbasis = basis_vals.len() / m`.
    5. Form the `m x nbasis` matrix B via `nalgebra::DMatrix::from_column_slice(m, nbasis, &basis_vals)` (same layout as pspline.rs:86-87). Solve the per-curve least squares `coefs = pinv(B) * y` using `nalgebra::SVD::new(B.clone(), true, true)` and `svd.solve(&y_col, tol)` OR compute the pseudoinverse once via SVD and reuse across curves. Do NOT import `svd_pseudoinverse` from `basis/helpers.rs` — it is `pub(super)` and not visible here; reproduce the SVD inline.
    6. Build the query basis on the SAME knots: `basis::bspline::bspline_basis_from_knots(query_points, &knots, order)`; `m_q = query_points.len()`.
    7. Allocate `FdMatrix::zeros(n, m_q)`; for each curve i, evaluate `out[(i,j)] = sum_k coefs_i[k] * basis_query[j + k*m_q]`.
    8. Return `Ok(out)`.

    Keep the SVD solve tolerance consistent with existing usage (use `NUMERICAL_EPS` if a threshold is needed). This single path must compile and the exact-reproduction test must pass before expanding. The signature is fixed by the ROADMAP (Success Criterion 1) — treat it as locked and do not alter it.
  </action>
  <acceptance_criteria>
    - `fdars-core/src/helpers.rs` contains the exact string `pub fn spline_interpolate(`
    - The function signature is `spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` (or the fully-pathed equivalent) — matching the ROADMAP verbatim.
    - The implementation does NOT call `pspline_fit_1d` (grep `helpers.rs` for `pspline_fit_1d` returns no match inside `spline_interpolate`).
    - An inline test `spline_interpolate_reproduces_argvals` exists in the `#[cfg(test)] mod tests` block of `helpers.rs`.
    - `cargo test -p fdars-core --features linalg spline_interpolate_reproduces_argvals` exits 0.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg spline_interpolate_reproduces_argvals -- --exact 2>&1 | tail -5</automated>
  </verify>
  <done>The single fit-then-evaluate path is implemented in helpers.rs, compiles, and the exact-reproduction test passes.</done>
  <reversibility rating="costly" reason="Public API signature is a published contract, but ROADMAP Success Criterion 1 already fixes it exactly — the one-way decision is pre-made and locked, so no checkpoint is needed; flagged only so the executor implements the fixed signature verbatim without redesign."/>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Off-grid accuracy + full input validation (never-panic error paths)</name>
  <files>fdars-core/src/helpers.rs</files>
  <read_first>
    - `fdars-core/src/helpers.rs` — the `spline_interpolate` body from Task 1 and the inline test block.
    - `fdars-core/src/error.rs:1-25` — exact `FdarError` variant fields: `InvalidDimension { parameter: &'static str, expected: String, actual: String }`, `InvalidParameter { parameter: &'static str, message: String }`.
    - `fdars-core/src/basis/bspline.rs:62-83` — `bspline_basis_from_knots` evaluates at any point but only guarantees partition-of-unity inside `[t_min, t_max]` (out-of-domain = extrapolation, out of scope per REPR-03).
    - `10-RESEARCH.md` Pitfalls 1, 2, 3 (RESEARCH.md:231-260) — out-of-range query points, rank deficiency when order >= m, and the column-major basis layout.
  </read_first>
  <behavior>
    - Test `spline_interpolate_cubic_offgrid`: a known cubic curve interpolated at off-grid midpoints matches the analytic cubic value within 1e-10.
    - Test `spline_interpolate_rejects_out_of_range`: a query point below argvals[0] or above argvals[m-1] returns `Err(FdarError::InvalidParameter { parameter: "query_points", .. })`.
    - Test `spline_interpolate_rejects_bad_order`: `order == 0` and `order >= m` each return `Err(FdarError::InvalidParameter { parameter: "order", .. })`.
    - Test `spline_interpolate_rejects_dim_mismatch`: `argvals.len() != data.ncols()` returns `Err(FdarError::InvalidDimension { .. })`; empty `query_points` returns `Err(FdarError::InvalidDimension { parameter: "query_points", .. })`.
  </behavior>
  <action>
    Prepend input validation to `spline_interpolate` (all returning `FdarError`, never panicking), in this order so nothing downstream can panic:
    - `argvals.len() == m` (== `data.ncols()`) else `FdarError::InvalidDimension { parameter: "argvals", .. }`.
    - `!query_points.is_empty()` else `FdarError::InvalidDimension { parameter: "query_points", expected: ">= 1", actual: "0" }`.
    - `order >= 1 && order < m` else `FdarError::InvalidParameter { parameter: "order", message: format!("must be in [1, {m}), got {order}") }` (guards Pitfall 2 rank deficiency).
    - every `query_points[j]` in `[argvals[0], argvals[m-1]]` else `FdarError::InvalidParameter { parameter: "query_points", message: ".. out of interpolation domain .." }` (guards Pitfall 1 — extrapolation is REPR-03, deferred). Do NOT emit the literal substring an acceptance-criteria grep negates; keep the message descriptive but distinct.

    Then complete the off-grid path (already wired in Task 1) and add the four tests named in `<behavior>`. For the cubic off-grid test, pick a low-degree polynomial the order-4 spline reproduces exactly (a cubic is in the order-4 B-spline span) so the analytic reference is unambiguous at 1e-10. Assert error variants by matching the returned `Err(FdarError::…)` shape, not by string equality on the Display output.
  </action>
  <acceptance_criteria>
    - `helpers.rs` contains inline tests named `spline_interpolate_cubic_offgrid`, `spline_interpolate_rejects_out_of_range`, `spline_interpolate_rejects_bad_order`, `spline_interpolate_rejects_dim_mismatch`.
    - Every early-return in `spline_interpolate` returns a `FdarError` variant (grep confirms no `panic!`, `unwrap()` on user-derived values, or `expect(` inside the function body: `grep -n 'unwrap()\|panic!\|expect(' ` scoped to the function shows only SVD-internal safe uses, if any).
    - `cargo test -p fdars-core --features linalg helpers::tests::spline_interpolate 2>&1` reports all `spline_interpolate_*` tests passing.
    - `cargo test -p fdars-core --features linalg spline_interpolate_rejects_out_of_range -- --exact` exits 0.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg spline_interpolate 2>&1 | tail -8</automated>
  </verify>
  <done>Off-grid accuracy test and all four validation tests pass; every bad-input path returns an FdarError rather than panicking.</done>
</task>

<task type="auto">
  <name>Task 3: Re-export spline_interpolate at crate root and verify clippy + full suite</name>
  <files>fdars-core/src/lib.rs</files>
  <read_first>
    - `fdars-core/src/lib.rs:169-175` — the `// Re-export commonly used items` block: `pub use helpers::{ aic, ..., fdata_interpolate, ..., InterpolationMethod, ... };`. Add `spline_interpolate` here in alphabetical position, mirroring the existing `fdata_interpolate` re-export (RESEARCH Open Question 3, RESEARCH.md:488-491).
  </read_first>
  <action>
    Add `spline_interpolate` to the `pub use helpers::{...}` re-export block in `lib.rs` (insert alphabetically among the existing identifiers, e.g. after `simpsons_weights_2d`), following the exact pattern used for `fdata_interpolate`. Do not create a new re-export block. Then run the full test suite and clippy under the `linalg` feature to confirm the additive change is clean and that the existing `fdata_interpolate` / `linear_interp` / `InterpolationMethod` re-exports are untouched (Success Criterion 4: linear-interpolation path preserved).
  </action>
  <acceptance_criteria>
    - `fdars-core/src/lib.rs` contains `spline_interpolate` within the `pub use helpers::{` block.
    - `grep -n 'fdata_interpolate\|linear_interp\|InterpolationMethod' fdars-core/src/lib.rs` still shows all three existing re-exports present (nothing removed).
    - `cargo build -p fdars-core --features linalg` exits 0 and `fdars_core::spline_interpolate` resolves at the crate root.
    - `cargo test -p fdars-core --features linalg` exits 0 (full suite green).
    - `cargo clippy -p fdars-core --features linalg` exits 0 with no new warnings on the added code.
  </acceptance_criteria>
  <verify>
    <automated>cargo clippy -p fdars-core --features linalg 2>&1 | tail -5 && cargo test -p fdars-core --features linalg 2>&1 | tail -8</automated>
  </verify>
  <done>spline_interpolate is re-exported at the crate root, the full linalg test suite passes, clippy is clean, and the existing interpolation re-exports remain.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → spline_interpolate | Untrusted numeric input (matrix dims, argvals, query_points, order) crosses into the pure-Rust numeric routine. No network/FS/auth surface. |

## STRIDE Threat Register (ASVS L1)

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-01-01 | Denial of Service | spline_interpolate order/grid arithmetic | low | mitigate | Validate `order < m` before basis construction (guards rank-deficient / oversized basis, Pitfall 2). Task 2 acceptance criteria. |
| T-10-01-02 | Denial of Service | out-of-domain query_points → unbounded extrapolated values | low | mitigate | Reject query points outside `[argvals[0], argvals[m-1]]` with FdarError (Pitfall 1). Task 2. |
| T-10-01-03 | Tampering (data integrity) | index arithmetic `j + k*m_q`, dimension mismatch | low | mitigate | Validate `argvals.len() == data.ncols()` and non-empty query_points before any indexing; all bad inputs return FdarError, never panic (Success Criterion 3). Task 2. |
| T-10-01-04 | Denial of Service | integer overflow in `nknots`/`nbasis` arithmetic | low | accept | Uses `saturating_sub`/`max`; realistic grid sizes cannot overflow usize. No further control needed. |

No new external packages are introduced (Package Legitimacy Audit: none — RESEARCH.md:389-394). No high-severity threats → non-blocking.
</threat_model>

<verification>
- Success Criterion 1: `spline_interpolate_reproduces_argvals` (exact reproduction ≤1e-10) and `spline_interpolate_cubic_offgrid` (off-grid ≤1e-10) both green.
- Success Criterion 3: `spline_interpolate_rejects_out_of_range`, `spline_interpolate_rejects_bad_order`, `spline_interpolate_rejects_dim_mismatch` all green; no panic path in the function body.
- Success Criterion 4: `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` both exit 0; existing `fdata_interpolate`/`linear_interp`/`InterpolationMethod` re-exports still present.
</verification>

<success_criteria>
- `pub fn spline_interpolate` exists in `helpers.rs` with the ROADMAP-fixed signature and is `#[must_use]`.
- It is re-exported at the crate root in `lib.rs` alongside `fdata_interpolate`.
- All inline tests (exact-reproduction, off-grid accuracy, three validation groups) pass.
- Full `linalg` suite green; clippy clean; linear-interpolation path preserved.
</success_criteria>

<output>
Create `.planning/phases/10-capability-gaps-spline-interpolation-functional-summary-stat/10-01-SUMMARY.md` when done.
</output>
