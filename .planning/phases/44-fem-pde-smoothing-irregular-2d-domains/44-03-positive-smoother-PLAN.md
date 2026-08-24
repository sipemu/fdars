---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: 03
type: execute
wave: 3
depends_on: ["44-02"]
files_modified:
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [REP-02-03]
estimate:
  tokens: 40000
  raw_tokens: 22000
  tasks: 2
  confidence: high
must_haves:
  truths:
    - "smooth_positive fits a nonnegative-guaranteed smoother by smoothing in the log domain and exponentiating — the fitted values are > 0 everywhere for any strictly-positive input"
    - "smooth_positive recovers a known positive curve within tolerance and reduces to the underlying log-domain B-spline smooth"
    - "Non-positive input (a data value <= 0) returns FdarError (no panic, no NaN from log)"
  artifacts:
    - "smooth_positive public fn + SmoothPositiveResult struct added to fdars-core/src/smooth_basis.rs"
    - "crate-root re-export of smooth_positive + SmoothPositiveResult in fdars-core/src/lib.rs; SmoothPositiveResult added to src/prelude.rs"
  key_links:
    - "smooth_positive delegates the actual smoothing to the existing smooth_basis on log(data), then exp-reconstructs — no new smoothing math, additive only"
---

<objective>
Deliver REP-02-03: a positive (nonnegative-guaranteed) smoother added additively to `smooth_basis.rs`. Smooth the response in the log domain with the existing `smooth_basis`, then exponentiate so the reconstructed fit is strictly positive by construction. Additive — zero changes to existing `smooth_basis*` signatures.

Purpose: gives users a smoother whose output respects a nonnegativity constraint (densities, intensities, concentrations), a capability fdars lacks; matches the log-domain positive-smoothing idiom.
Output: `smooth_positive` public fn + `SmoothPositiveResult` in `smooth_basis.rs`, crate-root + prelude re-exports, inline tests (positivity, recovery, non-positive-input error).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-CONTEXT.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-RESEARCH.md

@fdars-core/src/smooth_basis.rs
</context>

<artifacts_this_phase_produces>
New public symbols introduced by THIS plan, added to `smooth_basis.rs`:
- `SmoothPositiveResult { fitted: FdMatrix (n×m, strictly > 0), log_coefficients: FdMatrix (n×K), edf: f64, gcv: f64 }` — `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]`, conditional serde `cfg_attr` matching `SmoothBasisResult`.
- `smooth_positive(data: &FdMatrix, argvals: &[f64], fdpar: &FdPar) -> Result<SmoothPositiveResult, FdarError>` — `#[must_use]`.
- Crate-root re-export additions: `smooth_positive`, `SmoothPositiveResult`.
- Prelude addition: `SmoothPositiveResult`.

Reuses (existing, unchanged): `smooth_basis(data, argvals, fdpar) -> SmoothBasisResult`, `FdPar`, `BasisType`, `FdMatrix` (`column`, `column_mut`, `shape`, `from_column_major`, `zeros`).
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end smooth_positive — log-transform, delegate to smooth_basis, exp-reconstruct; positivity + recovery tests</name>
  <files>fdars-core/src/smooth_basis.rs</files>
  <read_first>
    - fdars-core/src/smooth_basis.rs:174-253 — `smooth_basis(data: &FdMatrix, argvals, fdpar)` signature + `SmoothBasisResult` fields (coefficients n×K, fitted n×m, edf, gcv). Mirror its validation + result-field style.
    - fdars-core/src/smooth_basis.rs:22-46 — `FdPar` / `BasisType`.
    - RESEARCH §4 (log-domain positive smoother: smooth log(y), exp-reconstruct; the exp of a real-valued smooth is strictly positive; note the retransformation bias caveat to document).
  </read_first>
  <action>
Add `SmoothPositiveResult` next to `SmoothBasisResult` in `smooth_basis.rs` with `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, and the same `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` line `SmoothBasisResult` uses. Fields: `fitted: FdMatrix` (n×m, strictly positive), `log_coefficients: FdMatrix` (n×K, the log-domain B-spline coefficients), `edf: f64`, `gcv: f64`.

Implement `pub fn smooth_positive(data: &FdMatrix, argvals: &[f64], fdpar: &FdPar) -> Result<SmoothPositiveResult, FdarError>` with `#[must_use = "expensive computation whose result should not be discarded"]`.

Validation at entry (before touching log): `let (n, m) = data.shape();` require `n > 0 && m > 0 && argvals.len() == m` (else `InvalidDimension`, mirroring smooth_basis). Scan every element: if any `data[(i,j)] <= 0.0` return `FdarError::InvalidParameter { parameter: "data", message: "smooth_positive requires strictly positive data (log-domain smoother); found a value <= 0".into() }` — this prevents `ln` producing NaN/-inf.

Build `log_data: FdMatrix` (n×m) with `log_data[(i,j)] = data[(i,j)].ln()`. Call `let inner = smooth_basis(&log_data, argvals, fdpar)?;`. Reconstruct the positive fit: `fitted[(i,j)] = inner.fitted[(i,j)].exp()` (strictly > 0 since exp of a finite real). Return `SmoothPositiveResult { fitted, log_coefficients: inner.coefficients, edf: inner.edf, gcv: inner.gcv }`.

Add a rustdoc `# Divergence / caveat` note: this is the standard log-domain positive smoother; `E[exp(fit)]` carries a small retransformation bias vs the conditional mean — documented, acceptable for v1 (RESEARCH §4).

Tracer tests (inline `#[cfg(test)] mod tests`, reuse `crate::test_helpers::uniform_grid`):
- `test_smooth_positive_is_positive`: build a single-curve `FdMatrix` (n=1, m≈41) sampling a positive function e.g. `2.0 + (2π t).sin()` (in (1,3), strictly > 0) plus tiny deterministic wiggle; build an `FdPar` with `BasisType::Bspline{order:4}`, `nbasis≈10`, moderate `lambda`, `lfd_order:2`, and its `penalty_matrix` via `bspline_penalty_matrix(argvals, nbasis, order, lfd_order)`; call `smooth_positive`; assert every `fitted[(0,j)] > 0.0` and finite.
- `test_smooth_positive_recovers_curve`: with the same positive target and a small lambda, assert mean-abs error between `fitted` and the true positive curve is below a tolerance (e.g. 0.2).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis::tests::test_smooth_positive_is_positive smooth_basis::tests::test_smooth_positive_recovers_curve 2>&1 | tail -20</automated>
  </verify>
  <done>smooth_positive smooths in log space and exp-reconstructs a strictly-positive fit that recovers a known positive curve; SmoothPositiveResult defined with correct derives.</done>
</task>

<task type="auto">
  <name>Task 2: Non-positive-input error path + crate-root/prelude re-exports</name>
  <files>fdars-core/src/smooth_basis.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/lib.rs:379 — existing `pub use smooth_basis::{ ... }` re-export block to extend.
    - fdars-core/src/prelude.rs — existing re-export style.
    - smooth_basis.rs Task 1 — the smooth_positive validation branch.
  </read_first>
  <action>
Extend the `pub use smooth_basis::{ ... }` block in `lib.rs` to include `smooth_positive` and `SmoothPositiveResult` (add to the existing list; do not remove anything). Add `SmoothPositiveResult` to the `smooth_basis::{...}` re-export in `src/prelude.rs`.

Add the error-path test:
- `test_smooth_positive_rejects_nonpositive`: build data with one element = 0.0 (or negative); assert `smooth_positive(...)` returns `Err(FdarError::InvalidParameter { parameter: "data", .. })` (no panic, no NaN).

Confirm `cargo doc`-level consistency is not needed here; just ensure the module compiles and the re-exports resolve.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis::tests::test_smooth_positive 2>&1 | tail -20</automated>
  </verify>
  <done>Non-positive input returns FdarError; smooth_positive + SmoothPositiveResult re-exported at crate root and prelude; module compiles.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → smooth_positive | Untrusted numeric response data crosses into a log transform + dense basis smooth |

Attack surface: none — pure in-process numeric library. Only numerical concern is `ln` of a non-positive value producing NaN/-inf.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-44-07 | Elevation (incorrect output) | ln(y) for y <= 0 → NaN propagates through the fit | medium | mitigate | validate all data > 0 at entry and return InvalidParameter before any ln |
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis` — all module tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
</verification>

<success_criteria>
- `smooth_positive` returns a strictly-positive fit for any strictly-positive input, recovering a known positive curve within tolerance.
- Non-positive input returns FdarError with no panic.
- `smooth_positive` + `SmoothPositiveResult` re-exported at crate root and prelude; existing `smooth_basis*` signatures unchanged.
</success_criteria>

<output>
Create `.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-03-SUMMARY.md` when done.
</output>
