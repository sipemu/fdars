# Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning
**Mode:** Autonomous smart-discuss (concrete requirements; no user grey-area pause needed)

<domain>
## Phase Boundary

Close three effort-S scikit-fda parity gaps in `fdars-core`, all additive and non-breaking:
- **FEAT-03 (PREP-03):** in-grid NaN imputation for a regular `FdMatrix`.
- **FEAT-04 (REPR-03):** composable `ExtrapolationPolicy` controlling out-of-range interpolation/evaluation behavior.
- **FEAT-05 (MISC-04):** functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance).
</domain>

<decisions>
## Implementation Decisions

### LOCKED (milestone + established pattern)
- **Non-breaking, additive.** Do NOT change existing public signatures (e.g. `fdata_interpolate`, `spline_interpolate`). Where a new option must reach an existing function, use the established Phase-12 pattern: a new `*_with_policy` wrapper or an `Option`-typed addition — never a breaking positional param. New functions are fine.
- **Every new public function returns `Result<_, FdarError>`** with dimension/parameter validation (never panic on input) — this OVERRIDES the audit backlog's sketch of bare-`f64` scoring returns.
- **FEAT-04 enum name & variants:** `ExtrapolationPolicy` with `Boundary` (clamp to nearest edge), `Exception` (return `FdarError` on out-of-range — name matches scikit-fda `ExceptionExtrapolation` and REQUIREMENTS, NOT the backlog's `Error`), `Fill(f64)` (constant fill), `Periodic` (wrap modulo domain). Derive `Debug, Clone, PartialEq` (+ conditional serde per crate convention).

### Claude's discretion (guided by backlog + conventions)
- **FEAT-03:** `impute_missing_values` returning `Result<FdMatrix, FdarError>` (or an in-place `&mut` variant — planner picks the cleaner Result-based signature). Provide at least two strategies via an `ImputationMethod` enum: `Linear` (reuse `helpers::linear_interp` between nearest non-NaN neighbors) and `Mean`/`Constant(f64)`. Validate + error on all-NaN curves or unsupported input. Location: `helpers.rs` (reuse `irreg_fdata/` interp infra if it fits).
- **FEAT-04:** thread the policy through the interpolation/evaluation path non-breakingly (wrapper or new fn). Reuse the existing spline/linear machinery from v0.15.0.
- **FEAT-05:** new `fdars-core/src/scoring.rs` module; `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, `functional_explained_variance` over `(y_true, y_pred, argvals)`, integrated over `argvals`, each `Result`-returning with shape validation; re-export at crate root. Reuse existing `r_squared`/integration helpers where sensible.
</decisions>

<code_context>
## Existing Code Insights

- `helpers.rs` — `linear_interp`, `fdata_interpolate` (currently silently clamps at boundary), `spline_interpolate` (v0.15.0, FEAT-01), `r_squared`/`r_squared_adj`, Simpson's-weight integration helpers.
- `irreg_fdata/` — existing irregular→regular conversion / kernel infra (candidate reuse for imputation).
- Convention: column-major `FdMatrix`, `Result<T, FdarError>` everywhere, `#[must_use]` on expensive fns, conditional serde derive, inline `#[cfg(test)] mod tests`, crate-root re-exports in `lib.rs` + prelude.
- **File-overlap note (for wave/plan sequencing):** FEAT-03 and FEAT-04 both touch `helpers.rs` — their plan writes MUST be serialized (no parallel worktree collision). FEAT-05 (new `scoring.rs`) is fully independent and can proceed in parallel.
</code_context>

<specifics>
## Specific Ideas

- Success criteria (ROADMAP): imputation returns Result, mean+linear strategies, rejects all-missing (FEAT-03); `ExtrapolationPolicy` enum threaded through interpolation, each variant tested incl. the `Exception` error path (FEAT-04); five `Result`-returning scoring fns verified against hand-computed references (FEAT-05).
- Tests: inline `#[cfg(test)]` per file; imputation reproduces known values on synthetic gaps + errors on invalid input; each ExtrapolationPolicy variant exercised; each scoring metric checked against a hand-computed value.
</specifics>

<deferred>
## Deferred Ideas

- Additional imputation strategies beyond mean/linear (spline-based, KNN) — future.
- Extrapolation policies beyond the four named variants — future.
- Additional metrics beyond the five named — future.
</deferred>
