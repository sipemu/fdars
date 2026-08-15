# Phase 21: Functional-Linear-Model Inference - Context

**Gathered:** 2026-08-16
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — method defaults set with grounded formulas + verified reuse anchors; harder null-distribution choices left to the planner/executor with strong test requirements.

<domain>
## Phase Boundary

Add formal functional-linear-model inference to the `inference/` module created in Phase 20. Covers INF-02: `flm_gof_test`, `flm_f_test` (both on a fitted `FregreLmResult`), and `oneway_anova_vstat` (asymptotic one-way functional ANOVA V-statistic, added alongside the existing permutation `fanova`). Additive/non-breaking, `Result`-returning, inline tests, crate-root re-exported. Final phase of v0.19.0.
</domain>

<decisions>
## Implementation Decisions

### `flm_f_test` — overall-significance F-test (default form)
- Test H0: the functional coefficient has no effect (FLM reduces to intercept-only). Standard residual-based F:
  `F = (R² / p) / ((1 − R²) / (n − p − 1))`, where `p = ncomp` (effective FPC parameters), `n` = sample size; equivalently from `SS_res = Σ residuals²` and `SS_null = Σ (yᵢ − ȳ)²` with `yᵢ = fitted_values[i] + residuals[i]`. p-value from the F(p, n−p−1) survival function.
- Reuse `FregreLmResult.{residuals, fitted_values, r_squared, ncomp}` directly (all public). Return `TestResult { statistic, p_value, .. }` (the Phase-20 struct; add fields only if additive).

### `flm_gof_test` — residual-based goodness-of-fit
- Report a goodness-of-fit statistic + p-value that **rejects when the FLM is mis-specified / inadequate and fails to reject a well-specified FLM.** Default: a residual-based lack-of-fit statistic (e.g. an F-form or a Delsol-Ferraty-Vieu-style no-effect statistic). **Planner/executor latitude on the exact null:** F-form, asymptotic approximation, or a residual bootstrap are all acceptable — **document the chosen method in rustdoc**, and the inline tests MUST validate correct rejection behavior on synthetic well-specified vs. mis-specified fits (not just that it runs).

### `oneway_anova_vstat` — asymptotic V-statistic (alongside `fanova`)
- Compute the functional one-way ANOVA V-statistic: `V = ∫ Σ_g n_g (x̄_g(t) − x̄(t))² dt` (Simpson-weighted via `helpers::simpsons_weights`). Provide an **asymptotic p-value** (the permutation counterpart already exists in `fanova`). The exact fdANOVA V-null is a weighted sum of χ²; a **scaled/approximate-χ² (Satterthwaite/Box) approximation is the acceptable default** — document it; a permutation fallback is allowed if the approximation proves unreliable (note it). Add as a NEW public fn; do NOT change `fanova`'s signature or behavior.
- Reuse the `pub(crate) integrated_f_statistic` helper / group-mean logic already in `function_on_scalar.rs` where it fits; keep it `pub(crate)`.

### Placement & API
- Add to the `inference/` module (e.g. a new `inference/flm.rs` + `inference/anova.rs` or fold into existing submodules — executor's discretion). Crate-root re-export `flm_gof_test`, `flm_f_test`, `oneway_anova_vstat`.
- Reuse the Phase-20 `TestResult` struct for the FLM tests (consistency); the V-stat may return `TestResult` or a small dedicated result — executor's discretion, additive only.

### Claude's Discretion
- Exact GoF null method + V-stat asymptotic approximation (with the mandated correctness tests), file split within `inference/`, and whether the V-stat reuses `TestResult`.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (verified)
- `FregreLmResult` (`scalar_on_function/mod.rs:54`) — fields: `intercept, beta_t, beta_se, gamma, fitted_values, residuals, r_squared, r_squared_adj, std_errors, ncomp, fpca, coefficients, residual_se, gcv, aic, bic`. All public — F-test/GoF read `residuals`/`fitted_values`/`r_squared`/`ncomp` directly.
- `scalar_on_function::fregre_lm(...)` — produces the `FregreLmResult` the tests fit against.
- `function_on_scalar::integrated_f_statistic(data, groups, labels) -> f64` (`:762`, now `pub(crate)` after Phase 20) + `fanova(...) -> FanovaResult` (permutation ANOVA; `FanovaResult` has `group_means, overall_mean, global_statistic, p_value, n_perm, n_groups`).
- `helpers::simpsons_weights`, `fdata::mean_1d` for the V-statistic integration.
- Phase-20 `inference::TestResult { statistic, p_value, n_perm }` — reuse for the FLM tests.
- χ²/F survival functions: Phase 20 added a self-contained χ² SF (regularized upper incomplete gamma + Lanczos ln_gamma) in `inference/` — reuse it; add an F-distribution SF the same way (no new crate dep).

### Established Patterns
- `Result<T, FdarError>`, input validation at entry, result structs `#[derive(Debug, Clone, PartialEq)]` + serde-gated, crate-root re-export. NO `#[must_use]` on `Result`-returning fns. Inline `#[cfg(test)] mod tests`. CI: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- Build/test: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for cargo (MEMORY). Code commits pass the pre-commit gate; `--no-verify` only for docs / spurious /tmp doctest-linking failures (with cargo test/clippy re-confirmed manually).

### Integration Points
- Consumes Phase-20 `inference/` module + `TestResult` + the χ²/F SF helper. Closes Area 5's two P1 table-stakes items (with INF-01).
</code_context>

<specifics>
## Specific Ideas

- Test correctness (mandatory): `flm_f_test` rejects a genuine functional effect (fit on y with real β) and fails to reject a null-effect fit (y ⊥ x); `flm_gof_test` fails to reject a well-specified FLM and flags a mis-specified one; `oneway_anova_vstat` agrees in direction with the existing permutation `fanova` (rejects real group differences, not equal groups) and is deterministic where applicable.
</specifics>

<deferred>
## Deferred Ideas

- Interval Testing Procedure (ITP) family → INF-03 (v2, deferred). This is the last phase of v0.19.0.
</deferred>
