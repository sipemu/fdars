# Phase 25: Functional GLM (Exponential Family) - Context

**Gathered:** 2026-08-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a **functional GLM for a scalar response** over functional predictors via a new public
`functional_glm(...)` entry point in `scalar_on_function/`, re-exported at the crate root. It
generalizes the existing `functional_logistic` IRLS-over-FPC-scores path to the four mainstream
exponential-family families through a `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` enum
(canonical link + variance function per family). `functional_glm(…, Binomial)` reproduces the
existing logistic fit. Additive/non-breaking: the `functional_logistic` public signature is
retained unchanged; no new crate dependency.

**Explicitly out of scope:** extra families (inverse-Gaussian, negative-binomial), configurable
/ non-canonical links, and any change to `functional_logistic`.

</domain>

<decisions>
## Implementation Decisions

### API Signature & GlmFamily Enum
- Signature **mirrors `functional_logistic` with `family` inserted after `y`**:
  `functional_glm(data, y, family, scalar_covariates, ncomp, max_iter, tol)
  -> Result<FunctionalGlmResult, FdarError>`. (The requirement's `functional_glm(data, y, family)`
  is illustrative shorthand for the core inputs; full params keep parity + enable IRLS reuse.)
- `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` — exactly these four, each carrying its
  **canonical link + variance function**: Binomial = logit / μ(1−μ), Poisson = log / μ,
  Gamma = inverse / μ², Gaussian = identity / 1.
- `functional_logistic` is **retained verbatim** (additive/non-breaking); `functional_glm(…, Binomial)`
  reproduces its fit.
- `GlmFamily` is **`#[non_exhaustive]`** (forward-compat; matches project public-enum convention).

### Result Struct
- Type name **`FunctionalGlmResult`** (mirrors `FunctionalLogisticResult`).
- Fields **generalize `FunctionalLogisticResult`**, family-agnostic:
  `{ intercept, beta_t, gamma, fitted_values (μ), linear_predictors (η), ncomp, coefficients,
  std_errors, log_likelihood, deviance, iterations, fpca, aic, bic, family }`. Classification-only
  fields (`probabilities`, `predicted_classes`, `accuracy`) are dropped in favor of the generic
  `fitted_values`.
- **Binomial parity (SC2)** is exposed via `intercept` / `beta_t` / `coefficients` + `fitted_values`
  (= P(Y=1) for Binomial) agreeing with `functional_logistic` within tolerance — no separate
  probabilities/classes field.
- Include **`deviance`** (GLM-standard goodness-of-fit) alongside retained `log_likelihood`/`aic`/`bic`.

### IRLS Numerics & Family Guards
- **Canonical links** as above (Gamma = inverse), with μ / variance **clamped away from the
  boundary** to keep inverse/log finite and avoid ÷0.
- **Per-family response-domain guards** → `FdarError::InvalidParameter` (never panic):
  Poisson `y ≥ 0` (integer-valued), Gamma `y > 0`, Binomial `y ∈ [0,1]`, Gaussian unrestricted.
- **Reuse the `functional_logistic` IRLS loop** over FPC scores from `fdata_to_pc_1d`; converge on
  deviance/coefficient change `< tol`, capped at `max_iter`; same default conventions as logistic.
- **Gaussian runs through the same IRLS uniformly** (weights ≡ 1, identity link → OLS in one step);
  no special-case branch.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scalar_on_function/logistic.rs`: `functional_logistic(data, y, scalar_covariates, ncomp,
  max_iter, tol) -> Result<FunctionalLogisticResult, FdarError>` (`logistic.rs:197`) — the IRLS
  loop, FPC-score design, std-error/AIC/BIC machinery to generalize. `predict_functional_logistic`
  is the prediction analog.
- `scalar_on_function/mod.rs:138` `FunctionalLogisticResult` — structural template for
  `FunctionalGlmResult` (intercept, beta_t, beta_se, gamma, ncomp, coefficients, std_errors,
  log_likelihood, iterations, fpca, aic, bic).
- `regression.rs`: `fdata_to_pc_1d` — FPCA-score design shared by the SoF regression family;
  `FpcaResult` embedded for projecting new data.
- `error.rs`: `FdarError::{InvalidParameter, InvalidDimension, ComputationFailed}`.

### Established Patterns
- All public fns return `Result<T, FdarError>`; dimension + parameter-range checks at entry.
- Public result structs derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde;
  `#[must_use]` on the fit fn. Public enums (`CovType`, `DepthMethod`, `CvCriterion`) are
  `#[non_exhaustive]`.
- Inline `#[cfg(test)] mod tests`; crate-root re-export in `src/lib.rs` (`pub use scalar_on_function::{…}`).

### Integration Points
- Add `functional_glm`, `GlmFamily`, `FunctionalGlmResult` to the `scalar_on_function` barrel
  (`mod.rs` `pub use`) and to `src/lib.rs` `pub use scalar_on_function::{…}` (additive lines only).

</code_context>

<specifics>
## Specific Ideas

- Binomial-parity test (SC2): `functional_glm(data, y, GlmFamily::Binomial, …)` coefficients /
  fitted values agree with `functional_logistic` on the same data within tolerance.
- Per-family recovery tests (SC3): Poisson counts under a log link, Gamma responses under the
  inverse link — recovered coefficients/predictions match the known generative signal within a
  stated tolerance.
- Error-path tests (SC4): dimension mismatch and out-of-domain responses (e.g. negative Poisson
  counts, non-positive Gamma responses) return the appropriate `FdarError`, no panic.
- R baselines matched by capability: `fda.usc` / `refund` GLM paths; document any link/variance
  convention divergence in rustdoc.

</specifics>

<deferred>
## Deferred Ideas

- Extra families (inverse-Gaussian, negative-binomial) and configurable / non-canonical links.
- Log-link Gamma alternative (canonical inverse link is the chosen default).
- Step-halving / line-search in the IRLS loop (reuse the existing convergence policy).

</deferred>
