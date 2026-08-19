# Phase 27: Elastic Multinomial Regression - Context

**Gathered:** 2026-08-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add **elastic multinomial (multi-class, K ≥ 2) logistic regression** over SRSF/SRVF space as a new
public entry point in `fdars-core/src/elastic_regression/logistic.rs`, re-exported at the crate
root, extending the existing binary elastic logistic, plus a `predict_elastic_multinomial`
companion. Closes the single elastic-logistic multi-class partial in fdars' otherwise-complete
elastic-regression family. Additive/non-breaking: the existing binary `elastic_logistic` /
`predict_elastic_logistic` signatures are unchanged; no new crate dependency.

**Explicitly out of scope:** joint softmax multinomial, multinomial elastic PCR / ordinal
variants, plotting.

</domain>

<decisions>
## Implementation Decisions

### Multinomial Approach & API
- **One-vs-rest (OvR):** fit K binary `elastic_logistic` models (class k vs rest, labels remapped
  to the binary encoding), reusing the existing SRVF/warping/IRLS machinery in `logistic.rs`
  UNCHANGED. Maximal reuse; K=2 reduces to the binary path.
- Entry point: **`elastic_multinomial(data: &FdMatrix, y: &[usize], argvals: &[f64], ncomp_beta,
  lambda, max_iter, tol) -> Result<ElasticMultinomialResult, FdarError>`** — mirrors the binary
  signature (labels `&[usize]` in `0..K` instead of `&[i8]`).
- Companion: **`predict_elastic_multinomial(fit, new_data, argvals) -> Vec<usize>`** (predicted
  labels), with normalized class probabilities exposed on the result.
- **K=2 agreement:** OvR with K=2 → predicted labels agree with binary `elastic_logistic` on
  separable data (SC3); documented in rustdoc.

### Result Struct
- New **`ElasticMultinomialResult`** (mirrors `ElasticLogisticResult` naming).
- Fields: **`{ n_classes, classes: Vec<usize>, class_models: Vec<ElasticLogisticResult>` (one OvR
  fit per class)`, train_probabilities: FdMatrix (n×K, row-normalized), predicted_classes:
  Vec<usize>, train_accuracy }`**.
- **Probability normalization:** the K one-vs-rest sigmoid outputs are **normalized row-wise to
  sum to 1** (class posterior).
- Include **`train_accuracy`** (mirrors the binary result's `accuracy`).
- Derive `Debug, Clone, PartialEq`; `#[non_exhaustive]`; conditional serde. `#[must_use]` on the fit fn.

### Numerics & Input Guards
- **Reuse binary `elastic_logistic` per class, K times, unchanged** — no edits to `logistic.rs`'s
  SRVF/warping/IRLS solver.
- Input guards: **K ≥ 2 distinct classes, labels form a contiguous `0..K`, `y.len() == n`** →
  `FdarError` (never panic).
- **Convergence/determinism:** pass `lambda` / `max_iter` / `tol` through to each OvR binary fit;
  deterministic (inherits the binary path's determinism).
- **Non-breaking:** binary `elastic_logistic` + `predict_elastic_logistic` signatures retained
  unchanged; multinomial is purely additive.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (VERIFIED present)
- `elastic_regression::elastic_logistic(data, y: &[i8], argvals, ncomp_beta, lambda, max_iter, tol)
  -> Result<ElasticLogisticResult, FdarError>` (`logistic.rs:53`) — the binary elastic-logistic
  fitter to call K times (OvR). `elastic_logistic_with_config` (`:148`) is a config variant.
- `elastic_regression::predict_elastic_logistic(fit, new_data, argvals) -> Vec<f64>` (`:174`) —
  per-class probability prediction for the OvR predict path.
- `ElasticLogisticResult` (`logistic.rs:14`) — the per-class model stored in `class_models`.
- Barrel re-export in `elastic_regression/mod.rs:21` (`pub use logistic::{...}`) — add the new
  symbols here + crate-root `lib.rs`.

### Established Patterns
- All public fns return `Result<T, FdarError>`; entry-point dimension/parameter validation.
- Public result structs derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde.
- Inline `#[cfg(test)] mod tests`; crate-root re-export in `src/lib.rs`.

### Integration Points
- Add `elastic_multinomial` + `predict_elastic_multinomial` + `ElasticMultinomialResult` to
  `elastic_regression/logistic.rs`, the `elastic_regression/mod.rs` barrel, and `src/lib.rs`
  (additive lines only). Do NOT modify existing `elastic_logistic` / `ElasticLogisticResult`.

</code_context>

<specifics>
## Specific Ideas

- Recovery/classification test (SC3): synthetic data with K well-separated per-class shape
  templates (e.g. bumps at different locations) → fitted model recovers the correct labels within
  a documented accuracy threshold; K=2 path agrees with binary `elastic_logistic`.
- Error-path tests (SC4): fewer than 2 classes, non-contiguous labels, label/curve-count mismatch,
  empty input → appropriate `FdarError`, no panic.
- R baseline matched by capability: `fdasrvf` elastic multinomial logistic regression. Document the
  OvR + row-normalization convention in rustdoc.

</specifics>

<deferred>
## Deferred Ideas

- Joint softmax multinomial (single joint optimizer over SRSF space).
- Multinomial elastic PCR / ordinal elastic regression.
- Per-class independent hyperparameter tuning.

</deferred>
