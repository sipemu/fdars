# Phase 55: Gram-Matrix Export - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — decisions resolved from `.planning/research/` + the Phase 54 seams. No open user decisions.

<domain>
## Phase Boundary

Deliver a split train/predict GAK Gram-matrix export API so users can drive an EXTERNAL precomputed-kernel SVM (`sklearn SVC(kernel='precomputed')` convention). fdars ships NO SVM (native kernel-SVM deferred to SVM-01). Functions live in `metric/gak.rs`, re-exported at the crate root. Additive/non-breaking, no new dependency.

In scope (GAK-05/06):
- **Train Gram export**: n_train × n_train, symmetric, PSD, unit diagonal — plus the training self-kernels needed for prediction normalization.
- **Predict Gram export**: n_test × n_train, cross-normalized against the STORED training self-kernels (not test-only self-kernels), same σ as train.

Out of scope: kernel-k-means (Phase 56), native SVM, any change to Phase 54's public kernel.
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + Phase 54 seams)

1. **Reuse the Phase 54 seam.** `pub(crate) fn loggak(x,y,sigma) -> f64` (unnormalized log-kernel) already exists in `metric/gak.rs`; the diagonal self-kernel of curve `i` is `loggak(x_i, x_i, sigma)`. Build both Gram matrices from `loggak` + a computed/stored diagonal, so normalization is `exp(loggak(a,b) - 0.5*(diag_a + diag_b))`. Do NOT recompute self-kernels twice (compute the diagonal once — O(n), not O(n²)).

2. **Split API + a result struct carrying the training diagonals** (this is what makes the cross-normalization bug impossible):
   - `pub struct GakGramTrain { pub gram: FdMatrix, pub(crate) log_self: Vec<f64>, pub sigma: f64 }` (n×n normalized Gram + per-training-curve unnormalized log self-kernels + the σ actually used). Derive `Debug, Clone, PartialEq`; serde-gated; `#[non_exhaustive]`. Expose `gram` (and read accessors) publicly; keep `log_self` `pub(crate)` (internal contract) — OR make it a public field if serde round-trip needs it; prefer `pub(crate)` with a `&self` accessor if one is needed.
   - `pub fn gak_gram_train(data: &FdMatrix, config: &GakConfig) -> Result<GakGramTrain, FdarError>` — resolves σ (via `sigma_gak` if `config.sigma` is None), builds the normalized symmetric-by-assignment PSD n×n Gram, and stores `log_self[i] = loggak(x_i,x_i,σ)` and `sigma`. `#[must_use]`.
   - `pub fn gak_gram_predict(train: &GakGramTrain, new_data: &FdMatrix) -> Result<FdMatrix, FdarError>` — returns an **n_test × n_train** matrix (rows = new/test curves, cols = training curves; ASSERT this orientation in a test) whose entry `(t,j) = exp(loggak(x_test_t, x_train_j, train.sigma) - 0.5*(loggak(x_test_t,x_test_t,train.sigma) + train.log_self[j]))`. Uses `train.sigma` and `train.log_self` — the STORED training diagonals — never test-set-only self-kernels. Every entry ∈ [0,1]. `#[must_use]`.

3. **Orientation is n_test × n_train** to match `SVC.predict(K)` where `K[i,j] = kernel(test_i, train_j)`. This must be asserted (shape test), because the silent-degradation bug is passing the transpose.

4. **`gak_gram_train` may internally reuse / share code with the Phase 54 `gak_gram_matrix`** (same normalized self-Gram); factor a shared helper if clean, but do not alter `gak_gram_matrix`'s existing public signature/behavior. Parallelize the predict cross-matrix with `iter_maybe_parallel!` (deterministic).

5. **Validation:** σ>0 (post-resolution), non-empty train + test, matching evaluation-grid width between train and test (same #cols). Return `FdarError::{InvalidDimension, InvalidParameter}` on violation.

6. **Re-exports:** add `gak_gram_train`, `gak_gram_predict`, `GakGramTrain` to `metric/mod.rs` and crate-root `lib.rs` alongside the existing gak items.
</decisions>

<code_context>
## Existing Code Insights
- `metric/gak.rs` seams: `pub(crate) fn loggak(x,y,sigma)->f64` (L108), `pub(crate) fn logsumexp3` (L73), `pub fn gak_gram_matrix` (L255, the self-Gram to mirror), `pub fn sigma_gak` (L191), `GakConfig { sigma: Option<f64> }`.
- Re-export site: `metric/mod.rs:141 pub use gak::{gak, gak_gram_matrix, sigma_gak, GakConfig};` — extend this line.
- `FdMatrix` column-major; construct via its `new`/from-rows API (see how `gak_gram_matrix` builds its output).
- `FdarError` in `src/error.rs`.
</code_context>

<specifics>
## Specific Ideas (verification hooks)
Tests the plan must include:
- `test_gram_train_shape_psd`: train Gram is n_train×n_train, symmetric, unit diagonal, PSD (min-eig ≥ −1e-8).
- `test_gram_predict_shape`: predict Gram is exactly n_test × n_train (assert both dims; a non-square test set catches a transpose).
- `test_gram_predict_normalized`: every predict entry ∈ [0,1]; a test curve equal to a training curve yields ≈1.0 in that column (cross-normalization correct).
- `test_gram_predict_uses_train_diag`: predicting with the SAME data as train reproduces the train Gram rows (predict(train_data) ≈ train.gram, within 1e-12) — proves the stored-diagonal normalization matches.
- `test_gram_predict_sigma_consistency`: predict uses train.sigma even if called with data whose own median σ differs.
- A rustdoc example: build train Gram + predict Gram → hand off to an external precomputed-kernel SVM (comment the sklearn call; do not depend on Python).
</specifics>

<deferred>
## Deferred Ideas
- Native in-crate kernel-SVM (SVM-01) — out of milestone scope.
- Caching/streaming Gram for very large n — future perf.
</deferred>
