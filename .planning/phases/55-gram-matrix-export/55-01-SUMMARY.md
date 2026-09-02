# Phase 55 — Summary 55-01: Gram-Matrix Export (GAK-05/06)

**Status:** Complete
**Commit (impl):** `30431aa2`

## Files Changed
- `fdars-core/src/metric/gak.rs` — added struct + 2 fns + shared helper + 6 tests + 2 doctests; refactored `gak_gram_matrix` onto the shared helper (public signature/behavior unchanged).
- `fdars-core/src/metric/mod.rs` — extended `pub use gak::{...}` re-export line.
- `fdars-core/src/lib.rs` — added the three new items to the `metric::{...}` crate-root re-export block.
- `.planning/phases/55-gram-matrix-export/55-01-PLAN.md` — task breakdown.

## Public API Added
- `pub struct GakGramTrain { pub gram: FdMatrix, pub(crate) log_self: Vec<f64>, pub sigma: f64, pub(crate) train_rows: Vec<Vec<f64>> }` — `Debug/Clone/PartialEq`, serde-gated, `#[non_exhaustive]`. Accessor `pub fn log_self(&self) -> &[f64]`.
- `pub fn gak_gram_train(data: &FdMatrix, config: &GakConfig) -> Result<GakGramTrain, FdarError>` (`#[must_use]`).
- `pub fn gak_gram_predict(train: &GakGramTrain, new_data: &FdMatrix) -> Result<FdMatrix, FdarError>` (`#[must_use]`) — returns **n_test × n_train**.

## Internal
- `fn build_train_gram(...) -> Result<(FdMatrix, Vec<f64> diag_log, f64 sigma, Vec<Vec<f64>> rows), FdarError>` — shared Gram core for `gak_gram_matrix` (discards extras) and `gak_gram_train` (keeps them). Keeps the two builders bit-identical, diagonal computed once (O(n)).

## Divergences from CONTEXT
- **`GakGramTrain` carries an extra private field `train_rows`.** CONTEXT decision 2 listed only `{gram, log_self, sigma}`, but `gak_gram_predict(train, new_data)` (the mandated signature) needs the *training curves* to evaluate the cross-kernel `loggak(x_test, x_train)` — they are not recoverable from the Gram. Added as `pub(crate)` under the already-required `#[non_exhaustive]`, so the public surface still matches CONTEXT (`gram`, `sigma` public; `log_self` via accessor). No behavioral or API-visibility divergence.
- Predict parallelizes over **test rows** (each row independent) via `iter_maybe_parallel!`, scattering row blocks into the column-major output — matches the "deterministic, seq==par" requirement.

## Tests + Results (all green)
Inline in `gak.rs mod tests`:
- `test_gram_train_shape_psd` — n×n, symmetric (bit-exact), unit diagonal, PSD (min-eig ≥ −1e-8 via `symmetric_eigenvalues`).
- `test_gram_predict_shape` — asserts exactly (3, 5) with n_test=3 ≠ n_train=5.
- `test_gram_predict_normalized` — all entries ∈ [0,1]; identical test curve → ≈1.0 in its column.
- `test_gram_predict_reproduces_train` — predict(train, train_data) ≈ train.gram within 1e-12.
- `test_gram_predict_sigma_consistency` — predict uses `train.sigma` (3.7) even when the test set's own median σ differs by >1; verified against a manual `loggak(...,3.7)` recompute.
- `test_gram_predict_empty_and_grid_errors` — empty test + grid-width mismatch → `InvalidDimension`.
- Doctests on `gak_gram_train` + `gak_gram_predict` — commented `SVC(kernel='precomputed')` handoff.

## Gate Tails
- `cargo test -p fdars-core --features linalg --lib gak`: **17 passed; 0 failed**.
- `cargo test -p fdars-core --features linalg --doc gak`: **4 passed; 0 failed** (2 new).
- `cargo test -p fdars-core gak` (default features): **17 passed; 0 failed**.
- `cargo fmt --check`: clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean (Finished, no warnings).

## Notes for Phase 56
- `GakGramTrain.train_rows` (pub(crate)) + `sigma` + `log_self()` give kernel-k-means everything it needs to run on a fixed precomputed Gram and to route out-of-sample curves via `gak_gram_predict` (same σ/normalization).
- No crate version bump (still 0.30.0 in-tree); milestone bump happens at ship.
