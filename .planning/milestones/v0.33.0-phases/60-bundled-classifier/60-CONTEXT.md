# Phase 60: Bundled ShapeletTransformClassifier - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` + Phase 57/58/59 API + the classification API. No open user decisions. FINAL phase of milestone v0.33.0.

<domain>
## Phase Boundary

Deliver the end-to-end `ShapeletTransformClassifier` (discover → transform → classify) + out-of-sample `predict`, plus the crate-root public re-exports for the whole `shapelet` module and a criterion benchmark. New `src/shapelet/classifier.rs`. Builds on Phase 57–59 + the existing `classification/` module. Additive/non-breaking, no new dependency.

In scope (SHP-07):
- **Fit** — `ShapeletTransformClassifier::fit`: discover shapelets (Phase 58) + transform training data (Phase 59) → n×K features, then fit an existing fdars classifier (kNN default, LDA optional) on those features.
- **Predict** — transform new curves through the stored shapelets → n_new×K, classify via the stored inner classifier.
- **Crate-root re-exports** — finalize `pub use shapelet::{…}` at `src/lib.rs` for all public shapelet items (distance, discovery, transform, classifier). This is the ONLY phase that touches the flat re-export surface.
- **Criterion benchmark** — a light `shapelet` bench (discover+transform+fit on a small synthetic set).

Out of scope: learning-shapelets; new classifiers.
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + the classification API)

1. **Config:** `pub struct ShapeletClassifierConfig { discovery: ShapeletDiscoveryConfig, classifier: ShapeletClassifier, ncomp: Option<usize> }` — Debug/Clone/PartialEq, serde-gated, `Default`. `pub enum ShapeletClassifier { Knn { k: usize }, Lda }` (`#[non_exhaustive]`, Default `Knn { k: 1 }` — canonical Hills/Lines 1-NN). `ncomp` = FPCA components for the inner classifier on the K-feature matrix; **default = None → use K** (full-rank; the FPCA rotation is then ~identity, so kNN/LDA operate on essentially the raw shapelet-distance features). Clamp to `min(K, n_train − 1)` to satisfy FPCA's `ncomp ≤ min(n,m)` bound.

2. **Inner classifier wiring:** both `fclassif_knn_fit(data, y, scalar_covariates, ncomp, k_nn)` and `fclassif_lda_fit(data, y, scalar_covariates, ncomp)` accept the feature matrix as `data` and run FPCA(ncomp) internally. The STC calls them with `data = features` (the n×K shapelet-distance matrix from Phase 59), `y = labels`, `scalar_covariates = None`, `ncomp = resolved`. **Document the divergence:** the inner FPCA runs on shapelet-distance features (not functional evaluation points); with `ncomp=K` it is full-rank and preserves the feature information — an intentional, documented reuse of the existing classifier machinery (sktime uses RotationForest here; fdars reuses kNN/LDA).

3. **Result:** `pub struct ShapeletClassifierFit { transform: ShapeletTransformFit, classifier: ClassifFit, config: ShapeletClassifierConfig }` — Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`; accessors `shapelets() -> &ShapeletSet`, `transform() -> &ShapeletTransformFit`, `classifier() -> &ClassifFit`, `train_accuracy() -> f64` (from `ClassifFit.result`).

4. **Fit API:** `pub fn shapelet_classifier_fit(data: &FdMatrix, labels: &[usize], config: &ShapeletClassifierConfig) -> Result<ShapeletClassifierFit, FdarError>` (`#[must_use]`): `shapelet_transform_fit` (Phase 59, discovers+transforms) → resolve ncomp = config.ncomp.unwrap_or(K).min(K).min(n−1).max(1) → dispatch on `config.classifier` to `fclassif_knn_fit`/`fclassif_lda_fit` on the features. Validation propagates from Phase 58/59 + the classifier (≥2 classes, labels length, etc.).

5. **Predict API:** `impl ShapeletClassifierFit { pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> }` (`#[must_use]`): `self.transform.transform(new_data)` → n_new×K features, then predict via the stored `ClassifFit`'s out-of-sample path — reuse the existing `FpcPredictor` machinery (`ClassifFit` implements it: `project` the feature matrix through the stored FPCA rotation → scores → `predict_from_scores` per row). Use the existing crate convenience for out-of-sample classification if one exists (check `explain_generic`/`FpcPredictor::project` + a predict loop); otherwise project + `predict_from_scores` row-by-row. Must reuse the SAME shapelets + z-normalization + FPCA rotation as the fit (no re-fitting).

6. **Crate-root re-exports (finalize):** in `src/lib.rs`, add `pub use shapelet::{ Shapelet, shapelet_distance, z_normalize_window, QualityMeasure, ShapeletDiscoveryConfig, ShapeletSet, discover_shapelets, ShapeletTransformFit, shapelet_transform, shapelet_transform_fit, ShapeletClassifier, ShapeletClassifierConfig, ShapeletClassifierFit, shapelet_classifier_fit };` (adjust to the exact exported names). Verify all are reachable from the crate root; keep `pub mod shapelet;` too. Optionally add to `prelude.rs` alongside other classifiers.

7. **Benchmark:** add `fdars-core/benches/shapelet.rs` (criterion) benchmarking `shapelet_classifier_fit` on a small synthetic 2-class set, plus a `[[bench]]` entry in `fdars-core/Cargo.toml` (`name = "shapelet"`, `harness = false`). Keep the dataset small so the bench is quick.
</decisions>

<code_context>
## Existing Code Insights
- Phase 59 `src/shapelet/transform.rs`: `ShapeletTransformFit` (`shapelets()`, `features()`, `transform(new_data)`), `shapelet_transform_fit(data, labels, config)`.
- `src/classification/fit.rs`: `fclassif_knn_fit(data, y, scalar_covariates, ncomp, k_nn) -> Result<ClassifFit,_>` (L231), `fclassif_lda_fit(data, y, scalar_covariates, ncomp)` (L76). `ClassifFit { result: ClassifResult, fpca_mean, fpca_rotation, fpca_scores, ncomp, method, fpca_int_weights }`; implements `FpcPredictor` (`predict_from_scores` L338, `project`). `ClassifResult` carries predicted labels + accuracy + confusion matrix.
- `src/explain_generic.rs` / `FpcPredictor` trait: `project()`, `predict_from_scores()`, `training_scores()` — the out-of-sample projection path.
- `src/lib.rs:445` `pub use classification::{…}` (mirror for the shapelet re-export block); `src/prelude.rs`.
- Conventions: config structs + Default, `#[must_use]`, serde-gated derives, `Result<_,FdarError>`.
- Benchmarks: existing `fdars-core/benches/*.rs` + `[[bench]]` entries in Cargo.toml (mirror one).
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md)
Tests the plan must include:
- `test_stc_fit_predict_end_to_end`: synthetic 2-class data with a class-discriminative motif → `fit` then `predict` on a held-out test split recovers the labels at high accuracy (well above chance); TRAIN/TEST discipline (discover on train only, predict on unseen test).
- `test_stc_knn_default`: default config uses kNN (1-NN); fit succeeds, `train_accuracy()` reported.
- `test_stc_lda_option`: `ShapeletClassifier::Lda` path fits + predicts without error.
- `test_stc_predict_consistency`: predicting the training curves reproduces the fit-time training predictions (same shapelets + classifier).
- `test_stc_validation`: <2 classes / label mismatch → error (propagated).
- `test_shapelet_reexports`: the crate-root re-exported names are reachable (a compile-level use of `fdars_core::{shapelet_classifier_fit, ShapeletClassifierConfig, …}`).
- Doctest on `shapelet_classifier_fit` with a train/test split (no external data).
</specifics>

<deferred>
## Deferred Ideas
- Learning-shapelets (LSH-01), multivariate/DTW shapelets, ROCKET — future milestones.
- Tuning ncomp/whether to bypass FPCA entirely for the feature classifier — a possible future refinement (identity projection).
</deferred>
