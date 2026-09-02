# Phase 60 Summary: Bundled ShapeletTransformClassifier (SHP-07)

**Completed:** 2026-09-02
**Requirements:** SHP-07
**Status:** Complete — all 4 ROADMAP success criteria PASS

> Note: the implementation was authored by the Phase 60 execution agent, which hit an account session limit immediately before running the final gates / writing this summary / committing. The orchestrator finished inline: ran the full gates (all green), authored this SUMMARY + VERIFICATION, and committed. No code was rewritten — the agent's `classifier.rs`/bench/re-exports were complete and correct.

## What was built

New `fdars-core/src/shapelet/classifier.rs` — the end-to-end bundled classifier wiring Phases 57–59 to the existing `classification/` module. Plus finalized crate-root re-exports and a criterion benchmark.

### Public API added
- `enum ShapeletClassifier { Knn { k: usize }, Lda }` — `#[non_exhaustive]`, `Default = Knn { k: 1 }` (canonical Hills/Lines 1-NN).
- `struct ShapeletClassifierConfig { discovery: ShapeletDiscoveryConfig, classifier: ShapeletClassifier, ncomp: Option<usize> }` — Debug/Clone/PartialEq, serde-gated, `Default`.
- `struct ShapeletClassifierFit { transform: ShapeletTransformFit, classifier: ClassifFit, config }` — `#[non_exhaustive]`; accessors `shapelets()`, `transform()`, `classifier()`, `train_accuracy()`; method `predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>`.
- `pub fn shapelet_classifier_fit(data: &FdMatrix, labels: &[usize], config: &ShapeletClassifierConfig) -> Result<ShapeletClassifierFit, FdarError>` (`#[must_use]`).

### Wiring
- `fit`: `shapelet_transform_fit` (Phase 59 discover+transform) → resolve `ncomp = config.ncomp.unwrap_or(K)` clamped to `min(K, n−1).max(1)` → dispatch on `ShapeletClassifier` to `fclassif_knn_fit(features, labels, None, ncomp, k)` / `fclassif_lda_fit(features, labels, None, ncomp)`.
- `predict`: `self.transform.transform(new_data)` → n_new×K features → predict via the stored `ClassifFit` out-of-sample path (FPCA projection + `predict_from_scores`), reusing the SAME shapelets + z-norm + FPCA rotation as the fit.

### Crate-root re-exports (finalized — the only phase that touches the flat surface)
`src/lib.rs`: `pub use shapelet::{ discover_shapelets, shapelet_classifier_fit, shapelet_distance, shapelet_transform, shapelet_transform_fit, z_normalize_into, z_normalize_window, QualityMeasure, Shapelet, ShapeletClassifier, ShapeletClassifierConfig, ShapeletClassifierFit, ShapeletDiscoveryConfig, ShapeletSet, ShapeletTransformFit };` (+ `prelude.rs`, + `mod.rs` module wiring).

### Benchmark
`fdars-core/benches/shapelet.rs` (criterion, `harness = false`) benchmarking `shapelet_classifier_fit` on a small synthetic 2-class set; `[[bench]] name = "shapelet"` added to `Cargo.toml`.

## Files changed
- Created: `fdars-core/src/shapelet/classifier.rs`, `fdars-core/benches/shapelet.rs`
- Modified: `fdars-core/src/shapelet/mod.rs`, `fdars-core/src/lib.rs`, `fdars-core/src/prelude.rs`, `fdars-core/Cargo.toml`

## Divergence
- **FPCA-on-features (documented):** the inner `fclassif_knn_fit`/`fclassif_lda_fit` run FPCA(ncomp) on the K-dimensional shapelet-distance feature matrix. With the default `ncomp = K` this is full-rank (≈identity), preserving the feature information. sktime uses RotationForest as the inner classifier; fdars intentionally reuses its existing kNN/LDA — documented in the module/fn docs.

## Gates (run inline by orchestrator)
- `cargo fmt --check`: clean
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean (bench compiles warning-free)
- `cargo test -p fdars-core --features linalg --lib shapelet`: 26 passed / 0 failed
- `cargo test -p fdars-core --lib shapelet` (default features): 26 passed / 0 failed
- `cargo test -p fdars-core --features linalg --doc shapelet`: 6 doctests passed
- No new crate dependency. Crate version left at 0.32.0 (milestone-end bump handled separately).
