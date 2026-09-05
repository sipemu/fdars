# Phase 73: VEESA Explainability Pipeline & Integration - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the VEESA explainability layer on top of Phase 72's jfPCA fit/transform seam
(requirements VEE-03, VEE-04, VEE-05): model-agnostic permutation feature importance
(PFI) over jfPCA PC scores, principal-direction reconstruction for visualization, and
a cohesive end-to-end pipeline with full re-exports and a running doctest.

Additive and non-breaking (protects R/WASM bindings + 28 examples), no new crate
dependency, reuse-first over `jfpca_model.rs` (Phase 72), `elastic_fpca.rs`,
`elastic_explain.rs`, and `explain/helpers/permutation.rs`.

**Scope fences (locked, from REQUIREMENTS out-of-scope):**
- VEE-F1: No new learner (random forest, etc.) — PFI stays model-agnostic.
- VEE-F2: No plotting/rendering — VEE-04 returns the curves; rendering is the caller's concern.
- No new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Model-agnostic PFI (VEE-03)
- Predictor is abstracted as a caller-supplied **prediction closure** `Fn(&FdMatrix) -> Vec<f64>`
  over PC scores — fully model-agnostic (NOT tied to `elastic_pcr` / no `FpcPredictor` coupling).
- Error metric: an `enum PfiMetric { Mse, Mae, Accuracy, ... }` plus a **custom-closure
  escape hatch** for arbitrary metrics.
- Determinism: `n_repeats` (default 10) + a `seed` parameter; permute each PC-score column
  with `StdRng::seed_from_u64(seed)` (per project RNG convention). Deterministic under seed.
- Code reuse: a new additive `elastic_pfi` function that reuses the column-shuffle helper in
  `explain/helpers/permutation.rs` if it can be lifted without breaking existing FPC callers;
  otherwise self-contained (thin additive layer alongside — never modify existing signatures).
- Known-signal gate: on a design where one PC carries the signal, that PC must rank above noise PCs.

### Principal-direction reconstruction (VEE-04)
- Output struct `PrincipalDirections { pc_index, c_values, amplitude_curves, phase_curves }`
  — splits each perturbation into its **amplitude (warped-function)** and **phase
  (warping-function)** parts; returns curves only (rendering is the caller's concern, VEE-F2).
- c multipliers: caller-supplied `&[f64]` (e.g. `[-2,-1,0,1,2]`); **`c = 0` reproduces the
  jfPCA mean function** (a make-or-break gate).
- Component selection: caller picks PC indices (or first-k).
- Reconstruction math: invert the jfPCA — perturb the score along the trained
  `vert_component`/`horiz_component` (weighted by `balance_c`), map SRSF→function for the
  amplitude part and recover the warping for the phase part, reusing existing `elastic_fpca`
  inversion helpers. Do NOT re-derive the inversion locally.

### Pipeline & Integration (VEE-05)
- A thin `veesa_pipeline` convenience helper tying **fit → transform → PFI** end-to-end.
- Module placement: new additive module(s) — `elastic_pfi.rs` (VEE-03) and principal-direction
  reconstruction in `jfpca_model.rs` or a new `veesa.rs`; do NOT cram into `elastic_explain.rs`.
- Full **crate-root (`lib.rs`) + prelude re-exports** for all new public items.
- A running **end-to-end module doctest** (fit → transform → PFI) under `cargo test --doc`.

### Claude's Discretion
- Exact struct field layout, the `PfiMetric` variant set, internal helper factoring, the
  precise home module for reconstruction (`jfpca_model.rs` vs new `veesa.rs`), and doctest
  fixture construction are at Claude's discretion within the above constraints and crate
  conventions (column-major `FdMatrix`, `Result<T, FdarError>`, `#[non_exhaustive]`,
  `#[must_use]`, conditional serde, `Debug/Clone/PartialEq`).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 72 seam** (`fdars-core/src/jfpca_model.rs`): `JfpcaModel` (stores `karcher_mean`,
  `mean_q`, `mean_psi`, `vert_component`, `horiz_component`, `balance_c`, `argvals`,
  `eigenvalues`, `ncomp`, `lambda`, `joint_result`, `training_gammas`, `training_aligned`),
  `jfpca_fit`, `JfpcaModel::transform`, `JfpcaModel::score_training`, `JfpcaTransform`.
- **`elastic_fpca.rs`**: joint-FPCA machinery + SRSF/inversion helpers; `pub(crate)`
  `build_combined_representation`; the eigenvector split (`vert_component`/`horiz_component`).
- **`explain/helpers/permutation.rs`**: column-shuffle / permutation helpers (currently FPC-specific;
  `pub(crate)` — candidate to lift for the generic PFI). `explain/importance.rs` +
  `explain/mod.rs`: `fpc_permutation_importance`, `FpcPermutationImportance` (the FPC analog —
  do NOT modify; the new PFI is a model-agnostic sibling).
- **`elastic_explain.rs`**: `elastic_pcr_attribution`, `ElasticAttributionResult` (existing
  elastic attribution — additive sibling, do not modify signatures).

### Established Patterns
- Column-major `FdMatrix`; `Result<T, FdarError>`; dimension checks at entry.
- Per-thread RNG seeding: `StdRng::seed_from_u64(seed + k)`.
- Result structs derive `Debug, Clone, PartialEq`; conditional serde; `#[non_exhaustive]`;
  `#[must_use]` on expensive computations.
- Crate-root re-exports in `src/lib.rs`; prelude re-exports in `src/prelude.rs`.

### Integration Points
- New public items (`elastic_pfi` / `PfiMetric` / `PfiResult`, `PrincipalDirections` +
  reconstruction fn, `veesa_pipeline`) re-exported from `lib.rs` + `prelude.rs`.
- Consumes Phase 72's `JfpcaModel` + `score_training()` (exact training coordinates for PFI).

</code_context>

<specifics>
## Specific Ideas

- PFI operates on PC scores (from `transform`/`score_training`), permuting each score column;
  the prediction closure maps a score matrix → predictions, so any external model (even one
  trained outside the crate) can be explained.
- Principal-direction `c=0` MUST reproduce the jfPCA mean exactly (known-answer gate).
- Confirm at plan time whether `explain/helpers/permutation.rs`'s shuffle can be lifted to a
  predictor-agnostic helper without breaking its existing `elastic_pcr`/FPC callers; if not,
  add a thin additive generic helper alongside (still additive/non-breaking).

</specifics>

<deferred>
## Deferred Ideas

- Native learner / random forest (VEE-F1) → future milestone.
- Plotting/rendering of principal directions + PFI (VEE-F2) → caller concern, out of scope.
- Elastic conformal anomaly detection → Phase 74.
- R/WASM binding exposure of the VEESA surface → future milestone (issue fdars-j75).

</deferred>
