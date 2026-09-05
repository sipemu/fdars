# Phase 73: VEESA Explainability Pipeline & Integration - Research

**Researched:** 2026-09-05
**Domain:** Rust elastic-FDA explainability — model-agnostic PFI, principal-direction reconstruction, pipeline integration
**Confidence:** HIGH (all claims grounded in file:line reads this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Model-agnostic PFI (VEE-03)**
- Predictor abstracted as caller-supplied `Fn(&FdMatrix) -> Vec<f64>` over PC scores — NOT tied to `elastic_pcr` / no `FpcPredictor` coupling.
- Error metric: `enum PfiMetric { Mse, Mae, Accuracy, ... }` + custom-closure escape hatch.
- Determinism: `n_repeats` (default 10) + `seed` parameter; permute with `StdRng::seed_from_u64(seed)`.
- Code reuse: new additive `elastic_pfi` function reusing column-shuffle helper in `explain/helpers/permutation.rs` if liftable without breaking existing callers; otherwise self-contained thin additive layer.
- Known-signal gate: on a design where one PC carries the signal, that PC must rank above noise PCs.

**Principal-direction reconstruction (VEE-04)**
- Output struct `PrincipalDirections { pc_index, c_values, amplitude_curves, phase_curves }` — splits each perturbation into amplitude (warped-function) and phase (warping-function) parts; returns curves only.
- `c = 0` reproduces the jfPCA mean function (make-or-break gate).
- Caller supplies `&[f64]` c multipliers (e.g. `[-2,-1,0,1,2]`).
- Reconstruction math: invert the jfPCA — perturb the score along trained `vert_component`/`horiz_component` (weighted by `balance_c`), map SRSF→function for amplitude part and recover warping for phase part, reusing existing `elastic_fpca` inversion helpers. Do NOT re-derive locally.

**Pipeline & Integration (VEE-05)**
- Thin `veesa_pipeline` convenience helper tying fit → transform → PFI end-to-end.
- Module placement: new additive module(s) — `elastic_pfi.rs` (VEE-03) and principal-direction reconstruction in `jfpca_model.rs` or a new `veesa.rs`; do NOT cram into `elastic_explain.rs`.
- Full crate-root (`lib.rs`) + prelude re-exports for all new public items.
- A running end-to-end module doctest under `cargo test --doc`.

### Claude's Discretion

- Exact struct field layout, the `PfiMetric` variant set, internal helper factoring, the precise home module for reconstruction (`jfpca_model.rs` vs new `veesa.rs`), and doctest fixture construction are at Claude's discretion within the above constraints and crate conventions.

### Deferred Ideas (OUT OF SCOPE)

- VEE-F1: No new learner (random forest, etc.) — PFI stays model-agnostic.
- VEE-F2: No plotting/rendering — VEE-04 returns the curves; rendering is the caller's concern.
- Elastic conformal anomaly detection → Phase 74.
- R/WASM binding exposure of the VEESA surface → future milestone (issue fdars-j75).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VEE-03 | Model-agnostic permutation feature importance over jfPCA PC scores; generic over any prediction closure; deterministic under seed; informative PC ranks above noise PCs | Permutation helpers liftable (see §Permutation Helper Liftability); `shuffle_global` is the key primitive; full design given in §VEE-03 |
| VEE-04 | Principal-direction reconstruction at μ ± c·σⱼ; split into amplitude and phase parts; `c=0` reproduces jfPCA mean | SRSF inversion path documented in §VEE-04; exact inversion formula given; all helper names + file:line cited |
| VEE-05 | Cohesive pipeline, re-exports, running module doctest | Re-export pattern documented in §VEE-05; lib.rs:509 and prelude.rs:75 are the insertion points |
</phase_requirements>

---

## Summary

Phase 73 builds the VEESA explainability layer directly on top of Phase 72's `JfpcaModel` seam. All three requirements are additive-only: no changes to existing public signatures, no new crate dependencies. The work is principally wiring and thin new modules.

**VEE-03** (model-agnostic PFI): The existing permutation primitives in `explain/helpers/permutation.rs` — specifically `shuffle_global` — are `pub(crate)` and can be called from a new sibling module `elastic_pfi.rs` without modification and without any risk of breaking existing FPC callers. The existing `fpc_permutation_importance` and `fpc_permutation_importance_logistic` in `importance.rs` use their own per-component advancing `StdRng` and are not modified. The new `elastic_pfi` function takes a `&FdMatrix` of PC scores (from `model.score_training()`) and a `Fn(&FdMatrix) -> Vec<f64>` prediction closure, so any external model can be explained.

**VEE-04** (principal-direction reconstruction): The full SRSF→function inversion path exists in `alignment/srsf.rs:65` (`srsf_inverse`) and `warping.rs:73,112` (`psi_to_gam`, `exp_map_sphere`). These are all public. The amplitude reconstruction is: perturb `mean_q[0..m]` by `c * sigma_j * vert_component[j, 0..m]`, invert via `srsf_inverse`, using the augmented dimension to recover `f0`. The phase reconstruction is: perturb `mean_psi` by `c * sigma_j * horiz_component[j, 0..m]` in the tangent space, exponentiate via `exp_map_sphere`, recover gamma via `psi_to_gam`. At `c = 0` both perturbations vanish and the amplitude path reduces exactly to the jfPCA mean curve.

**VEE-05** (pipeline + integration): The re-export pattern is established: `lib.rs:509` has `pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};` and `prelude.rs:75` has the same. New items follow this exact pattern. The convenience wrapper `veesa_pipeline` ties fit → `score_training()` → `elastic_pfi` end-to-end.

**Primary recommendation:** Place PFI in a new `elastic_pfi.rs` module; place principal-direction reconstruction as `impl JfpcaModel` methods in `jfpca_model.rs` (avoids a new file for a small addition). If reconstruction grows beyond ~150 lines, extract to `veesa.rs` per the crate's convention.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| jfPCA fit/transform | `jfpca_model.rs` | `elastic_fpca.rs` (internal) | Phase 72 seam owns the public contract |
| Model-agnostic PFI over PC scores | `elastic_pfi.rs` (new) | `explain/helpers/permutation.rs` (reuse) | Domain-separate from FPC-specific `importance.rs`; permutation primitive is shared |
| Principal-direction reconstruction | `jfpca_model.rs` impl block (or `veesa.rs`) | `alignment/srsf.rs`, `warping.rs` | Operates directly on `JfpcaModel` fields; belongs near the model |
| SRSF inversion (amplitude part) | `alignment/srsf.rs` (existing) | — | `srsf_inverse` is already public; do not re-implement |
| Phase reconstruction (psi→gam) | `warping.rs` (existing) | — | `exp_map_sphere`, `psi_to_gam` are already public |
| Pipeline convenience wrapper | `elastic_pfi.rs` or `jfpca_model.rs` | — | Thin delegating function; home is discretionary |
| Crate-root + prelude re-exports | `lib.rs`, `prelude.rs` | — | Follows established pattern |

---

## Standard Stack

No new crate dependencies. All primitives used are already in scope.

### Core Reused Functions

| Function | Visibility | File | Line | Purpose in Phase 73 |
|----------|------------|------|------|----------------------|
| `shuffle_global` | `pub(crate)` | `explain/helpers/permutation.rs` | 9 | Column-shuffle for unconditional PFI permutation |
| `clone_scores_matrix` | `pub(crate)` | `explain/helpers/projection.rs` | 40 | Clone score matrix before shuffling |
| `srsf_inverse` | `pub` | `alignment/srsf.rs` | 65 | Invert perturbed SRSF → amplitude curve |
| `exp_map_sphere` | `pub` | `warping.rs` | 112 | Exponentiate perturbed tangent vector → ψ |
| `psi_to_gam` | `pub` | `warping.rs` | 73 | Convert ψ → warping function γ |
| `JfpcaModel::score_training` | `pub` | `jfpca_model.rs` | 336 | Exact training PC scores for PFI |
| `JfpcaModel::transform` | `pub` | `jfpca_model.rs` | 267 | Out-of-sample PC scores |

### JfpcaModel Fields Used by Phase 73

All fields are `pub` on `JfpcaModel` [VERIFIED: `fdars-core/src/jfpca_model.rs:75-127`]:

```
pub karcher_mean: Vec<f64>,       // length m — the jfPCA mean curve
pub mean_q: Vec<f64>,             // length m+1 — augmented SRSF mean
pub mean_psi: Vec<f64>,           // length m — ψ Karcher mean on sphere
pub vert_component: FdMatrix,     // ncomp × (m+1) — amplitude eigenvectors
pub horiz_component: FdMatrix,    // ncomp × m — phase eigenvectors
pub balance_c: f64,               // phase-vs-amplitude balance weight
pub argvals: Vec<f64>,            // length m — evaluation grid
pub eigenvalues: Vec<f64>,        // length ncomp — σⱼ² values
pub ncomp: usize,
pub joint_result: JointFpcaResult, // contains training scores (n × ncomp)
pub training_gammas: FdMatrix,    // n_train × m — Karcher gammas (stored for exact round-trip)
pub training_aligned: FdMatrix,   // n_train × m — aligned training curves
```

**σⱼ source:** `model.eigenvalues[j]` is the variance explained (σⱼ²). Therefore `σⱼ = model.eigenvalues[j].sqrt()`.

---

## Permutation Helper Liftability Analysis

[VERIFIED: `fdars-core/src/explain/helpers/permutation.rs:1-21`]

`shuffle_global` signature:
```rust
pub(crate) fn shuffle_global(
    perm_scores: &mut FdMatrix,
    scores: &FdMatrix,
    k: usize,
    n: usize,
    rng: &mut StdRng,
)
```

`clone_scores_matrix` signature [VERIFIED: `fdars-core/src/explain/helpers/projection.rs:40-48`]:
```rust
pub(crate) fn clone_scores_matrix(scores: &FdMatrix, n: usize, ncomp: usize) -> FdMatrix
```

**Verdict: LIFTABLE WITHOUT MODIFICATION.** Both functions are `pub(crate)` and take only `FdMatrix` + primitive arguments — no `FregreLmResult`, no `FpcPredictor`, no domain-specific type. A new `elastic_pfi.rs` at crate level can call them directly via `use crate::explain::helpers::{shuffle_global, clone_scores_matrix};`. The existing `fpc_permutation_importance` callers in `importance.rs` are not touched.

**Critical provenance note on RNG convention:** The existing FPC PFI functions (`importance.rs:130`, `importance.rs:225`, `importance.rs:548`, `importance.rs:663`) use a **single advancing `StdRng`** across all components with a deferred migration note. The new `elastic_pfi` should use the **same advancing RNG pattern** (`StdRng::seed_from_u64(seed)` once, advance across components) for API consistency within the VEESA surface. The per-component reseed alternative (`seed_from_u64(seed + k)`) is explicitly tagged as behavior-changing and deferred. Do NOT introduce per-component reseeding.

---

## Architecture Patterns

### Recommended Project Structure (new files only)

```
fdars-core/src/
├── elastic_pfi.rs         # NEW: VEE-03 model-agnostic PFI + VEE-05 veesa_pipeline
├── jfpca_model.rs         # MODIFIED: add impl JfpcaModel { fn principal_directions(...) }
├── lib.rs                 # MODIFIED: pub mod elastic_pfi; + re-exports
└── prelude.rs             # MODIFIED: re-export new public items
```

### System Architecture Diagram

```
[Caller]
   │  raw curves (n × m)
   ▼
[jfpca_fit]          ─── jfpca_model.rs ───────────────────────────────
   │ JfpcaModel                                                         │
   │                                                                    │
   ├──► [score_training()]  ──► PC scores (n × ncomp) [exact, <1e-8]  │
   │                                       │                            │
   │    [transform(new)]   ──► JfpcaTransform.scores [out-of-sample]  │
   │                                                                    │
   └──► [principal_directions(pc_idx, c_values)]                       │
            │ perturb vert_component / horiz_component                  │
            │ srsf_inverse (amplitude) ─── alignment/srsf.rs:65        │
            │ exp_map_sphere + psi_to_gam (phase) ─── warping.rs:73,112│
            ▼                                                            │
        PrincipalDirections { amplitude_curves, phase_curves }         │
                                                                        │
─────────────────────── elastic_pfi.rs ─────────────────────────────────
   PC scores (n × ncomp)  +  Fn(&FdMatrix)->Vec<f64>  +  PfiMetric
   │
   ├── clone_scores_matrix  ─── explain/helpers/projection.rs:40
   ├── shuffle_global       ─── explain/helpers/permutation.rs:9
   │
   ▼
   ElasticPfiResult { importance, baseline_metric, permuted_metric }

[veesa_pipeline]  ─── fit → score_training → elastic_pfi (convenience wrapper)
```

### Pattern 1: Model-Agnostic PFI (VEE-03)

**What:** Permute each PC-score column in turn, measure metric degradation with caller's prediction closure.
**When to use:** Any trained predictor over jfPCA scores (elastic-PCR, logistic, external random forest, etc.).

```rust
// Source: design derived from explain/importance.rs:130-146 (existing advancing-RNG pattern)
// and explain/helpers/permutation.rs:9-21 (shuffle_global)

pub enum PfiMetric {
    Mse,
    Mae,
    Accuracy,
    Custom(Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>),
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElasticPfiResult {
    /// Metric drop per PC (length ncomp). Higher = more important.
    pub importance: Vec<f64>,
    /// Baseline metric (no permutation).
    pub baseline_metric: f64,
    /// Mean metric after permuting each PC (length ncomp).
    pub permuted_metric: Vec<f64>,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pfi(
    scores: &FdMatrix,          // n × ncomp — from score_training() or transform()
    y: &[f64],                  // n response values
    predict: impl Fn(&FdMatrix) -> Vec<f64>,  // maps scores → predictions
    metric: &PfiMetric,
    n_repeats: usize,           // default 10
    seed: u64,
) -> Result<ElasticPfiResult, FdarError> { ... }
```

**Key implementation notes:**
- `StdRng::seed_from_u64(seed)` once; advance across all components (mirrors existing FPC PFI pattern).
- Use `clone_scores_matrix(scores, n, ncomp)` + `shuffle_global(&mut perm, scores, k, n, &mut rng)`.
- Baseline metric computed ONCE before the loop (not re-derived per repeat).
- Metric computation: MSE = `mean((y[i] - pred[i])^2)`, MAE = `mean(|y[i] - pred[i]|)`, Accuracy = `mean(round(pred[i]) == y[i])`.
- `Custom` variant: `metric_fn(y, &predictions)` — caller supplies the aggregation.

### Pattern 2: Principal-Direction Reconstruction (VEE-04)

**What:** At each c-multiplier, perturb the joint PC basis and invert to get amplitude and phase curves.
**When to use:** Visualization of how PC direction j changes the underlying functions.

**Amplitude reconstruction formula:**

The joint SRSF representation for curve i has augmented SRSF `q_aug_centered[i, :]` and the k-th PC is `vert_component[k, 0..m+1]`. The jfPCA mean in function space is `model.karcher_mean`.

To reconstruct at `c * σⱼ` along PC `j` (amplitude part only):
1. The perturbed augmented SRSF direction: `delta_q[l] = c * sigma_j * model.vert_component[(j, l)]` for `l in 0..m`.
2. Perturbed SRSF: `q_perturbed[l] = model.mean_q[l] + delta_q[l]` for `l in 0..m`.
3. Recover `f0` from augmented dimension: `aug_val = model.mean_q[m] + c * sigma_j * model.vert_component[(j, m)]`; then `f0 = aug_val.signum() * aug_val * aug_val`.
4. Invert: `amplitude_curve = srsf_inverse(&q_perturbed[0..m], &model.argvals, f0)`.

**At `c = 0`:** `q_perturbed == model.mean_q[0..m]` and `f0 = model.mean_q[m].signum() * model.mean_q[m].powi(2)`, so `srsf_inverse` integrates the mean SRSF from the mean initial value — this MUST reproduce `model.karcher_mean` exactly (the make-or-break gate).

**Phase reconstruction formula:**

The phase perturbation operates in tangent-space (shooting-vector space):
1. Tangent perturbation: `v_perturbed[l] = c * sigma_j * model.horiz_component[(j, l)]` for `l in 0..m`.
2. Exponentiate from `mean_psi`: `psi_perturbed = exp_map_sphere(&model.mean_psi, &v_perturbed, &time)` where `time = (0..m).map(|i| i as f64 / (m-1) as f64)`.
3. Convert to warping: `gam_perturbed = psi_to_gam(&psi_perturbed, &time)`.
4. Scale to argvals domain: `phase_curve[l] = model.argvals[0] + gam_perturbed[l] * (model.argvals[m-1] - model.argvals[0])`.

**At `c = 0`:** `v_perturbed` is the zero vector; `exp_map_sphere` returns `mean_psi` unchanged (identity perturbation); `psi_to_gam` gives the identity warp.

```rust
// Source: warping.rs:73 (psi_to_gam), warping.rs:112 (exp_map_sphere),
//         alignment/srsf.rs:65 (srsf_inverse)

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrincipalDirections {
    /// PC index (0-based).
    pub pc_index: usize,
    /// c multipliers supplied by caller (length n_c).
    pub c_values: Vec<f64>,
    /// Amplitude curves at each c value (n_c × m).
    pub amplitude_curves: FdMatrix,
    /// Phase (warping) curves at each c value (n_c × m).
    pub phase_curves: FdMatrix,
}

// Method on JfpcaModel:
impl JfpcaModel {
    pub fn principal_directions(
        &self,
        pc_index: usize,
        c_values: &[f64],
    ) -> Result<PrincipalDirections, FdarError> { ... }
}
```

### Pattern 3: Pipeline Wrapper (VEE-05)

```rust
// veesa_pipeline — thin delegating convenience function

pub struct VeesaPipelineResult {
    pub model: JfpcaModel,
    pub training_scores: JfpcaTransform,   // from score_training()
    pub pfi: ElasticPfiResult,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn veesa_pipeline(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
    lambda: f64,
    max_iter: usize,
    y: &[f64],
    predict: impl Fn(&FdMatrix) -> Vec<f64>,
    metric: &PfiMetric,
    n_repeats: usize,
    seed: u64,
) -> Result<VeesaPipelineResult, FdarError> {
    let model = jfpca_fit(data, argvals, ncomp, balance_c, lambda, max_iter)?;
    let training_scores = model.score_training()?;
    let pfi = elastic_pfi(&training_scores.scores, y, predict, metric, n_repeats, seed)?;
    Ok(VeesaPipelineResult { model, training_scores, pfi })
}
```

### Pattern 4: Re-export (VEE-05)

Insertion points [VERIFIED: `fdars-core/src/lib.rs:508-509`]:
```rust
// In lib.rs — follow the jfpca_model block at line 508-509:
pub mod elastic_pfi;
pub use elastic_pfi::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric,
                      PrincipalDirections, VeesaPipelineResult};
```

Insertion point [VERIFIED: `fdars-core/src/prelude.rs:75`]:
```rust
// In prelude.rs — after the jfPCA line at 75:
pub use crate::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric,
                PrincipalDirections, VeesaPipelineResult};
```

### Anti-Patterns to Avoid

- **Re-implementing inversion:** Do not write a custom SRSF→function integrator. Use `srsf_inverse` from `alignment/srsf.rs:65` verbatim.
- **Modifying existing FPC PFI functions:** `fpc_permutation_importance` and `fpc_permutation_importance_logistic` in `importance.rs` are untouched. Phase 73 adds a SIBLING, not a replacement.
- **Per-component RNG reseed:** Do NOT use `StdRng::seed_from_u64(seed + k as u64)` for the component loop in `elastic_pfi`. The existing FPC functions use a single advancing RNG and the migration note at lines 126-129, 221-225, 543-547, 658-662 of `importance.rs` marks the per-component pattern as behavior-changing and deferred.
- **Putting reconstruction in `elastic_explain.rs`:** `elastic_explain.rs` is the elastic-PCR attribution module; reconstruction is a jfPCA concept. Keep them separate.
- **`PfiMetric::Custom` carrying a non-`Send+Sync` closure:** Any trait object stored in the enum must be `Send + Sync` to be compatible with potential parallel use in future milestones.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | File:Line |
|---------|-------------|-------------|-----------|
| SRSF→function inversion | Custom integrator | `srsf_inverse` | `alignment/srsf.rs:65` |
| Tangent-space exponentiation | Custom sphere math | `exp_map_sphere` | `warping.rs:112` |
| ψ → warping function | Custom cumtrapz | `psi_to_gam` | `warping.rs:73` |
| PC-score column shuffle | Custom index permutation | `shuffle_global` | `explain/helpers/permutation.rs:9` |
| Score matrix clone | Manual copy loop | `clone_scores_matrix` | `explain/helpers/projection.rs:40` |
| jfPCA fit | Inline Karcher + SVD | `jfpca_fit` + `JfpcaModel` | `jfpca_model.rs:175` |
| Exact training scores | re-running transform | `model.score_training()` | `jfpca_model.rs:336` |

**Key insight:** The VEESA explainability layer is entirely a composition problem. Every algorithm primitive already exists; Phase 73 only wires them together via the new model-agnostic closure interface.

---

## VEE-03: PFI Implementation Notes

### Known-Signal Test Design

The test must deterministically show that the informative PC ranks above noise PCs. Design:
1. Build a spanning multi-frequency fixture (follow `jfpca_model.rs:416-433` pattern — 4 harmonics, distinct per-curve amplitudes).
2. Fit `jfpca_fit` with `ncomp = 3` (or more).
3. Define a linear response tied to PC 0 only: `y[i] = scores[(i, 0)] + tiny_noise`.
4. Supply prediction closure: `|s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() }` (or a pre-trained linear model).
5. Run `elastic_pfi(..., n_repeats=20, seed=42)`.
6. Assert: `result.importance[0] > result.importance[1]` and `result.importance[0] > result.importance[2]`.

### Seed Determinism Test

Run `elastic_pfi` twice with the same seed. Assert `result1.importance == result2.importance` (exact bit equality — both deterministic from same seed).

### PfiMetric::Custom Escape Hatch

For regression metrics beyond MSE/MAE, the closure signature is:
```rust
Custom(Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>)
// Args: (y_true: &[f64], y_pred: &[f64]) -> metric_value: f64
// Convention: higher is BETTER (importance = baseline - permuted_mean)
```

---

## VEE-04: Principal-Direction Reconstruction Notes

### Exact c=0 Gate

The gate `c = 0` reproduces `model.karcher_mean` is provable from the inversion math:
- `q_perturbed = model.mean_q[0..m]` (the mean SRSF).
- `f0 = model.mean_q[m].signum() * model.mean_q[m].powi(2)` (the augmented mean dimension decoded to function initial value).
- `srsf_inverse(mean_q_1d, argvals, f0)` integrates `q(s)|q(s)|` from `f0` — this reconstructs the function corresponding to the mean SRSF, which IS `karcher_mean`.

**Implementation check:** After computing `amplitude_curves` at `c = 0`, assert element-wise: `|amplitude_curve[j] - model.karcher_mean[j]| < 1e-10` for all `j`.

### Shape of PrincipalDirections Output

For `n_c = len(c_values)` and `m = len(argvals)`:
- `amplitude_curves`: `FdMatrix` of shape `(n_c, m)` — row `i` is the amplitude curve at `c_values[i]`.
- `phase_curves`: `FdMatrix` of shape `(n_c, m)` — row `i` is the warping function at `c_values[i]`, on the same `argvals` domain.

### Eigenvalue Source

`sigma_j = model.eigenvalues[j].sqrt()` [VERIFIED: `fdars-core/src/jfpca_model.rs:112`, field `pub eigenvalues: Vec<f64>`] where `eigenvalues` carries variance-explained values (σⱼ² form from `svd_scores_and_eigenvalues` at `elastic_fpca.rs:789-792`: `sv^2 / (n-1)`). Therefore `sigma_j = eigenvalues[j].sqrt()`.

---

## VEE-05: Pipeline & Integration Notes

### Doctest Pattern

The end-to-end doctest should follow the `jfpca_model.rs` module-level doctest pattern (lines 29-56) and be placed in `elastic_pfi.rs` or the `veesa_pipeline` function's doc comment:

```rust
//! ```
//! use fdars_core::{jfpca_fit, elastic_pfi, PfiMetric};
//! use fdars_core::matrix::FdMatrix;
//! use std::f64::consts::PI;
//!
//! // Minimal spanning fixture
//! let n = 8; let m = 12;
//! let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();
//! let mut data = FdMatrix::zeros(n, m);
//! for i in 0..n {
//!     for j in 0..m {
//!         let t = argvals[j];
//!         data[(i, j)] = (1.0 + 0.3 * i as f64) * (2.0 * PI * t).sin()
//!                      + (0.5 - 0.05 * i as f64) * (4.0 * PI * t).cos();
//!     }
//! }
//! let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
//!
//! let model = jfpca_fit(&data, &argvals, 3, None, 0.0, 10)?;
//! let tr = model.score_training()?;
//! // Model-agnostic: any closure over scores
//! let pfi = elastic_pfi(
//!     &tr.scores, &y,
//!     |s: &FdMatrix| -> Vec<f64> { (0..n).map(|i| s[(i, 0)]).collect() },
//!     &PfiMetric::Mse,
//!     5, 42,
//! )?;
//! assert_eq!(pfi.importance.len(), model.ncomp);
//! # Ok::<(), fdars_core::FdarError>(())
//! ```
```

### lib.rs Module Declaration and Re-export

Current last elastic module declaration [VERIFIED: `fdars-core/src/lib.rs:141`]:
```rust
pub mod jfpca_model;
```

Add immediately after (or in the elastic analysis block):
```rust
pub mod elastic_pfi;
```

Current jfPCA re-export block [VERIFIED: `fdars-core/src/lib.rs:508-509`]:
```rust
// Re-export jfPCA fit/transform seam
pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};
```

Add new re-export after this block:
```rust
// Re-export VEESA explainability (Phase 73)
pub use elastic_pfi::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric,
                      PrincipalDirections, VeesaPipelineResult};
```

### prelude.rs Re-export

Current jfPCA prelude line [VERIFIED: `fdars-core/src/prelude.rs:75`]:
```rust
pub use crate::{jfpca_fit, JfpcaModel, JfpcaTransform};
```

Add immediately after:
```rust
pub use crate::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric,
                PrincipalDirections, VeesaPipelineResult};
```

---

## Common Pitfalls

### Pitfall 1: Wrong sigma_j Formula

**What goes wrong:** Using `model.eigenvalues[j]` directly as σⱼ (instead of its square root) produces perturbations ~sqrt(eigenvalue) times too large.
**Why it happens:** The eigenvalues stored are variance values (σ²), not standard deviations.
**How to avoid:** Always use `model.eigenvalues[j].sqrt()` as σⱼ. The gate `c=0` reproduces the mean regardless of this formula, so the gate does not catch this bug — add a separate test that checks reasonable perturbation magnitude at `c=1`.
**Warning signs:** Principal-direction curves look implausibly large or compressed at `c=±1`.

### Pitfall 2: Mis-indexing vert_component Augmented Dimension

**What goes wrong:** Using `vert_component[(j, 0..m)]` but forgetting the extra `m`-th column for `f0` recovery.
**Why it happens:** `vert_component` has shape `(ncomp, m+1)` [VERIFIED: `jfpca_model.rs:93`, shape is `ncomp × (m+1)`]. The m-th column encodes the augmented SRSF dimension.
**How to avoid:** Explicitly use `model.vert_component[(j, m)]` for the augmented-dimension perturbation that feeds the `f0` calculation.
**Warning signs:** `c=0` gate passes but `c≠0` amplitude curves have wrong initial values (drift accumulates from wrong `f0`).

### Pitfall 3: Phase Reconstruction Without Time-Grid Normalization

**What goes wrong:** Passing `&model.argvals` (possibly non-[0,1]) to `psi_to_gam` or `exp_map_sphere` instead of the normalized `time = [0/(m-1), 1/(m-1), ..., 1]`.
**Why it happens:** The warping math operates on the [0,1] normalized time grid; `psi_to_gam` returns a [0,1] result that must be scaled back.
**How to avoid:** Follow the exact pattern from `elastic_fpca.rs:217` and `jfpca_model.rs:292`:
  ```rust
  let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
  ```
  Then scale back: `phase_curve[l] = argvals[0] + gam[l] * (argvals[m-1] - argvals[0])`.
**Warning signs:** Phase curves run outside the argvals domain.

### Pitfall 4: Modifying Existing FPC PFI RNG Behavior

**What goes wrong:** Changing `importance.rs` to use a per-component reseed breaks the behavior guarantee noted in the deferred-migration comments.
**Why it happens:** The new generic path and the old FPC path differ in their RNG advancement pattern.
**How to avoid:** Do NOT touch `importance.rs` or `explain/helpers/permutation.rs` function bodies. Only add new callsites from `elastic_pfi.rs`.

### Pitfall 5: Clippy int_plus_one in Test Assertions

**What goes wrong:** `assert!(model.ncomp <= n - 1)` triggers `clippy::int_plus_one`.
**Why it happens:** Known clippy lint (encountered in Phase 72 per 72-01-SUMMARY.md).
**How to avoid:** Write `assert!(model.ncomp < n)` (equivalent, lint-clean).

### Pitfall 6: serde Feature Build Breakage

**What goes wrong:** Adding `#[derive(serde::Serialize, serde::Deserialize)]` to a struct that embeds a non-serde type.
**Why it happens:** Pre-existing serde break from Phase 60 (`ShapeletTransformClassifier` / `ClassifFit`). New structs must NOT embed non-serde types inside a serde-gated derive.
**How to avoid:** New result structs in this phase (`ElasticPfiResult`, `PrincipalDirections`, `VeesaPipelineResult`) contain only `FdMatrix`, `Vec<f64>`, `f64`, `usize`, `JfpcaModel`, `JfpcaTransform` — all of which already have conditional serde derives [VERIFIED: `jfpca_model.rs:74` has `#[cfg_attr(feature = "serde", ...)]`]. Use the same `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` pattern. The `PfiMetric::Custom` variant cannot be serde-derived (it contains a `Box<dyn Fn>`); gate the serde derive on a variant set that excludes `Custom`, or skip serde for `PfiMetric` entirely and document the limitation.

---

## Runtime State Inventory

This is a greenfield addition (new modules, additive re-exports). No runtime state migration required.

- **Stored data:** None — no new databases, collections, or persistent keys.
- **Live service config:** None — library crate, no external service configuration.
- **OS-registered state:** None — no task scheduler entries, daemon registrations.
- **Secrets/env vars:** None — no environment variable changes.
- **Build artifacts:** None — additive module additions do not create stale egg-info or binaries. If `target/` disk pressure is an issue before running tests, `rm -rf target/debug/{incremental,examples}` frees ~100GB per MEMORY.md.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo` / Rust toolchain | Build and test | Yes | 1.97.0 (dev) | — |
| `rand` crate | `StdRng::seed_from_u64` in `elastic_pfi.rs` | Yes (already in Cargo.lock) | 0.8 | — |
| `explain/helpers/permutation.rs` | `shuffle_global` reuse | Yes (same crate) | — | self-contained fallback |
| `/tmp` space | `cargo test --doc` doctest linking | Check before full suite | — | `cargo test --doc -p fdars-core` runs fewer examples |

---

## Validation Architecture

`workflow.nyquist_validation = true` — validation section required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(test)] mod tests` (inline) |
| Config file | None — no `test.toml`; uses `Cargo.toml` features |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel elastic_pfi 2>&1 \| tail -5` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1 \| tail -10` |
| Doc test command | `cargo test --doc -p fdars-core --features linalg,parallel 2>&1 \| tail -5` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VEE-03a | PFI seed-determinism: two runs with same seed produce identical importance | unit | `cargo test -p fdars-core --features linalg,parallel pfi_seed_determinism` | No — Wave 0 |
| VEE-03b | Informative PC ranks above noise PCs on known-signal design | unit | `cargo test -p fdars-core --features linalg,parallel pfi_known_signal_ranking` | No — Wave 0 |
| VEE-03c | `n_repeats = 0` returns `InvalidParameter` | unit | `cargo test -p fdars-core --features linalg,parallel pfi_rejects_zero_repeats` | No — Wave 0 |
| VEE-04a | `c = 0` amplitude curve reproduces `model.karcher_mean` within 1e-10 | unit (known-answer) | `cargo test -p fdars-core --features linalg,parallel principal_directions_c0_mean` | No — Wave 0 |
| VEE-04b | Output shapes: `amplitude_curves` and `phase_curves` are `(n_c, m)` | unit (structural) | `cargo test -p fdars-core --features linalg,parallel principal_directions_shapes` | No — Wave 0 |
| VEE-04c | `pc_index >= ncomp` returns `InvalidParameter` | unit | `cargo test -p fdars-core --features linalg,parallel principal_directions_rejects_bad_pc` | No — Wave 0 |
| VEE-05a | End-to-end module doctest: fit → score_training → elastic_pfi runs under `cargo test --doc` | doctest | `cargo test --doc -p fdars-core --features linalg,parallel` | No — Wave 0 |
| VEE-05b | Full suite non-regression: 2801 tests still pass after additive changes | integration | `cargo test -p fdars-core --features linalg,parallel` | Existing (count grows) |
| VEE-05c | clippy `--all-targets --features linalg,parallel -- -D warnings` clean | gate | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | Existing gate |
| VEE-05d | `cargo fmt --check` clean | gate | `cargo fmt -p fdars-core -- --check` | Existing gate |

### Known-Answer Tests Detail

**VEE-03b (known-signal ranking) fixture:**
```rust
// Follow spanning_fixture() from jfpca_model.rs:416 — 4 harmonics, n >= 8, m >= 12
// Response: y[i] = scores_training[(i, 0)] * 2.0  (pure PC-0 signal)
// Predict closure: |s| -> (0..n).map(|i| s[(i, 0)] * 2.0).collect()
// Assert: pfi.importance[0] > pfi.importance[1] && pfi.importance[0] > pfi.importance[2]
```

**VEE-04a (c=0 reproduces mean) test:**
```rust
let c_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
let pd = model.principal_directions(0, &c_values)?;
// Row 2 (c=0.0) of amplitude_curves must match model.karcher_mean
for j in 0..m {
    assert!((pd.amplitude_curves[(2, j)] - model.karcher_mean[j]).abs() < 1e-10,
            "c=0 amplitude curve deviates from karcher_mean at j={j}");
}
```

### Wave 0 Gaps

- [ ] `fdars-core/src/elastic_pfi.rs` — new module with all Phase 73 public items and inline tests
- [ ] `fdars-core/src/jfpca_model.rs` — add `principal_directions` method + tests in existing `mod tests`
- [ ] Module doctest for `veesa_pipeline` or `elastic_pfi` module-level
- [ ] No new test files needed — inline `#[cfg(test)] mod tests` per crate convention

*(No framework install required — existing `cargo test` infrastructure covers all requirements)*

---

## Security Domain

`security_enforcement = true` (ASVS Level 1).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Library crate — no user auth |
| V3 Session Management | No | Library crate — no sessions |
| V4 Access Control | No | Library crate — no ACLs |
| V5 Input Validation | Yes | `FdarError::InvalidDimension` + `InvalidParameter` at function entry |
| V6 Cryptography | No | No encryption; `StdRng` is not a cryptographic primitive and is used only for permutation |

### Input Validation Requirements (V5)

All new public functions must validate at entry and return `Result<T, FdarError>`:

| Function | Checks Required |
|----------|----------------|
| `elastic_pfi` | `scores.nrows() == y.len()`, `n_repeats >= 1`, `scores.nrows() >= 1`, `scores.ncols() >= 1` |
| `JfpcaModel::principal_directions` | `pc_index < self.ncomp`, `c_values` non-empty |
| `veesa_pipeline` | Delegates to `jfpca_fit` + `elastic_pfi` — their checks cover the inputs |

No cryptographic requirements. `StdRng::seed_from_u64(seed)` is a determinism tool, not a security primitive.

---

## Code Examples

### Full elastic_pfi Skeleton

```rust
// fdars-core/src/elastic_pfi.rs
// Source: design from explain/importance.rs advancing-RNG pattern (lines 130-146)
//         and explain/helpers/permutation.rs:shuffle_global + projection.rs:clone_scores_matrix

use crate::explain::helpers::{clone_scores_matrix, shuffle_global};
use crate::matrix::FdMatrix;
use crate::FdarError;
use rand::prelude::*;

pub enum PfiMetric {
    Mse,
    Mae,
    Accuracy,
    Custom(Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>),
}

fn compute_metric(y: &[f64], pred: &[f64], metric: &PfiMetric) -> f64 {
    let n = y.len();
    match metric {
        PfiMetric::Mse => y.iter().zip(pred).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>() / n as f64,
        PfiMetric::Mae => y.iter().zip(pred).map(|(&a, &b)| (a - b).abs()).sum::<f64>() / n as f64,
        PfiMetric::Accuracy => y.iter().zip(pred).filter(|(&a, &b)| (a - b.round()).abs() < 1e-10).count() as f64 / n as f64,
        PfiMetric::Custom(f) => f(y, pred),
    }
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pfi(
    scores: &FdMatrix,
    y: &[f64],
    predict: impl Fn(&FdMatrix) -> Vec<f64>,
    metric: &PfiMetric,
    n_repeats: usize,
    seed: u64,
) -> Result<ElasticPfiResult, FdarError> {
    let (n, ncomp) = scores.shape();
    // --- validation ---
    if n != y.len() { return Err(FdarError::InvalidDimension { ... }); }
    if n_repeats == 0 { return Err(FdarError::InvalidParameter { ... }); }

    let baseline_pred = predict(scores);
    let baseline_metric = compute_metric(y, &baseline_pred, metric);

    // Single advancing RNG — mirrors existing FPC PFI pattern (importance.rs:130)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_metric = 0.0;
        for _ in 0..n_repeats {
            let mut perm = clone_scores_matrix(scores, n, ncomp);
            shuffle_global(&mut perm, scores, k, n, &mut rng);
            let pred = predict(&perm);
            sum_metric += compute_metric(y, &pred, metric);
        }
        let mean_perm = sum_metric / n_repeats as f64;
        permuted_metric[k] = mean_perm;
        importance[k] = baseline_metric - mean_perm;
    }
    Ok(ElasticPfiResult { importance, baseline_metric, permuted_metric })
}
```

### principal_directions Skeleton

```rust
// In jfpca_model.rs — impl JfpcaModel
// Sources: srsf_inverse (alignment/srsf.rs:65), exp_map_sphere (warping.rs:112), psi_to_gam (warping.rs:73)

use crate::alignment::srsf_inverse;
use crate::warping::{exp_map_sphere, psi_to_gam};

impl JfpcaModel {
    pub fn principal_directions(
        &self,
        pc_index: usize,
        c_values: &[f64],
    ) -> Result<PrincipalDirections, FdarError> {
        if pc_index >= self.ncomp { return Err(FdarError::InvalidParameter { ... }); }
        let m = self.argvals.len();
        let n_c = c_values.len();
        let sigma_j = self.eigenvalues[pc_index].sqrt();
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let domain = self.argvals[m - 1] - self.argvals[0];

        let mut amplitude_curves = FdMatrix::zeros(n_c, m);
        let mut phase_curves = FdMatrix::zeros(n_c, m);

        for (ci, &c) in c_values.iter().enumerate() {
            // Amplitude part
            let q_perturbed: Vec<f64> = (0..m)
                .map(|l| self.mean_q[l] + c * sigma_j * self.vert_component[(pc_index, l)])
                .collect();
            let aug_val = self.mean_q[m] + c * sigma_j * self.vert_component[(pc_index, m)];
            let f0 = aug_val.signum() * aug_val * aug_val;
            let amp = srsf_inverse(&q_perturbed, &self.argvals, f0);
            for j in 0..m { amplitude_curves[(ci, j)] = amp[j]; }

            // Phase part
            let v_perturbed: Vec<f64> = (0..m)
                .map(|l| c * sigma_j * self.horiz_component[(pc_index, l)])
                .collect();
            let psi_p = exp_map_sphere(&self.mean_psi, &v_perturbed, &time);
            let gam = psi_to_gam(&psi_p, &time);
            for j in 0..m {
                phase_curves[(ci, j)] = self.argvals[0] + gam[j] * domain;
            }
        }
        Ok(PrincipalDirections { pc_index, c_values: c_values.to_vec(), amplitude_curves, phase_curves })
    }
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| FPC-specific PFI (tied to `FregreLmResult`) | Model-agnostic PFI via `Fn` closure (any predictor) | VEESA's intended generality; matches `sandialabs/veesa` R design |
| No principal-direction visualization | `JfpcaModel::principal_directions` with amplitude/phase split | Enables the μ ± c·σⱼ visualization from the VEESA paper |
| No pipeline wrapper | `veesa_pipeline` ties fit→scores→PFI | Convenience for end-users matching `prep_training_data` + `compute_pfi` in R |

**VEESA R package correspondence:**
- `prep_training_data` / `jfpca_fit` → `jfpca_fit` (Phase 72, done).
- `prep_testing_data` → `JfpcaModel::transform` (Phase 72, done).
- `compute_pfi` → `elastic_pfi` (Phase 73, VEE-03).
- Principal-direction visualization → `JfpcaModel::principal_directions` (Phase 73, VEE-04).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `srsf_inverse` at `c=0` exactly reproduces `karcher_mean` (within 1e-10) | VEE-04 Notes | c=0 gate fails; would require alternate mean reconstruction path |
| A2 | `exp_map_sphere` with a zero tangent vector returns `mean_psi` unchanged | VEE-04 Phase Notes | c=0 phase gate fails; inspect `warping.rs:112-124` — currently returns `psi.to_vec()` when `v_norm < 1e-10` which confirms this holds |

Note on A2: Reading `warping.rs:113-116` [VERIFIED: `fdars-core/src/warping.rs:113-116`]:
```rust
let v_norm = l2_norm_l2(v, time);
if v_norm < 1e-10 {
    psi.to_vec()
}
```
So at `c = 0`, `v_perturbed` is the zero vector, `v_norm = 0 < 1e-10`, and `exp_map_sphere` returns `mean_psi` directly. Then `psi_to_gam(mean_psi, time)` returns the identity warp. A1 is the only remaining assumption and is testable by the known-answer gate.

**If this table has only low-risk items:** Both assumptions are testable by the required gates (VEE-04a tests A1; A2 is already confirmed by reading the source).

---

## Open Questions

1. **`PfiMetric` for multi-class classification**
   - What we know: `Accuracy` works for any classification (binary or multi-class) if predictions are class labels.
   - What's unclear: Whether to add a `LogLoss` variant or leave multi-class probability metrics to the `Custom` escape hatch.
   - Recommendation: Start with `{Mse, Mae, Accuracy, Custom}`. Log-loss and other probabilistic metrics are well-served by `Custom`. Keep the enum small.

2. **`VeesaPipelineResult` serde-ability**
   - What we know: `JfpcaModel` and `JfpcaTransform` have conditional serde. `ElasticPfiResult` can too.
   - What's unclear: `VeesaPipelineResult` embeds `JfpcaModel`; the `Custom` variant of `PfiMetric` cannot be serialized.
   - Recommendation: Give `VeesaPipelineResult` the `#[cfg_attr(feature = "serde", ...)]` derive; the pipeline function does not need to store `PfiMetric`. Leave `PfiMetric` without serde (document the omission).

3. **Home module for `principal_directions`**
   - What we know: `jfpca_model.rs` is currently 646 lines; the reconstruction method adds ~60 lines + ~50 lines of tests.
   - What's unclear: Will the file exceed the ~700-line guideline triggering a split?
   - Recommendation: Add as `impl JfpcaModel` in `jfpca_model.rs`. If the file grows uncomfortably, create `veesa.rs` and re-export via `jfpca_model.rs`. This is Claude's discretion per the locked decisions.

---

## Sources

### Primary (HIGH confidence)

All findings below are [VERIFIED] from direct file reads in this session.

- [VERIFIED: `fdars-core/src/jfpca_model.rs:1-646`] — Complete Phase 72 seam: all field names, types, shapes, method signatures, scoring formula, test fixture pattern.
- [VERIFIED: `fdars-core/src/explain/helpers/permutation.rs:1-153`] — `shuffle_global` (line 9), `permute_component` (line 77), `shuffle_within_bins` (line 24): all `pub(crate)`, all take only `FdMatrix` + primitives — liftable without modification.
- [VERIFIED: `fdars-core/src/explain/helpers/projection.rs:1-134`] — `clone_scores_matrix` (line 40): `pub(crate)`, takes `&FdMatrix` — liftable.
- [VERIFIED: `fdars-core/src/explain/importance.rs:1-683`] — Existing FPC PFI functions: advancing RNG pattern confirmed at lines 130, 225, 548, 663; deferred-migration comment confirmed at lines 126-129, 221-225.
- [VERIFIED: `fdars-core/src/elastic_fpca.rs:1-999`] — `build_combined_representation` (line 914), `split_joint_eigenvectors` (line 818), `svd_scores_and_eigenvalues` (line 783-814, eigenvalue formula: `sv^2/(n-1)`), `warps_to_normalized_psi` (line 645), `shooting_vectors_from_psis` (line 705).
- [VERIFIED: `fdars-core/src/alignment/srsf.rs:65-76`] — `srsf_inverse` signature and formula: `f(t) = f0 + ∫q(s)|q(s)|ds`.
- [VERIFIED: `fdars-core/src/warping.rs:73-124`] — `psi_to_gam` (line 73), `inv_exp_map_sphere` (line 95), `exp_map_sphere` (line 112): all `pub`, signatures documented; zero-vector early-return confirmed at line 113-116.
- [VERIFIED: `fdars-core/src/lib.rs:141,508-509`] — `pub mod jfpca_model;` at line 141; `pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};` at lines 508-509.
- [VERIFIED: `fdars-core/src/prelude.rs:75`] — `pub use crate::{jfpca_fit, JfpcaModel, JfpcaTransform};` at line 75.
- [VERIFIED: `fdars-core/src/elastic_explain.rs:1-80`] — `elastic_pcr_attribution` / `ElasticAttributionResult` exist as the elastic-PCR attribution sibling; not to be modified.
- [VERIFIED: `fdars-core/src/alignment/mod.rs:139-146`] — `AlignmentSetResult { gammas, aligned_data, distances }` — confirms `align_to_target` return type fields.
- [VERIFIED: `.planning/phases/72-jfpca-fit-transform-seam/72-01-SUMMARY.md`] — Phase 72 gate results; score formula; round-trip note about Karcher centering.

### Secondary (MEDIUM confidence)

- [CITED: VEESA paper (Goode, Tucker & Ries, JDS)] — The VEESA pipeline design: `prep_training_data`, `prep_testing_data`, `compute_pfi`; principal-direction visualization at μ ± c·σⱼ.
- [CITED: `sandialabs/veesa` R package] — R reference for `compute_pfi` semantics (model-agnostic, metric-based).

---

## Metadata

**Confidence breakdown:**
- Phase 72 seam (signatures, fields): HIGH — file read verbatim.
- Permutation helper liftability: HIGH — file read verbatim, confirmed `pub(crate)` with no domain types.
- SRSF inversion path: HIGH — file read verbatim, formula confirmed.
- Phase reconstruction path: HIGH — `exp_map_sphere`, `psi_to_gam` read verbatim; zero-vector return confirmed.
- Sigma_j formula: HIGH — eigenvalue formula read from `svd_scores_and_eigenvalues` at elastic_fpca.rs:789-792.
- Re-export insertion points: HIGH — lib.rs:141,508-509 and prelude.rs:75 read verbatim.

**Research date:** 2026-09-05
**Valid until:** This research is grounded entirely in the current codebase; valid until any of the cited files change. No external packages involved — no staleness window.
