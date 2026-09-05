# Phase 74: Elastic Conformal Anomaly Detection — Research

**Researched:** 2026-09-05
**Domain:** Inductive conformal anomaly detection for functional data using elastic distances
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Elastic nonconformity scores (ECA-01)
- **Extend the existing `#[non_exhaustive] NonConformityScore`** in place with three elastic variants: `AmplitudeElastic`, `PhaseElastic`, `CombinedElastic`.
- `conformal_prediction_band` returns `FdarError::InvalidParameter` for these template-based variants (it has no reference template) — the band path stays otherwise unchanged.
- Score function: `elastic_nonconformity(curve, template, argvals, lambda, variant) -> f64` — non-negative, and **zero for a curve identical to the reference** (a make-or-break gate).
- Expose a `lambda` (warp penalty) parameter, default 0.0.
- Distance reuse: call `amplitude_distance` / `phase_distance_pair` / `elastic_distance` (`alignment/pairwise.rs`) verbatim; do NOT re-derive.

#### Inductive conformal anomaly detector (ECA-02)
- Reference template: **compute the calibration Karcher mean by default**; allow the caller to supply an explicit template.
- p-value: standard inductive conformal formula `(1 + #{calib_score >= test_score}) / (n_calib + 1)`.
- Flag rule: flag a curve when `p <= alpha` (equivalently, its score exceeds the (1-alpha) quantile of the calibration scores → the calibrated threshold).
- API shape: `elastic_conformal_anomaly(calibration, test, config)` taking a `ConformalAnomalyConfig` (variant / alpha / lambda / optional template) — builder-style config per crate convention.
- Behavioral gates: on exchangeable clean data the flag rate ≈ alpha (marginal validity); injected **magnitude** AND **shape** outliers are flagged.

#### Result type + integration (ECA-03)
- `ConformalAnomalyResult { p_values, scores, flags, threshold }` (per-curve p-values + nonconformity scores + boolean flags + the calibrated threshold).
- Module placement: new additive `tolerance/conformal_anomaly.rs`; leave `conformal.rs` band path unchanged.
- Full **crate-root (`lib.rs`) + prelude re-exports** for all new public items.
- A running **module doctest** (calibrate → flag) under `cargo test --doc`.

### Claude's Discretion
- Exact `ConformalAnomalyConfig` field layout + defaults, internal helper factoring, the home of the elastic-nonconformity dispatch (in `conformal_anomaly.rs`), and doctest fixture construction are at Claude's discretion within the above and crate conventions (column-major `FdMatrix`, `Result<T, FdarError>`, `#[non_exhaustive]`, `#[must_use]`, conditional serde, `Debug/Clone/PartialEq`).

### Deferred Ideas (OUT OF SCOPE)
- Full conditional / Mondrian conformal anomaly detection (class-conditional validity) — ECA-F1 → future milestone; v1 covers the inductive marginal case only.
- R/WASM binding exposure of the anomaly surface → future milestone (issue fdars-j75).
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ECA-01 | Elastic nonconformity scores — extend `NonConformityScore` with `AmplitudeElastic`, `PhaseElastic`, `CombinedElastic`; each non-negative and zero for a curve identical to the reference | Confirmed: all three distance functions verified (pairwise.rs:384–392); zero-for-identical validated via existing tests; `NonConformityScore` location confirmed (types.rs:26–33) |
| ECA-02 | Inductive conformal anomaly detector — calibrate scores on reference/calibration set, return per-curve p-values + anomaly flags at alpha; marginal validity + injected outlier gates | Confirmed: inductive p-value formula pinned; `karcher_mean` signature verified; `conformal_quantile` pattern extracted from `conformal/mod.rs:161–172`; `quantile_sorted` in `crate::helpers` reusable |
| ECA-03 | `ConformalAnomalyResult` type + crate-root/prelude re-exports + running module doctest | Confirmed: insertion points identified in `lib.rs:346–353` (tolerance block) and `prelude.rs:89–92` (tolerance block); `tolerance/mod.rs:17–42` wiring pattern documented |
</phase_requirements>

---

## Summary

Phase 74 adds inductive conformal anomaly detection for functional data by wiring three existing elastic-distance functions (`amplitude_distance`, `phase_distance_pair`, `elastic_distance`) into a new nonconformity-scoring layer, then building a calibration-and-flag pipeline on top. All algorithms already exist in the codebase; the work is purely additive: extend one enum, add one module, wire re-exports, and write the public API + doctest.

The critical shape of the implementation is: (1) extend `NonConformityScore` in `tolerance/types.rs` with three new elastic variants, (2) add arms to the `match score_type` in `tolerance/conformal.rs:nonconformity_score` returning `FdarError::InvalidParameter` (since the band function has no template), (3) implement `elastic_nonconformity` and `elastic_conformal_anomaly` in a new sibling file `tolerance/conformal_anomaly.rs`, and (4) wire `tolerance/mod.rs`, `lib.rs`, and `prelude.rs`.

**Primary recommendation:** Implement `conformal_anomaly.rs` as a self-contained sibling of `conformal.rs`. Use `crate::helpers::sort_nan_safe` + `crate::helpers::quantile_sorted` for the calibrated threshold (these are public and imported cleanly). Do NOT reuse `conformal/mod.rs::conformal_quantile` (it is `pub(super)` and inaccessible). The `karcher_mean` function in `alignment/karcher.rs` (re-exported via `alignment/mod.rs`) yields a `KarcherMeanResult` whose `mean` field (`Vec<f64>`) is the template curve.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Nonconformity scoring (elastic distance to template) | `tolerance/conformal_anomaly.rs` | `alignment/pairwise.rs` (distance calls) | Scoring lives in tolerance module; distance functions are reused from alignment |
| Calibration Karcher mean template | `alignment/karcher.rs` (existing) | `tolerance/conformal_anomaly.rs` (caller) | Template computation is alignment's job; anomaly module calls it |
| Inductive p-value + threshold computation | `tolerance/conformal_anomaly.rs` | `crate::helpers` (sort/quantile) | Math is local; numeric helpers from the shared helpers module |
| Enum extension (`NonConformityScore`) | `tolerance/types.rs` | `tolerance/conformal.rs` (new match arms) | Enum is defined in types.rs; all matchers must be updated |
| Public API surface | `tolerance/mod.rs`, `lib.rs`, `prelude.rs` | — | Standard crate re-export pattern |

---

## Standard Stack

### Core (all existing — no new dependencies)

| Component | Location | Purpose |
|-----------|----------|---------|
| `NonConformityScore` | `tolerance/types.rs:26–33` | Enum to extend with elastic variants |
| `amplitude_distance` | `alignment/pairwise.rs:384–386` | Amplitude nonconformity score |
| `phase_distance_pair` | `alignment/pairwise.rs:388–392` | Phase nonconformity score |
| `elastic_distance` | `alignment/pairwise.rs:103–105` | Combined nonconformity score |
| `karcher_mean` | `alignment/karcher.rs:293–301` | Calibration template computation |
| `sort_nan_safe` | `helpers.rs:10` | Sort scores for quantile computation |
| `quantile_sorted` | `helpers.rs:302–317` | Calibrated threshold via empirical quantile |
| `FdarError` | `error.rs` | Error return type for invalid operations |
| `FdMatrix` | `matrix.rs` | Column-major functional data matrix |

**No `Cargo.toml` change required.** [VERIFIED: fdars-core/Cargo.toml — no new deps]

---

## Detailed Source Analysis

### 1. `NonConformityScore` — Location, Definition, Exhaustive Matching

**File:** `fdars-core/src/tolerance/types.rs:26–33` [VERIFIED: tolerance/types.rs:26–33]

Verbatim definition:
```rust
/// Non-conformity score for conformal prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonConformityScore {
    /// Supremum norm: max_t |y(t) - center(t)|
    SupNorm,
    /// L2 norm: sqrt(sum (y(t) - center(t))^2)
    L2,
}
```

The enum is `#[non_exhaustive]` — adding new variants is a non-breaking change for downstream consumers. The executor must add the three new variants here:

```rust
/// Amplitude elastic distance to a reference template (Fisher-Rao metric).
AmplitudeElastic,
/// Phase elastic distance to a reference template (geodesic distance of optimal warp from identity).
PhaseElastic,
/// Combined elastic distance to a reference template.
CombinedElastic,
```

**All existing match sites must be updated:**

There is **one** match on `NonConformityScore` in the existing codebase:

- `tolerance/conformal.rs:17–26` — the private `nonconformity_score` function [VERIFIED: tolerance/conformal.rs:17–26]

Verbatim match block:
```rust
match score_type {
    NonConformityScore::SupNorm => (0..m)
        .map(|j| (data[(i, j)] - center[j]).abs())
        .fold(0.0_f64, f64::max),
    NonConformityScore::L2 => {
        let ss: f64 = (0..m).map(|j| (data[(i, j)] - center[j]).powi(2)).sum();
        ss.sqrt()
    }
}
```

The three new elastic arms must be added to this match returning `FdarError::InvalidParameter`. However, `nonconformity_score` currently returns `f64`, not `Result<f64, FdarError>`. The plan must either:

- (A) Change `nonconformity_score` to return `Result<f64, FdarError>` and propagate up through `conformal_prediction_band` — this would break the `Option<ToleranceBand>` return of `conformal_prediction_band` slightly, but since `conformal_prediction_band` already returns `Option`, the `?` operator cannot be used directly. Instead, return early with `None` for the invalid variants.
- (B) Keep `nonconformity_score` returning `f64` and add a pre-flight check at the top of `conformal_prediction_band` that returns `None` for elastic variants before calling `nonconformity_score`.

**Recommended approach (B):** Add an early guard at the start of `conformal_prediction_band` (before any computation):
```rust
// Elastic variants require a reference template — not supported by the band function
if matches!(score_type, NonConformityScore::AmplitudeElastic
    | NonConformityScore::PhaseElastic
    | NonConformityScore::CombinedElastic)
{
    return None;
}
```

This keeps `nonconformity_score` returning `f64` and avoids refactoring the band function. The `#[non_exhaustive]` attribute means Rust will **not** require exhaustive matching in external crates, but within the crate the match in `nonconformity_score` will produce a compile error for unhandled variants — which is caught by the CI gate. Add a `_ => unreachable!()` arm as a defensive catch since the early guard above makes the elastic arms unreachable.

### 2. Elastic Distance Functions — Exact Signatures and Semantics

All three functions live in `alignment/pairwise.rs` and are re-exported via `alignment/mod.rs`. [VERIFIED: alignment/pairwise.rs:103–105, 384–392; alignment/mod.rs:76–83]

**`elastic_distance`** (`alignment/pairwise.rs:103–105`):
```rust
pub fn elastic_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair(f1, f2, argvals, lambda).distance
}
```
- Argument order: `(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64)`
- Semantics: L2 distance between SRSFs of f1 and the optimally-warped f2 (Fisher-Rao metric); lambda penalizes warp deviation from identity
- Non-negative: confirmed by `assert!(result.distance >= 0.0)` in doctest
- Near-zero for identical: confirmed by existing test `test_elastic_distance_self_near_zero` (`d < 0.1` tolerance due to SRSF numerical discretization) [VERIFIED: alignment/tests.rs:612–624]
- Used for: `CombinedElastic` variant (combined amplitude + phase)

**`amplitude_distance`** (`alignment/pairwise.rs:384–386`):
```rust
pub fn amplitude_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_distance(f1, f2, argvals, lambda)
}
```
- Identical to `elastic_distance` — it is exactly the elastic distance after alignment, which IS the amplitude distance in the elastic framework
- Argument order: same as `elastic_distance`
- Used for: `AmplitudeElastic` variant

**`phase_distance_pair`** (`alignment/pairwise.rs:388–392`):
```rust
pub fn phase_distance_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    crate::warping::phase_distance(&alignment.gamma, argvals)
}
```
- Argument order: same as the others — `(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64)`
- Semantics: geodesic distance of the optimal warping from the identity warp on the Hilbert sphere
- Non-negative: confirmed by existing test `test_phase_distance_nonneg` [VERIFIED: alignment/tests.rs:1755–1767]
- `crate::warping::phase_distance` docstring: "Returns 0 for the identity warp" [VERIFIED: warping.rs:145]
- Used for: `PhaseElastic` variant

**Zero-for-identical caveat:** The existing test uses `d < 0.1` for `elastic_distance` self-distance, NOT exact zero. This is due to SRSF discretization on a finite grid. The ECA-01 gate "zero for a curve identical to the reference" should be implemented as `< 1e-10` (or appropriate small epsilon), not strict `== 0.0`. [VERIFIED: alignment/tests.rs:619–622]

### 3. `elastic_nonconformity` Dispatch

The function `elastic_nonconformity(curve: &[f64], template: &[f64], argvals: &[f64], lambda: f64, variant: NonConformityScore) -> Result<f64, FdarError>` should live in `tolerance/conformal_anomaly.rs`. Dispatch:

```rust
match variant {
    NonConformityScore::AmplitudeElastic =>
        Ok(crate::alignment::amplitude_distance(curve, template, argvals, lambda)),
    NonConformityScore::PhaseElastic =>
        Ok(crate::alignment::phase_distance_pair(curve, template, argvals, lambda)),
    NonConformityScore::CombinedElastic =>
        Ok(crate::alignment::elastic_distance(curve, template, argvals, lambda)),
    _ => Err(FdarError::InvalidParameter {
        parameter: "variant",
        message: "elastic_nonconformity requires an elastic NonConformityScore variant \
                  (AmplitudeElastic, PhaseElastic, or CombinedElastic)".to_string(),
    }),
}
```

### 4. `karcher_mean` — Signature, Return Type, Template Extraction

**File:** `alignment/karcher.rs:293–301` [VERIFIED: alignment/karcher.rs:270–301]

```rust
pub fn karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> KarcherMeanResult
```

Returns `KarcherMeanResult` (no `Result` wrapper — always returns a result, possibly not converged).

**`KarcherMeanResult`** fields [VERIFIED: alignment/mod.rs:148–167]:
```rust
pub struct KarcherMeanResult {
    pub mean: Vec<f64>,         // <-- this is the template curve (length m)
    pub mean_srsf: Vec<f64>,
    pub gammas: FdMatrix,
    pub aligned_data: FdMatrix,
    pub n_iter: usize,
    pub converged: bool,
    pub aligned_srsfs: Option<FdMatrix>,
}
```

**Template extraction:** `karcher.mean` is a `Vec<f64>` of length `m` (number of evaluation points). To use as template in `elastic_nonconformity`, pass `&karcher.mean` as the `template` argument.

**Default parameters for calibration:** Use `max_iter = 20`, `tol = 1e-4`, `lambda = 0.0` (matching `ElasticToleranceConfig::default()` [VERIFIED: tolerance/types.rs:97–110]).

### 5. Inductive Conformal Math — Pinned Formula

**Standard inductive (split) conformal p-value:**

Given calibration nonconformity scores `{a_1, ..., a_n_calib}` and a test score `a*`:
```
p_value = (1 + #{i : a_i >= a*}) / (n_calib + 1)
```

- `p_value` is in `(0, 1]`
- Flag when `p_value <= alpha`
- Calibrated threshold = the `(1 - alpha)` empirical quantile of the calibration scores
  - Equivalently, `threshold = quantile_sorted(&mut calib_scores_sorted, 1.0 - alpha)`
  - A test curve is flagged iff its score > threshold

**Marginal validity:** Under exchangeability, `P(flag | clean curve) ≤ alpha` (finite-sample guarantee). On test sets of clean exchangeable data, the empirical flag rate ≈ alpha. The test for this uses a large calibration set to reduce variance around alpha.

**Conformal quantile convention in this codebase:** The existing `conformal/mod.rs::conformal_quantile` uses `k = ceil((n+1) * (1-alpha))` and returns the k-th smallest score [VERIFIED: conformal/mod.rs:161–172]. The p-value formulation is equivalent: both correctly account for the `+1` correction that ensures the finite-sample guarantee. The anomaly detector should implement the p-value formula directly (clearer for the API) and derive the threshold from `quantile_sorted`.

**Implementation using `crate::helpers`:**
```rust
// Sort calibration scores (ascending)
crate::helpers::sort_nan_safe(&mut calib_scores);
// Calibrated threshold = (1-alpha) quantile
let threshold = crate::helpers::quantile_sorted(&calib_scores, 1.0 - alpha);
// For each test score a*:
let p_value = {
    let count = calib_scores.iter().filter(|&&a| a >= a_star).count();
    (1 + count) as f64 / (n_calib + 1) as f64
};
let flag = p_value <= alpha;
```

Note: `quantile_sorted` uses linear interpolation (not the exact order-statistic formula from `conformal_quantile`). For the threshold, linear interpolation is acceptable. For full rigor matching arXiv 2504.01172, use the order-statistic `ceil((n+1)*(1-alpha))`-th value, which is what `conformal_quantile` in `conformal/mod.rs` does — but that helper is `pub(super)`. Reproduce it inline:
```rust
fn calibrated_threshold(sorted_scores: &[f64], alpha: f64) -> f64 {
    let n = sorted_scores.len();
    let k = ((n + 1) as f64 * (1.0 - alpha)).ceil() as usize;
    if k > n { f64::INFINITY } else { sorted_scores[k.saturating_sub(1)] }
}
```

### 6. Module Integration Points

#### `tolerance/mod.rs` — Add submodule declaration and re-exports

Current mod.rs [VERIFIED: tolerance/mod.rs:1–43]:

```rust
mod conformal;
mod degras;
mod elastic;
// ... other submodules ...
mod types;

// ...

pub use conformal::conformal_prediction_band;
// ...
pub use types::{
    BandType, ElasticToleranceBandResult, ElasticToleranceConfig, EquivalenceBootstrap,
    EquivalenceTestResult, ExponentialFamily, MultiplierDistribution, NonConformityScore,
    PhaseToleranceBand, ToleranceBand,
};
```

**Add:**
```rust
mod conformal_anomaly;  // new sibling module

pub use conformal_anomaly::{
    elastic_conformal_anomaly, elastic_nonconformity, ConformalAnomalyConfig, ConformalAnomalyResult,
};
// NonConformityScore is already re-exported — the new variants are picked up automatically
```

#### `lib.rs` — Add to the tolerance re-export block

Current block (`lib.rs:346–353`) [VERIFIED: lib.rs:346–353]:
```rust
pub use tolerance::{
    conformal_prediction_band, elastic_tolerance_band, elastic_tolerance_band_with_config,
    equivalence_test, equivalence_test_one_sample, exponential_family_tolerance_band,
    fpca_tolerance_band, phase_tolerance_band, scb_mean_degras, BandType,
    ElasticToleranceBandResult, ElasticToleranceConfig, EquivalenceBootstrap,
    EquivalenceTestResult, ExponentialFamily, MultiplierDistribution, NonConformityScore,
    PhaseToleranceBand, ToleranceBand,
};
```

**Extend** (add to this block — keep existing items unchanged):
```rust
pub use tolerance::{
    // ... existing items ...
    elastic_conformal_anomaly, elastic_nonconformity, ConformalAnomalyConfig, ConformalAnomalyResult,
};
```

#### `prelude.rs` — Add to tolerance block

Current block (`prelude.rs:89–92`) [VERIFIED: prelude.rs:89–92]:
```rust
pub use crate::tolerance::{
    ElasticToleranceBandResult, ElasticToleranceConfig, PhaseToleranceBand, ToleranceBand,
};
```

**Extend:**
```rust
pub use crate::tolerance::{
    ElasticToleranceBandResult, ElasticToleranceConfig, PhaseToleranceBand, ToleranceBand,
    ConformalAnomalyConfig, ConformalAnomalyResult,
};
```

### 7. Config and Result Struct Conventions

#### `ConformalAnomalyConfig` — Template for builder-style config

Following `ElasticToleranceConfig` and `ConformalConfig` conventions [VERIFIED: tolerance/types.rs:77–110; conformal/mod.rs:116–135]:

```rust
/// Configuration for [`elastic_conformal_anomaly`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ConformalAnomalyConfig {
    /// Non-conformity score variant (must be an elastic variant).
    pub variant: NonConformityScore,
    /// Miscoverage level — flag when p_value <= alpha (default: 0.1).
    pub alpha: f64,
    /// Warp penalty passed to the elastic distance functions (default: 0.0).
    pub lambda: f64,
    /// Maximum iterations for Karcher mean (used when template is None; default: 20).
    pub max_iter: usize,
    /// Convergence tolerance for Karcher mean (default: 1e-4).
    pub tol: f64,
    /// Optional pre-computed template curve (length m).
    /// When None, the Karcher mean of the calibration set is used.
    pub template: Option<Vec<f64>>,
}

impl Default for ConformalAnomalyConfig {
    fn default() -> Self {
        Self {
            variant: NonConformityScore::CombinedElastic,
            alpha: 0.1,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
            template: None,
        }
    }
}
```

#### `ConformalAnomalyResult` — Per-curve outputs

```rust
/// Result of [`elastic_conformal_anomaly`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ConformalAnomalyResult {
    /// Conformal p-value for each test curve (length n_test).
    pub p_values: Vec<f64>,
    /// Nonconformity score for each test curve (length n_test).
    pub scores: Vec<f64>,
    /// Boolean anomaly flag for each test curve (true = anomaly; length n_test).
    pub flags: Vec<bool>,
    /// Calibrated threshold: (1-alpha) quantile of calibration scores.
    pub threshold: f64,
}
```

### 8. Public API Skeleton

```rust
/// Compute elastic nonconformity score for a single curve against a template.
///
/// Scores the curve against the template using the specified elastic distance variant.
/// Returns a non-negative value; near zero when the curve is identical to the template.
///
/// # Arguments
/// * `curve` — Curve to score (length m)
/// * `template` — Reference/template curve (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Warp penalty (0.0 = no penalty)
/// * `variant` — Must be `AmplitudeElastic`, `PhaseElastic`, or `CombinedElastic`
///
/// # Errors
/// Returns `FdarError::InvalidParameter` if `variant` is `SupNorm` or `L2`.
pub fn elastic_nonconformity(
    curve: &[f64],
    template: &[f64],
    argvals: &[f64],
    lambda: f64,
    variant: NonConformityScore,
) -> Result<f64, FdarError>
```

```rust
/// Inductive conformal anomaly detection for functional data using elastic distances.
///
/// Calibrates elastic nonconformity scores on `calibration`, then for each test
/// curve in `test` computes a conformal p-value and an anomaly flag at level `alpha`.
///
/// # Arguments
/// * `calibration` — Reference/calibration functional data (n_calib × m)
/// * `test` — Test functional data to score (n_test × m)
/// * `argvals` — Evaluation points (length m)
/// * `config` — [`ConformalAnomalyConfig`]
///
/// # Returns
/// [`ConformalAnomalyResult`] with per-curve p-values, scores, flags, and threshold.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_conformal_anomaly(
    calibration: &FdMatrix,
    test: &FdMatrix,
    argvals: &[f64],
    config: &ConformalAnomalyConfig,
) -> Result<ConformalAnomalyResult, FdarError>
```

---

## Architecture Patterns

### System Architecture Diagram

```
Caller
  │
  ├─ elastic_conformal_anomaly(calibration, test, argvals, config)
  │      │
  │      ├─ [template resolution]
  │      │    ├─ config.template.as_ref() → Some(t) → use directly
  │      │    └─ None → karcher_mean(calibration, argvals, max_iter, tol, lambda)
  │      │                └─ KarcherMeanResult.mean  (Vec<f64>)
  │      │
  │      ├─ [calibration scoring] for each calibration curve i:
  │      │    └─ elastic_nonconformity(curve_i, template, argvals, lambda, variant)
  │      │         ├─ AmplitudeElastic → amplitude_distance(curve, template, argvals, lambda)
  │      │         ├─ PhaseElastic    → phase_distance_pair(curve, template, argvals, lambda)
  │      │         └─ CombinedElastic → elastic_distance(curve, template, argvals, lambda)
  │      │
  │      ├─ [threshold] sort_nan_safe + calibrated_threshold(sorted_calib_scores, alpha)
  │      │
  │      └─ [test scoring] for each test curve j:
  │           ├─ elastic_nonconformity(test_j, template, ...)  → a*
  │           ├─ p_value = (1 + #{a_i >= a*}) / (n_calib + 1)
  │           └─ flag = p_value <= alpha
  │
  └─ ConformalAnomalyResult { p_values, scores, flags, threshold }
```

### Recommended Project Structure

```
fdars-core/src/tolerance/
├── conformal.rs          (EXISTING — unchanged, add early guard for elastic variants)
├── conformal_anomaly.rs  (NEW — all ECA logic here)
├── degras.rs
├── elastic.rs
├── equivalence.rs
├── exponential.rs
├── fpca.rs
├── helpers.rs
├── mod.rs                (wire conformal_anomaly mod + re-exports)
├── tests.rs
└── types.rs              (extend NonConformityScore)
```

### Anti-Patterns to Avoid

- **Reimplementing elastic distances:** Do NOT write any SRSF or DP-alignment logic in `conformal_anomaly.rs`. Call `crate::alignment::amplitude_distance` / `phase_distance_pair` / `elastic_distance` verbatim.
- **Using `conformal/mod.rs::conformal_quantile` from tolerance module:** It is `pub(super)`, not accessible. Replicate the order-statistic formula locally in `conformal_anomaly.rs` (4 lines).
- **Symmetric scoring:** The elastic distance is NOT symmetric (`elastic_distance(f1, f2)` ≠ `elastic_distance(f2, f1)` in general). Always score the **test curve** against the **template**: `elastic_distance(curve, template, ...)` where f1=curve, f2=template. (The alignment optimally warps f2 onto f1.)
- **Forgetting `#[non_exhaustive]` on result struct:** Both config and result must be `#[non_exhaustive]` for forward compatibility.
- **Missing `#[must_use]`:** `elastic_conformal_anomaly` is expensive; mark it.
- **Panic on invalid `variant`:** Return `FdarError::InvalidParameter`, never panic.
- **Using `f64::NAN` in scores:** Sort is NaN-safe via `sort_nan_safe`; do not propagate NaN from distance functions (they should be finite for valid inputs).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Elastic distance to template | Custom SRSF / DP alignment | `amplitude_distance`, `phase_distance_pair`, `elastic_distance` | Already tested, tuned, parallelized |
| Karcher mean template | Custom mean iteration | `crate::alignment::karcher_mean` | Mature implementation with convergence tracking |
| Sorting scores | Custom sort | `crate::helpers::sort_nan_safe` | NaN-safe, already imported elsewhere in tolerance module |
| Quantile extraction | Custom order statistics | `crate::helpers::quantile_sorted` or inline `calibrated_threshold` helper | Already public in `crate::helpers` |

---

## Common Pitfalls

### Pitfall 1: Elastic Distance Asymmetry
**What goes wrong:** `elastic_distance(curve, template)` ≠ `elastic_distance(template, curve)`. If the argument order is swapped, calibration and test scores are on different scales.
**Why it happens:** The DP warps `f2` onto `f1`; the optimization is asymmetric.
**How to avoid:** Always use `(curve, template, argvals, lambda)` — the test/calibration curve is `f1`, the template is `f2`. Verify: a curve scored against itself should give near-zero.
**Warning signs:** Self-scoring of calibration curves against the Karcher mean is not near-zero; flag rate is not ≈ alpha on clean data.

### Pitfall 2: Elastic Variants in `conformal_prediction_band`
**What goes wrong:** After adding the new enum variants, `conformal_prediction_band` passes an elastic variant to `nonconformity_score`, which will hit an unhandled arm.
**Why it happens:** The band function has no template; it was written before elastic variants existed.
**How to avoid:** Add an early guard at the top of `conformal_prediction_band` that returns `None` for elastic variants. Also add a `_ => unreachable!()` defensive arm to `nonconformity_score`'s match (the early guard makes it dead code, but the match must still compile).
**Warning signs:** Compile error: "non-exhaustive patterns: `AmplitudeElastic`, `PhaseElastic`, `CombinedElastic` not covered".

### Pitfall 3: Self-Distance Not Exactly Zero
**What goes wrong:** Test `elastic_nonconformity(curve, curve, ...)` expecting `== 0.0` but getting small non-zero values (~1e-2 to 1e-6).
**Why it happens:** SRSF computation and DP alignment use discrete grids; numerical round-off is unavoidable.
**How to avoid:** The zero-for-identical gate must use `< 1e-6` (or appropriate small epsilon), not exact equality. The existing test uses `< 0.1` [VERIFIED: alignment/tests.rs:619].
**Warning signs:** Test panics at `assert_eq!(score, 0.0)`.

### Pitfall 4: `conformal_quantile` Not Accessible
**What goes wrong:** Attempting to use `crate::conformal::conformal_quantile` from within `tolerance/conformal_anomaly.rs`.
**Why it happens:** `conformal_quantile` in `conformal/mod.rs` is `pub(super)`, not `pub(crate)`. [VERIFIED: conformal/mod.rs:161]
**How to avoid:** Implement `calibrated_threshold` inline in `conformal_anomaly.rs` using `crate::helpers::sort_nan_safe` + the 4-line order-statistic formula.

### Pitfall 5: Karcher Mean Argument Order
**What goes wrong:** `karcher_mean(calibration, argvals, lambda, max_iter, tol)` — wrong order.
**Why it happens:** Signature is `(data, argvals, max_iter, tol, lambda)`.
**How to avoid:** [VERIFIED: alignment/karcher.rs:293–301]. Always verify: `karcher_mean(calibration, argvals, 20, 1e-4, 0.0)`.

### Pitfall 6: Clippy Non-Exhaustive Match Warning
**What goes wrong:** Clippy warns or errors on the match in `nonconformity_score` after adding new variants.
**Why it happens:** CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
**How to avoid:** Update the match arms before running clippy. Add a `_ => unreachable!()` arm (or handle elastics with an early guard — see Pitfall 2).

---

## Code Examples

### Module Doctest (calibrate → flag)

```rust
//! ```
//! use fdars_core::simulation::{sim_fundata, EFunType, EValType};
//! use fdars_core::tolerance::{
//!     elastic_conformal_anomaly, ConformalAnomalyConfig, NonConformityScore,
//! };
//!
//! let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
//!
//! // Build a calibration set of clean curves
//! let calibration = sim_fundata(30, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
//! // Build a test set of clean curves (exchangeable with calibration)
//! let test = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(99));
//!
//! let mut config = ConformalAnomalyConfig::default();
//! config.variant = NonConformityScore::CombinedElastic;
//! config.alpha = 0.1;
//!
//! let result = elastic_conformal_anomaly(&calibration, &test, &t, &config).unwrap();
//! assert_eq!(result.p_values.len(), 20);
//! assert_eq!(result.flags.len(), 20);
//! assert!(result.threshold >= 0.0);
//! // On clean exchangeable data, flag rate should be approximately alpha
//! let flag_rate = result.flags.iter().filter(|&&f| f).count() as f64 / 20.0;
//! assert!(flag_rate <= 0.4, "Flag rate {} too high for alpha=0.1", flag_rate);
//! ```
```

### `elastic_nonconformity` zero-for-identical gate

```rust
let curve: Vec<f64> = (0..50).map(|i| (i as f64 / 49.0 * 6.0).sin()).collect();
let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
let score = elastic_nonconformity(&curve, &curve, &t, 0.0, NonConformityScore::CombinedElastic).unwrap();
assert!(score < 1e-6, "Self-score should be near zero, got {score}");
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + doctest |
| Config file | none (inline `#[cfg(test)] mod tests`) |
| Quick run command | `cargo test -p fdars-core --features linalg tolerance::conformal_anomaly` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Location |
|--------|----------|-----------|-------------------|----------|
| ECA-01 | `elastic_nonconformity` near-zero for identical curve | unit | `cargo test -p fdars-core --features linalg test_elastic_nonconformity_self_near_zero` | `tolerance/tests.rs` or inline in `conformal_anomaly.rs` |
| ECA-01 | `elastic_nonconformity` non-negative for all variants | unit | `cargo test -p fdars-core --features linalg test_elastic_nonconformity_nonneg` | `tolerance/tests.rs` |
| ECA-01 | `elastic_nonconformity` returns `Err` for `SupNorm`/`L2` variants | unit | `cargo test -p fdars-core --features linalg test_elastic_nonconformity_invalid_variant` | `tolerance/tests.rs` |
| ECA-02 | `conformal_prediction_band` returns `None` for elastic variants | regression | `cargo test -p fdars-core --features linalg test_conformal_band_rejects_elastic_variants` | `tolerance/tests.rs` |
| ECA-02 | Marginal validity: flag rate ≈ alpha on clean exchangeable data | statistical | `cargo test -p fdars-core --features linalg test_elastic_conformal_marginal_validity` | `tolerance/tests.rs` |
| ECA-02 | Magnitude outlier (scaled curve) flagged by `AmplitudeElastic` | unit | `cargo test -p fdars-core --features linalg test_elastic_conformal_magnitude_outlier` | `tolerance/tests.rs` |
| ECA-02 | Shape outlier (phase-shifted curve) flagged by `PhaseElastic` | unit | `cargo test -p fdars-core --features linalg test_elastic_conformal_shape_outlier` | `tolerance/tests.rs` |
| ECA-02 | `CombinedElastic` flags both magnitude and shape outliers | unit | `cargo test -p fdars-core --features linalg test_elastic_conformal_combined_outliers` | `tolerance/tests.rs` |
| ECA-03 | `ConformalAnomalyResult` fields have correct dimensions | unit | `cargo test -p fdars-core --features linalg test_conformal_anomaly_result_shape` | `tolerance/tests.rs` |
| ECA-03 | Module doctest runs under `cargo test --doc` | doctest | `cargo test -p fdars-core --features linalg --doc` | `tolerance/conformal_anomaly.rs` module docstring |

### Test Design Details

**Marginal validity test:** Use n_calib = 100, n_test = 100, alpha = 0.1 with `sim_fundata`. Expected: `flag_rate` in [0.0, 0.25] (generous tolerance for small test set). A tighter tolerance of [0.02, 0.20] is appropriate for 100 test curves with 100 calibration curves.

**Magnitude outlier:** Construct f_outlier = 5.0 * mean_curve (amplitude 5× the reference). Score with `AmplitudeElastic`. Assert score > threshold.

**Shape outlier:** Construct f_outlier by shifting argvals (phase shift ~0.2). Score with `PhaseElastic`. Assert score > threshold. Alternative: use `sim_fundata` with a different `EFunType` (Fourier vs Wiener) to get curves with genuinely different shape.

**Band regression test:** Confirm `conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::AmplitudeElastic, 42).is_none()` for all three elastic variants.

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg conformal_anomaly`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tolerance/conformal_anomaly.rs` — module does not exist yet; entire file is Wave 0
- [ ] Tests in `tolerance/tests.rs` — add 9 new test functions covering the test map above
- [ ] `tolerance/types.rs` — 3 new enum variants on `NonConformityScore`
- [ ] `tolerance/conformal.rs` — early guard for elastic variants + defensive match arm
- [ ] `tolerance/mod.rs` — `mod conformal_anomaly` + re-exports
- [ ] `lib.rs` — extend tolerance re-export block
- [ ] `prelude.rs` — extend tolerance re-export block

---

## Security Domain

> `security_enforcement` is enabled (absent key = enabled; config confirms `"security_enforcement": true`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (library, no auth) |
| V3 Session Management | no | n/a (pure computation) |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | `FdarError::InvalidDimension` checks at function entry (n_calib >= 1, m match, alpha in (0,1)) |
| V6 Cryptography | no | n/a |

### Known Threat Patterns for Rust numeric library

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Division by zero in p-value formula | Tampering | Guard `n_calib > 0` at function entry; return `Err(InvalidParameter)` |
| NaN propagation from distance functions | Tampering | `sort_nan_safe` handles NaN; validate `alpha > 0.0 && alpha < 1.0` |
| Integer overflow in `(n_calib + 1)` | Tampering | Use `as f64` cast, `usize` addition is safe for realistic sizes |
| Out-of-bounds row access in calibration loop | Tampering | `data.row(i)` panics if `i >= nrows`; use safe iteration over `0..n_calib` |

---

## Environment Availability

This phase is purely additive Rust code with no external tools required beyond the existing toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All compilation | yes | 1.97.0 | — |
| `cargo clippy --all-targets --features linalg,parallel` | CI gate | yes | bundled | — |
| `cargo fmt` | Per-commit | yes | bundled | — |

**No missing dependencies.** [ASSUMED — standard Rust toolchain availability on this machine]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `elastic_distance(curve, template)` with curve = template gives near-zero (< 1e-6 for typical smooth curves on 50+ grid points) | Pitfall 3, Test Design | Zero-for-identical gate may fail for certain curve types; set tolerance to 1e-4 as fallback |
| A2 | `amplitude_distance` is fully identical to `elastic_distance` in all semantic properties (it literally delegates) | Standard Stack | No risk — confirmed by reading the 3-line implementation [VERIFIED: pairwise.rs:384–386] |
| A3 | Marginal validity flag rate ≈ 0.10 with n_test=100 and n_calib=100 — wide enough tolerance [0.02, 0.20] to pass deterministically | Test Design | Statistical test may flake; seed-fix or widen tolerance if needed |
| A4 | `#[non_exhaustive]` allows adding variants to `NonConformityScore` without breaking existing downstream consumers | Integration | No risk — Rust language guarantee |
| A5 | The `conformal/mod.rs::conformal_quantile` function uses an order-statistic formula that is equivalent to the standard inductive conformal p-value | Math | Confirmed by reading implementation [VERIFIED: conformal/mod.rs:161–172]; no risk |

---

## Open Questions

1. **Argument order in `elastic_nonconformity`: (curve, template) vs (template, curve)?**
   - What we know: `elastic_distance(f1, f2)` optimally warps `f2` onto `f1`; scoring a test curve against the template means we want the template as the reference (warping the test curve onto it), i.e., `elastic_distance(template, curve)` OR `elastic_distance(curve, template)` — the choice affects asymmetry.
   - What's unclear: The arXiv paper (2504.01172) specifies scoring as "distance from curve to template"; which argument plays the reference role is convention-specific.
   - Recommendation: Use `elastic_distance(curve, template, argvals, lambda)` (curve = f1, template = f2) — this warps the template to fit the curve, which is the natural "how far is the test curve from the reference" interpretation. The zero-for-identical test will catch any argument-flip bug.

2. **Karcher mean of calibration vs separate template for calibration scoring?**
   - What we know: The CONTEXT.md decision says "compute the calibration Karcher mean" as the default template.
   - What's unclear: Should calibration curves be scored against the Karcher mean of the full calibration set (including themselves), or is there a leave-one-out concern?
   - Recommendation: Score all calibration curves against the Karcher mean computed from all of them. This is the standard inductive conformal setup where the template is computed on the calibration set itself; the finite-sample guarantee still holds because the test curves are independent of the calibration set. No LOO needed.

---

## Sources

### Primary (HIGH confidence — read this session via Read tool)

- `fdars-core/src/tolerance/types.rs:26–33` — `NonConformityScore` enum definition (verbatim)
- `fdars-core/src/tolerance/conformal.rs:1–117` — `conformal_prediction_band` full implementation (match sites, quantile convention)
- `fdars-core/src/tolerance/mod.rs:1–43` — module wiring and re-export pattern
- `fdars-core/src/alignment/pairwise.rs:103–105, 384–392` — `elastic_distance`, `amplitude_distance`, `phase_distance_pair` signatures
- `fdars-core/src/alignment/karcher.rs:293–301` — `karcher_mean` signature and return type
- `fdars-core/src/alignment/mod.rs:148–167` — `KarcherMeanResult` field definitions
- `fdars-core/src/helpers.rs:302–317` — `quantile_sorted` implementation
- `fdars-core/src/conformal/mod.rs:116–172` — `ConformalConfig` template + `conformal_quantile` implementation
- `fdars-core/src/lib.rs:346–353` — tolerance re-export block
- `fdars-core/src/prelude.rs:89–92` — prelude tolerance block
- `fdars-core/src/alignment/tests.rs:612–624, 1720–1767` — zero-for-identical and distance property tests
- `fdars-core/src/warping.rs:140–160` — `phase_distance` (Returns 0 for identity warp)

### Secondary (MEDIUM confidence)

- arXiv 2504.01172 (Adams, Berman, Michalenko & Tucker) — inductive conformal formula; not directly fetched this session but the formula is pinned in CONTEXT.md [ASSUMED from CONTEXT.md + standard conformal literature]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all functions read directly from source files this session
- Architecture: HIGH — all integration points verified from source
- Math: HIGH — inductive conformal formula is standard; codebase implementation read verbatim
- Pitfalls: HIGH — all pitfalls derived from reading actual source code (match sites, visibility modifiers, test tolerances)

**Research date:** 2026-09-05
**Valid until:** 2026-10-05 (stable codebase; 30-day window)
