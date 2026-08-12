# Phase 14: Shift Registration - Pattern Map

**Mapped:** 2026-08-12
**Files analyzed:** 4 new/modified files
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/alignment/shift.rs` | utility/algorithm | batch transform | `fdars-core/src/alignment/set.rs` | exact (same role: set-level registration; same data flow: parallel-per-curve collect + sequential assemble) |
| `fdars-core/src/alignment/quality.rs` | utility/algorithm | batch transform | `fdars-core/src/alignment/quality.rs` lines 63–161 (existing `alignment_quality`) | exact (same file; new functions sit beside `alignment_quality`, `warp_smoothness`) |
| `fdars-core/src/alignment/mod.rs` | config/barrel | — | `fdars-core/src/alignment/mod.rs` lines 86–107 (existing `pub use quality/set` blocks) | exact |
| `fdars-core/src/lib.rs` | config/barrel | — | `fdars-core/src/lib.rs` lines 139–170 (alignment `pub use` block) | exact |

---

## Pattern Assignments

### `fdars-core/src/alignment/shift.rs` (new file, algorithm, batch-transform)

**Analog:** `fdars-core/src/alignment/set.rs`

**Imports pattern** (`set.rs` lines 1–9):
```rust
use super::pairwise::elastic_align_pair;
use super::srsf::reparameterize_curve;
use super::{AlignmentResult, AlignmentSetResult};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

For `shift.rs`, adapt as:
```rust
//! Least-squares shift (rigid horizontal) registration.

use crate::error::FdarError;
use crate::helpers::{linear_interp, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

**Result struct pattern** (`alignment/mod.rs` lines 131–141):
```rust
/// Result of aligning a set of curves to a common target.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentSetResult {
    /// Warping functions (n × m).
    pub gammas: FdMatrix,
    /// Aligned curves (n × m).
    pub aligned_data: FdMatrix,
    /// Elastic distances for each curve.
    pub distances: Vec<f64>,
}
```

Model `ShiftRegistrationResult` on this exactly:
```rust
/// Result of least-squares shift registration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShiftRegistrationResult {
    /// Registered (shifted) functional data (n × m).
    pub registered_data: FdMatrix,
    /// Per-curve horizontal shifts δᵢ applied to each curve.
    pub shifts: Vec<f64>,
}
```

Note: `AlignmentSetResult` in `mod.rs` does NOT have `#[cfg_attr(feature = "serde", ...)]`. Per-convention check other result types (e.g. `BayesianAlignmentResult`), but the CONTEXT.md locks serde on this struct so include it.

**Core parallel-collect-then-assemble pattern** (`set.rs` lines 57–83 — the canonical template):
```rust
pub fn align_to_target(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> AlignmentSetResult {
    let (n, m) = data.shape();

    let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_align_pair(target, &fi, argvals, lambda)
        })
        .collect();

    let mut gammas = FdMatrix::zeros(n, m);
    let mut aligned_data = FdMatrix::zeros(n, m);
    let mut distances = Vec::with_capacity(n);

    for (i, r) in results.into_iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = r.gamma[j];
            aligned_data[(i, j)] = r.f_aligned[j];
        }
        distances.push(r.distance);
    }

    AlignmentSetResult { gammas, aligned_data, distances }
}
```

For `least_squares_shift_registration`, the parallel map collects `Vec<(f64, Vec<f64>)>` (shift, shifted_row), then a sequential loop assembles `FdMatrix`:
```rust
let results: Vec<(f64, Vec<f64>)> = iter_maybe_parallel!(0..n)
    .map(|i| {
        let row = data.row(i);
        let delta = golden_section_search(
            |d| l2_shift_objective(&row, argvals, &mean, &weights, d),
            -max_shift, max_shift, TOL, MAX_ITER,
        );
        let shifted: Vec<f64> = argvals.iter()
            .map(|&t| linear_interp(argvals, &row, t - delta))
            .collect();
        (delta, shifted)
    })
    .collect();

let mut registered_data = FdMatrix::zeros(n, m);
let mut shifts = Vec::with_capacity(n);
for (i, (delta, row)) in results.into_iter().enumerate() {
    for j in 0..m {
        registered_data[(i, j)] = row[j];
    }
    shifts.push(delta);
}
```

**Error handling / input validation pattern** (from `quality.rs` and `set.rs` style — no explicit example in set.rs since `align_to_target` returns bare struct, but new function returns `Result`):

Follow the `alignment_quality` function in `quality.rs` for shape extraction:
```rust
let (n, m) = data.shape();
// validate at entry:
if n == 0 || m == 0 {
    return Err(FdarError::InvalidDimension { ... });
}
if argvals.len() != m {
    return Err(FdarError::InvalidDimension { ... });
}
if argvals.len() < 2 {
    return Err(FdarError::InvalidParameter { ... });
}
if max_shift <= 0.0 {
    return Err(FdarError::InvalidParameter { ... });
}
```

**Doc comment + example pattern** (`set.rs` lines 23–50 — module-level doc + `# Examples` block with `cargo test`-runnable doctest):
```rust
/// Align all curves in `data` to a single target curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `target` — Target curve to align to (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// [`AlignmentSetResult`] with all warping functions, aligned curves, and distances.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::alignment::align_to_target;
/// ...
/// ```
```

**Anti-pattern: no `#[must_use]` on `Result`-returning functions.** `align_to_target` carries `#[must_use]` only because it returns a bare struct (line 50: `#[must_use = "expensive computation whose result should not be discarded"]`). `least_squares_shift_registration` returns `Result<_, _>`, which is already `#[must_use]` by std — omit the attribute.

**Inline test structure** (`set.rs` has no inline tests; tests are in `alignment/tests.rs`). For `shift.rs`, follow the CONTEXT.md decision to use inline `#[cfg(test)]`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    fn gaussian_bump(argvals: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        argvals.iter().map(|&t| (-(t - mu).powi(2) / (2.0 * sigma * sigma)).exp()).collect()
    }

    #[test]
    fn test_shift_already_aligned() { ... }
    // ... remaining tests
}
```

---

### `fdars-core/src/alignment/quality.rs` (modify — append three score functions)

**Analog:** `fdars-core/src/alignment/quality.rs` (same file; nearest neighbors are `alignment_quality` lines 63–161 and `warp_smoothness` lines 43–55)

**Imports already present** (`quality.rs` lines 1–8) — no new imports needed:
```rust
use crate::helpers::{gradient_uniform, l2_distance, simpsons_weights};
use crate::matrix::FdMatrix;
```

Add `use crate::error::FdarError;` at the top (not yet present — check: `quality.rs` does not currently import `FdarError` since existing functions return raw types). The three new score functions return `Result<f64, FdarError>`, so this import must be added.

**Pattern for `mean_1d` + `simpsons_weights` usage** (`quality.rs` lines 69–99):
```rust
let (n, m) = data.shape();
let weights = simpsons_weights(argvals);
let orig_mean = crate::fdata::mean_1d(data);
let total_var: f64 = (0..n)
    .map(|i| {
        let fi = data.row(i);
        let d = l2_distance(&fi, &orig_mean, &weights);
        d * d
    })
    .sum::<f64>()
    / n as f64;
```

`least_squares_score` is exactly this pattern without the `l2_distance` wrapper (expand `d*d` inline):
```rust
pub fn least_squares_score(registered: &FdMatrix, argvals: &[f64]) -> Result<f64, FdarError> {
    let (n, m) = registered.shape();
    // validate ...
    let weights = simpsons_weights(argvals);
    let mean = crate::fdata::mean_1d(registered);
    let score = (0..n)
        .map(|i| {
            let fi = registered.row(i);
            fi.iter().zip(mean.iter()).zip(weights.iter())
                .map(|((&a, &b), &w)| (a - b) * (a - b) * w)
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;
    Ok(score)
}
```

**Pattern for `gradient_uniform` (Sobolev term)** (`quality.rs` lines 43–55):
```rust
pub fn warp_smoothness(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 3 { return 0.0; }
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_prime = gradient_uniform(gamma, h);
    let gam_pprime = gradient_uniform(&gam_prime, h);
    let integrand: Vec<f64> = gam_pprime.iter().map(|&g| g * g).collect();
    crate::helpers::trapz(&integrand, argvals)
}
```

`sobolev_least_squares_score` uses `gradient_uniform` identically for the derivative term:
```rust
let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
let mean_prime = gradient_uniform(&mean, h);
// per curve:
let fi_row = registered.row(i);
let fi_prime = gradient_uniform(&fi_row.to_vec(), h);
```

**Return type deviation note** — document in rustdoc that new score functions return `Result<f64, FdarError>` while existing neighbors (`warp_complexity`, `warp_smoothness`, `pairwise_consistency`) return raw types. This is intentional per CONTEXT.md to enable dimension validation.

**No `#[must_use]` on new functions** — `warp_complexity` and `warp_smoothness` (`quality.rs` lines 38, 43) carry no `#[must_use]`. The new `Result`-returning functions also omit it (double `#[must_use]` lint).

---

### `fdars-core/src/alignment/mod.rs` (modify — register module + re-exports)

**Analog:** `fdars-core/src/alignment/mod.rs` (same file)

**Module registration pattern** (lines 14–40, existing `mod` declarations):
```rust
mod bayesian;
mod closed;
mod clustering;
// ...
mod quality;
mod robust_karcher;
mod set;
mod shape;
// ...
```

Add `mod shift;` in alphabetical order between `mod set;` and `mod shape;`.

**Re-export block for new items** — model on the existing `quality` and `set` re-export lines (lines 86–92):
```rust
pub use quality::{
    alignment_quality, pairwise_consistency, warp_complexity, warp_smoothness, AlignmentQuality,
};
// ...
pub use set::{align_to_target, elastic_decomposition, DecompositionResult};
```

Add new `pub use shift` block after `pub use set`:
```rust
pub use shift::{least_squares_shift_registration, ShiftRegistrationResult};
```

Extend the `pub use quality` block with three new score names:
```rust
pub use quality::{
    alignment_quality, least_squares_score, pairwise_consistency, pairwise_correlation_score,
    sobolev_least_squares_score, warp_complexity, warp_smoothness, AlignmentQuality,
};
```

**Serialization order constraint:** Both `mod.rs` re-export edits (shift + quality extension) must happen in the same plan step to avoid merge collision. The RESEARCH.md explicitly calls this out.

---

### `fdars-core/src/lib.rs` (modify — crate-root re-exports)

**Analog:** `fdars-core/src/lib.rs` lines 139–170

**Existing alignment `pub use` block** (lines 139–170 — complete excerpt):
```rust
pub use alignment::{
    align_to_target, alignment_quality, amplitude_distance, amplitude_self_distance_matrix,
    bayesian_align_pair, compose_warps, curve_geodesic, curve_geodesic_nd, cut_dendrogram,
    diagnose_alignment, diagnose_pairwise, elastic_align_pair, elastic_align_pair_closed,
    elastic_align_pair_constrained, elastic_align_pair_multires, elastic_align_pair_nd,
    elastic_align_pair_penalized, elastic_align_pair_with_landmarks, elastic_cross_distance_matrix,
    elastic_cross_distance_matrix_banded, elastic_cross_distance_matrix_with_band,
    elastic_decomposition, elastic_depth, elastic_distance, elastic_distance_closed,
    elastic_distance_nd, elastic_outlier_detection, elastic_partial_match,
    elastic_self_distance_matrix, elastic_self_distance_matrix_banded,
    elastic_self_distance_matrix_with_band, gauss_model, hierarchical_from_distances, horiz_fpns,
    invert_warp, joint_gauss_model, karcher_covariance_nd, karcher_mean, karcher_mean_banded,
    karcher_mean_closed, karcher_mean_nd, karcher_mean_with_band, karcher_median,
    kmedoids_from_distances, lambda_cv, orbit_representative, pairwise_consistency, pca_nd,
    peak_persistence, phase_boxplot, phase_distance_pair, phase_self_distance_matrix,
    reparameterize_curve, robust_karcher_mean, shape_confidence_interval, shape_distance,
    shape_mean, shape_self_distance_matrix, srsf_inverse, srsf_inverse_nd, srsf_transform,
    srsf_transform_nd, transfer_alignment, tsrvf_from_alignment, tsrvf_from_alignment_with_method,
    tsrvf_inverse, tsrvf_transform, tsrvf_transform_with_method, warp_complexity,
    warp_inverse_error, warp_smoothness, warp_statistics, AlignmentDiagnostic,
    AlignmentDiagnosticSummary, AlignmentQuality, AlignmentResult, AlignmentResultNd,
    AlignmentSetResult, BayesianAlignConfig, BayesianAlignmentResult, ClosedAlignmentResult,
    ClosedKarcherMeanResult, ConstrainedAlignmentResult, DecompositionResult, Dendrogram,
    DiagnosticConfig, ElasticDepthResult, ElasticOutlierConfig, ElasticOutlierResult, FpnsResult,
    GenerativeModelResult, GeodesicPath, GeodesicPathNd, KMedoidsConfig, KMedoidsResult,
    KarcherMeanResult, KarcherMeanResultNd, LambdaCvConfig, LambdaCvResult, Linkage,
    MultiresConfig, OrbitRepresentative, PartialMatchConfig, PartialMatchResult, PcaNdResult,
    PersistenceDiagramResult, PhaseBoxplot, RobustKarcherConfig, RobustKarcherResult,
    ShapeCiConfig, ShapeCiResult, ShapeDistanceResult, ShapeMeanResult, ShapeQuotient,
    TransferAlignConfig, TransferAlignResult, TransportMethod, TsrvfResult, WarpPenaltyType,
    WarpStatistics,
};
```

Add the following five items to this block (functions alphabetically with existing snake_case names; types alphabetically with existing PascalCase names):

Functions to insert (alphabetical position):
- `least_squares_score` — after `lambda_cv`
- `least_squares_shift_registration` — after `least_squares_score`
- `pairwise_correlation_score` — after `pairwise_consistency`
- `sobolev_least_squares_score` — after `shape_self_distance_matrix`

Type to insert:
- `ShiftRegistrationResult` — after `ShapeMeanResult`

**Convention note:** The existing block is one flat alphabetical list mixing snake_case functions and PascalCase types together. Insert new names in alphabetical order within that single flat list — do not create a separate block.

**Serialization order constraint:** `lib.rs` re-export edit must follow after `mod.rs` edits are complete (single serialized plan step covers both files, per RESEARCH.md pitfall 2).

---

## Shared Patterns

### Error handling (input validation)
**Source:** Inferred from `FdarError` usage across `fdars-core/src/scalar_on_function/fregre_lm.rs` and `classification/` — the pattern for `alignment/` specifically draws from `alignment_quality` implicit shape extraction (`quality.rs` lines 68–69).
**Apply to:** `least_squares_shift_registration`, `least_squares_score`, `pairwise_correlation_score`, `sobolev_least_squares_score`

Pattern:
```rust
let (n, m) = data.shape();
if n == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "non-empty matrix".to_string(),
        actual: format!("{}×{}", n, m),
    });
}
if argvals.len() != m {
    return Err(FdarError::InvalidDimension {
        parameter: "argvals",
        expected: m,
        actual: argvals.len(),
    });
}
```

### Simpson integration + mean combination
**Source:** `fdars-core/src/alignment/quality.rs` lines 69–99 (`alignment_quality`)
**Apply to:** `least_squares_score`, `sobolev_least_squares_score`
```rust
let weights = simpsons_weights(argvals);
let mean = crate::fdata::mean_1d(registered);
```

### Uniform-grid derivative
**Source:** `fdars-core/src/alignment/quality.rs` lines 43–55 (`warp_smoothness`)
**Apply to:** `sobolev_least_squares_score`
```rust
let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
let fi_prime = gradient_uniform(&fi_row, h);
```

### Parallel collect + sequential assemble
**Source:** `fdars-core/src/alignment/set.rs` lines 59–83 (`align_to_target`)
**Apply to:** `least_squares_shift_registration`
```rust
let results: Vec<_> = iter_maybe_parallel!(0..n).map(|i| { ... }).collect();
// then sequential loop to fill FdMatrix
for (i, item) in results.into_iter().enumerate() { ... }
```

### Module barrel re-export style
**Source:** `fdars-core/src/alignment/mod.rs` lines 86–107
**Apply to:** `alignment/mod.rs` extension
```rust
pub use quality::{existing_fns, new_fn1, new_fn2, ...};
pub use shift::{least_squares_shift_registration, ShiftRegistrationResult};
```

---

## No Analog Found

All four files have close analogs. No entries here.

---

## Metadata

**Analog search scope:** `fdars-core/src/alignment/` (set.rs, quality.rs, mod.rs), `fdars-core/src/lib.rs`
**Files scanned:** 4 (set.rs, quality.rs, alignment/mod.rs, lib.rs)
**Pattern extraction date:** 2026-08-12
