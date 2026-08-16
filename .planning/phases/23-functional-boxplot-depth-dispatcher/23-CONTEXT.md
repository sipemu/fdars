# Phase 23: Functional Boxplot & Depth Dispatcher - Context

**Gathered:** 2026-08-16
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — canonical functional-boxplot definition set as default; reuse anchors verified. Independent of Phase 22.

<domain>
## Phase Boundary

Two small additive capabilities (T-02), both wrapping existing depth functions: (1) a unified **`functional_depth(data, method: DepthMethod)`** dispatcher, and (2) the **depth-fence functional boxplot** with numeric central-region / whisker / outlier-flag outputs (no plotting). Covers T-02. Does NOT touch basis/smoothing (T-01 → Phase 22, done). Numeric-only.
</domain>

<decisions>
## Implementation Decisions

### `functional_depth` dispatcher
- `functional_depth(data: &FdMatrix, method: DepthMethod) -> Result<Vec<f64>, FdarError>` in `depth/` (new `depth/dispatch.rs`), computing **self-depth** (each curve's depth w.r.t. the sample, i.e. `data_obj == data_ori`). Returns a `Vec<f64>` (one depth per curve) — matches the `Vec<f64>` return of the underlying functions.
- `DepthMethod` enum (Debug/Clone/Copy/PartialEq, `#[non_exhaustive]`) carrying per-method params: `FraimanMuniz { scale: bool }`, `Band`, `ModifiedBand`, `RandomProjection { nproj: usize, seed: u64 }`. Dispatch to the existing `fraiman_muniz_1d(data, data, scale)`, `band_1d(data, data)`, `modified_band_1d(data, data)`, `random_projection_1d_seeded(data, data, nproj, seed)`.
- Crate-root re-export `functional_depth` + `DepthMethod`.

### Depth-fence functional boxplot (canonical López-Pintado–Romo / Sun–Genton)
- `functional_boxplot(data: &FdMatrix, method: DepthMethod, factor: f64) -> Result<FunctionalBoxplotResult, FdarError>` (default `method = ModifiedBand`, `factor = 1.5`). Algorithm (numeric outputs only):
  1. Rank curves by `functional_depth`; the **median curve** = deepest.
  2. **50% central region** = pointwise envelope (min/max at each t) of the deepest 50% of curves.
  3. **Fence/whiskers** = central region inflated by `factor × (central-region width)` at each t.
  4. **Outliers** = curves that exceed the fence at *any* evaluation point (`Vec<usize>` of row indices).
- `FunctionalBoxplotResult { median: Vec<f64>, central_lower: Vec<f64>, central_upper: Vec<f64>, whisker_lower: Vec<f64>, whisker_upper: Vec<f64>, outliers: Vec<usize>, depths: Vec<f64> }` (derive Debug/Clone/PartialEq, serde-gated). Place in `outliers.rs` (alongside `outliergram`/`magnitude_shape_outlyingness`) or a new `depth/boxplot.rs` — executor's discretion. Crate-root re-export.
- **Latitude:** the backlog also phrased the whisker as "1.5×IQR of depth values" — the canonical *envelope-inflation* form above is the default (it is *the* functional boxplot); an alternative depth-IQR outlier flag may be added if trivial, but the envelope form is required. Document the definition in rustdoc.

### Conventions
- `Result<_, FdarError>`, validate inputs (non-empty, ≥ a few curves for a meaningful central region); inline `#[cfg(test)] mod tests`; crate-root re-export; existing depth functions' signatures unchanged (dispatcher only wraps them). NO `#[must_use]` on `Result` fns. Numeric only — NO plotting.

### Claude's Discretion
- File placement (`outliers.rs` vs `depth/boxplot.rs`), the exact `FunctionalBoxplotResult` field set, and whether to add the optional depth-IQR variant.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (verified)
- Depth functions (→ `Vec<f64>`, `(data_obj, data_ori, …)`): `fraiman_muniz_1d` (`depth/fraiman_muniz.rs:32`, has `scale: bool`), `band_1d` / `modified_band_1d` (`depth/band.rs:30/43`), `random_projection_1d` / `random_projection_1d_seeded` (`depth/random_projection.rs:32/38`, `nproj`, seeded variant for determinism).
- `outliers.rs` neighbors: `outliergram` → `OutligramResult` (`:278`), `magnitude_shape_outlyingness` → `MagnitudeShapeResult` (`:352`), `detect_outliers_lrt`. The functional boxplot is a NEW numeric outlier detector alongside these.
- `FdMatrix` row methods for per-curve/per-column access; `fdata::mean_1d`.

### Established Patterns
- `Result<T, FdarError>`, input validation, inline tests, crate-root re-export, serde-gated result structs. CI: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. Build/test: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` (MEMORY). Seeded RNG (`random_projection_1d_seeded`) for determinism.

### Integration Points
- Independent of Phase 22 (disjoint modules). Both additive. Final phase of v0.20.0.
</code_context>

<specifics>
## Specific Ideas

- Test correctness (mandatory): `functional_depth(data, DepthMethod::FraimanMuniz{scale})` == `fraiman_muniz_1d(data, data, scale)` (and the same per-method equality for Band/ModifiedBand/RandomProjection with a fixed seed); the functional boxplot **flags a planted gross outlier curve** and does not flag inliers; median == the deepest curve; central region brackets the median; deterministic under a fixed seed for the RandomProjection method; invalid input → `Err`.
</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of the boxplot → out of scope (numeric outputs only).
- Other depth methods / combinators (EXPL-01 pluggable-metric depth) → later backlog item.
</deferred>
